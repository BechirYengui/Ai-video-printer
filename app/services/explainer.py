"""YVP : vidéos explicatives animées (schémas, flèches, sous-titres mot à mot).

L'IA ne dessine pas : elle écrit un *plan de scènes* (éléments, échanges,
phrase dite à chaque scène) que le moteur resource/explainer/engine.html anime
de façon fiable. Étapes :

1. faits : l'IA propose des articles Wikipédia, on garde ceux qui existent
   vraiment ; leur résumé sert de référence (pas de faits plutôt que des faux) ;
2. plan : écrit par le modèle choisi dans les paramètres, puis vérifié et
   réparé (types d'icônes, échanges cohérents avec la voix…) ;
3. icônes : bibliothèque maison + 5 166 icônes Tabler (MIT) ; l'IA choisit
   parmi les meilleures candidates ;
4. voix : edge-tts, avec le minutage de chaque mot ;
5. rendu : Chrome/Edge isolé (sans réseau) capture chaque image, ffmpeg
   assemble le MP4 1080×1920.

Le rendu tourne dans un fil d'exécution à part : la WebUI suit sa progression
via ``job_status``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from typing import Callable
from urllib.parse import urlencode
from uuid import uuid4

from loguru import logger

from app.utils import utils

KINDS = ["phone", "laptop", "browser", "server", "database", "cloud", "lock", "user",
         "globe", "api", "router", "cache", "shield", "file"]
# Types inventés par le modèle → icône la plus proche de la bibliothèque maison.
SYNONYMS = {
    "network": "globe", "internet": "globe", "web": "globe", "wifi": "router", "smartphone": "phone",
    "mobile": "phone", "app": "phone", "application": "phone", "computer": "laptop", "pc": "laptop",
    "desktop": "laptop", "db": "database", "storage": "database", "data": "database", "security": "shield",
    "firewall": "shield", "encryption": "lock", "password": "lock", "client": "user", "person": "user",
    "people": "user", "website": "browser", "webpage": "browser", "code": "api", "backend": "server",
    "document": "file", "memory": "cache",
}
# Mots courants absents des noms/tags Tabler → mot qui y figure.
ICON_WORDS = {"parcel": "package", "colis": "package", "order": "shopping", "customer": "user",
              "money": "cash", "notification": "bell", "email": "mail"}
RETURN_WORDS = re.compile(
    r"\b(renvoi\w*|retourn\w*|répond\w*|reviens?\w*|send\w* back|return\w*|repl\w*|respond\w*)", re.I
)
LANGUAGES = {
    "fr": ("French", "fr-FR-DeniseNeural"), "en": ("English", "en-US-AvaMultilingualNeural"),
    "es": ("Spanish", "es-ES-ElviraNeural"), "de": ("German", "de-DE-KatjaNeural"),
    "it": ("Italian", "it-IT-ElsaNeural"), "pt": ("Portuguese", "pt-BR-FranciscaNeural"),
    "ar": ("Arabic", "ar-SA-ZariyahNeural"), "tr": ("Turkish", "tr-TR-EmelNeural"),
    "ru": ("Russian", "ru-RU-SvetlanaNeural"), "zh": ("Chinese", "zh-CN-XiaoxiaoNeural"),
}
MAX_ELEMENTS = 5
FPS = 30
GAP = 0.25

PROMPT = """You plan a 20-second animated explainer video (vertical, TikTok style) about: "{topic}".
Narration language: {language}. Write ALL texts (title, labels, narration) in {language}.

Return ONLY a JSON object with this exact shape:
{{
  "title": "short catchy title, max 6 words",
  "elements": [
    {{"id": "a", "kind": "phone", "label": "max 2 words", "icon": "1 to 3 English keywords"}}
  ],
  "scenes": [
    {{"narration": "one spoken sentence, 8 to 14 words",
      "show": ["a"],
      "flow": {{"from": "a", "to": "b", "label": "max 3 words"}},
      "highlight": "b"}}
  ]
}}

Rules:
- 2 to 5 elements. "kind" is one of: {kinds}. If none fits (a drone, a car, a parcel, a satellite...),
  use "kind": "other". In every case "icon" gives 1 to 3 English keywords describing the object
  (e.g. "drone", "delivery truck", "charging station"): an icon library is searched with them.
- 4 or 5 scenes. Total narration about 45 words (20 seconds when read aloud).
- "show" lists the elements visible from this scene on (introduce them progressively).
- "flow" (optional, or null) is a message or data moving between two shown elements, with a short label
  such as a protocol or data name. Use flows to explain how things communicate.
- "highlight" (optional, or null) is the element the narration talks about.
- Tell ONE clear story in order, like a teacher: e.g. the request goes from the client to the server,
  the server asks the database, the database answers the server, the server answers the client.
  Each flow must make sense technically; reuse the same elements instead of adding vague ones.
- The first scene hooks the viewer; the last scene concludes.
"""


class ExplainerError(Exception):
    """Plan impossible à obtenir ou rendu impossible (message montrable)."""


def output_dir() -> str:
    override = os.environ.get("YVP_EXPLAINER_DIR", "").strip()
    if override:
        # Tests et instances d'essai : jamais les vidéos de l'utilisateur.
        os.makedirs(override, exist_ok=True)
        return override
    return utils.storage_dir("explainer", create=True)


def resource(name: str) -> str:
    return os.path.join(utils.resource_dir("explainer"), name)


def language_of(script_language: str) -> tuple[str, str, str]:
    """(code, nom anglais, voix Edge par défaut) pour la langue du script."""
    code = str(script_language or "fr").split("-")[0].lower()
    name, voice = LANGUAGES.get(code, ("English", "en-US-AvaMultilingualNeural"))
    return code, name, voice


# --- Modèle IA ------------------------------------------------------------------------
def _ask_json(prompt: str) -> dict:
    """Réponse JSON du modèle choisi dans les paramètres (Ollama, en ligne…)."""
    from app.config import config
    from app.services import llm

    provider = str(config.app.get("llm_provider", "")).lower()
    model = str(config.app.get(f"{provider}_model_name", "")).lower()
    if provider == "ollama" and model.startswith("qwen3"):
        # Interrupteur de qwen3 : réponse directe, sans phase de réflexion
        # (120 s → ~15 s pour un plan, pour une qualité équivalente ici).
        prompt += "\n/no_think"
    response = llm._generate_response(prompt)
    if not isinstance(response, str) or response.startswith("Error: "):
        raise ExplainerError(f"le modèle IA n'a pas répondu : {response}")
    text = llm._strip_code_fence(response)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def fetch_facts(topic: str, script_language: str) -> tuple[str, list[str]]:
    """Résumés Wikipédia des articles que l'IA juge utiles (et qui existent)."""
    code, language, _ = language_of(script_language)

    def api(params: dict) -> dict:
        url = f"https://{code}.wikipedia.org/w/api.php?" + urlencode({**params, "format": "json"})
        request = urllib.request.Request(
            url, headers={"User-Agent": "YVP-explainer/1.0 (local educational video tool; based on MoneyPrinterTurbo)"}
        )
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=15) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code != 429 or attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))  # Wikipédia limite le débit

    def words(text: str) -> set[str]:
        return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}

    try:
        wanted = _ask_json(
            f'Topic of an educational video: "{topic}". Give the 1 or 2 most relevant {language} '
            'Wikipedia article titles that explain it (exact article names, e.g. "Domain Name System"). '
            'Return ONLY JSON like {"titles": ["..."]}.'
        ).get("titles") or []
        titles = []
        for title in [str(t) for t in wanted][:2]:
            pages = api({"action": "query", "titles": title, "redirects": 1})["query"]["pages"]
            found = next((p["title"] for p in pages.values() if "missing" not in p and "invalid" not in p), None)
            if not found:
                hits = api({"action": "query", "list": "search", "srsearch": title, "srlimit": 1})["query"]["search"]
                if hits and words(hits[0]["title"]) & words(title):
                    found = hits[0]["title"]
            if found and found not in titles:
                titles.append(found)
        if not titles:
            return "", []
        pages = api({"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
                     "exsentences": 6, "titles": "|".join(titles)})["query"]["pages"]
        text = "\n\n".join(f"{p['title']}: {p.get('extract', '').strip()}" for p in pages.values() if p.get("extract"))
        return text[:1800], titles
    except Exception as exc:  # pas de réseau, API indisponible, réponse inattendue…
        logger.warning(f"explainer: Wikipedia facts unavailable: {exc}")
        return "", []


# --- Icônes ---------------------------------------------------------------------------
_TABLER: dict | None = None


def _tabler() -> dict:
    global _TABLER
    if _TABLER is None:
        with open(resource("tabler-outline.json"), encoding="utf-8") as handle:
            _TABLER = json.load(handle)
    return _TABLER


def icon_candidates(*hints: str, limit: int = 5) -> list[str]:
    words = [w for h in hints for w in re.findall(r"[a-z0-9]+", str(h or "").lower()) if len(w) > 2]
    words = [ICON_WORDS.get(w, w) for w in words]
    if not words:
        return []
    scored = []
    for name, icon in _tabler().items():
        parts = name.split("-")
        score = 12 if name == "-".join(words) else 0
        score += sum(4 for w in words if w in parts)
        score += sum(3 for w in words if w in icon["t"])
        score += sum(1 for w in words if w in name and w not in parts)
        if parts[-1] == "off" and "off" not in words:
            score -= 6  # variantes barrées (« drone-off »)
        score -= len(parts) * 0.1
        if score >= 2.5:
            scored.append((score, name))
    return [name for _, name in sorted(scored, reverse=True)[:limit]]


def icon_svg(name: str) -> str:
    paths = _tabler()[name]["s"]
    return (f'<g transform="translate(14 14) scale(3)" fill="none" stroke-linecap="round" stroke-linejoin="round">'
            f'<g stroke="#8C70FF" stroke-width="3.6" opacity=".45">{paths}</g>'
            f'<g stroke="#F1EDFF" stroke-width="1.7">{paths}</g></g>')


def _keywords(value) -> str:
    """Mots-clés d'icône en texte simple (le modèle renvoie parfois une liste)."""
    if isinstance(value, (list, tuple)):
        value = " ".join(str(v) for v in value)
    return " ".join(re.sub(r"[\[\]'\",]", " ", str(value or "")).split())[:40]


def set_element_icon(element: dict, keywords: str) -> None:
    """Icône d'un élément d'après des mots-clés (bibliothèque maison d'abord)."""
    keywords = _keywords(keywords)
    word = SYNONYMS.get(keywords.lower(), keywords.lower())
    for key in ("svg", "icon_name", "candidates"):
        element.pop(key, None)
    element["icon"] = keywords
    if word in KINDS:
        element["kind"] = word
        return
    candidates = icon_candidates(keywords) or icon_candidates(element.get("label", ""))
    if candidates:
        element.update(kind="custom", candidates=candidates, icon_name=candidates[0], svg=icon_svg(candidates[0]))
    else:
        element["kind"] = "generic"


def choose_icons(plan: dict, topic: str) -> None:
    """L'IA choisit la plus juste parmi les icônes candidates de chaque élément."""
    pending = [e for e in plan["elements"] if len(e.get("candidates") or []) > 1]
    if not pending:
        return
    lines = [
        f'- element "{e["id"]}" labelled "{e["label"]}" ({e.get("icon", "")}): '
        + "; ".join(f"{n} ({', '.join(_tabler()[n]['t'][:6])})" for n in e["candidates"])
        for e in pending
    ]
    prompt = (f'Video topic: "{topic}". For each element, pick the icon that represents it best.\n'
              + "\n".join(lines)
              + '\nReturn ONLY JSON like {"choices": {"element id": "icon name"}} using the exact icon names.')
    try:
        choices = _ask_json(prompt).get("choices") or {}
    except Exception as exc:
        logger.warning(f"explainer: icon choice unavailable: {exc}")
        return
    for e in pending:
        pick = choices.get(e["id"])
        if pick in e["candidates"]:
            e["icon_name"], e["svg"] = pick, icon_svg(pick)


# --- Plan -------------------------------------------------------------------------------
def repair_plan(raw: dict) -> dict:
    """Vérifie et répare un plan (IA ou édité) ; lève ExplainerError s'il est inutilisable."""
    elements, ids = [], set()
    for item in (raw.get("elements") or [])[:MAX_ELEMENTS]:
        if not isinstance(item, dict):
            continue
        eid = str(item.get("id") or f"e{len(elements)}")[:20]
        if eid in ids:
            continue
        element = {"id": eid, "label": " ".join(str(item.get("label") or eid).split()[:3])[:22]}
        if item.get("svg") and item.get("icon_name") in _tabler():
            # Élément déjà résolu (plan édité) : on garde l'icône choisie.
            element.update(kind="custom", svg=icon_svg(item["icon_name"]), icon_name=item["icon_name"],
                           icon=_keywords(item.get("icon")), candidates=item.get("candidates") or [item["icon_name"]])
        else:
            raw_kind = str(item.get("kind") or "").lower()
            kind = SYNONYMS.get(raw_kind, raw_kind)
            if kind in KINDS:
                element.update(kind=kind, icon=_keywords(item.get("icon")) or kind)
            else:
                set_element_icon(element, _keywords(item.get("icon")) or raw_kind or element["label"])
        elements.append(element)
        ids.add(eid)
    if len(elements) < 2:
        raise ExplainerError("le plan doit contenir au moins 2 éléments")

    scenes, shown, pairs = [], [], set()
    for item in (raw.get("scenes") or [])[:6]:
        if not isinstance(item, dict):
            continue
        narration = " ".join(str(item.get("narration") or "").split())
        if len(narration.split()) < 3:
            continue
        for eid in item.get("show") or []:
            if eid in ids and eid not in shown:
                shown.append(eid)
        flow = item.get("flow") if isinstance(item.get("flow"), dict) else None
        if flow and (flow.get("from") not in ids or flow.get("to") not in ids or flow.get("from") == flow.get("to")):
            flow = None
        if flow:
            for end in (flow["from"], flow["to"]):
                if end not in shown:
                    shown.append(end)
            flow = {"from": flow["from"], "to": flow["to"],
                    "label": " ".join(str(flow.get("label") or "").split()[:4])[:26]}
            # « renvoie la réponse » → la flèche revient vers celui qui a demandé.
            if RETURN_WORDS.search(narration):
                askers = [a for (a, b) in pairs if b == flow["from"] and a != flow["from"]]
                if askers and flow["to"] not in askers:
                    flow["to"] = askers[0]
            flow["reverse"] = (flow["to"], flow["from"]) in pairs
            pairs.add((flow["from"], flow["to"]))
        highlight = item.get("highlight") if item.get("highlight") in shown else None
        if flow and highlight not in (flow["from"], flow["to"]):
            highlight = flow["to"]
        scenes.append({"narration": narration, "show": list(shown), "flow": flow, "highlight": highlight})
    if not 2 <= len(scenes) <= 6:
        raise ExplainerError(f"le plan doit contenir de 2 à 6 scènes ({len(scenes)} utilisables)")
    order = {eid: i for i, eid in enumerate(shown)}
    elements = sorted((e for e in elements if e["id"] in order), key=lambda e: order[e["id"]])
    title = " ".join(str(raw.get("title") or "").split()[:8]) or "Explication"
    plan = {"title": title, "elements": elements, "scenes": scenes}
    if raw.get("sources"):
        plan["sources"] = list(raw["sources"])
    return plan


def generate_plan(topic: str, script_language: str, use_facts: bool = True) -> dict:
    """Plan complet (faits, scènes, icônes) ; jusqu'à 3 essais."""
    topic = " ".join(str(topic or "").split())
    if not topic:
        raise ExplainerError("sujet vide")
    _, language, _ = language_of(script_language)
    facts, sources = fetch_facts(topic, script_language) if use_facts else ("", [])
    prompt = PROMPT.format(topic=topic, language=language, kinds=", ".join(KINDS))
    if facts:
        prompt += ("\nReference facts (from Wikipedia). Stay consistent with them, simplify for a young "
                   "audience, and do not invent technical details that contradict them:\n" + facts + "\n")
    last_error = None
    for attempt in range(1, 4):
        try:
            plan = repair_plan(_ask_json(prompt))
            choose_icons(plan, topic)
            plan["sources"] = sources
            plan["topic"] = topic
            logger.info(f"explainer: plan ready (attempt {attempt}): {len(plan['elements'])} elements, "
                        f"{len(plan['scenes'])} scenes, sources={sources}")
            return plan
        except (ExplainerError, ValueError, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning(f"explainer: plan rejected (attempt {attempt}): {exc}")
    raise ExplainerError(f"aucun plan utilisable après 3 essais : {last_error}")


# --- Voix ---------------------------------------------------------------------------------
async def _speak(text: str, voice: str, rate: str, path: str) -> list[dict]:
    import edge_tts

    words = []
    communicate = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    with open(path, "wb") as handle:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                handle.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                start = chunk["offset"] / 1e7
                words.append({"text": chunk["text"], "start": start, "end": start + chunk["duration"] / 1e7})
    if not words or not os.path.getsize(path):
        raise ExplainerError(f"la voix {voice} n'a rien produit (langue du texte différente ?)")
    return words


def _duration(path: str) -> float:
    out = subprocess.run([utils.get_ffmpeg_binary(), "-hide_banner", "-i", path],
                         capture_output=True, text=True).stderr
    match = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    if not match:
        raise ExplainerError("durée de la voix illisible")
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)


def edge_voice_for(voice_name: str, script_language: str) -> str:
    """Voix Edge à utiliser : celle choisie si elle convient, sinon celle de la langue."""
    from app.services import voice as voice_service

    name = voice_service.parse_voice_name(voice_name or "")
    is_edge = bool(re.match(r"^[a-z]{2,3}-[A-Z]{2}-\w+Neural$", name))
    if is_edge and not voice_service.voice_language_mismatch(name, script_language):
        return name
    return language_of(script_language)[2]


def _voice_over(plan: dict, voice: str, rate: float, work: str) -> str:
    rate_text = f"{int(round((rate - 1) * 100)):+d}%"
    t, parts = 0.4, []
    for i, scene in enumerate(plan["scenes"]):
        path = os.path.join(work, f"scene{i}.mp3")
        words = asyncio.run(_speak(scene["narration"], voice, rate_text, path))
        length = _duration(path)
        scene["start"], scene["end"] = t, t + length + GAP
        scene["words"] = [{"text": w["text"], "start": t + w["start"], "end": t + w["end"]} for w in words]
        parts.append((path, t))
        t = scene["end"]
    plan["scenes"][-1]["end"] += 0.8  # dernière image tenue un instant
    audio = os.path.join(work, "voice.m4a")
    inputs, filters = [], []
    for i, (path, start) in enumerate(parts):
        inputs += ["-i", path]
        filters.append(f"[{i}:a]adelay={int(start * 1000)}|{int(start * 1000)}[a{i}]")
    mix = "".join(f"[a{i}]" for i in range(len(parts)))
    subprocess.run([utils.get_ffmpeg_binary(), "-y", "-loglevel", "error", *inputs, "-filter_complex",
                    ";".join(filters) + f";{mix}amix=inputs={len(parts)}:normalize=0[out]",
                    "-map", "[out]", "-c:a", "aac", "-b:a", "160k", audio], check=True, capture_output=True)
    return audio


# --- Rendu ----------------------------------------------------------------------------------
def _launch_browser(playwright):
    """Chrome sous Linux, Edge sous Windows : le navigateur déjà installé."""
    errors = []
    for channel in ("chrome", "msedge", None):
        try:
            return playwright.chromium.launch(channel=channel, headless=True) if channel else \
                playwright.chromium.launch(headless=True)
        except Exception as exc:
            errors.append(f"{channel or 'chromium'}: {str(exc).splitlines()[0]}")
    raise ExplainerError("aucun navigateur pour le rendu (Chrome ou Edge) : " + " | ".join(errors))


def render(plan: dict, voice_name: str, script_language: str, rate: float = 1.0,
           progress: Callable[[str, float], None] | None = None) -> str:
    """Rend la vidéo ; renvoie le chemin du MP4 (storage/explainer)."""
    from playwright.sync_api import sync_playwright

    report = progress or (lambda phase, fraction: None)
    plan = json.loads(json.dumps(plan))  # copie : on y ajoute le minutage
    work = tempfile.mkdtemp(prefix="yvp-explainer-")
    voice = edge_voice_for(voice_name, script_language)
    report("voice", 0.0)
    audio = _voice_over(plan, voice, rate, work)
    total = plan["scenes"][-1]["end"]
    frames = int(total * FPS)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = os.path.join(output_dir(), f"explainer-{stamp}-{uuid4().hex[:6]}.mp4")
    temp = f"{out}.part.mp4"
    ffmpeg = subprocess.Popen([
        utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", str(FPS), "-i", "-", "-i", audio,
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
        "-af", "apad", "-c:a", "aac", "-b:a", "160k", "-t", f"{frames / FPS:.3f}",
        "-movflags", "+faststart", temp,
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    errors = []
    try:
        with sync_playwright() as p:
            browser = _launch_browser(p)
            context = browser.new_context(viewport={"width": 1080, "height": 1920}, offline=True)
            # Isolé : seul le moteur local est chargé, aucun accès réseau.
            context.route("**/*", lambda r: r.continue_() if r.request.url.startswith("file:") else r.abort())
            page = context.new_page()
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto("file:///" + resource("engine.html").replace(os.sep, "/").lstrip("/"))
            page.evaluate("plan => window.init(plan)", plan)
            for i in range(frames):
                page.evaluate(f"window.seek({i / FPS})")
                ffmpeg.stdin.write(page.screenshot(type="jpeg", quality=93))
                if i % 15 == 0:
                    report("render", i / frames)
            browser.close()
        ffmpeg.stdin.close()
        if ffmpeg.wait() != 0 or errors:
            raise ExplainerError(f"rendu impossible : {errors[:2] or ffmpeg.stderr.read().decode()[-300:]}")
        os.replace(temp, out)
    finally:
        if ffmpeg.poll() is None:
            ffmpeg.kill()
        if os.path.exists(temp):
            os.remove(temp)
    with open(out.replace(".mp4", ".plan.json"), "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    report("done", 1.0)
    logger.info(f"explainer: video ready: {out}")
    return out


# --- Tâches de rendu en arrière-plan ---------------------------------------------------------
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def start_render(plan: dict, voice_name: str, script_language: str, rate: float = 1.0) -> str:
    job_id = uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "phase": "voice", "fraction": 0.0, "started": time.time()}

    def update(**fields):
        with _jobs_lock:
            _jobs[job_id].update(fields)

    def run():
        try:
            path = render(plan, voice_name, script_language, rate,
                          progress=lambda phase, fraction: update(phase=phase, fraction=fraction))
            update(status="done", path=path, fraction=1.0)
        except Exception as exc:
            logger.exception(f"explainer: render failed: {exc}")
            update(status="failed", error=str(exc))

    threading.Thread(target=run, name=f"explainer-{job_id[:6]}", daemon=True).start()
    return job_id


def job_status(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


# --- Vidéos produites -------------------------------------------------------------------------
_NAME_RE = re.compile(r"^explainer-\d{8}-\d{6}-[0-9a-f]{6}\.mp4$")


def list_videos() -> list[dict]:
    folder = output_dir()
    items = []
    for name in sorted((n for n in os.listdir(folder) if _NAME_RE.match(n)), reverse=True):
        path = os.path.join(folder, name)
        title = ""
        try:
            with open(path.replace(".mp4", ".plan.json"), encoding="utf-8") as handle:
                title = json.load(handle).get("title", "")
        except (OSError, ValueError):
            pass
        items.append({"name": name, "path": path, "title": title, "size": os.path.getsize(path),
                      "created": datetime.fromtimestamp(os.path.getmtime(path))})
    return items


def delete_video(name: str) -> None:
    if not _NAME_RE.match(name or ""):
        raise ExplainerError(f"vidéo inconnue : {name}")
    path = os.path.join(output_dir(), name)
    for target in (path, path.replace(".mp4", ".plan.json")):
        if os.path.exists(target):
            os.remove(target)
