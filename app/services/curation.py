"""YVP : sélection des visuels par un modèle de vision local et repli sur des
photos libres de droits (Openverse, qui indexe aussi Wikimedia Commons).

Le principe : les banques de vidéos renvoient beaucoup de candidats par mot-clé,
mais leur ordre ne tient pas compte de la phrase réellement prononcée. Un petit
modèle de vision local (Ollama) décrit la miniature de chaque candidat — il voit
bien mais note mal — puis le LLM principal juge ces descriptions par rapport au
passage du script. Si aucun clip n'est assez pertinent pour un passage, on cherche
une photo sous licence libre, jugée de la même façon, que le montage animera par
un zoom lent.

Tout est optionnel et dégradé en douceur : sans Ollama, sans modèle de vision ou
en cas d'erreur réseau, la liste d'origine est conservée telle quelle.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List

import requests
from loguru import logger

from app.config import config
from app.models.schema import MaterialInfo, VideoAspect
from app.utils import utils

DEFAULT_VISION_MODEL = "gemma3:4b"
DEFAULT_MIN_SCORE = 5
DEFAULT_MAX_CANDIDATES = 6
OPENVERSE_API_URL = "https://api.openverse.org/v1/images/"
_SCORE_PATTERN = re.compile(r"\b(10|[0-9])\b")
_THUMBNAIL_TIMEOUT = (10, 20)
_VISION_TIMEOUT = (10, 120)
_DESCRIBE_PROMPT = (
    "Describe in one short sentence what this image shows: main subject, action "
    "and setting. Mention any visible text or logo. No preamble."
)


@dataclass
class CurationContext:
    """Ce que le modèle de vision doit savoir pour juger un visuel."""

    video_subject: str = ""
    video_script: str = ""
    match_script_order: bool = False
    search_terms: List[str] = field(default_factory=list)
    # Plans tirés des photos/vidéos de l'utilisateur (user_assets.AssetShot).
    user_shots: List[Any] = field(default_factory=list)
    # Niveau de qualité de la tâche (quality.PRESETS) ; None = config.toml seule.
    quality: str | None = None


def _app_config() -> dict:
    return config.app if isinstance(config.app, dict) else {}


def is_enabled() -> bool:
    return bool(_app_config().get("curation_enabled", False))


def openverse_fallback_enabled() -> bool:
    return bool(_app_config().get("curation_openverse_fallback", True))


def _min_score() -> int:
    try:
        return max(0, min(10, int(_app_config().get("curation_min_score", DEFAULT_MIN_SCORE))))
    except (TypeError, ValueError):
        return DEFAULT_MIN_SCORE


def _max_candidates() -> int:
    try:
        return max(1, int(_app_config().get("curation_max_candidates", DEFAULT_MAX_CANDIDATES)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_CANDIDATES


def _vision_model() -> str:
    return str(_app_config().get("curation_vision_model") or DEFAULT_VISION_MODEL).strip()


def _ollama_native_base_url() -> str:
    """L'API native d'Ollama (/api/chat) accepte les images en base64."""
    base_url = str(_app_config().get("ollama_base_url") or "").strip()
    if not base_url:
        base_url = config.get_default_ollama_base_url()
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        base_url = base_url[: -len("/v1")]
    return base_url


def vision_model_available() -> bool:
    """Vérifie une seule fois par appel que le modèle de vision est installé."""
    try:
        response = requests.get(f"{_ollama_native_base_url()}/api/tags", timeout=(5, 10))
        response.raise_for_status()
        names = {m.get("name", "") for m in response.json().get("models", [])}
    except Exception as exc:
        logger.warning(f"curation disabled: Ollama unreachable ({type(exc).__name__}: {exc})")
        return False
    model = _vision_model()
    candidates = {model, f"{model}:latest"} if ":" not in model else {model}
    if not names & candidates:
        logger.warning(f"curation disabled: vision model {model!r} is not installed in Ollama")
        return False
    return True


def split_script_for_terms(script: str, term_count: int) -> List[str]:
    """Associe à chaque mot-clé le passage du script qu'il illustre (ordre du script)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", script or "") if s.strip()]
    if term_count <= 0 or not sentences:
        return [""] * max(term_count, 0)
    passages = []
    for index in range(term_count):
        # Plus de mots-clés que de phrases : plusieurs mots-clés partagent une phrase.
        start = min(round(index * len(sentences) / term_count), len(sentences) - 1)
        end = max(start + 1, round((index + 1) * len(sentences) / term_count))
        passages.append(" ".join(sentences[start:end]))
    return passages


def _description_cache_path(thumbnail_url: str) -> Path:
    digest = hashlib.sha256(f"{_vision_model()}|{thumbnail_url}".encode("utf-8")).hexdigest()
    return Path(utils.storage_dir("cache_curation", create=True)) / f"{digest}.txt"


def _fetch_image_b64(url: str) -> str | None:
    try:
        response = requests.get(
            url,
            timeout=_THUMBNAIL_TIMEOUT,
            headers={"User-Agent": "YenguiVideoPrinter/1.0 (+curation)"},
            proxies=config.proxy,
        )
        response.raise_for_status()
        if not response.content:
            return None
        return base64.b64encode(response.content).decode("ascii")
    except Exception as exc:
        logger.debug(f"thumbnail download failed: {type(exc).__name__}: {exc}")
        return None


def describe_image(thumbnail_url: str) -> str | None:
    """Le modèle de vision décrit ce que montre l'image (il voit bien, mais juge mal).

    Les descriptions ne dépendent que de l'image : elles sont mises en cache sur
    disque, une nouvelle vidéo sur un sujet proche ne les recalcule pas.
    """
    cache_path = _description_cache_path(thumbnail_url)
    try:
        if cache_path.exists():
            cached = cache_path.read_text(encoding="utf-8").strip()
            if cached:
                return cached
    except OSError:
        pass

    image_b64 = _fetch_image_b64(thumbnail_url)
    if not image_b64:
        return None
    description = ask_vision_model(image_b64, _DESCRIBE_PROMPT, num_predict=60)
    if not description:
        return None
    try:
        cache_path.write_text(description, encoding="utf-8")
    except OSError:
        pass
    return description


def ask_vision_model(image_b64: str, prompt: str, num_predict: int = 60) -> str | None:
    """Pose une question au modèle de vision sur une image (base64), sans cache."""
    payload = {
        "model": _vision_model(),
        "messages": [{"role": "user", "content": prompt, "images": [image_b64]}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }
    try:
        response = requests.post(
            f"{_ollama_native_base_url()}/api/chat", json=payload, timeout=_VISION_TIMEOUT
        )
        response.raise_for_status()
        description = response.json().get("message", {}).get("content", "")
    except Exception as exc:
        logger.warning(f"image description failed: {type(exc).__name__}: {exc}")
        return None
    return " ".join(str(description).split())[:400] or None


def parse_score(text: str) -> int | None:
    """Extrait une note 0-10 isolée ; ignore le raisonnement éventuel."""
    if not isinstance(text, str):
        return None
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    match = _SCORE_PATTERN.search(text)
    return int(match.group(1)) if match else None


def build_judge_prompt(context: CurationContext, shot: dict) -> str:
    """Un appel court par passage : un gros prompt groupé fait décrocher les petits modèles."""
    lines = [
        "You are a video editor choosing B-roll for a short vertical video"
        + (f' about "{context.video_subject}".' if context.video_subject else "."),
        f'Shot to fill: "{shot["term"]}"',
    ]
    if shot.get("passage"):
        lines.append(f'Narration: "{shot["passage"]}"')
    lines.append("Candidates:")
    for index, description in enumerate(shot["descriptions"], start=1):
        lines.append(f"{index}. {description}")
    lines.append(
        "Score each candidate from 0 to 10 for how well it illustrates the narration "
        "(0 = unrelated, 3 = same broad theme only, 6 = clearly related, "
        "9 = exactly what is said). Penalize visible text or logos."
    )
    if context.video_subject:
        lines.append(
            f'A candidate unrelated to the video topic ("{context.video_subject}") '
            "scores 2 at most, even if it matches a word of the shot."
        )
    if _judge_is_qwen3():
        # Sans « réflexion » (≈ 50 s par appel), qwen3 note au hasard ; une raison
        # courte avant chaque note suffit à retrouver le même classement en ≈ 5 s.
        lines.append("First give each candidate a reason of at most 8 words, then its score.")
        lines.append('Reply with JSON only: {"reasons": ["one short reason per candidate"], '
                     '"scores": [one integer per candidate, in order]}')
        lines.append("/no_think")
    else:
        lines.append('Reply with JSON only: {"scores": [one integer per candidate, in order]}')
    return "\n".join(lines)


def _judge_is_qwen3() -> bool:
    app_config = _app_config()
    provider = str(app_config.get("llm_provider") or "")
    model = str(app_config.get(f"{provider}_model_name") or "")
    return provider == "ollama" and model.lower().startswith("qwen3")


def parse_judge_scores(text: str, expected: int) -> List[int | None]:
    """Lit {"scores": [...]} ; à défaut, la dernière liste de nombres de la bonne taille."""
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    raw: Any = None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            raw = data.get("scores") if isinstance(data, dict) else None
        except ValueError:
            raw = None
    if not isinstance(raw, list):
        lists = [
            [int(n) for n in re.findall(r"\d+", found)]
            for found in re.findall(r"\[([\d\s,]+)\]", text)
        ]
        lists = [values for values in lists if len(values) == expected]
        raw = lists[-1] if lists else None
    scores: List[int | None] = []
    for index in range(expected):
        value = raw[index] if isinstance(raw, list) and index < len(raw) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 10:
            scores.append(int(round(value)))
        else:
            scores.append(None)
    return scores


def judge_shots(context: CurationContext, shots: List[dict]) -> dict[str, List[int | None]]:
    """Le LLM principal note les descriptions, passage par passage."""
    if not shots:
        return {}
    from app.services import llm  # import tardif : llm importe beaucoup de dépendances

    scores = {}
    for shot in shots:
        try:
            response = llm._generate_response(build_judge_prompt(context, shot))
        except Exception as exc:
            logger.warning(f"curation judge failed: {type(exc).__name__}: {exc}")
            response = ""
        if isinstance(response, str) and response.startswith("Error:"):
            logger.warning(f"curation judge failed: {response[:200]}")
            response = ""
        scores[shot["id"]] = parse_judge_scores(response, len(shot["descriptions"]))
    return scores


def _thumbnail_of(item: MaterialInfo) -> str:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    thumbnail = source.get("thumbnail")
    return thumbnail if isinstance(thumbnail, str) and thumbnail.startswith("https://") else ""


def _describe_candidates(
    items: List[MaterialInfo], used: set, limit: int | None = None
) -> List[tuple[MaterialInfo, str]]:
    """Décrit les premiers candidats encore libres (ceux qui ont une miniature)."""
    limit = _max_candidates() if limit is None else limit
    described = []
    for item in items:
        if len(described) >= limit:
            break
        if _asset_key(item) in used:
            continue
        thumbnail = _thumbnail_of(item)
        if not thumbnail:
            continue
        description = describe_image(thumbnail)
        if not description:
            continue
        tags = (item.source_info or {}).get("tags") if isinstance(item.source_info, dict) else ""
        if isinstance(tags, str) and tags:
            description = f"{description} (tags: {tags[:120]})"
        described.append((item, description))
    return described


def _asset_key(item: MaterialInfo) -> str:
    source = item.source_info if isinstance(item.source_info, dict) else {}
    provider = str(source.get("provider") or item.provider or "")
    asset_id = source.get("asset_id")
    return f"{provider}:{asset_id}" if asset_id not in (None, "") else item.url


def _target_size(video_aspect: VideoAspect) -> tuple[int, int]:
    return VideoAspect(video_aspect).to_resolution()


def _openverse_thumbnail(url: str, proxy_thumbnail: Any) -> str:
    """Vignette 500 px (taille standard imposée par Wikimedia) quand c'est possible ; sinon celle d'Openverse."""
    match = re.match(r"^https://upload\.wikimedia\.org/wikipedia/commons/(\w/\w\w)/([^/?#]+)$", url)
    if match:
        folder, name = match.groups()
        return f"https://upload.wikimedia.org/wikipedia/commons/thumb/{folder}/{name}/500px-{name}"
    return proxy_thumbnail if isinstance(proxy_thumbnail, str) else ""


def search_openverse_images(term: str, video_aspect: VideoAspect, limit: int = 12) -> List[MaterialInfo]:
    """Photos libres de droits, utilisables commercialement et modifiables."""
    target_width, target_height = _target_size(video_aspect)
    params = {
        "q": term,
        "license_type": "commercial,modification",
        "size": "large",
        "mature": "false",
        "page_size": limit,
    }
    try:
        response = requests.get(
            OPENVERSE_API_URL,
            params=params,
            timeout=(10, 30),
            headers={"User-Agent": "YenguiVideoPrinter/1.0 (+openverse)"},
            proxies=config.proxy,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        logger.warning(f"openverse search failed for {term!r}: {type(exc).__name__}: {exc}")
        return []

    items = []
    for result in results if isinstance(results, list) else []:
        if not isinstance(result, dict):
            continue
        url, thumbnail = result.get("url"), result.get("thumbnail")
        width, height = result.get("width") or 0, result.get("height") or 0
        if not (isinstance(url, str) and url.startswith("https://")):
            continue
        # Recadrage centré sans agrandissement : l'image doit couvrir la cible.
        if not (isinstance(width, int) and isinstance(height, int)):
            continue
        if width < target_width or height < target_height:
            continue
        items.append(
            MaterialInfo(
                provider="openverse",
                url=url,
                duration=60,
                source_info={
                    "provider": "openverse",
                    "kind": "image",
                    "search_term": term,
                    "asset_id": str(result.get("id") or ""),
                    "source_page": result.get("foreign_landing_url"),
                    "thumbnail": _openverse_thumbnail(url, thumbnail),
                    "creator": {"name": result.get("creator") or ""},
                    "license": result.get("license") or "",
                    "license_version": result.get("license_version") or "",
                    "license_url": result.get("license_url") or "",
                    "attribution": result.get("attribution") or "",
                    "origin": result.get("source") or "",
                    "rendition": {"id": "original", "width": width, "height": height},
                },
            )
        )
    return items


def _apply_scores(described, scores, min_score):
    """Classe les candidats décrits : bons (≥ seuil), passables (≥ seuil - 2), rejetés."""
    good, fair = [], []
    # Pas de zip : un juge qui renvoie moins de notes ne doit faire perdre aucun candidat.
    for index, (item, _) in enumerate(described):
        score = scores[index] if index < len(scores) else None
        if isinstance(item.source_info, dict) and score is not None:
            item.source_info = dict(item.source_info)
            item.source_info["curation_score"] = score
        if score is None:
            fair.append((-1, item))
        elif score >= min_score:
            good.append((score, item))
        elif score >= min_score - 2:
            fair.append((score, item))
        # sinon : rejeté (hors sujet), jamais utilisé
    good.sort(key=lambda entry: -entry[0])
    fair.sort(key=lambda entry: -entry[0])
    return [item for _, item in good], [item for _, item in fair]


def curate_search_results(
    results: Iterable[tuple[str, List[MaterialInfo]]],
    context: CurationContext | None,
    video_aspect: VideoAspect,
) -> list[tuple[str, List[MaterialInfo]]]:
    """Réordonne les candidats de chaque mot-clé selon leur pertinence.

    1. le modèle de vision décrit les miniatures des premiers candidats ;
    2. le LLM principal note toutes les descriptions en un seul appel ;
    3. si aucun clip n'atteint le seuil pour un passage, on décrit et note des
       photos libres Openverse ;
    4. les candidats non retenus restent en fin de liste comme dernier recours,
       pour que la vidéo ne manque jamais d'images.
    """
    results = list(results)
    if not context or not is_enabled() or not results:
        return results
    limit, use_openverse = _max_candidates(), openverse_fallback_enabled()
    if context.quality:
        from app.services import quality  # import tardif : évite un cycle

        level = quality.preset(context.quality)
        if not level.curation:
            logger.info(f"curation skipped: quality={context.quality}")
            return results
        limit = min(limit, level.max_candidates)
        use_openverse = use_openverse and level.openverse
    if not vision_model_available():
        return results

    terms = [term for term, _ in results]
    passages = (
        split_script_for_terms(context.video_script, len(terms))
        if context.match_script_order
        else [""] * len(terms)
    )
    min_score = _min_score()
    used: set = set()

    described_by_term = []
    for term, items in results:
        described = _describe_candidates(items, used, limit)
        described_by_term.append(described)
    shots = [
        {"id": f"S{index}", "term": term, "passage": passage,
         "descriptions": [d for _, d in described]}
        for index, ((term, _), passage, described) in enumerate(
            zip(results, passages, described_by_term), start=1
        )
        if described
    ]
    scores = judge_shots(context, shots)

    goods, fairs, spares = [], [], []
    for index, ((term, items), described) in enumerate(zip(results, described_by_term), start=1):
        good, fair = _apply_scores(described, scores.get(f"S{index}", []), min_score)
        good = [item for item in good if _asset_key(item) not in used]
        for item in good:
            used.add(_asset_key(item))
        judged = {id(item) for item, _ in described}
        goods.append(good)
        fairs.append([item for item in fair if _asset_key(item) not in used])
        # Candidats jamais évalués : dernier recours seulement.
        spares.append([i for i in items if id(i) not in judged and _asset_key(i) not in used])

    missing = [position for position, good in enumerate(goods) if not good]
    if missing and use_openverse:
        photo_shots, photo_described = [], {}
        for position in missing:
            term, passage = terms[position], passages[position]
            described = _describe_candidates(
                search_openverse_images(term, video_aspect), used, limit
            )
            if described:
                shot_id = f"P{position}"
                photo_described[shot_id] = (position, described)
                photo_shots.append({"id": shot_id, "term": term, "passage": passage,
                                    "descriptions": [d for _, d in described]})
        photo_scores = judge_shots(context, photo_shots)
        for shot_id, (position, described) in photo_described.items():
            good, _ = _apply_scores(described, photo_scores.get(shot_id, []), min_score)
            good = [item for item in good if _asset_key(item) not in used][:2]
            for item in good:
                used.add(_asset_key(item))
            if good:
                goods[position] = good
                logger.info(f"curation: no relevant clip for {terms[position]!r}, "
                            f"using {len(good)} openverse photo(s)")

    # Toujours rien : emprunter un bon clip en surplus d'un autre passage
    # plutôt que de montrer un hors-sujet.
    for position in [p for p, good in enumerate(goods) if not good]:
        donors = sorted(range(len(goods)), key=lambda p: -len(goods[p]))
        for donor in donors:
            if donor != position and len(goods[donor]) > 2:
                borrowed = goods[donor].pop()
                goods[position] = [borrowed]
                logger.info(f"curation: {terms[position]!r} borrows a relevant clip from {terms[donor]!r}")
                break

    curated: list[tuple[str, List[MaterialInfo]]] = []
    for position, term in enumerate(terms):
        items = goods[position] + fairs[position]
        curated.append((term, items or spares[position]))

    for term, items in curated:
        top = [str((i.source_info or {}).get("curation_score", "-")) for i in items[:4]]
        logger.info(f"curation: {term!r} -> best scores {', '.join(top)}")
    return curated


def credits_lines(material_sources: List[dict[str, Any]]) -> List[str]:
    """Lignes de crédits à coller dans la description de la vidéo."""
    lines, seen = [], set()
    for source in material_sources:
        if not isinstance(source, dict):
            continue
        provider = source.get("provider") or ""
        if provider == "user":
            continue  # les assets de l'utilisateur n'ont pas de crédit à citer
        creator = (source.get("creator") or {}).get("name") if isinstance(source.get("creator"), dict) else ""
        page = source.get("source_page") or ""
        key = (provider, page or source.get("asset_id"))
        if key in seen:
            continue
        seen.add(key)
        if source.get("attribution"):
            lines.append(str(source["attribution"]))
            continue
        parts = [p for p in (creator and f"by {creator}", f"via {provider.capitalize()}" if provider else "", page) if p]
        if parts:
            lines.append(" — ".join(parts))
    return lines
