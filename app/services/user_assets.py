"""YVP : photos et vidéos de l'utilisateur pour nourrir le montage.

1. Le modèle de vision local regarde chaque asset (une photo, ou quelques images
   prises dans une vidéo). La légende facultative apporte ce que l'image ne dit
   pas : un nom, un lieu, un message.
2. Ces descriptions servent au LLM pour écrire un script qui parle des assets.
3. Chaque plan est placé sur le passage du script qu'il illustre le mieux ; la
   source de vidéos choisie (Pixabay…) comble les autres passages.

Une vidéo longue donne jusqu'à trois plans pris au début, au milieu et à la fin :
le montage ne garde qu'un plan court par fichier, une seule séquence de 30 s
serait sinon réduite à ses trois premières secondes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List

from loguru import logger
from PIL import Image, ImageOps

from app.models.schema import MaterialInfo, UserAsset
from app.services import curation, material_upload
from app.utils import file_security, utils

MAX_ASSETS = 20
MAX_TOTAL_BYTES = 300 * 1024 * 1024
MAX_SHOTS_PER_VIDEO = 3
_VISION_IMAGE_SIZE = 896
_DESCRIBE_PROMPT = (
    "Describe in one or two short sentences what this image shows: main subject, "
    "action, setting and mood. Mention any visible text or logo. No preamble."
)


@dataclass
class AssetShot:
    """Un plan utilisable : une photo, ou un extrait d'une vidéo de l'utilisateur."""

    asset_path: str
    kind: str  # "image" ou "video"
    caption: str = ""
    start: float = 0.0
    duration: float = 0.0
    description: str = ""

    @property
    def label(self) -> str:
        return self.caption or Path(self.asset_path).name

    def brief(self) -> str:
        """Ce que le LLM sait du plan : la légende d'abord, puis ce que voit l'IA."""
        parts = [p for p in (self.caption.strip(), self.description.strip()) if p]
        return " | ".join(parts) or Path(self.asset_path).name


def _kind_of(path: str) -> str | None:
    """Mêmes extensions que l'upload de fichiers locaux."""
    extension = f".{utils.parse_extension(path)}"
    if extension in material_upload.SUPPORTED_IMAGE_EXTENSIONS:
        return "image"
    if extension in material_upload.SUPPORTED_VIDEO_EXTENSIONS:
        return "video"
    return None


def resolve_assets(assets: Iterable[UserAsset | dict] | None) -> List[UserAsset]:
    """Ne garde que les fichiers sûrs (dans storage/local_videos), dans les limites."""
    base_dir = utils.storage_dir("local_videos", create=True)
    resolved, total_bytes = [], 0
    for asset in assets or []:
        if isinstance(asset, dict):
            asset = UserAsset(**asset)
        try:
            path = file_security.resolve_path_within_directory(base_dir, asset.path)
        except (ValueError, OSError) as exc:
            logger.warning(f"user asset rejected: {asset.path!r} ({exc})")
            continue
        if _kind_of(path) is None:
            logger.warning(f"user asset rejected (unsupported type): {path}")
            continue
        size = os.path.getsize(path)
        if len(resolved) >= MAX_ASSETS or total_bytes + size > MAX_TOTAL_BYTES:
            logger.warning(f"user asset skipped (limit of {MAX_ASSETS} files / "
                           f"{MAX_TOTAL_BYTES // 2**20} MB reached): {path}")
            continue
        total_bytes += size
        resolved.append(UserAsset(path=path, caption=(asset.caption or "").strip()))
    return resolved


def _video_duration(path: str) -> float:
    from app.services import video  # import tardif : video charge MoviePy

    clip = video._open_video_clip_quietly(path)
    try:
        return float(clip.duration or 0)
    finally:
        video.close_clip(clip)


def plan_shots(assets: Iterable[UserAsset], clip_duration: float) -> List[AssetShot]:
    """Une photo donne un plan ; une vidéo jusqu'à trois, répartis sur sa durée."""
    clip_duration = max(1.0, float(clip_duration))
    shots: List[AssetShot] = []
    for asset in assets:
        kind = _kind_of(asset.path)
        if kind == "image":
            shots.append(AssetShot(asset.path, "image", asset.caption, 0.0, clip_duration))
            continue
        try:
            total = _video_duration(asset.path)
        except Exception as exc:
            logger.warning(f"user video unreadable, skipped: {asset.path} ({exc})")
            continue
        if not math.isfinite(total) or total <= 0:
            continue
        count = max(1, min(MAX_SHOTS_PER_VIDEO, int(total // clip_duration)))
        length = min(clip_duration, total)
        last_start = max(0.0, total - length)
        for index in range(count):
            start = last_start * index / (count - 1) if count > 1 else 0.0
            shots.append(AssetShot(asset.path, "video", asset.caption, round(start, 2), length))
    return shots


def _file_key(path: str) -> str:
    stat = os.stat(path)
    return hashlib.sha256(f"{path}|{stat.st_size}|{stat.st_mtime_ns}".encode()).hexdigest()[:24]


def _jpeg_b64(image: Image.Image) -> str:
    image = image.convert("RGB")
    image.thumbnail((_VISION_IMAGE_SIZE, _VISION_IMAGE_SIZE))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _shot_image_b64(shot: AssetShot) -> str:
    if shot.kind == "image":
        with Image.open(shot.asset_path) as image:
            return _jpeg_b64(ImageOps.exif_transpose(image))
    from app.services import video

    clip = video._open_video_clip_quietly(shot.asset_path)
    try:
        middle = min(shot.start + shot.duration / 2, max(0.0, float(clip.duration) - 0.05))
        return _jpeg_b64(Image.fromarray(clip.get_frame(middle)))
    finally:
        video.close_clip(clip)


def _description_cache_path(shot: AssetShot) -> Path:
    key = f"{curation._vision_model()}|{_file_key(shot.asset_path)}|{shot.start}"
    digest = hashlib.sha256(key.encode()).hexdigest()
    return Path(utils.storage_dir("cache_curation", create=True)) / f"user-{digest}.txt"


def describe_shot(shot: AssetShot) -> str:
    """Ce que voit le modèle de vision ; mis en cache tant que le fichier ne change pas."""
    cache_path = _description_cache_path(shot)
    try:
        cached = cache_path.read_text(encoding="utf-8").strip()
        if cached:
            shot.description = cached
            return cached
    except OSError:
        pass
    try:
        image_b64 = _shot_image_b64(shot)
    except Exception as exc:
        logger.warning(f"cannot read user asset for analysis: {shot.asset_path} ({exc})")
        return ""
    description = curation.ask_vision_model(image_b64, _DESCRIBE_PROMPT, num_predict=90) or ""
    if description:
        try:
            cache_path.write_text(description, encoding="utf-8")
        except OSError:
            pass
    shot.description = description
    return description


def describe_shots(shots: List[AssetShot]) -> List[AssetShot]:
    """Analyse tous les plans ; sans modèle de vision, seules les légendes servent."""
    if not shots or not curation.vision_model_available():
        return shots
    for index, shot in enumerate(shots, start=1):
        describe_shot(shot)
        logger.info(f"user asset {index}/{len(shots)} ({shot.kind}): {shot.brief()[:160]}")
    return shots


def prepare(assets: Iterable[UserAsset | dict] | None, clip_duration: float) -> List[AssetShot]:
    """Valide, découpe et analyse les assets d'une tâche."""
    resolved = resolve_assets(assets)
    if not resolved:
        return []
    logger.info(f"\n\n## analysing {len(resolved)} user asset(s)")
    return describe_shots(plan_shots(resolved, clip_duration))


def visual_brief(shots: List[AssetShot]) -> str:
    """Liste des visuels à couvrir, pour le prompt d'écriture du script."""
    return "\n".join(f"{index}. {shot.brief()}" for index, shot in enumerate(shots, start=1))


def build_assignment_prompt(shots: List[AssetShot], passages: List[str], subject: str) -> str:
    lines = [
        "You are a video editor. Each visual below must appear in a short video"
        + (f' about "{subject}".' if subject else "."),
        "Narration passages, in order:",
    ]
    lines += [f"P{index}. {passage}" for index, passage in enumerate(passages, start=1)]
    lines.append("User visuals:")
    lines += [f"V{index}. {shot.brief()}" for index, shot in enumerate(shots, start=1)]
    lines.append(
        "For each visual, give the number of the passage it illustrates best. "
        "Every visual must be placed. Spread visuals over different passages when "
        "they fit equally well."
    )
    if curation._judge_is_qwen3():
        # Comme pour la notation des clips : une raison courte remplace la
        # « réflexion » de qwen3, dix fois plus lente.
        lines.append("First give each visual a reason of at most 8 words, then its passage.")
        lines.append('Reply with JSON only: {"reasons": ["one short reason per visual"], '
                     '"passages": [one passage number per visual, in order]}')
        lines.append("/no_think")
    else:
        lines.append('Reply with JSON only: {"passages": [one passage number per visual, in order]}')
    return "\n".join(lines)


def parse_assignment(text: str, shot_count: int, passage_count: int) -> List[int | None]:
    """Numéros de passage (base 0) par plan ; None si absent ou invalide."""
    if not isinstance(text, str):
        text = ""
    text = re.sub(r"<think\b[^>]*>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    raw: Any = None
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            raw = data.get("passages") if isinstance(data, dict) else None
        except ValueError:
            raw = None
    result: List[int | None] = []
    for index in range(shot_count):
        value = raw[index] if isinstance(raw, list) and index < len(raw) else None
        if isinstance(value, str) and value.strip().upper().lstrip("P").isdigit():
            value = int(value.strip().upper().lstrip("P"))
        if isinstance(value, int) and not isinstance(value, bool) and 1 <= value <= passage_count:
            result.append(value - 1)
        else:
            result.append(None)
    return result


def _even_positions(shot_count: int, passage_count: int) -> List[int]:
    """Répartition régulière, dans l'ordre d'ajout : le repli sans LLM."""
    return [min(passage_count - 1, index * passage_count // shot_count) for index in range(shot_count)]


def assign_shots(shots: List[AssetShot], passages: List[str], subject: str = "") -> List[int]:
    """Passage (base 0) de chaque plan : choisi par le LLM, sinon réparti dans l'ordre."""
    if not shots or not passages:
        return []
    fallback = _even_positions(len(shots), len(passages))
    if len(passages) == 1:
        return [0] * len(shots)
    from app.services import llm  # import tardif : llm importe beaucoup de dépendances

    try:
        response = llm._generate_response(build_assignment_prompt(shots, passages, subject))
    except Exception as exc:
        logger.warning(f"user asset placement failed: {type(exc).__name__}: {exc}")
        response = ""
    if isinstance(response, str) and response.startswith("Error:"):
        logger.warning(f"user asset placement failed: {response[:200]}")
        response = ""
    chosen = parse_assignment(response, len(shots), len(passages))
    positions = [c if c is not None else fallback[i] for i, c in enumerate(chosen)]
    return _spread(positions, len(passages))


def _spread(positions: List[int], passage_count: int) -> List[int]:
    """Un passage ne garde qu'un asset tant qu'un voisin proche est libre.

    Le montage prend un plan par passage et par tour : un deuxième asset sur le
    même passage passerait après tous les autres passages, hors de son contexte.
    """
    positions = list(positions)
    for _ in range(len(positions)):
        counts = [positions.count(p) for p in range(passage_count)]
        crowded = [i for i, p in enumerate(positions) if counts[p] > 1]
        if not crowded:
            break
        moved = False
        for index in reversed(crowded):
            current = positions[index]
            free = [p for p in range(passage_count) if counts[p] == 0]
            if not free:
                break
            nearest = min(free, key=lambda p: (abs(p - current), p))
            if abs(nearest - current) <= 1:
                positions[index] = nearest
                counts[current] -= 1
                counts[nearest] += 1
                moved = True
        if not moved:
            break
    return positions


def _trimmed_clip_path(shot: AssetShot) -> str:
    name = f"user-{_file_key(shot.asset_path)}-{shot.start:.2f}-{shot.duration:.2f}.mp4"
    return os.path.join(utils.storage_dir("cache_videos", create=True), name)


def _trim_video(shot: AssetShot) -> str:
    """Extrait le plan ; le côté court est limité à 1080 px pour un rendu léger."""
    target = _trimmed_clip_path(shot)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return target
    temp = f"{target}.part.mp4"
    command = [
        utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
        "-ss", f"{shot.start:.2f}", "-i", shot.asset_path, "-t", f"{shot.duration:.2f}",
        "-an", "-vf",
        "scale='if(lt(iw,ih),min(iw,1080),-2)':'if(lt(iw,ih),-2,min(ih,1080))'",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        temp,
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, timeout=300)
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    return target


def to_material(shot: AssetShot) -> MaterialInfo:
    """Le plan sous la forme attendue par le téléchargement des visuels."""
    # Plusieurs plans d'une même vidéo : l'URL doit rester unique, le
    # téléchargement écarte les doublons d'URL.
    url = shot.asset_path if shot.kind == "image" else f"{shot.asset_path}#t={shot.start}"
    return MaterialInfo(
        provider="user",
        url=url,
        duration=max(1, int(math.ceil(shot.duration))),
        source_info={
            "provider": "user",
            "path": shot.asset_path,
            "kind": shot.kind,
            "asset_id": f"{Path(shot.asset_path).name}@{shot.start}",
            "start": shot.start,
            "clip_duration": shot.duration,
            "description": shot.brief()[:300],
        },
    )


def save_user_material(item: MaterialInfo, clip_duration: int) -> str:
    """Rend le plan prêt à monter : photo animée par un zoom lent, ou extrait vidéo."""
    from app.services import video

    source = item.source_info if isinstance(item.source_info, dict) else {}
    path = str(source.get("path") or item.url)
    if source.get("kind") == "image":
        return video.render_image_zoom_video(path, max(1, int(clip_duration)))
    shot = AssetShot(
        asset_path=path,
        kind="video",
        start=float(source.get("start") or 0.0),
        duration=float(source.get("clip_duration") or clip_duration),
    )
    return _trim_video(shot)


def merge_into_groups(
    results: List[tuple[str, List[MaterialInfo]]],
    context: curation.CurationContext | None,
) -> List[tuple[str, List[MaterialInfo]]]:
    """Place chaque plan de l'utilisateur en tête du passage qu'il illustre."""
    shots = list(getattr(context, "user_shots", None) or [])
    results = list(results)
    if not shots or not results:
        return results
    terms = [term for term, _ in results]
    passages = curation.split_script_for_terms(context.video_script, len(terms))
    passages = [p or t for p, t in zip(passages, terms)]
    positions = assign_shots(shots, passages, context.video_subject)
    placed: dict[int, List[MaterialInfo]] = {}
    for shot, position in zip(shots, positions):
        placed.setdefault(position, []).append(to_material(shot))
    for position, items in sorted(placed.items()):
        labels = ", ".join(str(i.source_info["asset_id"]) for i in items)
        logger.info(f"user assets: {terms[position]!r} <- {labels}")
    return [(term, placed.get(position, []) + items) for position, (term, items) in enumerate(results)]
