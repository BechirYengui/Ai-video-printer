"""YVP : studio caméra — vidéos où l'utilisateur parle devant un fond choisi.

L'enregistrement et le remplacement du fond se font dans le navigateur
(webui/static/camera_studio.js). Ici on ne fait que :
1. convertir la prise du navigateur (WebM ou MP4 à cadence variable) en MP4
   H.264/AAC standard, lisible partout et téléchargeable ;
2. lister et supprimer les enregistrements (storage/recordings) ;
3. découper un extrait vers storage/local_videos pour le réutiliser dans le
   montage, comme une vidéo importée.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List
from uuid import uuid4

from loguru import logger

from app.services import material_upload
from app.utils import file_security, utils

MAX_RECORDING_BYTES = 200 * 1024 * 1024
MAX_RECORDING_SECONDS = 180
MAX_CLIP_SECONDS = 30
_SOURCE_EXTENSIONS = {"video/webm": ".webm", "video/mp4": ".mp4"}
_NAME_RE = re.compile(r"^rec-\d{8}-\d{6}-[0-9a-f]{6}-(portrait|landscape)\.mp4$")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")


class RecordingError(Exception):
    """Prise invalide ou conversion impossible (message montrable à l'utilisateur)."""


@dataclass
class Recording:
    name: str
    path: str
    size: int
    duration: float
    created: datetime
    aspect: str  # "portrait" ou "landscape"


def _user_videos_dir() -> str:
    """Dossier « Vidéos » de l'utilisateur (Vidéos, Videos, Movies…)."""
    if sys.platform.startswith("linux"):
        try:
            completed = subprocess.run(
                ["xdg-user-dir", "VIDEOS"], capture_output=True, text=True, timeout=5
            )
            folder = completed.stdout.strip()
            if completed.returncode == 0 and folder and os.path.abspath(folder) != os.path.expanduser("~"):
                return folder
        except (OSError, subprocess.SubprocessError):
            pass
    return os.path.join(os.path.expanduser("~"), "Movies" if sys.platform == "darwin" else "Videos")


def recordings_dir() -> str:
    """Vidéos/YVP : là où l'utilisateur retrouve ses vidéos sans chercher.

    Repli sur storage/recordings si ce dossier n'est pas accessible. Les
    enregistrements faits avant ce changement y sont déplacés une fois.
    """
    legacy = utils.storage_dir("recordings")
    override = os.environ.get("YVP_RECORDINGS_DIR", "").strip()
    if override:
        # Tests et instances d'essai : jamais les vidéos de l'utilisateur.
        os.makedirs(override, exist_ok=True)
        return override
    folder = os.path.join(_user_videos_dir(), "YVP")
    try:
        os.makedirs(folder, exist_ok=True)
    except OSError as exc:
        logger.warning(f"videos folder unavailable ({exc}), using {legacy}")
        os.makedirs(legacy, exist_ok=True)
        return legacy
    if os.path.isdir(legacy):
        for name in os.listdir(legacy):
            target = os.path.join(folder, name)
            if _NAME_RE.match(name) and not os.path.exists(target):
                shutil.move(os.path.join(legacy, name), target)
                logger.info(f"camera recording moved to {target}")
    return folder


def _ffmpeg(args: List[str], timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(
        [utils.get_ffmpeg_binary(), "-hide_banner", *args],
        capture_output=True,
        timeout=timeout,
    )


def probe_duration(path: str) -> float:
    """Durée lue par ffmpeg (évite de charger MoviePy pour une simple liste)."""
    completed = _ffmpeg(["-i", path], timeout=30)
    match = _DURATION_RE.search(completed.stderr.decode("utf-8", "replace"))
    if not match:
        return 0.0
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def save_recording(data: bytes, mime: str, aspect: str) -> Recording:
    """Convertit la prise du navigateur en MP4 et l'enregistre."""
    if aspect not in ("portrait", "landscape"):
        raise RecordingError(f"format inconnu : {aspect}")
    if not data:
        raise RecordingError("enregistrement vide")
    if len(data) > MAX_RECORDING_BYTES:
        raise RecordingError(f"enregistrement trop lourd (> {MAX_RECORDING_BYTES // 2**20} Mo)")
    base_mime = (mime or "").split(";")[0].strip().lower()
    source_ext = _SOURCE_EXTENSIONS.get(base_mime)
    if source_ext is None:
        raise RecordingError(f"type de vidéo non pris en charge : {mime}")

    folder = recordings_dir()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"rec-{stamp}-{uuid4().hex[:6]}-{aspect}.mp4"
    target = os.path.join(folder, name)
    source = os.path.join(folder, f".{name}.source{source_ext}")
    temp = os.path.join(folder, f".{name}.part.mp4")
    try:
        with open(source, "wb") as handle:
            handle.write(data)
        # Cadence fixe 30 i/s : MediaRecorder produit une cadence variable que
        # certains lecteurs et MoviePy gèrent mal.
        completed = _ffmpeg([
            "-y", "-loglevel", "error", "-i", source,
            "-t", str(MAX_RECORDING_SECONDS),
            "-vf", "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
            "-movflags", "+faststart",
            temp,
        ])
        if completed.returncode != 0 or not os.path.exists(temp):
            logger.error(f"camera recording conversion failed: {completed.stderr.decode('utf-8', 'replace')[-800:]}")
            raise RecordingError("conversion de la vidéo impossible")
        os.replace(temp, target)
    finally:
        for leftover in (source, temp):
            if os.path.exists(leftover):
                os.remove(leftover)
    logger.info(f"camera recording saved: {target}")
    return _describe(target)


def _describe(path: str) -> Recording:
    name = os.path.basename(path)
    stat = os.stat(path)
    return Recording(
        name=name,
        path=path,
        size=stat.st_size,
        duration=probe_duration(path),
        created=datetime.fromtimestamp(stat.st_mtime),
        aspect="portrait" if name.endswith("-portrait.mp4") else "landscape",
    )


def list_recordings() -> List[Recording]:
    """Enregistrements du plus récent au plus ancien."""
    folder = recordings_dir()
    names = sorted((n for n in os.listdir(folder) if _NAME_RE.match(n)), reverse=True)
    return [_describe(os.path.join(folder, n)) for n in names]


def _resolve(name: str) -> str:
    if not _NAME_RE.match(name or ""):
        raise RecordingError(f"enregistrement inconnu : {name}")
    try:
        return file_security.resolve_path_within_directory(recordings_dir(), name)
    except (ValueError, OSError) as exc:
        raise RecordingError(f"enregistrement introuvable : {name}") from exc


def delete_recording(name: str) -> None:
    path = _resolve(name)
    os.remove(path)
    logger.info(f"camera recording deleted: {path}")


def cut_recording(name: str, start: float, end: float, mode: str = "keep") -> Recording:
    """Nouvelle version de l'enregistrement, l'original reste intact.

    - ``keep``   : garde seulement [start, end] (couper le début et la fin) ;
    - ``remove`` : retire [start, end] (un passage raté au milieu).
    """
    path = _resolve(name)
    duration = probe_duration(path)
    start = max(0.0, float(start))
    end = min(float(end), duration) if duration else float(end)
    if end - start < 0.2:
        raise RecordingError("passage trop court")
    if mode == "keep":
        filters = [
            "-ss", f"{start:.3f}", "-i", path, "-t", f"{end - start:.3f}",
        ]
        kept = end - start
    elif mode == "remove":
        kept = (duration or end) - (end - start)
        if kept < 0.5:
            raise RecordingError("il ne resterait presque rien de la vidéo")
        # Deux morceaux recollés, image et son ensemble.
        parts, labels = [], []
        for index, (a, b) in enumerate(((0.0, start), (end, None))):
            if b is not None and b - a < 0.05:
                continue
            bound = f":end={b:.3f}" if b is not None else ""
            parts.append(
                f"[0:v]trim=start={a:.3f}{bound},setpts=PTS-STARTPTS[v{index}];"
                f"[0:a]atrim=start={a:.3f}{bound},asetpts=PTS-STARTPTS[a{index}]"
            )
            labels.append(f"[v{index}][a{index}]")
        graph = ";".join(parts) + ";" + "".join(labels) + f"concat=n={len(labels)}:v=1:a=1[v][a]"
        filters = ["-i", path, "-filter_complex", graph, "-map", "[v]", "-map", "[a]"]
    else:
        raise RecordingError(f"mode de découpe inconnu : {mode}")

    aspect = "portrait" if name.endswith("-portrait.mp4") else "landscape"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = os.path.join(recordings_dir(), f"rec-{stamp}-{uuid4().hex[:6]}-{aspect}.mp4")
    temp = f"{target}.part.mp4"
    try:
        completed = _ffmpeg([
            "-y", "-loglevel", "error", *filters,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            temp,
        ])
        if completed.returncode != 0 or not os.path.exists(temp):
            logger.error(f"camera recording cut failed: {completed.stderr.decode('utf-8', 'replace')[-800:]}")
            raise RecordingError("découpe de la vidéo impossible")
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    logger.info(f"camera recording cut ({mode} {start:.1f}-{end:.1f}s, ~{kept:.1f}s kept): {target}")
    return _describe(target)


def extract_clip(name: str, start: float, end: float) -> str:
    """Copie [start, end] dans storage/local_videos ; renvoie le chemin du clip."""
    path = _resolve(name)
    start = max(0.0, float(start))
    length = float(end) - start
    if length <= 0:
        raise RecordingError("extrait vide")
    if length > MAX_CLIP_SECONDS:
        raise RecordingError(f"extrait trop long (max {MAX_CLIP_SECONDS} s)")
    target = os.path.join(
        material_upload.uploaded_material_dir(),
        f"camera-{uuid4().hex}.mp4",
    )
    temp = f"{target}.part.mp4"
    try:
        completed = _ffmpeg([
            "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", path, "-t", f"{length:.2f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            temp,
        ], timeout=300)
        if completed.returncode != 0 or not os.path.exists(temp):
            logger.error(f"camera clip extraction failed: {completed.stderr.decode('utf-8', 'replace')[-800:]}")
            raise RecordingError("découpe de l'extrait impossible")
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    return target
