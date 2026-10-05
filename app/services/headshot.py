"""YVP : photo de profil LinkedIn créée à partir de plusieurs photos de l'utilisateur.

La génération (PhotoMaker V2 sur SDXL) demande PyTorch CUDA et environ 8 Go de
modèles : elle tourne dans un environnement à part (.headshot-venv, installé par
tools/headshot/setup.sh) et dans un processus détaché (tools/headshot/worker.py).
Ici on ne fait que :
1. installer cet environnement à la demande et suivre l'installation ;
2. préparer les photos (orientation, taille) et le prompt du style choisi ;
3. lancer, suivre et annuler la génération (storage/headshot/<tâche>) ;
4. lister et supprimer les résultats.

Les photos restent sur le PC ; le worker les efface dès la génération finie.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Tuple
from uuid import uuid4

import requests
from loguru import logger
from PIL import Image, ImageOps, UnidentifiedImageError

from app.utils import utils

MIN_PHOTOS = 5
MAX_PHOTOS = 20
# Au-delà, la ressemblance ne progresse plus et la mémoire GPU déborde.
MAX_USED_PHOTOS = 10
MIN_USABLE_PHOTOS = 3
MAX_PHOTO_SIDE = 1024
MAX_IMAGES = 2
BASE_MODEL = "SG161222/RealVisXL_V4.0"
# (côté en pixels, étapes) : LinkedIn affiche 400×400, 768 suffit largement.
QUALITIES = {"fast": (768, 20), "high": (1024, 30)}
# Minutes par image mesurées sur une GTX 1070 (estimation pour l'utilisateur).
MINUTES_PER_IMAGE = {"fast": 3, "high": 9}
GENDERS = {"man": "a man", "woman": "a woman"}
OUTFITS = {
    "suit": "wearing a tailored dark suit jacket and a crisp white shirt",
    "blazer": "wearing a navy blazer over a light shirt, business casual",
    "smart_casual": "wearing a smart casual fine knit sweater",
}
BACKGROUNDS = {
    "studio": "neutral light grey studio background",
    "office": "softly blurred modern bright office background",
    "outdoor": "softly blurred green outdoor background, bokeh",
    "white": "plain white background",
}
NEGATIVE_PROMPT = (
    "nsfw, lowres, bad anatomy, bad hands, text, watermark, logo, blurry, cartoon, painting, "
    "illustration, 3d render, deformed, extra fingers, sunglasses, hat, heavy makeup, plastic skin"
)
PHOTO_EXTENSIONS = ("jpg", "jpeg", "png", "webp")
_JOB_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{6}$")
_OUTPUT_RE = re.compile(r"^headshot-\d+\.png$")


class HeadshotError(Exception):
    """Demande impossible (message montrable à l'utilisateur)."""


@dataclass
class HeadshotResult:
    job_id: str
    created: datetime
    images: List[str]
    style: dict


# --- Emplacements -----------------------------------------------------------------------------
def headshot_dir() -> str:
    override = os.environ.get("YVP_HEADSHOT_DIR", "").strip()
    if override:
        # Tests et instances d'essai : jamais les photos de l'utilisateur.
        os.makedirs(override, exist_ok=True)
        return override
    return utils.storage_dir("headshot", create=True)


def venv_dir() -> str:
    return os.environ.get("YVP_HEADSHOT_VENV", "").strip() or os.path.join(utils.root_dir(), ".headshot-venv")


def venv_python() -> str:
    if sys.platform.startswith("win"):
        return os.path.join(venv_dir(), "Scripts", "python.exe")
    return os.path.join(venv_dir(), "bin", "python")


def _tool(name: str) -> str:
    return os.path.join(utils.root_dir(), "tools", "headshot", name)


def _install_log() -> str:
    return os.path.join(headshot_dir(), "install.log")


def _job_dir(job_id: str) -> str:
    if not _JOB_RE.match(job_id or ""):
        raise HeadshotError(f"tâche inconnue : {job_id}")
    return os.path.join(headshot_dir(), job_id)


# --- Environnement de génération --------------------------------------------------------------
def has_nvidia_gpu() -> bool:
    if not shutil.which("nvidia-smi"):
        return False
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def is_installed() -> bool:
    """setup.sh / setup.ps1 déposent ce marqueur une fois tous les imports vérifiés."""
    return os.path.isfile(venv_python()) and os.path.isfile(os.path.join(venv_dir(), ".yvp-ready"))


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
        if pid <= 0:
            return False
        if sys.platform.startswith("win"):
            return _windows_pid_alive(pid)
        os.kill(pid, 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    # Processus terminé mais pas encore récupéré par son parent (Streamlit).
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as handle:
            return handle.read().split(") ", 1)[1][:1] != "Z"
    except (OSError, IndexError):
        return True


def _windows_pid_alive(pid: int) -> bool:
    # Surtout pas os.kill(pid, 0) : sous Windows, il termine le processus.
    import ctypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not handle:
        return False
    try:
        code = ctypes.c_ulong()
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == 259  # STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _detached() -> dict:
    """Processus sans fenêtre de console, qui ne dépend pas de la webui."""
    if sys.platform.startswith("win"):
        return {"creationflags": subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _kill_tree(pid: int) -> None:
    if not pid:
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=30,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError, subprocess.SubprocessError):
        pass


def install_status() -> dict:
    """{"state": "ready" | "running" | "failed" | "missing", "log": dernières lignes}."""
    if is_installed():
        return {"state": "ready", "log": ""}
    log = ""
    try:
        with open(_install_log(), encoding="utf-8", errors="replace") as handle:
            log = "".join(handle.readlines()[-8:])
    except OSError:
        pass
    pid_file = _install_log() + ".pid"
    try:
        with open(pid_file, encoding="utf-8") as handle:
            pid = handle.read().strip()
    except OSError:
        pid = ""
    if pid and _pid_alive(pid):
        return {"state": "running", "log": log}
    return {"state": "failed" if log else "missing", "log": log}


def start_install() -> None:
    if install_status()["state"] == "running":
        return
    if sys.platform.startswith("win"):
        command = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", _tool("setup.ps1")]
    else:
        command = ["bash", _tool("setup.sh")]
    with open(_install_log(), "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, cwd=utils.root_dir(), **_detached(),
        )
    with open(_install_log() + ".pid", "w", encoding="utf-8") as handle:
        handle.write(str(process.pid))


# --- Préparation ------------------------------------------------------------------------------
def build_prompt(gender: str, outfit: str, background: str) -> Tuple[str, str]:
    """Prompt PhotoMaker : le mot « img » doit suivre le mot qui désigne la personne."""
    if gender not in GENDERS or outfit not in OUTFITS or background not in BACKGROUNDS:
        raise HeadshotError(f"style inconnu : {gender}, {outfit}, {background}")
    prompt = (
        f"professional linkedin headshot portrait photo of {GENDERS[gender]} img, {OUTFITS[outfit]}, "
        f"{BACKGROUNDS[background]}, soft studio lighting, sharp focus, natural skin texture, "
        "confident friendly smile, looking at the camera, head and shoulders, 85mm lens, high quality"
    )
    return prompt, NEGATIVE_PROMPT


def prepare_photo(data: bytes) -> Image.Image:
    """Photo redressée (EXIF des téléphones), en RGB, plus grand côté ≤ 1024 px."""
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HeadshotError("unreadable") from exc
    image.thumbnail((MAX_PHOTO_SIDE, MAX_PHOTO_SIDE), Image.LANCZOS)
    return image


def free_gpu_memory() -> None:
    """Décharge les modèles qu'Ollama garde en mémoire vidéo : SDXL a besoin des 8 Go."""
    from app.services.curation import _ollama_native_base_url

    base = _ollama_native_base_url()
    try:
        loaded = requests.get(f"{base}/api/ps", timeout=(3, 5)).json().get("models", [])
        for model in loaded:
            requests.post(f"{base}/api/generate", json={"model": model.get("name"), "keep_alive": 0},
                          timeout=(3, 30))
            logger.info(f"headshot: unloaded Ollama model {model.get('name')}")
    except Exception as exc:
        logger.debug(f"headshot: Ollama not reachable, nothing to unload ({exc})")


def worker_env() -> dict:
    env = dict(os.environ)
    # start.sh masque CUDA pour Ollama (CUDA_VISIBLE_DEVICES=-1) : le worker en a besoin.
    if env.get("CUDA_VISIBLE_DEVICES", "").strip() in ("", "-1"):
        env.pop("CUDA_VISIBLE_DEVICES", None)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # glibc garde pour lui la mémoire des gros tenseurs libérés (poids passés sur la
    # carte) : au-delà de 1 Mo, allocation à part, rendue au système dès la libération.
    # Mesuré au chargement : 5,4 → 2,7 Go de RAM.
    env.setdefault("MALLOC_MMAP_THRESHOLD_", "1048576")
    return env


def _scope_prefix(job_id: str) -> List[str]:
    """Le worker dans sa propre « scope » systemd. Quand la RAM sature, systemd-oomd tue
    tout un groupe : sans ça, c'était celui de YVP (webui + Ollama) ou le navigateur ;
    là c'est le worker, et YVP affiche « arrêtée brusquement » au lieu de disparaître."""
    if not sys.platform.startswith("linux") or not shutil.which("systemd-run"):
        return []
    try:
        probe = subprocess.run(["systemd-run", "--user", "--scope", "--quiet", "--collect", "true"],
                               capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return []
    if probe.returncode != 0:  # pas de session systemd utilisateur (ssh, conteneur…)
        return []
    # --scope exécute la commande elle-même : le PID reste celui du worker.
    return ["systemd-run", "--user", "--scope", "--quiet", "--collect", f"--unit=yvp-headshot-{job_id}"]


# --- Tâches -----------------------------------------------------------------------------------
def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def running_job() -> Optional[str]:
    for job_id in sorted(os.listdir(headshot_dir()), reverse=True):
        if _JOB_RE.match(job_id) and (job_status(job_id) or {}).get("status") == "running":
            return job_id
    return None


def start_job(photos: Iterable[Tuple[str, bytes]], gender: str, outfit: str, background: str,
              count: int = 2, quality: str = "fast") -> str:
    photos = list(photos)
    if len(photos) < MIN_PHOTOS:
        raise HeadshotError("too_few_photos")
    if len(photos) > MAX_PHOTOS:
        raise HeadshotError("too_many_photos")
    if quality not in QUALITIES or not 1 <= int(count) <= MAX_IMAGES:
        raise HeadshotError(f"réglage inconnu : {quality}, {count}")
    if not is_installed():
        raise HeadshotError("not_installed")
    if running_job():
        raise HeadshotError("busy")
    prompt, negative = build_prompt(gender, outfit, background)

    job_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid4().hex[:6]}"
    job_dir = _job_dir(job_id)
    photos_dir = os.path.join(job_dir, "photos")
    os.makedirs(photos_dir)
    names, labels = [], {}
    try:
        for index, (label, data) in enumerate(photos):
            name = f"photo-{index + 1:02d}.jpg"
            prepare_photo(data).save(os.path.join(photos_dir, name), quality=95)
            names.append(name)
            labels[name] = os.path.basename(label or name)
    except HeadshotError:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    size, steps = QUALITIES[quality]
    job = {
        "photos": names, "labels": labels, "prompt": prompt, "negative_prompt": negative,
        "count": int(count), "size": size, "steps": steps, "seed": random.randrange(2**31),
        "base_model": BASE_MODEL, "max_photos": MAX_USED_PHOTOS, "min_usable": MIN_USABLE_PHOTOS,
        "style": {"gender": gender, "outfit": outfit, "background": background, "quality": quality},
    }
    with open(os.path.join(job_dir, "job.json"), "w", encoding="utf-8") as handle:
        json.dump(job, handle, ensure_ascii=False, indent=2)

    # Statut provisoire, le temps que le worker importe PyTorch et prenne la main.
    with open(os.path.join(job_dir, "status.json"), "w", encoding="utf-8") as handle:
        json.dump({"status": "running", "phase": "start", "fraction": 0.0,
                   "started": time.time(), "rejected": [], "outputs": []}, handle)
    free_gpu_memory()
    with open(os.path.join(job_dir, "worker.log"), "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            _scope_prefix(job_id) + [venv_python(), _tool("worker.py"), job_dir], stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=worker_env(), cwd=job_dir, **_detached(),
        )
    with open(os.path.join(job_dir, "worker.pid"), "w", encoding="utf-8") as handle:
        handle.write(str(process.pid))
    logger.info(f"headshot: job {job_id} started ({len(names)} photos, {count}×{quality})")
    return job_id


def _worker_pid(job_id: str) -> int:
    try:
        with open(os.path.join(_job_dir(job_id), "worker.pid"), encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return 0


def job_status(job_id: str) -> Optional[dict]:
    job_dir = _job_dir(job_id)
    status = _read_json(os.path.join(job_dir, "status.json"))
    if not status or status.get("status") != "running":
        return status
    pid = _worker_pid(job_id)
    # Sans PID depuis plus de 30 s : le lancement a échoué.
    lost = not pid and time.time() - status.get("started", 0) > 30
    if lost or (pid and not _pid_alive(pid)):
        # Worker tué (manque de mémoire, redémarrage…) sans avoir pu l'écrire.
        status.update(status="failed", error=status.get("error") or "crashed")
    return status


def cancel_job(job_id: str) -> None:
    status = job_status(job_id)
    if not status or status.get("status") != "running":
        return
    _kill_tree(_worker_pid(job_id))
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)


def list_results() -> List[HeadshotResult]:
    results = []
    for job_id in sorted(os.listdir(headshot_dir()), reverse=True):
        if not _JOB_RE.match(job_id):
            continue
        job_dir = os.path.join(headshot_dir(), job_id)
        images = sorted(
            (os.path.join(job_dir, n) for n in os.listdir(job_dir) if _OUTPUT_RE.match(n)),
            key=lambda p: int(re.search(r"\d+", os.path.basename(p)).group()),
        )
        if not images:
            continue
        job = _read_json(os.path.join(job_dir, "job.json")) or {}
        results.append(HeadshotResult(
            job_id=job_id, created=datetime.strptime(job_id[:15], "%Y%m%d-%H%M%S"),
            images=images, style=job.get("style", {}),
        ))
    return results


def delete_job(job_id: str) -> None:
    cancel_job(job_id)
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)


def linkedin_jpeg(path: str, size: int = 800) -> bytes:
    """Carré en JPEG, le format que LinkedIn attend pour une photo de profil."""
    with Image.open(path) as image:
        image = ImageOps.fit(image.convert("RGB"), (size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=92)
        return buffer.getvalue()
