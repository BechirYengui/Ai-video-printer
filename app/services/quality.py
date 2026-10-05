"""YVP : niveaux de qualité et estimation du temps de génération.

Le temps d'une génération se concentre sur deux étapes : la sélection des
visuels par l'IA (le modèle de vision décrit des dizaines de miniatures) et le
montage (décodage des clips sources, encodage). Les niveaux jouent sur ces deux
leviers ; la résolution de sortie ne change pas, les sous-titres restent nets.

L'estimation part de valeurs mesurées sur un PC modeste, puis s'ajuste à la
machine : chaque étape terminée met à jour une moyenne glissante, stockée dans
storage/eta_history.json.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Set

from loguru import logger

from app.utils import utils

QUALITY_LEVELS = ("low", "medium", "high")
DEFAULT_QUALITY = "high"  # comportement historique pour l'API et la CLI


@dataclass(frozen=True)
class QualityPreset:
    encoder_preset: str  # preset libx264 : ultrafast < veryfast < medium
    curation: bool  # sélection des visuels par l'IA
    max_candidates: int  # miniatures décrites par mot-clé
    openverse: bool  # photos libres en secours


PRESETS: Dict[str, QualityPreset] = {
    "low": QualityPreset("ultrafast", curation=False, max_candidates=0, openverse=False),
    "medium": QualityPreset("veryfast", curation=True, max_candidates=3, openverse=False),
    "high": QualityPreset("medium", curation=True, max_candidates=6, openverse=True),
}


def normalize(level: str | None) -> str:
    level = str(level or "").strip().lower()
    return level if level in PRESETS else DEFAULT_QUALITY


def preset(level: str | None) -> QualityPreset:
    return PRESETS[normalize(level)]


# --- Estimation du temps -------------------------------------------------------

STAGES = ("assets", "script", "terms", "audio", "subtitle", "materials", "render")

# Secondes par unité. Unités : plan analysé (assets), mot-clé (materials),
# seconde de vidéo (render), sinon l'étape entière. Valeurs mesurées sur un
# i7-8750H + GTX 1070 avec Ollama local.
_DEFAULT_RATES = {
    "assets": 12.0,
    "script": 30.0,
    "terms": 15.0,
    "audio": 8.0,
    "subtitle": 4.0,
    "materials:low": 12.0,
    "materials:medium": 55.0,
    "materials:high": 110.0,
    "render:low": 4.0,
    "render:medium": 6.0,
    "render:high": 9.0,
}
_PER_QUALITY = {"materials", "render"}
_HISTORY_WEIGHT = 0.4  # part de la dernière mesure dans la moyenne glissante
_history_lock = threading.Lock()


def _history_path() -> str:
    return os.path.join(utils.storage_dir(), "eta_history.json")


def _load_history() -> Dict[str, float]:
    try:
        with open(_history_path(), encoding="utf-8") as handle:
            data = json.load(handle)
        return {k: float(v) for k, v in data.items() if isinstance(v, (int, float)) and v > 0}
    except (OSError, ValueError, TypeError):
        return {}


def _rate_key(stage: str, quality: str) -> str:
    return f"{stage}:{quality}" if stage in _PER_QUALITY else stage


def stage_rate(stage: str, quality: str) -> float:
    key = _rate_key(stage, normalize(quality))
    return _load_history().get(key, _DEFAULT_RATES[key])


def record(stage: str, quality: str, seconds: float, units: float) -> None:
    """Ajuste la moyenne glissante d'une étape avec la durée observée."""
    # Moins d'une seconde : étape simulée (tests) ou servie par un cache, pas
    # représentative du temps qu'elle prend vraiment.
    if units <= 0 or seconds < 1.0:
        return
    key = _rate_key(stage, normalize(quality))
    with _history_lock:
        history = _load_history()
        previous = history.get(key, _DEFAULT_RATES[key])
        history[key] = round(previous * (1 - _HISTORY_WEIGHT) + (seconds / units) * _HISTORY_WEIGHT, 3)
        try:
            temp = f"{_history_path()}.tmp"
            with open(temp, "w", encoding="utf-8") as handle:
                json.dump(history, handle, indent=2)
            os.replace(temp, _history_path())
        except OSError as exc:
            logger.debug(f"eta history not saved: {exc}")


@dataclass
class Workload:
    """Taille du travail : ce qui fait varier la durée des étapes."""

    video_seconds: float = 30.0
    terms: int = 8
    shots: int = 0
    # Étapes sans travail (script fourni, mots-clés fournis…) : ni estimées, ni
    # mesurées, sinon leur durée quasi nulle fausserait la moyenne.
    skipped: Set[str] = field(default_factory=set)

    def units(self, stage: str) -> float:
        if stage in self.skipped:
            return 0.0
        if stage == "assets":
            return float(self.shots)
        if stage == "materials":
            return float(max(1, self.terms))
        if stage == "render":
            return float(max(5.0, self.video_seconds))
        return 1.0


def estimate_seconds(quality: str, workload: Workload, stages=STAGES) -> float:
    return sum(stage_rate(stage, quality) * workload.units(stage) for stage in stages)


class EtaTracker:
    """Suit l'avancement d'une tâche et publie le temps restant estimé.

    L'estimation est gardée en mémoire : la WebUI exécute les tâches dans un
    thread du même processus et la lit pour afficher le compte à rebours.
    """

    def __init__(self, task_id: str, quality: str, workload: Workload):
        self.task_id = task_id
        self.quality = normalize(quality)
        self.workload = workload
        self.started_at = time.time()
        self.stage: str | None = None
        self.stage_started_at = self.started_at
        # Part réellement faite de l'étape en cours (0 = inconnue), remontée
        # par les étapes longues (préparation des plans, rendu image par image).
        self.stage_fraction = 0.0
        with _live_lock:
            _trackers[str(task_id)] = self

    def _close_stage(self, now: float) -> None:
        if self.stage is not None:
            record(self.stage, self.quality, now - self.stage_started_at,
                   self.workload.units(self.stage))

    def remaining(self, now: float | None = None) -> float:
        now = now or time.time()
        if self.stage is None:
            return estimate_seconds(self.quality, self.workload)
        index = STAGES.index(self.stage)
        spent = now - self.stage_started_at
        if self.stage_fraction >= 0.05:
            # Avancement mesuré : on extrapole à partir du temps déjà passé,
            # bien plus juste que la moyenne des tâches précédentes.
            left_in_current = spent * (1 - self.stage_fraction) / self.stage_fraction
        else:
            current = stage_rate(self.stage, self.quality) * self.workload.units(self.stage)
            left_in_current = max(0.0, current - spent)
        return left_in_current + estimate_seconds(self.quality, self.workload, STAGES[index + 1:])

    def fields(self, now: float | None = None) -> dict:
        now = now or time.time()
        return {
            "eta_seconds": int(round(self.remaining(now))),
            "eta_at": round(now, 1),
            "started_at": round(self.started_at, 1),
            # Étape en cours : affichée quand l'estimation est dépassée.
            "stage": self.stage,
        }

    def begin(self, stage: str) -> None:
        now = time.time()
        self._close_stage(now)
        self.stage, self.stage_started_at = stage, now
        self.stage_fraction = 0.0
        self._publish(self.fields(now))

    def advance(self, fraction: float) -> None:
        """Avancement réel (0…1) de l'étape en cours."""
        try:
            fraction = max(0.0, min(1.0, float(fraction)))
        except (TypeError, ValueError):
            return
        if fraction > self.stage_fraction:
            self.stage_fraction = fraction
            self._publish(self.fields())

    def finish(self) -> None:
        now = time.time()
        self._close_stage(now)
        self.stage = None
        self._publish({"eta_seconds": 0, "eta_at": round(now, 1),
                       "started_at": round(self.started_at, 1)})
        with _live_lock:
            _trackers.pop(str(self.task_id), None)

    def _publish(self, fields: dict) -> None:
        with _live_lock:
            _live_eta[str(self.task_id)] = dict(fields)


_live_eta: Dict[str, dict] = {}
_trackers: Dict[str, EtaTracker] = {}
_live_lock = threading.Lock()


def report_stage_progress(task_id: str, fraction: float) -> None:
    """Remonte l'avancement réel de l'étape en cours (sans effet si inconnue)."""
    with _live_lock:
        tracker = _trackers.get(str(task_id))
    if tracker is not None:
        tracker.advance(fraction)


def stage_progress_callback(task_id: str, start: float = 0.0, end: float = 1.0):
    """Rappel 0…1 d'une sous-partie [start, end] de l'étape en cours."""
    def report(fraction) -> None:
        try:
            report_stage_progress(task_id, start + (end - start) * float(fraction))
        except (TypeError, ValueError):
            pass
    return report


def read_eta(task_id: str) -> dict:
    with _live_lock:
        return dict(_live_eta.get(str(task_id), {}))


def time_progress(task: dict, now: float | None = None) -> tuple[int, int] | None:
    """(pourcentage basé sur le temps, secondes restantes) pour l'affichage."""
    try:
        eta_seconds = float(task["eta_seconds"])
        eta_at = float(task["eta_at"])
        started_at = float(task["started_at"])
    except (KeyError, TypeError, ValueError):
        return None
    now = now or time.time()
    remaining = max(0.0, eta_seconds - (now - eta_at))
    elapsed = max(0.0, now - started_at)
    total = elapsed + remaining
    percent = int(99 * elapsed / total) if total > 0 else 0
    return max(1, min(99, percent)), int(round(remaining))


def elapsed_seconds(task: dict, now: float | None = None) -> int:
    """Temps écoulé depuis le début de la tâche (0 si inconnu)."""
    try:
        return max(0, int((now or time.time()) - float(task["started_at"])))
    except (KeyError, TypeError, ValueError):
        return 0


def format_duration(seconds: float) -> str:
    """« 45 s », « 3 min », « 1 h 05 » : un ordre de grandeur, pas un chrono."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{max(5, int(round(seconds / 5.0)) * 5)} s"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d}"
