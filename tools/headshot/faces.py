"""YVP : détection de visage et empreinte d'identité, sans le paquet insightface.

insightface 0.7.3 n'a pas de paquet Windows et se compile avec Visual C++ ;
on n'utilise pourtant que deux de ses modèles ONNX (pack buffalo_l) :
- det_10g.onnx  : SCRFD, cadre et 5 points du visage ;
- w600k_r50.onnx : ArcFace, empreinte de 512 valeurs (celle qu'attend PhotoMaker V2).
Portage fidèle de insightface (model_zoo/scrfd.py, arcface_onnx.py,
utils/face_align.py, licence MIT), vérifié contre lui sous Linux.
"""

import os
import time
import zipfile
from dataclasses import dataclass

import cv2
import numpy as np

PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
DETECTOR, RECOGNIZER = "det_10g.onnx", "w600k_r50.onnx"
# Les 5 points (yeux, nez, coins de la bouche) d'un visage ArcFace de 112 px.
ARCFACE_POINTS = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                           [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)


@dataclass
class Face:
    bbox: np.ndarray       # x0, y0, x1, y1
    kps: np.ndarray        # 5 × (x, y)
    det_score: float
    embedding: np.ndarray = None


def models_dir() -> str:
    # Même dossier qu'insightface : rien à retélécharger là où il a déjà servi.
    return os.path.join(os.path.expanduser("~"), ".insightface", "models", "buffalo_l")


def ensure_models() -> str:
    folder = models_dir()
    if all(os.path.isfile(os.path.join(folder, name)) for name in (DETECTOR, RECOGNIZER)):
        return folder
    import requests

    os.makedirs(folder, exist_ok=True)
    archive = os.path.join(folder, "buffalo_l.zip.part")
    last_error = None
    for _ in range(20):  # connexion lente et coupures : on reprend où on en était
        try:
            done = os.path.getsize(archive) if os.path.exists(archive) else 0
            headers = {"Range": f"bytes={done}-"} if done else {}
            with requests.get(PACK_URL, headers=headers, stream=True, timeout=(10, 60)) as response:
                if response.status_code == 416:  # déjà complet
                    break
                response.raise_for_status()
                mode = "ab" if done and response.status_code == 206 else "wb"
                with open(archive, mode) as handle:
                    for chunk in response.iter_content(1 << 20):
                        handle.write(chunk)
            break
        except Exception as exc:
            last_error = exc
            time.sleep(10)
    else:
        raise RuntimeError(f"téléchargement impossible : buffalo_l ({last_error})")
    with zipfile.ZipFile(archive) as pack:
        for member in pack.namelist():
            name = os.path.basename(member)
            if name in (DETECTOR, RECOGNIZER):
                with pack.open(member) as source, open(os.path.join(folder, name), "wb") as target:
                    target.write(source.read())
    os.remove(archive)
    return folder


def _session(path, providers):
    import onnxruntime

    available = onnxruntime.get_available_providers()
    providers = [p for p in providers if p in available] or ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in providers and hasattr(onnxruntime, "preload_dlls"):
        try:  # Windows : bibliothèques CUDA livrées avec PyTorch
            onnxruntime.preload_dlls()
        except Exception:
            pass
    try:
        return onnxruntime.InferenceSession(path, providers=providers)
    except Exception:
        # CUDA indisponible pour onnxruntime : quelques secondes de plus sur le processeur.
        return onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])


def _similarity_transform(src, dst):
    """Matrice 2×3 (rotation, échelle, translation) de src vers dst : Umeyama,
    comme skimage.transform.SimilarityTransform qu'utilise insightface."""
    src_mean, dst_mean = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - src_mean, dst - dst_mean
    cov = dst_c.T @ src_c / len(src)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[1] = -1
    u, s, vt = np.linalg.svd(cov)
    rotation = u @ np.diag(d) @ vt
    scale = (s * d).sum() / src_c.var(axis=0).sum()
    matrix = np.zeros((2, 3))
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = dst_mean - scale * rotation @ src_mean
    return matrix


class FaceAnalyzer:
    def __init__(self, providers=("CUDAExecutionProvider", "CPUExecutionProvider"), det_size=640,
                 det_thresh=0.5, nms_thresh=0.4):
        folder = ensure_models()
        self.detector = _session(os.path.join(folder, DETECTOR), providers)
        self.recognizer = _session(os.path.join(folder, RECOGNIZER), providers)
        self.det_size, self.det_thresh, self.nms_thresh = det_size, det_thresh, nms_thresh
        self._det_outputs = [o.name for o in self.detector.get_outputs()]
        self._batched = len(self.detector.get_outputs()[0].shape) == 3

    def get(self, bgr):
        """Visages d'une image BGR (comme cv2.imread), avec leur empreinte."""
        faces = self.detect(bgr)
        for face in faces:
            matrix = _similarity_transform(face.kps.astype(np.float64), ARCFACE_POINTS.astype(np.float64))
            aligned = cv2.warpAffine(bgr, matrix, (112, 112), borderValue=0.0)
            blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (112, 112), (127.5,) * 3, swapRB=True)
            face.embedding = self.recognizer.run(None, {self.recognizer.get_inputs()[0].name: blob})[0].flatten()
        return faces

    def detect(self, bgr):
        size = self.det_size
        ratio = bgr.shape[0] / bgr.shape[1]
        height, width = (size, int(size / ratio)) if ratio > 1 else (int(size * ratio), size)
        scale = height / bgr.shape[0]
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:height, :width] = cv2.resize(bgr, (width, height))
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (size, size), (127.5,) * 3, swapRB=True)
        outputs = self.detector.run(self._det_outputs, {self.detector.get_inputs()[0].name: blob})

        scores_all, boxes_all, points_all = [], [], []
        for level, stride in enumerate((8, 16, 32)):
            pick = (lambda i: outputs[i][0]) if self._batched else (lambda i: outputs[i])
            scores, boxes, points = pick(level), pick(level + 3) * stride, pick(level + 6) * stride
            cells = size // stride
            centers = np.stack(np.mgrid[:cells, :cells][::-1], axis=-1).astype(np.float32)
            centers = np.repeat((centers * stride).reshape(-1, 2), 2, axis=0)  # 2 ancres par case
            keep = np.where(scores >= self.det_thresh)[0]
            scores_all.append(scores[keep])
            boxes_all.append(np.hstack([centers - boxes[:, :2], centers + boxes[:, 2:]])[keep])
            points_all.append((np.tile(centers, 5) + points).reshape(-1, 5, 2)[keep])

        scores = np.vstack(scores_all).ravel()
        order = scores.argsort()[::-1]
        boxes = (np.vstack(boxes_all) / scale)[order]
        points = (np.vstack(points_all) / scale)[order]
        scores = scores[order]
        keep = self._nms(boxes, scores)
        return [Face(bbox=boxes[i], kps=points[i], det_score=float(scores[i])) for i in keep]

    def _nms(self, boxes, scores):
        x0, y0, x1, y1 = boxes.T
        areas = (x1 - x0 + 1) * (y1 - y0 + 1)
        order, keep = np.arange(len(scores)), []
        while order.size:
            i, rest = order[0], order[1:]
            keep.append(i)
            w = np.maximum(0.0, np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]) + 1)
            h = np.maximum(0.0, np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]) + 1)
            overlap = w * h / (areas[i] + areas[rest] - w * h)
            order = rest[overlap <= self.nms_thresh]
        return keep
