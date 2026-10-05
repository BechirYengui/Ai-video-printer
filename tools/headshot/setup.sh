#!/usr/bin/env bash
# YVP : installe l'environnement « Photo de profil LinkedIn » (PyTorch CUDA,
# diffusers, PhotoMaker V2) dans .headshot-venv (sans insightface : voir faces.py), à part de l'application :
# plusieurs Go que seuls les PC avec une carte NVIDIA utilisent.
# Les modèles (≈ 8 Go) se téléchargent ensuite à la première génération.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="${YVP_HEADSHOT_VENV:-$DIR/.headshot-venv}"
export UV_HTTP_TIMEOUT=300 UV_HTTP_RETRIES=8 UV_CONCURRENT_DOWNLOADS=2

command -v uv >/dev/null || { echo "uv introuvable"; exit 1; }
command -v git >/dev/null || { echo "git introuvable (nécessaire pour PhotoMaker)"; exit 1; }

rm -f "$VENV/.yvp-ready"
[ -x "$VENV/bin/python" ] || uv venv -q --python 3.11 "$VENV"
# Connexion instable : on retente, uv reprend depuis son cache.
for attempt in 1 2 3 4; do
  echo ">> installation, tentative $attempt"
  # onnxruntime-gpu 1.24+ cible CUDA 13, qui ne gère plus les cartes Pascal (GTX 10xx).
  if uv pip install -q --python "$VENV/bin/python" "torch==2.14.1" \
       --index-url https://download.pytorch.org/whl/cu126 \
     && uv pip install -q --python "$VENV/bin/python" numpy pillow "diffusers>=0.30" \
       transformers accelerate safetensors peft omegaconf einops huggingface_hub \
       "onnxruntime-gpu==1.23.2" opencv-python-headless requests \
       "git+https://github.com/TencentARC/PhotoMaker.git" \
     && "$VENV/bin/python" "$DIR/tools/headshot/worker.py" --check; then
    touch "$VENV/.yvp-ready"
    echo ">> environnement prêt : $VENV"
    exit 0
  fi
  sleep 10
done
echo ">> installation impossible (voir les erreurs ci-dessus)"
exit 1
