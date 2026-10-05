#!/usr/bin/env bash
# Lance Ollama (local, dossier ./ollama) puis YenguiVideoPrinter (YVP, webui). Ctrl+C arrête tout.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MPT_OLLAMA_MODEL:-qwen3:8b}"

export OLLAMA_MODELS="$DIR/ollama/models"
export OLLAMA_HOST="127.0.0.1:11434"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-10m}"
export PATH="$DIR/ollama/bin:$PATH"
# GTX 1070 (Pascal, cc 6.1) : les libs CUDA d'Ollama ne la supportent plus,
# on masque CUDA et on passe par Vulkan pour garder le calcul sur la GPU.
export CUDA_VISIBLE_DEVICES=-1
export OLLAMA_VULKAN=1
mkdir -p "$OLLAMA_MODELS" "$DIR/logs"

STARTED_OLLAMA=0
if ! curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
  echo ">> Démarrage d'Ollama..."
  ollama serve >"$DIR/logs/ollama.log" 2>&1 &
  OLLAMA_PID=$!
  STARTED_OLLAMA=1
  trap '[ "$STARTED_OLLAMA" = 1 ] && kill "$OLLAMA_PID" 2>/dev/null || true' EXIT INT TERM
  for _ in $(seq 1 60); do
    curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1 && break
    sleep 1
  done
  curl -fs "http://$OLLAMA_HOST/api/version" >/dev/null || { echo "Ollama n'a pas démarré, voir logs/ollama.log"; exit 1; }
fi

if ! ollama list | awk 'NR>1{print $1}' | grep -qx "$MODEL"; then
  echo ">> Téléchargement du modèle $MODEL (première fois seulement)..."
  # curl + DoH plutôt que `ollama pull`, qui casse avec le DNS local instable
  "$DIR/tools/ollama-fetch.sh" "$MODEL"
fi

echo ">> Ollama prêt ($MODEL). Lancement de YenguiVideoPrinter (YVP) sur http://127.0.0.1:8501"
cd "$DIR"
sh webui.sh
