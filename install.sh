#!/usr/bin/env bash
# Installation de YenguiVideoPrinter (YVP) sous Linux, pendant de Installer-YVP.bat :
# 1. uv (gestionnaire Python) ;
# 2. Python 3.11 + dépendances (dans .venv du projet) ;
# 3. config.toml ;
# 4. Ollama portable dans ./ollama (pas de service, rien au démarrage du PC) ;
# 5. modèle IA dans ./ollama/models ;
# 6. icône dans le menu des applications et le dock.
# Relançable sans risque : chaque étape déjà faite est sautée.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL="${MPT_OLLAMA_MODEL:-qwen3:8b}"
cd "$DIR"
step() { printf '\n\033[36m>> %s\033[0m\n' "$1"; }

for tool in curl git zstd; do
  command -v "$tool" >/dev/null || { echo "$tool est requis : sudo apt install $tool"; exit 1; }
done

step "uv (gestionnaire Python)"
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"

step "Python et dépendances (quelques minutes la première fois)"
"$UV" sync --frozen --no-dev --python 3.11

step "Configuration"
if [ ! -f config.toml ]; then
  cp config.example.toml config.toml
  echo "config.toml créé depuis config.example.toml."
fi

step "Ollama (moteur IA local)"
if [ ! -x ollama/bin/ollama ]; then
  mkdir -p ollama
  curl -fL --retry 5 -C - -o ollama/ollama-linux-amd64.tar.zst \
    https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
  tar --zstd -xf ollama/ollama-linux-amd64.tar.zst -C ollama
  rm -f ollama/ollama-linux-amd64.tar.zst
fi

step "Modèle IA $MODEL (environ 5 Go la première fois)"
MANIFEST="ollama/models/manifests/registry.ollama.ai/library/${MODEL%%:*}/${MODEL#*:}"
[ -f "$MANIFEST" ] || tools/ollama-fetch.sh "$MODEL"

step "Icône YVP"
if command -v gsettings >/dev/null && command -v google-chrome >/dev/null; then
  bash tools/desktop/install.sh
else
  echo "Pas de bureau GNOME ou pas de Google Chrome : lancez YVP avec ./start.sh"
fi

printf '\n\033[32mInstallation terminée !\033[0m Lancez YVP depuis son icône ou avec ./start.sh\n'
