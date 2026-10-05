#!/usr/bin/env bash
# Fabrique dist/YVP-Windows.zip à copier sur un PC Windows 11.
# Contient le code, config.toml et les scripts Windows ; pas de .venv, ni
# d'Ollama, ni de storage (l'installateur Windows télécharge ce qu'il faut).
# Option --with-model : embarque aussi le modèle Ollama déjà téléchargé ici
# (évite ~5 Go de téléchargement sur l'autre PC).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODEL="${MPT_OLLAMA_MODEL:-qwen3:8b}"
WITH_MODEL=0; [ "${1:-}" = "--with-model" ] && WITH_MODEL=1
OUT="$ROOT/dist/YVP-Windows.zip"
mkdir -p "$ROOT/dist"; rm -f "$OUT"
cd "$ROOT"
"$ROOT/.venv/bin/python" - "$OUT" "$WITH_MODEL" "$MODEL" <<'PY'
import json, os, subprocess, sys, zipfile
out, with_model, model = sys.argv[1], sys.argv[2] == "1", sys.argv[3]
files = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"],
                       capture_output=True, check=True).stdout.decode().split("\0")
files = [f for f in files if f and os.path.isfile(f) and not f.startswith("dist/")]
# Ignorés par git mais indispensables : réglages et thème Streamlit (dont le
# service des fichiers du studio caméra).
for extra in ("config.toml", ".streamlit/config.toml"):
    if os.path.isfile(extra) and extra not in files:
        files.append(extra)

def crlf(path):
    data = open(path, "rb").read().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
    return data

def bom(data):
    return data if data.startswith(b"\xef\xbb\xbf") else b"\xef\xbb\xbf" + data

with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
    for f in files:
        arc = "YVP/" + f
        if f.endswith(".ps1"):      # PowerShell 5.1 a besoin du BOM pour les accents
            z.writestr(arc, bom(crlf(f)))
        elif f.endswith(".bat") or (f.endswith(".txt") and f.startswith("tools/windows/")):
            z.writestr(arc, crlf(f))
        else:
            z.write(f, arc)
    # À la racine, là où on les voit tout de suite.
    z.writestr("YVP/LISEZ-MOI.txt", bom(crlf("tools/windows/LISEZ-MOI.txt")))
    if with_model:
        name, _, tag = model.partition(":"); tag = tag or "latest"
        base = "ollama/models"
        man = f"{base}/manifests/registry.ollama.ai/library/{name}/{tag}"
        z.write(man, "YVP/" + man)
        m = json.load(open(man))
        for layer in m["layers"] + [m["config"]]:
            blob = f"{base}/blobs/" + layer["digest"].replace(":", "-")
            z.write(blob, "YVP/" + blob, compress_type=zipfile.ZIP_STORED)
print(f"{len(files)} fichiers")
PY
ls -lh "$OUT"
