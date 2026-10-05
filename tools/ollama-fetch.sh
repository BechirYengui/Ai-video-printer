#!/usr/bin/env bash
# Télécharge un modèle Ollama avec curl et l'installe dans ./ollama/models.
# Robuste aux coupures : morceaux en parallèle, reprise par morceau, relances
# sans limite, DNS via DoH (contourne le résolveur local instable).
# Usage: tools/ollama-fetch.sh qwen3:8b   (relançable à tout moment, reprend où il en est)
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1%%:*}"; TAG="${1#*:}"; [ "$TAG" = "$1" ] && TAG=latest
PARTS="${PARTS:-4}"
REG="https://registry.ollama.ai/v2/library/$NAME"
MODELS="$DIR/ollama/models"
CURL=(curl -fsL --doh-url https://cloudflare-dns.com/dns-query --connect-timeout 20 --speed-limit 10000 --speed-time 45)
mkdir -p "$MODELS/blobs" "$MODELS/manifests/registry.ollama.ai/library/$NAME"

# Relance une commande sans limite, attente croissante plafonnée à 60 s.
retry() {
  local wait=5
  until "$@"; do sleep "$wait"; wait=$(( wait < 60 ? wait * 2 : 60 )); done
}

manifest="$MODELS/.manifest-$NAME-$TAG"
retry "${CURL[@]}" -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' "$REG/manifests/$TAG" -o "$manifest"

# Télécharge l'octet [start, end] dans $out, en reprenant ce qui est déjà là.
fetch_range() {
  local digest=$1 start=$2 end=$3 out=$4 have
  local want=$(( end - start + 1 ))
  while :; do
    have=$(stat -c %s "$out" 2>/dev/null || echo 0)
    [ "$have" -ge "$want" ] && return 0
    "${CURL[@]}" -r "$(( start + have ))-$end" "$REG/blobs/$digest" >>"$out" || sleep 5
  done
}

# Chaque couche du manifeste : "digest taille" (la config a sa propre taille aussi).
layers=$(tr -d '\n ' <"$manifest" | grep -o '"digest":"sha256:[0-9a-f]\{64\}","size":[0-9]\+\|"size":[0-9]\+,"digest":"sha256:[0-9a-f]\{64\}"' \
  | sed -E 's/.*(sha256:[0-9a-f]{64}).*/\1/;' | sort -u)
for d in $layers; do
  hex="${d#sha256:}"; out="$MODELS/blobs/sha256-$hex"
  if [ -f "$out" ] && echo "$hex  $out" | sha256sum -c --quiet - 2>/dev/null; then echo "déjà là: ${hex:0:12}"; continue; fi
  size=$(tr -d '\n ' <"$manifest" | grep -o "\"size\":[0-9]\+,\"digest\":\"$d\"\|\"digest\":\"$d\",\"size\":[0-9]\+" | grep -o '"size":[0-9]\+' | cut -d: -f2)
  n=$PARTS; [ "$size" -lt 50000000 ] && n=1
  chunk=$(( (size + n - 1) / n ))
  echo "${hex:0:12}: $(( size / 1048576 )) Mo en $n morceau(x)"
  pids=()
  for i in $(seq 0 $(( n - 1 ))); do
    s=$(( i * chunk )); e=$(( s + chunk - 1 )); [ "$e" -ge "$size" ] && e=$(( size - 1 ))
    fetch_range "$d" "$s" "$e" "$out.part$i" & pids+=($!)
  done
  # Affiche la progression jusqu'à la fin de tous les morceaux.
  while kill -0 "${pids[@]}" 2>/dev/null; do
    got=$(stat -c %s "$out".part* 2>/dev/null | paste -sd+ | bc); echo "  ${hex:0:12}: $(( got / 1048576 ))/$(( size / 1048576 )) Mo"; sleep 60
  done
  wait "${pids[@]}"
  for i in $(seq 0 $(( n - 1 ))); do cat "$out.part$i"; done >"$out.tmp"
  if echo "$hex  $out.tmp" | sha256sum -c --quiet -; then
    mv "$out.tmp" "$out"; rm -f "$out".part*; echo "OK: ${hex:0:12}"
  else
    # On garde les morceaux : une relance ne retéléchargera que ce qui manque.
    rm -f "$out.tmp"; echo "ERREUR: empreinte invalide pour ${hex:0:12} (morceaux conservés)"; exit 1
  fi
done

mv "$manifest" "$MODELS/manifests/registry.ollama.ai/library/$NAME/$TAG"
echo "Modèle $NAME:$TAG installé."
