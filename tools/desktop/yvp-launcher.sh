#!/usr/bin/env bash
# Lanceur bureau de YenguiVideoPrinter (YVP).
# Démarre Ollama + la webui (via start.sh), ouvre une fenêtre dédiée,
# et arrête tout dès que cette fenêtre est fermée (ou à la fermeture de session).
set -uo pipefail
DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../.." && pwd)"
STATE="${XDG_STATE_HOME:-$HOME/.local/state}/yvp"
PROFILE="$STATE/browser-profile"
LOG="$DIR/logs/desktop.log"
mkdir -p "$STATE" "$DIR/logs"

notify() { command -v notify-send >/dev/null && notify-send -a YVP -i "$DIR/tools/desktop/yvp.svg" "YVP" "$1" || true; }
fail() { zenity --error --title=YVP --width=500 --text="$1\n\n$(tail -n 15 "$LOG" 2>/dev/null | sed 's/[<>&]//g')" 2>/dev/null; exit 1; }

# Une seule instance : si YVP tourne déjà, on rouvre juste la fenêtre.
exec 9>"$STATE/lock"
if ! flock -n 9; then
  URL="$(cat "$STATE/url" 2>/dev/null || echo http://127.0.0.1:8501)"
  exec google-chrome --user-data-dir="$PROFILE" --class=YVP --disable-features=Translate --app="$URL"
fi

# Streamlit ne doit pas ouvrir d'onglet tout seul : c'est nous qui ouvrons la fenêtre.
export STREAMLIT_SERVER_HEADLESS=true
: >"$LOG"
setsid "$DIR/start.sh" >>"$LOG" 2>&1 < /dev/null &
SERVER_PID=$!   # setsid => PID == PGID : on peut tuer tout le groupe (webui + ollama)

cleanup() {
  trap - EXIT INT TERM HUP
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 10); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
  fi
  rm -f "$STATE/url"
}
trap cleanup EXIT INT TERM HUP

notify "Démarrage en cours…"
URL=""
for _ in $(seq 1 900); do   # jusqu'à 15 min (premier téléchargement du modèle)
  kill -0 "$SERVER_PID" 2>/dev/null || fail "YVP n'a pas pu démarrer (voir logs/desktop.log)."
  URL="$(grep -o 'WebUI address: http://[0-9.:]*' "$LOG" | tail -n1 | cut -d' ' -f3)"
  [ -n "$URL" ] && curl -fs "$URL/_stcore/health" >/dev/null 2>&1 && break
  URL=""
  sleep 1
done
[ -n "$URL" ] || fail "YVP ne répond pas après 15 minutes."
echo "$URL" >"$STATE/url"

# Profil de navigateur dédié => ce processus reste vivant tant que la fenêtre est ouverte.
# Pas de traduction automatique : elle renommait l'app (« Imprimante vidéo Yengui »).
# Le drapeau ne suffit pas si « Toujours traduire » a été cliqué un jour : on l'efface.
pgrep -f -- "--user-data-dir=$PROFILE" >/dev/null || \
  "$DIR/.venv/bin/python" "$DIR/tools/desktop/browser_prefs.py" "$PROFILE" >>"$LOG" 2>&1
google-chrome --user-data-dir="$PROFILE" --class=YVP --no-first-run --disable-features=Translate \
  --no-default-browser-check --app="$URL" >>"$LOG" 2>&1
# Si une fenêtre YVP était déjà ouverte, Chrome lui confie la nouvelle et rend la
# main tout de suite : on attend donc tant qu'un Chrome de ce profil tourne,
# sinon le serveur serait arrêté alors que la fenêtre est encore là.
while pgrep -f -- "--user-data-dir=$PROFILE" >/dev/null; do sleep 2; done
# Fenêtre fermée : le trap EXIT arrête la webui et Ollama.
