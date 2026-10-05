#!/usr/bin/env bash
# Retire YVP du dock et du menu des applications (le projet n'est pas touché).
set -euo pipefail
FAVS="$(gsettings get org.gnome.shell favorite-apps | sed "s/, 'yvp.desktop'//; s/'yvp.desktop', //; s/'yvp.desktop'//")"
gsettings set org.gnome.shell favorite-apps "$FAVS"
rm -f "$HOME/.local/share/applications/yvp.desktop" "$HOME/.local/share/icons/hicolor/scalable/apps/yvp.svg"
echo "YVP retiré du dock."
