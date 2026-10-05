#!/usr/bin/env bash
# Installe l'icône YVP dans le menu des applications et l'épingle au dock Ubuntu.
# Rien ne démarre automatiquement : YVP ne tourne que lorsqu'on clique sur l'icône.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"
mkdir -p "$APPS" "$ICONS"
chmod +x "$HERE/yvp-launcher.sh" "$HERE/../../start.sh"
cp "$HERE/yvp.svg" "$ICONS/yvp.svg"

cat >"$APPS/yvp.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=YenguiVideoPrinter
GenericName=YVP
Comment=Générateur de vidéos (démarre au clic, s'arrête à la fermeture)
Exec=$HERE/yvp-launcher.sh
Icon=yvp
Terminal=false
Categories=AudioVideo;Video;
StartupNotify=true
StartupWMClass=YVP
DESKTOP
update-desktop-database "$APPS" 2>/dev/null || true
gtk-update-icon-cache -q "$HOME/.local/share/icons/hicolor" 2>/dev/null || true

# Épingler au dock sans toucher aux autres favoris.
FAVS="$(gsettings get org.gnome.shell favorite-apps)"
if [[ "$FAVS" != *"'yvp.desktop'"* ]]; then
  gsettings set org.gnome.shell favorite-apps "${FAVS%]*}, 'yvp.desktop']"
fi
echo "YVP installé et épinglé au dock."
