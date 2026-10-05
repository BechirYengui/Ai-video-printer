"""Coupe la traduction automatique dans le profil navigateur dédié à YVP.

Un clic sur « Toujours traduire l'anglais » reste enregistré dans le profil et
passe avant les balises notranslate de la page : le titre devenait
« Imprimante vidéo Yengui ». Lancé par les lanceurs (Linux et Windows) avant
d'ouvrir la fenêtre, navigateur fermé : il réécrit ses préférences en quittant.

Usage : python browser_prefs.py <dossier_du_profil>
"""

import json
import os
import sys


def disable_translation(profile_dir: str) -> None:
    path = os.path.join(profile_dir, "Default", "Preferences")
    try:
        with open(path, encoding="utf-8") as handle:
            prefs = json.load(handle)
    except (OSError, ValueError):
        return  # premier lancement : le drapeau --disable-features=Translate suffit
    # account_values : réglages synchronisés depuis le compte Google du profil.
    scopes = [prefs]
    if isinstance(prefs.get("account_values"), dict):
        scopes.append(prefs["account_values"])
    translate = prefs.setdefault("translate", {})
    if translate.get("enabled") is False and not any(scope.get("translate_allowlists") for scope in scopes):
        return
    translate["enabled"] = False
    for scope in scopes:
        scope.pop("translate_allowlists", None)
    tmp = path + ".yvp-tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(prefs, handle, ensure_ascii=False, separators=(",", ":"))
    os.replace(tmp, path)


if __name__ == "__main__":
    try:
        disable_translation(sys.argv[1])
    except Exception as exc:  # ne jamais empêcher YVP de démarrer pour ça
        print(f"browser_prefs: {exc}", file=sys.stderr)
