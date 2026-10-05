"""Interdit la traduction automatique dès le HTML servi par Streamlit.

La page posait déjà translate="no" en JavaScript, mais une fois chargée : avec
« Toujours traduire » (Chrome, Edge), le titre avait déjà eu le temps de devenir
« Imprimante vidéo Yengui ». On l'écrit donc dans streamlit/static/index.html
lui-même, avant le lancement du serveur. À refaire après chaque `uv sync` qui
réinstalle Streamlit : les lanceurs l'appellent à chaque démarrage.

Usage : python notranslate.py   (avec le Python de l'application)
"""

import os
import sys

MARK = '<meta name="google" content="notranslate" />'


def patch(path: str) -> bool:
    with open(path, encoding="utf-8") as handle:
        page = handle.read()
    if MARK in page:
        return False
    page = page.replace('<html lang="en">', '<html lang="en" translate="no" class="notranslate">', 1)
    page = page.replace("<head>", "<head>\n    " + MARK, 1)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(page)
    return True


if __name__ == "__main__":
    try:
        import streamlit

        patch(os.path.join(os.path.dirname(streamlit.__file__), "static", "index.html"))
    except Exception as exc:  # ne jamais empêcher YVP de démarrer pour ça
        print(f"notranslate: {exc}", file=sys.stderr)
