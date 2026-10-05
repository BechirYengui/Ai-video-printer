import os
import tempfile

# Les tests ne doivent jamais lire ni écrire les vidéos de l'utilisateur
# (Vidéos/YVP) : le studio caméra range les siennes dans un dossier jetable.
os.environ.setdefault(
    "YVP_RECORDINGS_DIR", tempfile.mkdtemp(prefix="yvp-test-recordings-")
)
os.environ.setdefault(
    "YVP_EXPLAINER_DIR", tempfile.mkdtemp(prefix="yvp-test-explainer-")
)
os.environ.setdefault(
    "YVP_HEADSHOT_DIR", tempfile.mkdtemp(prefix="yvp-test-headshot-")
)
