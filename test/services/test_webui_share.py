import ast
import json
import re
from pathlib import Path

from streamlit.testing.v1 import AppTest

from app.services import llm

ROOT_DIR = Path(__file__).resolve().parents[2]
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"
SHARE_NAMES = {
    "SHARE_PLATFORMS",
    "_default_share_metadata",
    "_format_share_caption",
    "_render_share_panel",
}


def _share_source():
    """Extrait du WebUI le code du panneau de partage, sans exécuter la page."""
    tree = ast.parse(WEBUI_MAIN.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in SHARE_NAMES)
        or (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in SHARE_NAMES for t in node.targets)
        )
    ]
    assert len(nodes) == len(SHARE_NAMES)
    return ast.unparse(ast.Module(body=nodes, type_ignores=[]))


def _share_namespace():
    namespace = {}
    exec(_share_source(), namespace)
    return namespace


def test_share_platforms_cover_the_four_networks_with_llm_specs():
    platforms = _share_namespace()["SHARE_PLATFORMS"]
    labels = [label for _, label, _, _ in platforms]
    assert labels == ["YouTube", "TikTok", "Instagram", "Facebook"]
    for platform, _, url, _ in platforms:
        assert platform in llm.SOCIAL_PLATFORMS
        assert url.startswith("https://")


def test_share_texts_are_translated_in_french_and_english():
    platforms = _share_namespace()["SHARE_PLATFORMS"]
    keys = set(
        ast.literal_eval(match)
        for match in re.findall(r'tr\(\s*("[^"]+")', _share_source())
    ) | {hint for *_, hint in platforms}
    for lang in ("fr", "en"):
        translation = json.loads(
            (ROOT_DIR / "webui" / "i18n" / f"{lang}.json").read_text(encoding="utf-8")
        )["Translation"]
        assert keys <= translation.keys(), keys - translation.keys()


def test_share_caption_appends_hashtags_after_a_blank_line():
    fmt = _share_namespace()["_format_share_caption"]
    assert fmt({"caption": "Le café ☕", "hashtags": ["#cafe", "#sante"]}) == (
        "Le café ☕\n\n#cafe #sante"
    )
    assert fmt({"caption": "Sans tags", "hashtags": []}) == "Sans tags"


def test_default_share_metadata_uses_subject_then_script():
    default = _share_namespace()["_default_share_metadata"]
    assert default("café", "Le script.")["title"] == "café"
    assert default("", "Le script.")["title"] == "Le script."
    assert default(None, None) == {"title": "", "caption": "", "hashtags": []}


def _panel_app():
    script = f'''
import html
import streamlit as st
from types import SimpleNamespace

opened = st.session_state.setdefault("opened", [])
calls = st.session_state.setdefault("llm_calls", [])
tr = lambda key: key
open_task_folder = opened.append

def _fake_metadata(video_subject, video_script, platform):
    calls.append(platform)
    return {{"title": "Titre IA " + platform, "caption": "Légende", "hashtags": ["#ia"]}}

llm = SimpleNamespace(generate_social_metadata=_fake_metadata)
{_share_source()}
_render_share_panel("task-1", "/tmp/final-1.mp4", "café", "Le script.", key="k")
'''
    return AppTest.from_string(script).run()


def test_share_panel_renders_upload_links_and_copyable_text():
    at = _panel_app()
    assert not at.exception
    # Un réseau à la fois : le bouton « Publier » suit le réseau choisi.
    for platform, _, url, _ in _share_namespace()["SHARE_PLATFORMS"]:
        at.session_state["k_platform"] = platform
        at.run()
        assert not at.exception
        assert [b.proto.url for b in at.get("link_button")] == [url]
        # Avant génération IA : sujet en titre, script en description.
        assert [c.value for c in at.code][:2] == ["café", "Le script."]


def test_share_panel_buttons_generate_text_and_open_folder():
    at = _panel_app()
    at.session_state["k_platform"] = "tiktok"
    at.run()
    at.button(key="k_generate_tiktok").click().run()
    assert at.session_state["llm_calls"] == ["tiktok"]
    assert "Titre IA tiktok" in [c.value for c in at.code]
    assert "Légende\n\n#ia" in [c.value for c in at.code]

    at.button(key="k_open_folder").click().run()
    assert at.session_state["opened"] == ["task-1"]
