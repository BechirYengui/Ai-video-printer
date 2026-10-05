import os
import tempfile
import unittest
from unittest.mock import patch

from app.services import explainer


def _raw(**overrides):
    plan = {
        "title": "Le DNS",
        "elements": [
            {"id": "a", "kind": "phone", "label": "Téléphone", "icon": ["phone", "app"]},
            {"id": "b", "kind": "router", "label": "Routeur"},
            {"id": "c", "kind": "other", "label": "Drone", "icon": "delivery drone"},
            {"id": "d", "kind": "database", "label": "Base"},
        ],
        "scenes": [
            {"narration": "Tu tapes une adresse web.", "show": ["a"], "flow": None},
            {"narration": "Le téléphone demande au routeur.", "show": ["a", "b"],
             "flow": {"from": "a", "to": "b", "label": "Requête"}},
            {"narration": "Le routeur interroge le drone.", "show": ["c"],
             "flow": {"from": "b", "to": "c", "label": "Question"}},
            {"narration": "Le drone trouve et renvoie la réponse.", "show": ["d"],
             "flow": {"from": "c", "to": "d", "label": "Réponse"}, "highlight": "d"},
        ],
    }
    plan.update(overrides)
    return plan


class TestExplainerPlan(unittest.TestCase):
    def test_icons_come_from_the_library_and_keywords_are_plain_text(self):
        plan = explainer.repair_plan(_raw())
        by_id = {e["id"]: e for e in plan["elements"]}
        self.assertEqual(by_id["a"]["kind"], "phone")
        self.assertEqual(by_id["a"]["icon"], "phone app")  # liste du modèle → texte
        self.assertEqual(by_id["c"]["kind"], "custom")
        self.assertEqual(by_id["c"]["icon_name"], "drone")
        self.assertIn("<path", by_id["c"]["svg"])

    def test_a_returned_answer_goes_back_to_who_asked_and_is_highlighted(self):
        last = explainer.repair_plan(_raw())["scenes"][-1]
        # « renvoie » : la flèche revient vers le routeur (qui a demandé), pas vers la base.
        self.assertEqual((last["flow"]["from"], last["flow"]["to"]), ("c", "b"))
        self.assertTrue(last["flow"]["reverse"])
        self.assertEqual(last["highlight"], "b")

    def test_unusable_plans_are_rejected(self):
        with self.assertRaises(explainer.ExplainerError):
            explainer.repair_plan(_raw(elements=[{"id": "a", "kind": "phone", "label": "Seul"}]))
        with self.assertRaises(explainer.ExplainerError):
            explainer.repair_plan(_raw(scenes=[{"narration": "Une seule scène ici.", "show": ["a"]}]))

    def test_edited_plan_keeps_the_chosen_icon(self):
        plan = explainer.repair_plan(_raw())
        again = explainer.repair_plan(plan)
        self.assertEqual([e.get("icon_name") for e in again["elements"]],
                         [e.get("icon_name") for e in plan["elements"]])

    def test_voice_falls_back_to_the_script_language(self):
        self.assertEqual(explainer.edge_voice_for("fr-BE-CharlineNeural-Female", "fr-FR"), "fr-BE-CharlineNeural")
        # Voix française pour un script anglais : voix anglaise par défaut.
        self.assertEqual(explainer.edge_voice_for("fr-BE-CharlineNeural-Female", "en-US"), "en-US-AvaMultilingualNeural")
        self.assertEqual(explainer.edge_voice_for("siliconflow:x:alex-Male", "fr-FR"), "fr-FR-DeniseNeural")


class TestExplainerVideos(unittest.TestCase):
    def test_videos_are_listed_and_deleted_by_known_names_only(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(explainer, "output_dir", return_value=tmp):
            name = "explainer-20261004-120000-abcdef.mp4"
            open(os.path.join(tmp, name), "wb").close()
            open(os.path.join(tmp, "autre.mp4"), "wb").close()
            self.assertEqual([v["name"] for v in explainer.list_videos()], [name])
            with self.assertRaises(explainer.ExplainerError):
                explainer.delete_video("../autre.mp4")
            explainer.delete_video(name)
            self.assertEqual(explainer.list_videos(), [])
            self.assertTrue(os.path.exists(os.path.join(tmp, "autre.mp4")))


if __name__ == "__main__":
    unittest.main()
