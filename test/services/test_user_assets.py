import os
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from app.models.schema import MaterialInfo, UserAsset
from app.services import curation, llm, material, user_assets
from app.utils import utils


class TestPlanShots(unittest.TestCase):
    def test_photo_gives_one_shot_and_long_video_three_spread_shots(self):
        assets = [UserAsset(path="/x/photo.jpg", caption="Notre boutique"),
                  UserAsset(path="/x/clip.mp4")]
        with patch.object(user_assets, "_video_duration", return_value=30.0):
            shots = user_assets.plan_shots(assets, clip_duration=3)

        self.assertEqual([s.kind for s in shots], ["image", "video", "video", "video"])
        self.assertEqual(shots[0].caption, "Notre boutique")
        # Début, milieu et fin de la vidéo, 3 s chacun.
        self.assertEqual([s.start for s in shots[1:]], [0.0, 13.5, 27.0])
        self.assertTrue(all(s.duration == 3 for s in shots[1:]))

    def test_short_video_gives_a_single_shot_of_its_length(self):
        with patch.object(user_assets, "_video_duration", return_value=2.0):
            shots = user_assets.plan_shots([UserAsset(path="/x/court.mov")], 3)
        self.assertEqual([(s.start, s.duration) for s in shots], [(0.0, 2.0)])

    def test_unreadable_video_is_skipped(self):
        with patch.object(user_assets, "_video_duration", side_effect=OSError("bad")):
            self.assertEqual(user_assets.plan_shots([UserAsset(path="/x/bad.mp4")], 3), [])


class TestResolveAssets(unittest.TestCase):
    def test_only_files_inside_local_videos_are_kept(self):
        local_dir = utils.storage_dir("local_videos", create=True)
        inside = os.path.join(local_dir, "test-user-asset.png")
        Image.new("RGB", (8, 8)).save(inside)
        try:
            with tempfile.NamedTemporaryFile(suffix=".png") as outside:
                kept = user_assets.resolve_assets([
                    {"path": inside, "caption": "  ok  "},
                    {"path": outside.name},
                    {"path": "../../etc/passwd"},
                ])
            self.assertEqual([a.caption for a in kept], ["ok"])
            self.assertEqual(os.path.realpath(kept[0].path), os.path.realpath(inside))
        finally:
            os.remove(inside)

    def test_file_count_limit(self):
        local_dir = utils.storage_dir("local_videos", create=True)
        paths = []
        try:
            for index in range(3):
                path = os.path.join(local_dir, f"test-limit-{index}.png")
                Image.new("RGB", (8, 8)).save(path)
                paths.append(path)
            with patch.object(user_assets, "MAX_ASSETS", 2):
                kept = user_assets.resolve_assets([{"path": p} for p in paths])
            self.assertEqual(len(kept), 2)
        finally:
            for path in paths:
                os.remove(path)


class TestAssignment(unittest.TestCase):
    def test_parse_assignment_accepts_numbers_and_p_labels(self):
        parsed = user_assets.parse_assignment(
            '<think>x</think>{"passages": [2, "P1", 9, null]}', 4, 3
        )
        self.assertEqual(parsed, [1, 0, None, None])

    def test_even_positions_follow_upload_order(self):
        self.assertEqual(user_assets._even_positions(3, 6), [0, 2, 4])
        self.assertEqual(user_assets._even_positions(4, 2), [0, 0, 1, 1])

    def test_spread_moves_a_duplicate_to_a_free_neighbour_only(self):
        # Deux assets sur P2 alors que P3 est libre : le second glisse sur P3.
        self.assertEqual(user_assets._spread([1, 1, 0], 3), [1, 2, 0])
        # Aucun voisin libre : on garde le choix du LLM.
        self.assertEqual(user_assets._spread([0, 0, 1, 3], 5), [0, 0, 1, 3])

    def test_llm_choice_is_used_and_missing_answers_fall_back(self):
        shots = [user_assets.AssetShot("/x/a.jpg", "image", "a"),
                 user_assets.AssetShot("/x/b.jpg", "image", "b")]
        with patch.object(llm, "_generate_response", return_value='{"passages": [3]}'):
            positions = user_assets.assign_shots(shots, ["p1", "p2", "p3"], "sujet")
        # Plan 1 : choix du LLM (P3) ; plan 2 : réparti dans l'ordre (P2).
        self.assertEqual(positions, [2, 1])

    def test_prompt_lists_captions_and_descriptions(self):
        shot = user_assets.AssetShot("/x/a.jpg", "image", "Café Yengui", description="a shop")
        prompt = user_assets.build_assignment_prompt([shot], ["Bienvenue."], "café")
        self.assertIn("V1. Café Yengui | a shop", prompt)
        self.assertIn("P1. Bienvenue.", prompt)


class TestMergeIntoGroups(unittest.TestCase):
    def test_assets_go_first_on_their_passage(self):
        stock = [("t1", [MaterialInfo(provider="pixabay", url="https://s/1")]),
                 ("t2", [MaterialInfo(provider="pixabay", url="https://s/2")])]
        shot = user_assets.AssetShot("/x/a.jpg", "image", "boutique", duration=3)
        context = curation.CurationContext(
            video_subject="café", video_script="Un. Deux.", user_shots=[shot]
        )
        with patch.object(user_assets, "assign_shots", return_value=[1]):
            merged = user_assets.merge_into_groups(stock, context)

        self.assertEqual([i.url for i in merged[0][1]], ["https://s/1"])
        self.assertEqual([i.provider for i in merged[1][1]], ["user", "pixabay"])
        self.assertEqual(merged[1][1][0].source_info["kind"], "image")

    def test_shots_of_one_video_keep_distinct_urls(self):
        shots = [user_assets.AssetShot("/x/v.mp4", "video", start=t, duration=3)
                 for t in (0.0, 13.5, 27.0)]
        urls = {user_assets.to_material(shot).url for shot in shots}
        self.assertEqual(len(urls), 3)

    def test_video_shot_is_trimmed_from_its_original_file(self):
        item = user_assets.to_material(
            user_assets.AssetShot("/x/v.mp4", "video", start=13.5, duration=3)
        )
        with patch.object(user_assets, "_trim_video", return_value="/c/cut.mp4") as trim:
            self.assertEqual(user_assets.save_user_material(item, 3), "/c/cut.mp4")
        shot = trim.call_args.args[0]
        self.assertEqual((shot.asset_path, shot.start, shot.duration), ("/x/v.mp4", 13.5, 3.0))

    def test_without_assets_results_are_unchanged(self):
        stock = [("t1", [MaterialInfo(url="https://s/1")])]
        self.assertEqual(user_assets.merge_into_groups(stock, None), stock)


class TestIntegration(unittest.TestCase):
    def test_script_prompt_includes_the_user_visuals(self):
        prompt = llm.build_script_prompt("café", visual_brief="1. Café Yengui | a shop")
        self.assertIn("# Visuals Provided by the User:", prompt)
        self.assertIn("1. Café Yengui | a shop", prompt)
        self.assertNotIn("Visuals Provided", llm.build_script_prompt("café"))

    def test_user_material_is_rendered_locally_not_downloaded(self):
        item = user_assets.to_material(
            user_assets.AssetShot("/x/a.jpg", "image", duration=3)
        )
        with patch.object(user_assets, "save_user_material", return_value="/x/a.mp4") as save:
            self.assertEqual(material._save_material(item, "", 3), "/x/a.mp4")
        save.assert_called_once_with(item, 3)

    def test_user_assets_have_no_credit_line(self):
        self.assertEqual(curation.credits_lines([{"provider": "user", "asset_id": "a"}]), [])


if __name__ == "__main__":
    unittest.main()
