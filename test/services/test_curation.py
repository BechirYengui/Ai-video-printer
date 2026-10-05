"""Tests YVP : recadrage des clips, sélection par modèle de vision, photos libres."""

import unittest
from unittest.mock import MagicMock, patch

from app.models.schema import MaterialInfo, VideoAspect
from app.services import curation, material


def _item(asset_id, thumbnail="https://cdn.example.com/t.jpg", provider="pixabay"):
    return MaterialInfo(
        provider=provider,
        url=f"https://cdn.example.com/{asset_id}.mp4",
        duration=20,
        source_info={"provider": provider, "asset_id": asset_id, "thumbnail": thumbnail},
    )


class TestScoreParsing(unittest.TestCase):
    def test_parses_plain_and_wrapped_scores(self):
        self.assertEqual(curation.parse_score("7"), 7)
        self.assertEqual(curation.parse_score("Score: 10/10"), 10)
        self.assertEqual(curation.parse_score("<think>maybe 3</think>8"), 8)

    def test_rejects_answers_without_score(self):
        self.assertIsNone(curation.parse_score("no idea"))
        self.assertIsNone(curation.parse_score(None))


class TestScriptPassages(unittest.TestCase):
    def test_each_term_gets_its_part_of_the_script(self):
        script = "Un. Deux. Trois. Quatre."
        self.assertEqual(
            curation.split_script_for_terms(script, 2), ["Un. Deux.", "Trois. Quatre."]
        )

    def test_more_terms_than_sentences_still_returns_one_passage_per_term(self):
        passages = curation.split_script_for_terms("Une seule phrase.", 3)
        self.assertEqual(len(passages), 3)
        self.assertTrue(all(passages))


class TestCropRule(unittest.TestCase):
    def test_4k_landscape_covers_portrait_target_without_upscale(self):
        self.assertTrue(
            material._covers_target_without_upscale(3840, 2160, VideoAspect.portrait)
        )

    def test_1080p_landscape_would_need_upscale(self):
        self.assertFalse(
            material._covers_target_without_upscale(1920, 1080, VideoAspect.portrait)
        )

    def test_cached_landscape_items_are_kept_only_when_crop_is_enabled(self):
        item = _item("a")
        item.source_info["rendition"] = {"width": 3840, "height": 2160}
        with patch.dict(material.config.app, {"stock_allow_crop": False}):
            self.assertEqual(material._filter_materials_by_aspect([item], VideoAspect.portrait), [])
        with patch.dict(material.config.app, {"stock_allow_crop": True}):
            self.assertEqual(
                material._filter_materials_by_aspect([item], VideoAspect.portrait), [item]
            )


class TestPixabayCrop(unittest.TestCase):
    def _response(self):
        response = MagicMock(status_code=200, headers={"content-type": "application/json"})
        response.json.return_value = {
            "hits": [
                {
                    "id": 1,
                    "duration": 20,
                    "tags": "coffee, cup",
                    "videos": {
                        "large": {"url": "https://cdn.pixabay.com/v/1_large.mp4", "width": 3840, "height": 2160, "thumbnail": "https://cdn.pixabay.com/v/1_large.jpg"},
                        "small": {"url": "https://cdn.pixabay.com/v/1_small.mp4", "width": 1920, "height": 1080, "thumbnail": "https://cdn.pixabay.com/v/1_small.jpg"},
                    },
                }
            ]
        }
        return response

    def test_landscape_4k_clip_is_accepted_for_portrait_only_with_crop(self):
        with patch.object(material, "get_api_key", return_value="k"), patch.object(
            material, "_is_cloudflare_challenge", return_value=False
        ), patch.object(material.requests, "get", return_value=self._response()):
            with patch.dict(material.config.app, {"stock_allow_crop": False}):
                self.assertEqual(material.search_videos_pixabay("coffee", 5), [])
            with patch.dict(material.config.app, {"stock_allow_crop": True}):
                items = material.search_videos_pixabay("coffee", 5)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].url, "https://cdn.pixabay.com/v/1_large.mp4")
        self.assertEqual(items[0].source_info["thumbnail"], "https://cdn.pixabay.com/v/1_small.jpg")
        self.assertEqual(items[0].source_info["tags"], "coffee, cup")


class TestJudgeParsing(unittest.TestCase):
    def test_reads_json_scores_and_ignores_reasoning(self):
        text = '<think>maybe {"scores": [1, 1]}</think> {"scores": [4, 9]}'
        self.assertEqual(curation.parse_judge_scores(text, 2), [4, 9])

    def test_falls_back_to_last_list_of_the_right_size(self):
        text = "S1 notes: [3, 5]. Final Scores: [6, 0]"
        self.assertEqual(curation.parse_judge_scores(text, 2), [6, 0])

    def test_invalid_answer_gives_no_scores(self):
        self.assertEqual(curation.parse_judge_scores("not json", 2), [None, None])
        self.assertEqual(curation.parse_judge_scores('{"scores": [12, "x"]}', 2), [None, None])

    def test_prompt_contains_the_shot_and_its_narration(self):
        context = curation.CurationContext(video_subject="café")
        prompt = curation.build_judge_prompt(
            context, {"id": "S1", "term": "coffee", "passage": "Le café.", "descriptions": ["a cup"]}
        )
        self.assertIn('Shot to fill: "coffee"', prompt)
        self.assertIn('Narration: "Le café."', prompt)
        self.assertIn("1. a cup", prompt)

    def test_qwen3_judge_gives_short_reasons_instead_of_thinking(self):
        context = curation.CurationContext(video_subject="café")
        shot = {"id": "S1", "term": "coffee", "passage": "", "descriptions": ["a cup", "a car"]}
        with patch.object(curation, "_judge_is_qwen3", return_value=True):
            prompt = curation.build_judge_prompt(context, shot)
        self.assertIn('"reasons"', prompt)
        self.assertTrue(prompt.endswith("/no_think"))
        answer = '{"reasons": ["coffee cup, 9 out of 10", "unrelated car"], "scores": [9, 1]}'
        self.assertEqual(curation.parse_judge_scores(answer, 2), [9, 1])

    def test_wikimedia_images_use_the_official_thumbnail(self):
        self.assertEqual(
            curation._openverse_thumbnail(
                "https://upload.wikimedia.org/wikipedia/commons/5/57/Cup.jpg", "https://api.openverse.org/t"
            ),
            "https://upload.wikimedia.org/wikipedia/commons/thumb/5/57/Cup.jpg/500px-Cup.jpg",
        )


class TestCurateSearchResults(unittest.TestCase):
    def setUp(self):
        self.context = curation.CurationContext(video_subject="café", video_script="Le café.")

    def _run(self, results, judge, config_overrides=None):
        app_config = {"curation_enabled": True, "curation_min_score": 5,
                      "curation_max_candidates": 6, "curation_openverse_fallback": False}
        app_config.update(config_overrides or {})
        with patch.dict(curation.config.app, app_config), patch.object(
            curation, "vision_model_available", return_value=True
        ), patch.object(curation, "describe_image", side_effect=lambda url: f"desc {url}"), patch.object(
            curation, "judge_shots", side_effect=judge
        ):
            return curation.curate_search_results(results, self.context, VideoAspect.portrait)

    @staticmethod
    def _ids(items):
        return [i.source_info["asset_id"] for i in items]

    def test_best_scored_clips_come_first(self):
        results = [("coffee", [_item("a"), _item("b"), _item("c")])]
        curated = self._run(results, lambda ctx, shots: {"S1": [3, 9, 7]})
        self.assertEqual(self._ids(curated[0][1]), ["b", "c", "a"])

    def test_a_clip_is_never_selected_for_two_terms(self):
        results = [("coffee", [_item("a")]), ("cup", [_item("a"), _item("b")])]
        curated = self._run(results, lambda ctx, shots: {"S1": [9], "S2": [8]})
        self.assertEqual(self._ids(curated[1][1]), ["b"])

    def test_disabled_curation_returns_results_untouched(self):
        results = [("coffee", [_item("a")])]
        with patch.dict(curation.config.app, {"curation_enabled": False}):
            self.assertEqual(
                curation.curate_search_results(results, self.context, VideoAspect.portrait),
                results,
            )

    def test_unavailable_vision_model_keeps_original_order(self):
        results = [("coffee", [_item("a"), _item("b")])]
        with patch.dict(curation.config.app, {"curation_enabled": True}), patch.object(
            curation, "vision_model_available", return_value=False
        ):
            self.assertEqual(
                curation.curate_search_results(results, self.context, VideoAspect.portrait),
                results,
            )

    def test_openverse_photo_is_used_when_no_clip_is_relevant(self):
        photo = _item("p", provider="openverse")
        photo.source_info["kind"] = "image"

        def judge(ctx, shots):
            return {shot["id"]: ([8] if shot["id"].startswith("P") else [2]) for shot in shots}

        with patch.object(curation, "search_openverse_images", return_value=[photo]):
            curated = self._run([("heart health", [_item("a")])], judge,
                                {"curation_openverse_fallback": True})
        # Le clip noté 2/10 est hors sujet : il est écarté, la photo le remplace.
        self.assertEqual(self._ids(curated[0][1]), ["p"])

    def test_off_topic_clips_are_dropped_but_fair_ones_are_kept(self):
        results = [("coffee", [_item("a"), _item("b"), _item("c")])]
        curated = self._run(results, lambda ctx, shots: {"S1": [8, 4, 1]})
        self.assertEqual(self._ids(curated[0][1]), ["a", "b"])

    def test_term_without_relevant_clip_borrows_from_a_rich_term(self):
        results = [
            ("coffee", [_item("a"), _item("b"), _item("c")]),
            ("energy", [_item("w")]),
        ]
        curated = self._run(results, lambda ctx, shots: {"S1": [9, 8, 7], "S2": [0]})
        self.assertEqual(self._ids(curated[1][1]), ["c"])
        self.assertEqual(self._ids(curated[0][1]), ["a", "b"])

    def test_failed_judge_keeps_every_candidate(self):
        results = [("coffee", [_item("a"), _item("b")])]
        curated = self._run(results, lambda ctx, shots: {})
        self.assertEqual(self._ids(curated[0][1]), ["a", "b"])


class TestOpenverseAndCredits(unittest.TestCase):
    def test_small_photos_are_skipped_and_licences_are_kept(self):
        response = MagicMock()
        response.json.return_value = {
            "results": [
                {"id": "big", "url": "https://upload.wikimedia.org/big.jpg", "width": 4000, "height": 3000,
                 "creator": "Ana", "license": "by", "license_version": "2.0",
                 "license_url": "https://creativecommons.org/licenses/by/2.0/",
                 "attribution": '"Cup" by Ana is licensed under CC BY 2.0.',
                 "foreign_landing_url": "https://commons.wikimedia.org/x", "thumbnail": "https://api.openverse.org/t"},
                {"id": "small", "url": "https://upload.wikimedia.org/small.jpg", "width": 800, "height": 600},
            ]
        }
        with patch.object(curation.requests, "get", return_value=response):
            items = curation.search_openverse_images("cup", VideoAspect.portrait)
        self.assertEqual([i.source_info["asset_id"] for i in items], ["big"])
        self.assertEqual(items[0].source_info["kind"], "image")

        record = material._material_source_record(items[0], "/tmp/x.mp4")
        self.assertEqual(record["license"], "by")
        self.assertEqual(curation.credits_lines([record]), ['"Cup" by Ana is licensed under CC BY 2.0.'])

    def test_credits_for_stock_clips_name_the_creator_and_page(self):
        lines = curation.credits_lines(
            [{"provider": "pixabay", "creator": {"name": "Bob"}, "source_page": "https://pixabay.com/v/1"}]
        )
        self.assertEqual(lines, ["by Bob — via Pixabay — https://pixabay.com/v/1"])


if __name__ == "__main__":
    unittest.main()
