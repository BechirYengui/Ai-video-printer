import os
import tempfile
import unittest
from unittest.mock import patch

from app.models.schema import MaterialInfo, VideoParams
from app.services import curation, llm, quality, video


class _TempHistory(unittest.TestCase):
    """Chaque test a son propre historique : le vrai n'est jamais touché."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        path = os.path.join(self._dir.name, "eta_history.json")
        patcher = patch.object(quality, "_history_path", return_value=path)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._dir.cleanup)


class TestPresets(unittest.TestCase):
    def test_levels_trade_speed_for_quality(self):
        low, medium, high = (quality.preset(level) for level in ("low", "medium", "high"))
        self.assertFalse(low.curation)
        self.assertEqual((medium.max_candidates, medium.openverse), (3, False))
        self.assertEqual((high.max_candidates, high.openverse), (6, True))
        self.assertEqual(
            [p.encoder_preset for p in (low, medium, high)], ["ultrafast", "veryfast", "medium"]
        )

    def test_unknown_level_falls_back_to_historical_behaviour(self):
        self.assertEqual(quality.normalize("ultra"), "high")
        self.assertEqual(VideoParams(video_subject="x").video_quality, "high")


class TestEstimates(_TempHistory):
    def test_low_is_faster_than_medium_faster_than_high(self):
        work = quality.Workload(video_seconds=30, terms=8)
        low, medium, high = (quality.estimate_seconds(l, work) for l in quality.QUALITY_LEVELS)
        self.assertLess(low, medium)
        self.assertLess(medium, high)

    def test_skipped_stages_cost_nothing(self):
        full = quality.estimate_seconds("low", quality.Workload())
        no_script = quality.estimate_seconds("low", quality.Workload(skipped={"script"}))
        self.assertAlmostEqual(full - no_script, quality.stage_rate("script", "low"))

    def test_history_learns_from_real_runs_but_ignores_instant_stages(self):
        quality.record("script", "high", seconds=0.01, units=1)  # simulé : ignoré
        self.assertEqual(quality.stage_rate("script", "high"), 30.0)
        quality.record("script", "high", seconds=10, units=1)
        self.assertAlmostEqual(quality.stage_rate("script", "high"), 30 * 0.6 + 10 * 0.4)

    def test_tracker_counts_down_and_publishes_for_the_ui(self):
        tracker = quality.EtaTracker("task-eta", "low", quality.Workload(video_seconds=10, terms=2))
        with patch.object(quality.time, "time", return_value=1000.0):
            tracker.started_at = 1000.0
            tracker.begin("script")
        before = quality.read_eta("task-eta")["eta_seconds"]
        with patch.object(quality.time, "time", return_value=1010.0):
            tracker.begin("terms")
        after = quality.read_eta("task-eta")
        self.assertLess(after["eta_seconds"], before)
        percent, remaining = quality.time_progress(after, now=1010.0)
        self.assertEqual(remaining, after["eta_seconds"])
        self.assertTrue(1 <= percent <= 99)

    def test_overdue_task_still_says_which_stage_is_running(self):
        tracker = quality.EtaTracker("task-late", "low", quality.Workload(video_seconds=10, terms=2))
        with patch.object(quality.time, "time", return_value=1000.0):
            tracker.started_at = 1000.0
            tracker.begin("render")
        eta = quality.read_eta("task-late")
        self.assertEqual(eta["stage"], "render")
        # Bien après l'estimation : plus de temps restant, mais l'étape et le
        # temps écoulé restent connus pour l'affichage.
        percent, remaining = quality.time_progress(eta, now=5000.0)
        self.assertEqual((percent, remaining), (99, 0))
        self.assertEqual(quality.elapsed_seconds(eta, now=5000.0), 4000)
        self.assertEqual(quality.elapsed_seconds({}), 0)

    def test_measured_progress_drives_the_remaining_time(self):
        tracker = quality.EtaTracker("task-measured", "low", quality.Workload(video_seconds=10, terms=2))
        with patch.object(quality.time, "time", return_value=1000.0):
            tracker.started_at = 1000.0
            tracker.begin("render")
        # 120 s passées pour 25 % du rendu : il en reste environ 360 s, quelle
        # que soit l'estimation de départ.
        with patch.object(quality.time, "time", return_value=1120.0):
            quality.report_stage_progress("task-measured", 0.25)
        self.assertAlmostEqual(quality.read_eta("task-measured")["eta_seconds"], 360, delta=1)
        # Sous-partie [0.55, 1] : 50 % de la sous-partie = 77,5 % de l'étape.
        with patch.object(quality.time, "time", return_value=1155.0):
            quality.stage_progress_callback("task-measured", 0.55, 1.0)(0.5)
        self.assertAlmostEqual(quality.read_eta("task-measured")["eta_seconds"], 155 * 0.225 / 0.775, delta=1)
        tracker.finish()
        quality.report_stage_progress("task-measured", 0.9)  # tâche finie : ignoré
        self.assertEqual(quality.read_eta("task-measured")["eta_seconds"], 0)

    def test_time_progress_needs_eta_fields(self):
        self.assertIsNone(quality.time_progress({}))

    def test_format_duration_gives_an_order_of_magnitude(self):
        self.assertEqual(quality.format_duration(42), "40 s")
        self.assertEqual(quality.format_duration(185), "3 min")
        self.assertEqual(quality.format_duration(3900), "1 h 05")


class TestQualityInPipeline(unittest.TestCase):
    def test_low_quality_skips_ai_footage_selection(self):
        results = [("t1", [MaterialInfo(url="https://a/1")])]
        context = curation.CurationContext(video_subject="x", quality="low")
        with patch.object(curation, "is_enabled", return_value=True), \
             patch.object(curation, "vision_model_available") as vision:
            self.assertEqual(curation.curate_search_results(results, context, "9:16"), results)
        vision.assert_not_called()

    def test_medium_quality_describes_at_most_three_candidates(self):
        items = [MaterialInfo(url=f"https://a/{i}", source_info={"thumbnail": f"https://t/{i}"})
                 for i in range(6)]
        context = curation.CurationContext(video_subject="x", quality="medium")
        with patch.object(curation, "is_enabled", return_value=True), \
             patch.object(curation, "vision_model_available", return_value=True), \
             patch.object(curation, "describe_image", return_value="a cup") as describe, \
             patch.object(curation, "judge_shots", return_value={}):
            curation.curate_search_results([("t1", items)], context, "9:16")
        self.assertEqual(describe.call_count, 3)

    def test_encoder_preset_applies_to_software_encoders_only(self):
        with patch.object(video, "_get_effective_video_codec", return_value="libx264"):
            self.assertEqual(video.encoder_preset_options("veryfast"), {"preset": "veryfast"})
        with patch.object(video, "_get_effective_video_codec", return_value="h264_nvenc"):
            self.assertEqual(video.encoder_preset_options("veryfast"), {})


class TestTargetDuration(unittest.TestCase):
    def test_prompt_asks_for_the_matching_word_count(self):
        prompt = llm.build_script_prompt("café", language="fr", target_seconds=20)
        self.assertIn("about 20 seconds when read aloud, so about 50 words", prompt)
        self.assertNotIn("read aloud", llm.build_script_prompt("café"))

    def test_languages_without_spaces_count_characters(self):
        self.assertEqual(llm.target_script_length(10, "zh-CN"), 45)
        self.assertEqual(llm.script_length("咖啡 很好。", "zh-CN"), 5)

    def test_too_long_script_is_cut_at_a_sentence_end(self):
        script = "Un deux trois. Quatre cinq six sept. Huit neuf."
        self.assertEqual(llm.fit_script_to_length(script, 5, "fr"), "Un deux trois.")
        self.assertEqual(llm.fit_script_to_length(script, 50, "fr"), script)

    def test_generated_script_is_fitted_to_the_target(self):
        long_script = " ".join(f"Phrase numéro {i} avec quelques mots." for i in range(40))
        with patch.object(llm, "_generate_response", return_value=long_script):
            script = llm.generate_script("café", language="fr", target_seconds=10)
        limit = int(llm.target_script_length(10, "fr") * 1.15)
        self.assertLessEqual(llm.script_length(script, "fr"), limit)
        self.assertTrue(script.endswith("."))


if __name__ == "__main__":
    unittest.main()
