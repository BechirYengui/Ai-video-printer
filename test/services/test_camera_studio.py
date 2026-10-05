import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from app.services import camera_studio
from app.utils import utils


def _make_browser_take(path: str, seconds: float = 3.0) -> None:
    """Petite vidéo WebM (VP8 + Opus), comme celle produite par MediaRecorder."""
    subprocess.run(
        [
            utils.get_ffmpeg_binary(), "-y", "-loglevel", "error",
            "-f", "lavfi", "-i", f"testsrc=size=360x640:rate=24:duration={seconds}",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
            "-c:v", "libvpx", "-b:v", "300k", "-c:a", "libopus", "-shortest", path,
        ],
        check=True,
        capture_output=True,
    )


class TestCameraStudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recordings = os.path.join(self.tmp.name, "recordings")
        self.local_videos = os.path.join(self.tmp.name, "local_videos")
        os.makedirs(self.recordings)
        os.makedirs(self.local_videos)
        patches = [
            patch.object(camera_studio, "recordings_dir", return_value=self.recordings),
            patch.object(
                camera_studio.material_upload, "uploaded_material_dir", return_value=self.local_videos
            ),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def _take(self, seconds=3.0) -> bytes:
        source = os.path.join(self.tmp.name, "take.webm")
        _make_browser_take(source, seconds)
        with open(source, "rb") as handle:
            return handle.read()

    def test_take_is_converted_listed_cut_and_deleted(self):
        rec = camera_studio.save_recording(self._take(), "video/webm;codecs=vp8,opus", "portrait")

        self.assertTrue(rec.name.endswith("-portrait.mp4"))
        self.assertEqual(rec.aspect, "portrait")
        self.assertAlmostEqual(rec.duration, 3.0, delta=0.3)
        # Plus aucun fichier temporaire à côté du MP4.
        self.assertEqual(os.listdir(self.recordings), [rec.name])
        self.assertEqual([r.name for r in camera_studio.list_recordings()], [rec.name])

        clip = camera_studio.extract_clip(rec.name, 0.5, 2.0)
        self.assertEqual(os.path.dirname(clip), self.local_videos)
        self.assertAlmostEqual(camera_studio.probe_duration(clip), 1.5, delta=0.3)

        camera_studio.delete_recording(rec.name)
        self.assertEqual(camera_studio.list_recordings(), [])
        # L'extrait déjà envoyé au montage reste utilisable.
        self.assertTrue(os.path.exists(clip))

    def test_invalid_takes_are_rejected(self):
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.save_recording(b"", "video/webm", "portrait")
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.save_recording(b"x", "image/png", "portrait")
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.save_recording(b"x", "video/webm", "square")
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.save_recording(b"not a video", "video/webm", "landscape")
        self.assertEqual(os.listdir(self.recordings), [])

    def test_only_known_recording_names_can_be_deleted_or_cut(self):
        outside = os.path.join(self.tmp.name, "secret.mp4")
        open(outside, "wb").close()
        for name in ("../secret.mp4", outside, "rec-x.mp4", ""):
            with self.assertRaises(camera_studio.RecordingError):
                camera_studio.delete_recording(name)
            with self.assertRaises(camera_studio.RecordingError):
                camera_studio.extract_clip(name, 0, 1)
        self.assertTrue(os.path.exists(outside))

    def test_cut_keeps_or_removes_a_passage_and_leaves_the_original(self):
        rec = camera_studio.save_recording(self._take(4.0), "video/webm", "portrait")
        kept = camera_studio.cut_recording(rec.name, 1.0, 2.5, "keep")
        removed = camera_studio.cut_recording(rec.name, 1.0, 2.5, "remove")
        self.assertAlmostEqual(kept.duration, 1.5, delta=0.3)
        self.assertAlmostEqual(removed.duration, 2.5, delta=0.3)
        self.assertEqual(kept.aspect, "portrait")
        # Son conservé dans les deux versions ; l'original est intact.
        for cut in (kept, removed):
            info = camera_studio._ffmpeg(["-i", cut.path], timeout=30).stderr.decode()
            self.assertIn("Audio:", info)
        self.assertEqual(
            {r.name for r in camera_studio.list_recordings()}, {rec.name, kept.name, removed.name}
        )
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.cut_recording(rec.name, 0, 3.9, "remove")
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.cut_recording("../x.mp4", 0, 1, "keep")

    def test_clip_length_is_limited(self):
        rec = camera_studio.save_recording(self._take(1.0), "video/webm", "landscape")
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.extract_clip(rec.name, 2.0, 1.0)
        with self.assertRaises(camera_studio.RecordingError):
            camera_studio.extract_clip(rec.name, 0, camera_studio.MAX_CLIP_SECONDS + 1)



class TestRecordingsDir(unittest.TestCase):
    def test_override_keeps_tests_away_from_the_user_videos(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.dict(os.environ, {"YVP_RECORDINGS_DIR": tmp}), \
                patch.object(camera_studio, "_user_videos_dir", side_effect=AssertionError):
            self.assertEqual(camera_studio.recordings_dir(), tmp)

    def test_recordings_go_to_videos_yvp_and_old_ones_are_moved(self):
        with tempfile.TemporaryDirectory() as tmp:
            videos = os.path.join(tmp, "Vidéos")
            legacy = os.path.join(tmp, "storage", "recordings")
            os.makedirs(legacy)
            old = "rec-20260101-120000-abcdef-portrait.mp4"
            open(os.path.join(legacy, old), "wb").close()
            open(os.path.join(legacy, "autre.txt"), "wb").close()
            with patch.object(camera_studio, "_user_videos_dir", return_value=videos), \
                    patch.object(camera_studio.utils, "storage_dir", return_value=legacy), \
                    patch.dict(os.environ, {"YVP_RECORDINGS_DIR": ""}):
                folder = camera_studio.recordings_dir()
            self.assertEqual(folder, os.path.join(videos, "YVP"))
            self.assertEqual(os.listdir(folder), [old])
            # Seuls les enregistrements YVP sont déplacés.
            self.assertEqual(os.listdir(legacy), ["autre.txt"])


if __name__ == "__main__":
    unittest.main()
