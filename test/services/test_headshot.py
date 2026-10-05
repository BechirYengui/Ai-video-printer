import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from PIL import Image

from app.services import headshot


def _jpeg(width=1600, height=1200, orientation=None) -> bytes:
    image = Image.new("RGB", (width, height), (200, 120, 80))
    buffer = io.BytesIO()
    exif = Image.Exif()
    if orientation:
        exif[0x0112] = orientation
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


class TestHeadshot(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.venv = os.path.join(self.tmp.name, "venv")
        env = patch.dict(os.environ, {"YVP_HEADSHOT_DIR": os.path.join(self.tmp.name, "jobs"),
                                      "YVP_HEADSHOT_VENV": self.venv})
        env.start()
        self.addCleanup(env.stop)
        gpu = patch.object(headshot, "free_gpu_memory")
        gpu.start()
        self.addCleanup(gpu.stop)

    def _install_fake_venv(self, worker_code="import time; time.sleep(30)"):
        """Un « python » qui ignore le worker et exécute worker_code à sa place."""
        os.makedirs(os.path.join(self.venv, "bin"))
        script = os.path.join(self.venv, "bin", "python")
        with open(script, "w") as handle:
            handle.write(f"#!/bin/sh\nexec {sys.executable} -c '{worker_code}'\n")
        os.chmod(script, 0o755)
        open(os.path.join(self.venv, ".yvp-ready"), "w").close()

    def test_prompt_places_trigger_word_after_the_person(self):
        prompt, negative = headshot.build_prompt("woman", "blazer", "office")
        self.assertIn("of a woman img,", prompt)
        self.assertIn(headshot.OUTFITS["blazer"], prompt)
        self.assertIn(headshot.BACKGROUNDS["office"], prompt)
        self.assertIn("watermark", negative)
        with self.assertRaises(headshot.HeadshotError):
            headshot.build_prompt("man", "pyjamas", "studio")

    def test_photo_is_rotated_from_exif_and_downscaled(self):
        # Orientation 6 : le téléphone était tenu en portrait.
        image = headshot.prepare_photo(_jpeg(1600, 1200, orientation=6))
        self.assertEqual(image.size, (768, 1024))
        self.assertEqual(image.mode, "RGB")
        with self.assertRaises(headshot.HeadshotError):
            headshot.prepare_photo(b"not an image")

    def test_worker_env_restores_cuda_hidden_for_ollama(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "-1"}):
            self.assertNotIn("CUDA_VISIBLE_DEVICES", headshot.worker_env())
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            self.assertEqual(headshot.worker_env()["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(headshot.worker_env()["MALLOC_MMAP_THRESHOLD_"], "1048576")

    def test_worker_scope_falls_back_without_systemd(self):
        with patch.object(headshot.shutil, "which", return_value=None):
            self.assertEqual(headshot._scope_prefix("job"), [])
        failed = subprocess.CompletedProcess([], 1)
        with patch.object(headshot.shutil, "which", return_value="/usr/bin/systemd-run"), \
                patch.object(headshot.subprocess, "run", return_value=failed):
            self.assertEqual(headshot._scope_prefix("job"), [])

    def test_windows_install_uses_powershell_without_console(self):
        launched = []

        class FakeProcess:
            pid = 4242

            def __init__(self, command, **kwargs):
                launched.append((command, kwargs))

        with patch.object(headshot.sys, "platform", "win32"), \
                patch.object(headshot.subprocess, "Popen", FakeProcess), \
                patch.object(headshot.subprocess, "CREATE_NO_WINDOW", 0x08000000, create=True), \
                patch.object(headshot.subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200, create=True), \
                patch.object(headshot, "_windows_pid_alive", return_value=False):
            headshot.start_install()
        command, kwargs = launched[0]
        self.assertEqual(command[0], "powershell.exe")
        self.assertTrue(command[-1].endswith(os.path.join("tools", "headshot", "setup.ps1")))
        self.assertEqual(kwargs["creationflags"], 0x08000000 | 0x200)
        self.assertNotIn("start_new_session", kwargs)

    def test_start_job_validates_photos_and_installation(self):
        photos = [(f"p{i}.jpg", _jpeg()) for i in range(headshot.MIN_PHOTOS)]
        with self.assertRaisesRegex(headshot.HeadshotError, "too_few_photos"):
            headshot.start_job(photos[:-1], "man", "suit", "studio")
        with self.assertRaises(headshot.HeadshotError):
            headshot.start_job(photos, "man", "suit", "studio", count=headshot.MAX_IMAGES + 1)
        with self.assertRaisesRegex(headshot.HeadshotError, "not_installed"):
            headshot.start_job(photos, "man", "suit", "studio")
        self._install_fake_venv()
        with self.assertRaisesRegex(headshot.HeadshotError, "unreadable"):
            headshot.start_job(photos[:-1] + [("bad.jpg", b"xx")], "man", "suit", "studio")
        # La tâche refusée ne laisse rien derrière elle.
        self.assertEqual(os.listdir(headshot.headshot_dir()), [])

    def test_job_runs_detached_and_can_be_cancelled(self):
        self._install_fake_venv()
        photos = [(f"/home/me/p{i}.jpg", _jpeg()) for i in range(headshot.MIN_PHOTOS)]
        job_id = headshot.start_job(photos, "man", "suit", "studio", count=2, quality="high")

        job_dir = os.path.join(headshot.headshot_dir(), job_id)
        with open(os.path.join(job_dir, "job.json")) as handle:
            job = json.load(handle)
        self.assertEqual((job["count"], job["size"], job["steps"]), (2, 1024, 30))
        self.assertEqual(len(os.listdir(os.path.join(job_dir, "photos"))), headshot.MIN_PHOTOS)
        self.assertEqual(job["labels"]["photo-01.jpg"], "p0.jpg")
        self.assertEqual(headshot.job_status(job_id)["status"], "running")
        self.assertEqual(headshot.running_job(), job_id)
        with self.assertRaisesRegex(headshot.HeadshotError, "busy"):
            headshot.start_job(photos, "man", "suit", "studio")

        headshot.cancel_job(job_id)
        self.assertIsNone(headshot.running_job())
        self.assertFalse(os.path.exists(job_dir))

    def test_dead_worker_is_reported_as_crashed(self):
        self._install_fake_venv("import sys; sys.exit(3)")
        photos = [(f"p{i}.jpg", _jpeg()) for i in range(headshot.MIN_PHOTOS)]
        job_id = headshot.start_job(photos, "man", "suit", "studio")
        deadline = time.time() + 10
        while headshot.job_status(job_id)["status"] == "running" and time.time() < deadline:
            time.sleep(0.1)
        status = headshot.job_status(job_id)
        self.assertEqual((status["status"], status["error"]), ("failed", "crashed"))

    def test_results_are_listed_downloaded_square_and_deleted(self):
        job_id = "20261005-101500-abcdef"
        job_dir = os.path.join(headshot.headshot_dir(), job_id)
        os.makedirs(job_dir)
        for index in (2, 1, 10):
            Image.new("RGB", (768, 768), "grey").save(os.path.join(job_dir, f"headshot-{index}.png"))
        with open(os.path.join(job_dir, "job.json"), "w") as handle:
            json.dump({"style": {"outfit": "suit", "background": "studio"}}, handle)
        os.makedirs(os.path.join(headshot.headshot_dir(), "20261005-090000-000000"))  # sans image

        results = headshot.list_results()
        self.assertEqual([r.job_id for r in results], [job_id])
        self.assertEqual([os.path.basename(p) for p in results[0].images],
                         ["headshot-1.png", "headshot-2.png", "headshot-10.png"])
        self.assertEqual(results[0].style["outfit"], "suit")

        with Image.open(io.BytesIO(headshot.linkedin_jpeg(results[0].images[0]))) as image:
            self.assertEqual((image.format, image.size), ("JPEG", (800, 800)))

        headshot.delete_job(job_id)
        self.assertEqual(headshot.list_results(), [])
        with self.assertRaises(headshot.HeadshotError):
            headshot.delete_job("../../etc")


class TestWorkerFaceChecks(unittest.TestCase):
    """face_issue est pur : testable sans PyTorch ni modèles."""

    @classmethod
    def setUpClass(cls):
        import importlib.util

        path = os.path.join(os.path.dirname(headshot.__file__), "..", "..", "tools", "headshot", "worker.py")
        spec = importlib.util.spec_from_file_location("headshot_worker", path)
        cls.worker = importlib.util.module_from_spec(spec)
        with patch.object(sys, "argv", ["worker.py", tempfile.gettempdir()]):
            spec.loader.exec_module(cls.worker)

    def _face(self, size):
        return type("Face", (), {"bbox": (0, 0, size, size)})()

    def test_identity_photos_are_cropped_on_the_face(self):
        image = Image.new("RGB", (768, 1024))
        crop = self.worker.face_crop(image, (300, 200, 500, 450))  # visage 200×250
        self.assertEqual(crop.size, (500, 500))
        # Visage au bord : le carré reste dans la photo.
        self.assertEqual(self.worker.face_crop(image, (0, 0, 400, 600)).size, (768, 768))
        self.assertEqual((self.worker.merge_step(20), self.worker.merge_step(30)), (3, 4))

    def test_face_issues(self):
        self.assertEqual(self.worker.face_issue([]), "no_face")
        self.assertEqual(self.worker.face_issue([self._face(60)]), "face_too_small")
        self.assertEqual(self.worker.face_issue([self._face(300), self._face(200)]), "several_faces")
        # Un visage lointain à l'arrière-plan ne gêne pas.
        self.assertEqual(self.worker.face_issue([self._face(300), self._face(80)]), "")


if __name__ == "__main__":
    unittest.main()
