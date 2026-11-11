"""First-run recovery and compatibility with the installed Hub download API.

No model weights or network access are needed. Run with .venv/bin/python to
include the real huggingface_hub integration checks.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


WORKER = Path(__file__).resolve().parents[1] / "src-tauri/python/transcribe_worker.py"
SPEC = importlib.util.spec_from_file_location("download_worker", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class ModelDownloadRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.model_dir = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, {"WHISPER_MODEL_DIR": str(self.model_dir)}))
        self.model = Mock()
        self.hub = SimpleNamespace(
            snapshot_download=Mock(side_effect=FileNotFoundError("uncached")),
            hf_hub_download=Mock(side_effect=self.download),
        )
        self.enterContext(patch.dict(sys.modules, {
            "faster_whisper": SimpleNamespace(WhisperModel=self.model),
            "huggingface_hub": self.hub,
        }))
        self.stdout = self.enterContext(redirect_stdout(io.StringIO()))

    def download(self, repo, name, *, local_dir, tqdm_class):
        path = Path(local_dir) / name
        path.write_bytes(b"fixture")
        return str(path)

    def test_retry_completes_download_interrupted_after_weights(self):
        def interrupted(repo, name, **kwargs):
            if name == "tokenizer.json":
                raise OSError("connection lost")
            return self.download(repo, name, **kwargs)

        self.hub.hf_hub_download.side_effect = interrupted
        with self.assertRaisesRegex(RuntimeError, "model download failed: connection lost"):
            worker._load_model("cpu", "int8")
        self.model.assert_not_called()
        self.assertTrue((self.model_dir / "model.bin").is_file())

        self.hub.hf_hub_download.side_effect = self.download
        worker._load_model("cpu", "int8")
        self.assertTrue(all((self.model_dir / name).is_file() for name in worker.MODEL_FILES))
        self.model.assert_called_once_with(str(self.model_dir), device="cpu", compute_type="int8")
        events = [json.loads(line)["event"] for line in self.stdout.getvalue().splitlines()]
        self.assertEqual(events, ["downloading_model", "downloading_model", "loading_model"])

    def test_complete_dedicated_model_needs_no_hub_access(self):
        for name in worker.MODEL_FILES:
            (self.model_dir / name).write_bytes(b"fixture")
        worker._load_model("cpu", "int8")
        self.model.assert_called_once_with(str(self.model_dir), device="cpu", compute_type="int8")
        self.hub.snapshot_download.assert_not_called()
        self.hub.hf_hub_download.assert_not_called()

    def test_complete_hub_snapshot_is_loaded_by_local_path(self):
        cached = self.model_dir / "snapshot"
        cached.mkdir()
        for name in worker.MODEL_FILES:
            (cached / name).write_bytes(b"fixture")
        self.hub.snapshot_download.side_effect = None
        self.hub.snapshot_download.return_value = str(cached)
        worker._load_model("cpu", "int8")
        self.model.assert_called_once_with(str(cached), device="cpu", compute_type="int8")
        self.hub.snapshot_download.assert_called_once_with(
            worker.MODEL_REPO, local_files_only=True, allow_patterns=worker.MODEL_FILES)
        self.hub.hf_hub_download.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "requires the app's Python environment")
class InstalledHubCompatibilityTests(unittest.TestCase):
    def test_download_transport_default_is_set_before_hub_import(self):
        script = """
import json, runpy, sys
runpy.run_path(sys.argv[1], run_name="download_worker")
import faster_whisper
from huggingface_hub import constants
from huggingface_hub.utils._runtime import is_xet_available
print(json.dumps({"disabled": constants.HF_HUB_DISABLE_XET,
                  "available": is_xet_available()}))
"""
        for override in (None, "0"):
            with self.subTest(override=override):
                env = os.environ.copy()
                env.pop("HF_HUB_DISABLE_XET", None)
                if override is not None:
                    env["HF_HUB_DISABLE_XET"] = override
                result = subprocess.run(
                    [sys.executable, "-c", script, str(WORKER)],
                    env=env, check=True, capture_output=True, text=True, timeout=30)
                state = json.loads(result.stdout)
                self.assertEqual(state["disabled"], override is None)
                if override is None:
                    self.assertFalse(state["available"])

    def test_http_download_resumes_and_reports_each_byte_once(self):
        import httpx
        from huggingface_hub import file_download
        from huggingface_hub.utils import _http

        chunk = b"x" * 1_000_000
        requests = []

        class InterruptedStream(httpx.SyncByteStream):
            def __iter__(self):
                yield chunk
                raise httpx.ReadTimeout("simulated interrupted download")

        def respond(request):
            requests.append(request.headers.get("Range"))
            if len(requests) == 1:
                return httpx.Response(200, headers={"Content-Length": "3000000"},
                                      stream=InterruptedStream())
            return httpx.Response(206, headers={
                "Content-Length": "2000000",
                "Content-Range": "bytes 1000000-2999999/3000000",
            }, content=chunk * 2)

        stdout = io.StringIO()
        downloaded = io.BytesIO()
        with httpx.Client(transport=httpx.MockTransport(respond)) as client, \
             patch.object(_http, "get_session", return_value=client), \
             patch.object(file_download.constants, "DOWNLOAD_CHUNK_SIZE", len(chunk)), \
             patch.object(file_download.time, "sleep"), \
             patch.object(file_download.logger, "warning"), \
             patch.object(worker, "time", SimpleNamespace(
                 monotonic=Mock(side_effect=[1.0, 2.0, 3.0]))), \
             redirect_stdout(stdout):
            file_download.http_get(
                "https://fixture.invalid/model.bin", downloaded,
                expected_size=3_000_000, displayed_filename="model.bin",
                tqdm_class=worker._JsonBar)

        self.assertEqual(requests, [None, "bytes=1000000-"])
        self.assertEqual(downloaded.getvalue(), chunk * 3)
        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([event["bytes"] for event in events], [1_000_000, 2_000_000, 3_000_000])
        self.assertTrue(all(event["event"] == "download_progress" for event in events))
        self.assertTrue(all(event["total"] == 3_000_000 for event in events))

    def test_only_complete_hub_snapshot_is_reused_offline(self):
        from huggingface_hub import constants

        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(constants, "HF_HUB_CACHE", tmp), \
             patch("socket.create_connection", side_effect=AssertionError("network access")):
            repo = Path(tmp) / "models--Systran--faster-whisper-large-v3"
            snapshot = repo / "snapshots" / ("a" * 40)
            snapshot.mkdir(parents=True)
            (repo / "refs").mkdir()
            (repo / "refs/main").write_text(snapshot.name)
            (snapshot / "model.bin").write_bytes(b"fixture")
            self.assertIsNone(worker._hf_cached_model_dir())
            for name in worker.MODEL_FILES:
                (snapshot / name).write_bytes(b"fixture")
            self.assertEqual(worker._hf_cached_model_dir(), str(snapshot))

    def test_xet_reports_one_consistent_byte_stream(self):
        try:
            from huggingface_hub.utils._xet_progress_reporting import XetDownloadProgressReporter
        except ImportError:
            self.skipTest("Hub version predates the dual Xet reporter")

        stdout = io.StringIO()
        with redirect_stdout(stdout), \
             patch.object(worker.time, "monotonic", side_effect=[1.0, 2.0, 3.0, 4.0]):
            with XetDownloadProgressReporter(
                reconstruction_desc="model.bin: reconstructing file",
                transfer_desc="model.bin: downloading bytes", total=3_000_000,
                log_level=20, tqdm_class=worker._JsonBar,
            ) as reporter:
                for reconstructed, transferred in ((1_000_000, 500_000), (3_000_000, 1_500_000)):
                    reporter.update_progress(SimpleNamespace(
                        total_bytes_completed=reconstructed,
                        total_transfer_bytes_completed=transferred,
                        total_bytes_completion_rate=1_000_000,
                        total_transfer_bytes_completion_rate=500_000,
                        total_bytes=3_000_000,
                    ))
        events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual([event["bytes"] for event in events], [1_000_000, 3_000_000])
        self.assertTrue(all(event["event"] == "download_progress" for event in events))
        self.assertTrue(all(event["total"] == 3_000_000 for event in events))


if __name__ == "__main__":
    unittest.main()
