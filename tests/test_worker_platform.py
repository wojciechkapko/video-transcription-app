"""Platform/device checks without model downloads or GPU dependencies."""
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


WORKER = Path(__file__).resolve().parents[1] / "src-tauri/python/transcribe_worker.py"
SPEC = importlib.util.spec_from_file_location("transcribe_worker", WORKER)
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


class DeviceSelectionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {}, clear=True))
        self.device_count = Mock(return_value=1)
        self.enterContext(patch.dict(sys.modules, {
            "ctranslate2": SimpleNamespace(get_cuda_device_count=self.device_count),
        }))
        self.enterContext(patch.object(worker.sys, "platform", "linux"))
        self.enterContext(patch.object(worker, "CUDA_FAMILIES_RESOLVED", None))
        self.load_library = self.enterContext(patch.object(worker.ctypes, "CDLL"))

    def test_linux_gpu_needs_all_compute_libraries(self):
        self.assertEqual(worker.choose_device(), "cuda")
        self.assertEqual(
            [call.args[0] for call in self.load_library.call_args_list],
            ["libcublasLt.so.12", "libcublas.so.12", "libcudart.so.12", "libcudnn.so.9"],
        )

    def test_linux_missing_library_falls_back_to_cpu(self):
        for missing in ("libcublasLt.so.12", "libcublas.so.12",
                        "libcudart.so.12", "libcudnn.so.9"):
            with self.subTest(missing=missing):
                def load_library(library):
                    if library == missing:
                        raise OSError("shared library unavailable")
                self.load_library.side_effect = load_library
                self.assertEqual(worker.choose_device(), "cpu")

    def test_no_gpu_skips_library_probe(self):
        self.device_count.return_value = 0
        self.assertEqual(worker.choose_device(), "cpu")
        self.load_library.assert_not_called()

    def test_driver_probe_failure_falls_back_to_cpu(self):
        self.device_count.side_effect = RuntimeError("driver unavailable")
        self.assertEqual(worker.choose_device(), "cpu")
        self.load_library.assert_not_called()

    def test_explicit_device_overrides_probes(self):
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), patch.dict(os.environ, {"TRANSCRIBE_DEVICE": device}):
                self.assertEqual(worker.choose_device(), device)
        self.device_count.assert_not_called()
        self.load_library.assert_not_called()

    def test_windows_bootstrap_gate_is_preserved(self):
        with patch.object(worker.sys, "platform", "win32"):
            for resolved, expected in (([], "cpu"), (["cublas", "cudnn"], "cpu"),
                                       (["cublas", "cudnn", "cudart"], "cuda")):
                with self.subTest(resolved=resolved), patch.object(worker, "CUDA_FAMILIES_RESOLVED", resolved):
                    self.assertEqual(worker.choose_device(), expected)
        self.load_library.assert_not_called()

    def test_probe_keeps_boolean_contract(self):
        self.load_library.side_effect = OSError("shared library unavailable")
        self.assertEqual(worker._probe_device(), {
            "event": "probe_result", "device": "cpu", "cuda_available": False,
        })


class CudaInstallerTests(unittest.TestCase):
    def test_non_windows_rejected_before_location_or_download(self):
        with patch.object(worker.os, "name", "posix"), \
             patch.object(worker, "_cuda_dir") as cuda_dir, \
             patch.object(worker.os, "makedirs") as makedirs, \
             patch.object(worker, "_resolve_wheel") as resolve, \
             patch.object(worker, "_download_wheel") as download, \
             patch.object(worker, "_extract_wheel") as extract:
            with self.assertRaisesRegex(RuntimeError, "supported only on Windows"):
                worker._install_cuda()
            for operation in (cuda_dir, makedirs, resolve, download, extract):
                operation.assert_not_called()

    def test_windows_still_requires_install_location(self):
        with patch.object(worker.os, "name", "nt"), \
             patch.object(worker, "_cuda_dir", return_value=None), \
             patch.object(worker.os, "makedirs") as makedirs:
            with self.assertRaisesRegex(RuntimeError, "LOCALAPPDATA not set"):
                worker._install_cuda()
            makedirs.assert_not_called()


if __name__ == "__main__":
    unittest.main()
