"""Regression fixtures for NVIDIA's internal SBSA wheel tag; no GPU required."""
import base64
import csv
import hashlib
import importlib.metadata
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep_modly import native_metadata
import setup as installer


class NativeMetadataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.info = self.root / "nvidia_cusparselt_cu13-0.8.0.dist-info"
        self.info.mkdir()
        (self.info / "METADATA").write_text("Name: nvidia-cusparselt-cu13\nVersion: 0.8.0\n")
        self.wheel = self.info / "WHEEL"
        self.wheel.write_text("Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-manylinux2014_sbsa\n")
        self.lib = self.root / "nvidia/cusparselt/lib/libcusparseLt.so.0"
        self.lib.parent.mkdir(parents=True)
        self.lib.write_bytes(b"\x7fELF\x02\x01\x01" + b"\0" * 9 + b"\x03\0\xb7\0" + b"\0" * 44)
        self.record = self.info / "RECORD"
        self.refresh_record()
        self.dist = importlib.metadata.Distribution.at(self.info)
        for target, value in (("sys.platform", "linux"), ("platform.machine", None), ("importlib.metadata.distribution", None), ("ctypes.CDLL", None)):
            mock = patch("acestep_modly.native_metadata." + target, value) if value else patch("acestep_modly.native_metadata." + target)
            active = mock.start()
            self.addCleanup(mock.stop)
            if target == "platform.machine":
                active.return_value = "aarch64"
            elif target == "importlib.metadata.distribution":
                active.return_value = self.dist
                self.lookup = active
            elif target == "ctypes.CDLL":
                self.load = active

    @staticmethod
    def digest(data):
        return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()

    def refresh_record(self):
        with self.record.open("w", newline="") as file:
            writer = csv.writer(file)
            for path in (self.wheel, self.lib, self.info / "METADATA"):
                data = path.read_bytes()
                writer.writerow([path.relative_to(self.root).as_posix(), self.digest(data), str(len(data))])
            writer.writerow([self.record.relative_to(self.root).as_posix(), "", ""])

    def test_repairs_exact_tag_and_record_idempotently(self):
        library_before = self.lib.read_bytes()
        self.assertTrue(native_metadata.repair_cusparselt_tag("cu130"))
        self.assertIn(b"Tag: py3-none-manylinux2014_aarch64", self.wheel.read_bytes())
        rows = list(csv.reader(io.StringIO(self.record.read_text())))
        row = next(row for row in rows if row[0].endswith("/WHEEL"))
        self.assertEqual(row[1:], [self.digest(self.wheel.read_bytes()), str(self.wheel.stat().st_size)])
        self.assertEqual(self.lib.read_bytes(), library_before)
        before = (self.wheel.read_bytes(), self.record.read_bytes())
        self.assertFalse(native_metadata.repair_cusparselt_tag("cu130"))
        self.assertEqual(before, (self.wheel.read_bytes(), self.record.read_bytes()))
        self.load.assert_called_with(str(self.lib))

    def test_non_target_lanes_and_platforms_do_not_inspect_or_modify(self):
        for flavor, system, machine in (("cpu", "linux", "aarch64"), ("cu128", "linux", "aarch64"), ("cu130", "linux", "x86_64"), ("cu130", "win32", "aarch64"), ("cu130", "darwin", "arm64")):
            with self.subTest(flavor=flavor, system=system, machine=machine), patch.object(native_metadata.sys, "platform", system), patch.object(native_metadata.platform, "machine", return_value=machine):
                self.assertFalse(native_metadata.repair_cusparselt_tag(flavor))
        self.lookup.assert_not_called()
        self.load.assert_not_called()

    def test_unknown_identity_version_and_tags_fail_without_modification(self):
        original = self.wheel.read_bytes()
        for metadata, wheel in (("Name: other\nVersion: 0.8.0\n", original), ("Name: nvidia-cusparselt-cu13\nVersion: 0.9.0\n", original), ("Name: nvidia-cusparselt-cu13\nVersion: 0.8.0\n", original.replace(b"sbsa", b"x86_64")), ("Name: nvidia-cusparselt-cu13\nVersion: 0.8.0\n", original + b"Tag: py3-none-any\n")):
            with self.subTest(metadata=metadata, wheel=wheel):
                (self.info / "METADATA").write_text(metadata)
                self.wheel.write_bytes(wheel)
                self.refresh_record()
                before = (self.wheel.read_bytes(), self.record.read_bytes())
                with self.assertRaises(RuntimeError):
                    native_metadata.repair_cusparselt_tag("cu130")
                self.assertEqual(before, (self.wheel.read_bytes(), self.record.read_bytes()))

    def test_invalid_elf_and_loader_failure_leave_metadata_untouched(self):
        original = self.lib.read_bytes()
        before = (self.wheel.read_bytes(), self.record.read_bytes())
        for data in (b"not ELF", original[:4] + b"\x01" + original[5:], original[:5] + b"\x02" + original[6:], original[:18] + b"\x3e\0" + original[20:]):
            with self.subTest(data=data):
                self.lib.write_bytes(data)
                with self.assertRaises(RuntimeError):
                    native_metadata.repair_cusparselt_tag("cu130")
                self.assertEqual(before, (self.wheel.read_bytes(), self.record.read_bytes()))
        self.lib.write_bytes(original)
        self.load.side_effect = OSError("unresolved symbol")
        with self.assertRaisesRegex(OSError, "unresolved symbol"):
            native_metadata.repair_cusparselt_tag("cu130")
        self.assertEqual(before, (self.wheel.read_bytes(), self.record.read_bytes()))

    def test_inconsistent_or_duplicate_record_fails_closed(self):
        original = self.record.read_bytes()
        for data in (original.replace(b"sha256=", b"sha512=", 1), original + original.splitlines(keepends=True)[0]):
            self.record.write_bytes(data)
            with self.assertRaises(RuntimeError):
                native_metadata.repair_cusparselt_tag("cu130")

    def test_interrupted_pair_update_can_recover_original_record(self):
        self.wheel.write_bytes(self.wheel.read_bytes().replace(b"manylinux2014_sbsa", b"manylinux2014_aarch64"))
        self.assertTrue(native_metadata.repair_cusparselt_tag("cu130"))
        self.assertFalse(native_metadata.repair_cusparselt_tag("cu130"))


class SetupNativeGateTests(unittest.TestCase):
    def test_target_venv_repair_precedes_strict_pip_and_conflicts_are_fatal(self):
        lane = {"system": "linux", "arch": "arm64", "python": "3.12", "flavor": "cu130", "accelerator": "cuda", "index": "https://download.pytorch.org/whl/cu130"}
        target = installer.ROOT / "venv/bin/python"
        seen = []

        def run(command, stage, **kwargs):
            seen.append([str(item) for item in command])
            if command[1:] == ["-m", "pip", "check"]:
                raise subprocess.CalledProcessError(1, command, stderr="unrelated dependency conflict")

        with patch.object(installer.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, f"{sys.version_info.major}.{sys.version_info.minor}\n")), patch.object(installer, "select_lane", return_value=lane), patch.object(installer, "probe_cuda_version", return_value=130), patch.object(installer, "resolve_models_root", return_value=Path("/unused-models")), patch.object(installer, "ensure_source"), patch.object(installer, "ensure_venv", return_value=target), patch.object(installer, "torch_requirements", return_value=["torch==2.10.0+cu130"]), patch.object(installer, "run", side_effect=run):
            with self.assertRaises(subprocess.CalledProcessError):
                installer.setup({"python_exe": sys.executable, "ext_dir": str(installer.ROOT)})
        self.assertEqual(seen[-2:], [[str(target), "-m", "acestep_modly.native_metadata", "--flavor", "cu130"], [str(target), "-m", "pip", "check"]])
        self.assertFalse(any("acestep_modly.provision" in cmd for cmd in seen))


if __name__ == "__main__":
    unittest.main()
