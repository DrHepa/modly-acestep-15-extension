"""Host protocol and manifest validation, no model weights or network required."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from acestep_modly.constants import NODE_ID, ROOT
from acestep_modly.parameters import normalize
from acestep_modly.platforms import select_lane, torch_requirements
from setup import parse_args


def payload(**params):
    return {"nodeId": NODE_ID, "input": {"text": "Quiet acoustic jazz"}, "params": params}


class ContractTests(unittest.TestCase):
    def test_manifest_defaults_and_connected_text(self):
        manifest = json.loads((ROOT / "manifest.json").read_text())
        self.assertEqual((manifest["type"], manifest["author"], manifest["entry"]), ("process", "DrHepa", "processor.py"))
        self.assertEqual((manifest["nodes"][0]["input"], manifest["nodes"][0]["output"]), ("text", "audio"))
        values = normalize(payload(caption="Ignored fallback"))
        self.assertEqual(values["caption"], "Quiet acoustic jazz")
        self.assertEqual(values["lyrics"], "[Instrumental]")
        self.assertFalse(values["thinking"])
        self.assertEqual(values["duration"], 30)

    def test_lyrics_and_boolean_selects(self):
        values = normalize(payload(instrumental=False, thinking=True, lyrics="[Verse]\\nHello"))
        self.assertEqual(values["lyrics"], "[Verse]\nHello")
        self.assertTrue(values["thinking"])

    def test_bad_inputs_rejected(self):
        for key, value in [("duration", 0), ("duration", 11.5), ("seed", True), ("duration", "nan"), ("bpm", 5), ("device", "mps"), ("lyrics", 10)]:
            with self.subTest(key=key, value=value), self.assertRaises((ValueError, TypeError)):
                normalize(payload(**{key: value}))
        request = payload()
        request["input"]["nodeId"] = "unrelated-node"
        with self.assertRaises(ValueError):
            normalize(request)

    def test_all_twelve_dependency_lanes(self):
        for system, arch in (("win32", "x64"), ("linux", "x64"), ("linux", "arm64")):
            for version in ((3, 11), (3, 12)):
                for accelerator in ("cpu", "cuda"):
                    lane = select_lane({"accelerator": accelerator}, system=system, machine=arch, version=version, driver_cuda=130)
                    self.assertEqual(lane["arch"], arch)
                    self.assertEqual(len(torch_requirements(lane)), 3)
                    if accelerator == "cuda":
                        self.assertEqual(lane["flavor"], "cu130" if arch == "arm64" else "cu128")

    def test_arm_uses_real_driver_not_capped_host_value(self):
        context = {"accelerator": "cuda", "cuda_version": 128}
        lane = select_lane(context, system="linux", machine="aarch64", version=(3, 12), driver_cuda=130)
        self.assertEqual(lane["flavor"], "cu130")
        with self.assertRaises(ValueError):
            select_lane(context, system="linux", machine="aarch64", version=(3, 12), driver_cuda=128)

    def test_unsupported_platform_and_abi_fail(self):
        for system, arch, version in (("darwin", "arm64", (3, 12)), ("win32", "arm64", (3, 12)), ("linux", "x64", (3, 13))):
            with self.assertRaises(ValueError):
                select_lane({}, system=system, machine=arch, version=version)

    def test_both_setup_argument_contracts(self):
        value = {"python_exe": sys.executable, "ext_dir": str(ROOT), "gpu_sm": 75}
        self.assertEqual(parse_args(["setup.py", json.dumps(value)]), value)
        self.assertEqual(parse_args(["setup.py", sys.executable, str(ROOT), "75", "128"])["cuda_version"], 128)

    def test_real_entry_errors_are_json_and_nonzero(self):
        for request in ("not json", json.dumps({"nodeId": "bad", "input": {}})):
            result = subprocess.run([sys.executable, "processor.py"], cwd=ROOT, input=request + "\n", text=True, capture_output=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            messages = [json.loads(line) for line in result.stdout.splitlines()]
            terminals = [m for m in messages if m["type"] in ("done", "error")]
            self.assertEqual(len(terminals), 1)
            self.assertEqual(terminals[0]["type"], "error")

    def test_native_stdout_stderr_and_single_terminal(self):
        code = '''
import os
from acestep_modly.protocol import Protocol
p = Protocol()
with p.capture_logs():
    print("Python output")
    os.write(1, b"native stdout\\n")
    os.write(2, b"native stderr\\n")
p.progress(50, "half")
p.progress(10, "earlier upstream phase")
p.emit({"type": "done", "result": {"filePath": "/contract-only.wav"}})
p.emit({"type": "error", "message": "must not emit a second terminal"})
'''
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True, check=True, timeout=30)
        messages = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual({m["message"] for m in messages if m["type"] == "log"}, {"Python output", "native stdout", "native stderr"})
        self.assertEqual([m["percent"] for m in messages if m["type"] == "progress"], [50, 50])
        self.assertEqual(sum(m["type"] in ("done", "error") for m in messages), 1)

    def test_unicode_logs_override_legacy_windows_codepage(self):
        code = '''
from acestep_modly.protocol import Protocol, configure_stdio
configure_stdio()
p = Protocol()
with p.capture_logs():
    print("Music: \\u97f3\\u697d \\U0001f3b5")
p.emit({"type":"done", "result": {"text":"ok"}})
'''
        env = {**os.environ, "PYTHONIOENCODING": "cp1252", "PYTHONUTF8": "0"}
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, check=True, timeout=30)
        messages = [json.loads(line) for line in result.stdout.decode("utf-8").splitlines()]
        self.assertIn("Music: 音楽 🎵", [m.get("message") for m in messages])


if __name__ == "__main__":
    unittest.main()
