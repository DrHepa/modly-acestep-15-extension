"""Real upstream API/audio integration with ONLY the model inference boundary replaced.

Enable with ACESTEP_TEST_UPSTREAM=1 in a prepared dependency environment.
These are contract tests, not evidence of model quality or GPU qualification.
"""
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from acestep_modly.parameters import normalize
from acestep_modly.constants import NODE_ID
from acestep_modly.vendor import ensure_source
from acestep_modly.offline import configure


@unittest.skipUnless(os.environ.get("ACESTEP_TEST_UPSTREAM") == "1", "Enable upstream integration tests explicitly")
class UpstreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_source(lambda _: None)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        configure(cls.root / "checkpoints", cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_real_orchestrator_seed_params_and_wav32(self):
        import torch
        import soundfile as sf
        from acestep.handler import AceStepHandler
        from acestep_modly.runtime import generate
        observed = {}
        original = AceStepHandler.generate_music

        def inference_boundary(self, **kwargs):
            observed.update(kwargs)
            return {"success": True, "audios": [{"tensor": torch.zeros((2, 4800)), "sample_rate": 48000}], "extra_outputs": {}}

        inference_boundary.__signature__ = inspect.signature(original)
        values = normalize({"nodeId": NODE_ID, "input": {"text": "Acoustic jazz"}, "params": {"duration": 10, "seed": 42, "device": "cpu"}})
        output_dir = self.root / "output"
        with patch.object(AceStepHandler, "initialize_service", return_value=("Test-only model boundary", True)) as init, patch.object(AceStepHandler, "generate_music", inference_boundary):
            output, metadata = generate(values, self.root / "checkpoints", output_dir, self.root, lambda *_: None, lambda _: None)
        self.assertEqual(init.call_args.kwargs["config_path"], "acestep-v15-turbo")
        self.assertEqual(init.call_args.kwargs["vae_checkpoint"], "official")
        self.assertEqual(observed["captions"], "Acoustic jazz")
        self.assertEqual(observed["seed"], "42")
        self.assertEqual(observed["audio_duration"], 10)
        self.assertEqual(observed["inference_steps"], 8)
        self.assertEqual(observed["batch_size"], 1)
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(sf.info(output).subtype, "FLOAT")
        self.assertEqual((sf.info(output).channels, sf.info(output).samplerate), (2, 48000))

    def test_native_bf16_gate_rejects_turing_emulation(self):
        from acestep.gpu_config import cuda_supports_bfloat16
        import torch
        with patch.object(torch.cuda, "is_available", return_value=True), patch.object(torch.cuda, "get_device_capability", return_value=(7, 5)), patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertFalse(cuda_supports_bfloat16())

    def test_wrapper_propagates_real_upstream_failure(self):
        from acestep.handler import AceStepHandler
        from acestep_modly.runtime import generate
        values = normalize({"nodeId": NODE_ID, "input": {"text": "Acoustic jazz"}, "params": {"duration": 10, "device": "cpu"}})
        with patch.object(AceStepHandler, "initialize_service", return_value=("Model initialization failed", False)):
            with self.assertRaisesRegex(RuntimeError, "Model initialization failed"):
                generate(values, self.root / "checkpoints", self.root / "failed", self.root, lambda *_: None, lambda _: None)


if __name__ == "__main__":
    unittest.main()
