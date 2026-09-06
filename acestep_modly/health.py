"""Real no-weights import/native/audio smoke check, run before model provisioning."""
import argparse
import tempfile
from pathlib import Path

from .offline import configure, deny_network
from .protocol import configure_stdio


def check(accelerator="cpu"):
    with tempfile.TemporaryDirectory(prefix="ace-step-health-") as temp:
        root = Path(temp)
        configure(root / "checkpoints", root)
        deny_network()
        import torch
        import torchaudio
        import soundfile as sf
        import transformers
        import diffusers
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
        from acestep.inference import GenerationParams, GenerationConfig, generate_music
        from acestep.models.turbo.modeling_acestep_v15_turbo import AceStepConditionGenerationModel
        from acestep.audio_utils import AudioSaver
        from pytorch_wavelets import DWT1DForward
        if accelerator == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was selected but native torch cannot use this GPU/driver")
        device = "cuda" if accelerator == "cuda" else "cpu"
        value = torch.ones((1, 1, 32), device=device)
        assert torch.isfinite(DWT1DForward(J=1).to(device)(value)[0]).all()
        assert torch.isfinite(torchaudio.functional.resample(value, 24000, 48000)).all()
        # Confirms the same upstream WAV32 path used for generated music, no torchcodec/FFmpeg.
        out = AudioSaver().save_audio(torch.zeros((2, 4800)), root / "health.wav", format="wav32")
        info = sf.info(out)
        assert info.frames == 4800 and info.channels == 2
        AceStepHandler()
        LLMHandler()
        assert GenerationConfig(audio_format="wav32").audio_format == "wav32"
        assert callable(generate_music) and AceStepConditionGenerationModel is not None
        print(f"Health OK: torch {torch.__version__}, transformers {transformers.__version__}, diffusers {diffusers.__version__}; {device} kernels; stereo WAV32. No model weights loaded.", flush=True)


def main():
    configure_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--accelerator", choices=("cpu", "cuda"), default="cpu")
    check(parser.parse_args().accelerator)


if __name__ == "__main__":
    main()
