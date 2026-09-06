"""Real ACE-Step third-party inference API adapter (one local WAV per run)."""
from pathlib import Path

from .assets import require_assets
from .constants import DIT_MODEL, LM_MODEL
from .vendor import sync_model_code


def generate(values, checkpoints, output_dir, temporary, progress, log):
    import torch
    import soundfile as sf
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
    from acestep.gpu_config import get_global_gpu_config, cuda_supports_bfloat16

    class LocalHandler(AceStepHandler):
        def _ensure_models_present(self, **kwargs):
            require_assets(checkpoints)
            return None

        @staticmethod
        def _sync_model_code_if_needed(config_path, checkpoint_path):
            sync_model_code(checkpoint_path, verify_only=True)

    device = values["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable. Repair the CUDA runtime/driver or select CPU.")
    offload = device == "cuda" and values["memory_mode"] != "resident"
    if values["memory_mode"] == "auto" and device == "cuda":
        required_gb = 24 if values["thinking"] else 16
        offload = torch.cuda.mem_get_info()[0] < required_gb * 1024**3
    if device == "cuda":
        limits = get_global_gpu_config()
        maximum = limits.max_duration_with_lm if values["thinking"] else limits.max_duration_without_lm
        if values["duration"] > maximum:
            raise ValueError(f"Requested duration exceeds ACE-Step's {maximum}s limit for this GPU/mode; reduce duration or disable the planner")
    from acestep.constants import VALID_LANGUAGES
    if values["vocal_language"] not in VALID_LANGUAGES:
        raise ValueError("Unknown vocal_language; use an upstream language code such as en, es, ja, or unknown")
    log(f"Device: {device}; CPU offload: {offload}; planner: {values['thinking']}; sampler: Turbo 8 steps")
    handler, planner = LocalHandler(), None
    progress(10, "Loading local Turbo, VAE and text encoder")
    status, ok = handler.initialize_service(
        project_root=str(temporary), config_path=DIT_MODEL, device=device,
        use_flash_attention=False, compile_model=False, offload_to_cpu=offload,
        offload_dit_to_cpu=offload, quantization=None, use_mlx_dit=False,
        vae_checkpoint="official",
    )
    if not ok:
        raise RuntimeError(status)
    log(status)
    if values["thinking"]:
        progress(25, "Loading local 1.7B music planner (may take 1-2 minutes)")
        planner = LLMHandler()
        # LM fp16 is unsafe on older CUDA GPUs; use fp32 there, bf16 on supported GPUs.
        dtype = torch.bfloat16 if device == "cuda" and cuda_supports_bfloat16() else torch.float32
        status, ok = planner.initialize(checkpoint_dir=str(checkpoints), lm_model_path=LM_MODEL,
                                        backend="pt", device=device, offload_to_cpu=offload, dtype=dtype)
        if not ok:
            raise RuntimeError(status)
        log(status)
    params = GenerationParams(
        task_type="text2music", caption=values["caption"], lyrics=values["lyrics"],
        instrumental=values["instrumental"], duration=values["duration"], seed=values["seed"],
        bpm=values["bpm"] or None, keyscale=values["keyscale"], timesignature=values["timesignature"],
        vocal_language=values["vocal_language"], thinking=values["thinking"],
        lm_temperature=values["lm_temperature"], use_cot_metas=values["thinking"],
        use_cot_caption=False, use_cot_language=values["thinking"],
        inference_steps=8,
    )
    config = GenerationConfig(batch_size=1, use_random_seed=values["seed"] == -1,
                              seeds=None if values["seed"] == -1 else [values["seed"]], audio_format="wav32")
    progress(30, "Generating music")

    def upstream_progress(value, desc="Generating", **kwargs):
        if isinstance(value, (tuple, list)):
            value = value[0] / max(value[1], 1)
        progress(30 + max(0.0, min(1.0, float(value))) * 65, desc)

    result = generate_music(handler, planner, params, config, save_dir=str(output_dir), progress=upstream_progress)
    if not result.success:
        raise RuntimeError(result.error or result.status_message or "Upstream generation failed")
    if len(result.audios) != 1 or not result.audios[0].get("path"):
        raise RuntimeError("ACE-Step did not save the requested WAV output; inspect generation logs")
    output = Path(result.audios[0]["path"]).resolve()
    if output_dir.resolve() not in output.parents or not output.is_file() or output.suffix.lower() != ".wav":
        raise RuntimeError("Upstream returned an invalid/out-of-workspace audio path")
    info = sf.info(str(output))
    if info.frames <= 0 or info.samplerate != 48000 or info.channels != 2:
        raise RuntimeError("Generated WAV has invalid length, sample rate or channel count")
    log(f"WAV ready: {info.duration:.2f}s, {info.samplerate}Hz, {info.channels} channels")
    return output, {"seed": result.audios[0].get("params", {}).get("seed", values["seed"]), "duration_seconds": info.duration}
