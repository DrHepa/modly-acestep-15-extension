# Validation record — 2026-09-06

This record distinguishes implementation, real dependency tests, contract tests
with a controlled model boundary, and hardware/model tests not yet performed.
It is not a claim that music generation has been qualified on every platform.

## Audited sources

| Source | Pinned revision / relevant code |
| --- | --- |
| Modly 0.4.2 main | [`1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65`](https://github.com/lightningpixel/modly/tree/1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65) |
| Process protocol | [`electron/main/process-runner.ts`](https://github.com/lightningpixel/modly/blob/1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65/electron/main/process-runner.ts) |
| Install/setup and logs | [`electron/main/ipc-handlers.ts`](https://github.com/lightningpixel/modly/blob/1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65/electron/main/ipc-handlers.ts) |
| Settings and models path | [`electron/main/settings-store.ts`](https://github.com/lightningpixel/modly/blob/1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65/electron/main/settings-store.ts) |
| ACE-Step source / dependency policy | [`ca1e85fe9430179831e6bc6be790c332190a3866`](https://github.com/ace-step/ACE-Step-1.5/tree/ca1e85fe9430179831e6bc6be790c332190a3866), `pyproject.toml`, `uv.lock` |
| Actual inference API | [`acestep/inference.py`](https://github.com/ace-step/ACE-Step-1.5/blob/ca1e85fe9430179831e6bc6be790c332190a3866/acestep/inference.py), `handler.py`, `llm_inference.py`, `audio_utils.py` |
| Official model snapshot | [`19671f406d603126926c1b7e2adc169acbcade22`](https://huggingface.co/ACE-Step/Ace-Step1.5/tree/19671f406d603126926c1b7e2adc169acbcade22) |

The host was read directly, not inferred from another extension's manifest.
DrHepa's Qwen3-TTS and Kokoro-ONNX process extensions were also inspected as
comparators. No host files, existing extensions or public repositories were
modified as part of this package.

## Host contract decisions

- `type: process`, explicit Python entry, one `text` → `audio` node. There are
  no speculative model-runner or HF-download fields in the manifest.
- Setup accepts upstream's single JSON argument and the legacy positional form.
  Modly invokes Python setup for process extensions and forwards both log streams.
- Generation receives `input`, `params`, `nodeId`, `workspaceDir`, `tempDir` and
  emits JSON `progress`, `log`, `done.result.filePath`, or `error.message`.
- Process stderr is buffered by the audited host. The adapter captures Python
  and native fd 1/2 output into live JSON logs, including Unicode-safe output.
- `models_dir` is not in the current process/setup payload. Settings must be
  bound to the installed extension, or an explicit override/successful prior
  setup binding must resolve it. Missing/ambiguous configuration fails closed.
- Process setup downloads weights because the user explicitly requested this
  lifecycle. It is not the model-node UI download contract.
- The host's Python-process `terminate()` is empty. The README records this
  limitation instead of claiming workflow cancellation support.

## Tests actually executed

| Check | Python 3.11.15, Linux x64 CPU | Python 3.12.13, Linux x64 CPU |
| --- | --- | --- |
| Install selected CPU torch/torchaudio/torchvision and inference dependencies | Passed | Passed |
| Dependency consistency | Passed | Passed |
| Real imports: handler, LM, Turbo class, Transformers, Diffusers | Passed | Passed |
| Real DWT / torchaudio resample native operations | Passed | Passed |
| Upstream AudioSaver stereo 48 kHz WAV32 write/read | Passed | Passed |
| Complete unittest suite with upstream tests enabled | 22 passed | 22 passed |

The suite covers schema defaults and input validation; all 12 CPU/CUDA lane
selections; unsupported platforms/ABIs; the capped Modly ARM64 driver value;
setup argument formats; real entry-point failure protocol; native stdout/stderr
capture; monotonic progress; one terminal message; Windows-codepage Unicode
handling; settings relocation; path traversal/symlink/reparse rejection;
SHA-256/Git-blob checks; first download vs update reuse; corrupt-file recovery;
missing local weights; HTTP Range resume via a real local HTTP server; native
BF16 capability gating; upstream error propagation; and the real upstream
orchestrator's parameter/seed forwarding and WAV serialization.

The orchestrator integration test replaces model initialization and the expensive
DiT neural inference call with a controlled tensor fixture. All surrounding
upstream orchestration and WAV saving are real. It is **not** an inference or
audio-quality test and its output is not shipped as a generated example.

An additional live HF check fetched only the official snapshot's README and
Qwen embedding tokenizer JSON, verified the Git-blob and LFS SHA-256 identities,
then repeated provisioning with a downloader that fails on any call. Result:
first run 2 downloaded; second run **2 reused, 0 downloads**. No neural checkpoint
weights were fetched for that test.

The skill's real-entry process test passed the expected-error path. The generic
v0.4 validator is used with `--allow-io audio` because current main genuinely
supports audio; old image/text/mesh-only defaults are not the live host schema.

## Platform evidence

`scripts/check_wheels.py` queried official PyTorch indices. All **36 checks**
passed: three native packages × three platform targets × two Python ABIs ×
two CPU/CUDA variants. Direct requirement releases were also checked against
PyPI metadata for Python compatibility and platform/pure-Python wheels.

| Target | Configured dependency route | Real target qualification |
| --- | --- | --- |
| Windows x64, Python 3.11 / 3.12 | torch 2.7.1, torchvision 0.22.1, cu128 or CPU | Not executed on a Windows host |
| Linux x64, Python 3.11 / 3.12 | torch 2.10.0, torchvision 0.25.0, cu128 or CPU | CPU dependency/kernel/audio checks executed; CUDA not executed |
| Linux ARM64, Python 3.11 / 3.12 | torch 2.10.0, torchvision 0.25.0, cu130 or CPU | Not executed on an ARM64 host |

An independent fresh-context review caught the capped ARM64 CUDA report,
BF16-emulation detection on older GPUs and an unproven storage-path fallback.
All three were corrected and covered by regression tests. Source pinning,
model paths, API calls and IPC were reviewed; this does not replace hardware
qualification.

## Required real acceptance before declaring full platform support

1. Install from GitHub after publication, or local import + Repair; confirm the
   exact configured `models_dir` in setup logs.
2. Complete the real snapshot download and record successful setup/native checks.
3. Generate the supplied 10-second instrumental smoke input with seed 42.
   Confirm audible music, the Modly audio preview and a valid stereo WAV.
4. Test vocal mode with supplied lyrics, then enable the 1.7B planner on a
   sufficiently provisioned target. Confirm no generation-time network access.
5. Run Repair/update again; confirm **25 reused, 0 downloaded** for an unchanged
   intact weight lock. Interrupt a download and verify Repair resumes it.
6. Repeat the runtime tests on physical Windows x64, Linux x64 CUDA and Linux
   ARM64 CUDA, for both Python ABIs before marking those lanes qualified.

The repository contains no model weights. The results above record validation
before publication. The included CI workflow covers the six CPU platform/ABI
combinations; current remote results are available in
[GitHub Actions](https://github.com/DrHepa/modly-acestep-15-extension/actions).
