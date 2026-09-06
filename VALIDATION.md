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
comparators. The initial package audit did not modify host files, existing
extensions or public repositories. The runtime repair validation below used
an updated copy of this extension in Modly's extensions directory.

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

### Linux ARM64 cuSPARSELt setup regression

The published `nvidia_cusparselt_cu13-0.8.0-py3-none-manylinux2014_aarch64.whl`
contains the sole internal tag `py3-none-manylinux2014_sbsa`. Pip accepts the
filename at install time but rejects this internal tag during `pip check`.
The narrowly scoped repair runs under the extension venv on Linux aarch64 cu130
only. It validates the distribution name/version, sole known tag, WHEEL/RECORD
hash and size, ELF64 little-endian AArch64 shared-library header and actual
native load before replacing the tag with `manylinux2014_aarch64` and updating
RECORD. Unrelated pip failures are not filtered or suppressed.

Regression evidence on Linux ARM64, system Python 3.12: the new test module was
first run RED with the missing implementation, then all **7 regression tests
passed**. Full default discovery ran **29 tests: 26 passed, 3 opt-in upstream
tests skipped**. The existing local HTTP-resume test required host loopback
access; the sandbox-only run failed on socket permissions, not an assertion.
Compile checks and `git diff --check` also passed.

The new tests cover non-target platform/lane no-ops, unknown identity/version/
tags, invalid native headers, loader failure, RECORD corruption/duplicates,
correct hashes and idempotency, recovery from an interrupted metadata pair
update, target-venv invocation ordering, and fatal unrelated pip conflicts.
They use synthetic package fixtures and a mocked native loader: this is
regression proof, **not** successful Modly setup, GPU inference or audio-quality
proof. The separate real runtime evidence follows.

### Real GB10 runtime acceptance — 2026-09-06

Tested the copied extension in Modly's runtime directory using its own CPython
3.12 venv on Linux ARM64/NVIDIA GB10, with torch **2.10.0+cu130**:

| Check | Measured result |
| --- | --- |
| Initial setup | Exit 0; **25 downloaded, 0 reused**; real imports, CUDA kernels and stereo WAV32 health passed |
| Strict `pip check` | Exit 0; no dependency conflicts suppressed |
| Actual processor inference | 10 seconds, seed 42, instrumental, planner Off; CUDA BF16, SDPA, Turbo 8 steps; Automatic memory mode selected CPU offload |
| Generated WAV | **480,000 frames, 48,000 Hz, stereo, FLOAT**, 10.0 seconds; all samples finite; peak **0.8912509**, RMS **0.0749941** |
| Process protocol | **222 NDJSON messages**, exactly one terminal `done`; monotonic progress ending at 100 |
| Repair | Exit 0; **25 reused, 0 downloaded**; unchanged weight modification times |
| Post-Repair integrity | Corrected aarch64 WHEEL tag; RECORD verification had zero errors; copied source hashes unchanged |

The input was `Warm acoustic jazz, brushed drums, mellow piano, relaxed evening
atmosphere`. This run used real local checkpoints and neural inference, not the
controlled model boundary used in the unit tests. Local setup/Repair logs and
`inference-evidence.json` / `final-runtime-evidence.json` record these results;
the generated WAV SHA-256 is
`f84fb60da989884c640acd26cc29bd5a76749163373bb152573ec549b2645c71`.
For this initial 10-second instrumental check, listening/audio quality and Modly
UI end-to-end behavior were not assessed. The subsequent vocal/planner listening
acceptance below is separate evidence; neither run qualifies other hardware or
Python 3.11 ARM64.

### Full-song vocals and planner demo — 2026-09-06

A subsequent real Modly workflow generated the published **Worlds We Make** take
on the same GB10/Linux ARM64/Python 3.12 runtime. The sanitized
[generation sidecar](assets/demo/worlds-we-make.json) retains the exact caption,
lyrics, requested parameters, actual seed/duration and pinned upstream revision.

| Check | Evidence / scope |
| --- | --- |
| Recorded generation settings | 120 s; seed **1786369090**; **115 BPM**, A minor, 4/4; English vocal mode; CUDA, CPU offload; planner **On**, temperature **0.85** |
| Original output | Stereo 48,000 Hz FLOAT WAV; **5,760,000 frames**, **120.0 s**, **46,080,088 bytes** |
| Vocals / lyric coverage / opening | **USER-VERIFIED by listening:** this take sings the complete supplied song without the earlier approximately 30-second intro |
| Planner comparison | User reported skipped opening lyrics in planner-Off takes, then complete lyrics after enabling the 1.7B planner; this is sample-specific, not a general alignment benchmark |
| Public preview | **4,005,621 bytes**; full 120.0 s; H.264 High, yuv420p, 1280×720, 2 fps; AAC-LC stereo 48 kHz, encoded at 256 kb/s; faststart MP4 |
| Preview integrity | Full FFmpeg audio/video decode completed without errors; cover inspected visually; original WAV SHA-256 unchanged before/after encoding |

Original WAV SHA-256:
`6d501ce8aedec7efd4587114cce0b1c49eff92f299be411fd00c3319df20fe14`

Published MP4 SHA-256:
`3e4f247c03d7c1cf88dee89f04a238d29816793edc5b952e6520e746ea013c16`

The MP4 is a **lossy listening preview**, not the float master or a new generation.
No sections were removed/reordered and no normalization, remixing or gain changes
were applied. FFmpeg 6.1.1 was extracted from the official public Ubuntu ARM64
package into a temporary directory for this encode; it was not installed into
the extension or added as a runtime dependency. The original WAV stays outside
the repository. The cover is original text/vector-style artwork, not a claim to
be an official Modly logo.

This evidence updates the earlier “vocals and planner inference untested” status.
It does **not** establish universal lyric adherence, exact vocal identity, quality
across seeds, clean GitHub installation, complete UI lifecycle/cancellation,
absence of generation-time network activity, or another hardware/ABI lane.

### Cross-platform availability and remaining qualification

`scripts/check_wheels.py` queried official PyTorch indices. All **36 checks**
passed: three native packages × three platform targets × two Python ABIs ×
two CPU/CUDA variants. Direct requirement releases were also checked against
PyPI metadata for Python compatibility and platform/pure-Python wheels.

| Target | Configured dependency route | Real target qualification |
| --- | --- | --- |
| Windows x64, Python 3.11 / 3.12 | torch 2.7.1, torchvision 0.22.1, cu128 or CPU | Not executed on a Windows host |
| Linux x64, Python 3.11 / 3.12 | torch 2.10.0, torchvision 0.25.0, cu128 or CPU | CPU dependency/kernel/audio checks executed; CUDA not executed |
| Linux ARM64, Python 3.11 / 3.12 | torch 2.10.0, torchvision 0.25.0, cu130 or CPU | GB10/Python 3.12 cu130 setup, Repair and instrumental inference passed; full-song planner/vocals user-listened; Python 3.11 and CPU runtime untested |

An independent fresh-context review caught the capped ARM64 CUDA report,
BF16-emulation detection on older GPUs and an unproven storage-path fallback.
All three were corrected and covered by regression tests. Source pinning,
model paths, API calls and IPC were reviewed; this does not replace hardware
qualification.

## Required real acceptance before declaring full platform support

The GB10/Python 3.12 runs above cover setup, native checks, instrumental
file/protocol generation, intact-weight Repair and one user-listened planner/vocal
song for that target only. The remaining UI lifecycle and platform checks are
not implied by these results.

1. Install from GitHub after publication, or local import + Repair; confirm the
   exact configured `models_dir` in setup logs.
2. Complete the real snapshot download and record successful setup/native checks.
3. Generate the supplied 10-second instrumental smoke input with seed 42.
   Confirm audible music, the Modly audio preview and a valid stereo WAV.
4. Extend the vocal/planner acceptance beyond the published GB10 sample to more
   seeds, lyrics and targets. Independently confirm no generation-time network
   access; the accepted listening test did not measure network activity.
5. Run Repair/update again; confirm **25 reused, 0 downloaded** for an unchanged
   intact weight lock. Interrupt a download and verify Repair resumes it.
6. Repeat the runtime tests on physical Windows x64, Linux x64 CUDA and Linux
   ARM64 CUDA, for both Python ABIs before marking those lanes qualified.

The repository contains no model weights. The results above distinguish initial
prepublication checks from subsequent runtime and listening evidence. The included CI workflow covers the six CPU platform/ABI
combinations; current remote results are available in
[GitHub Actions](https://github.com/DrHepa/modly-acestep-15-extension/actions).
