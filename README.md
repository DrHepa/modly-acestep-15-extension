# ACE-Step 1.5 Music for Modly

A local **process** extension by **DrHepa**. One Text to Music node uses the real
ACE-Step 1.5 Turbo inference API to produce instrumental music or songs from a
description and optional lyrics. This is music generation, not a speech TTS or
voice-cloning adapter.

`setup.py` prepares an isolated runtime and downloads the pinned model files to
Modly's persistent models directory. `processor.py` only reads local weights;
it never downloads them. This follows the current process contract, not the
separate weight-download UI used by model extensions.

Target: upstream Modly **0.4.2**, audited at
[`1476fd0`](https://github.com/lightningpixel/modly/tree/1476fd0b1c19c9ab177c1ca3ee4d1842119e9f65).
See [VALIDATION.md](VALIDATION.md) for the exact evidence and untested targets.

## Listen to a generated song

**Worlds We Make** — a two-minute epic synth-pop song about creating worlds with
Modly, with original English lyrics and a requested female lead vocal.

https://github.com/user-attachments/assets/1a6ccf6c-75fa-4387-85ae-b88966342111

Press play and use the speaker control to unmute if the preview starts silently.

[Play / download the MP4 preview](assets/demo/worlds-we-make.mp4) ·
[Prompt and generation parameters](assets/demo/worlds-we-make.json) ·
[Lyrics](assets/demo/worlds-we-make-lyrics.txt)

Generated locally on **NVIDIA GB10 / Linux ARM64 / Python 3.12**, with **120 s,
115 BPM, A minor, 4/4, seed 1786369090, CUDA, CPU offload**, and the **1.7B planner
On at temperature 0.85**. Turbo uses 8 steps. 
That is sample-specific listening acceptance, not a guarantee for every song.

The **4.01 MB MP4** is a lossy AAC listening preview with a static cover. It
contains the full generated 120 seconds, without rearrangement or normalization;
the original float WAV is preserved locally and is not bundled in this repository.

## Installation

### Local package

1. Extract this repository as `modly-acestep-15-extension` in the Extensions
   directory shown in Modly Settings, or use Modly's local-extension import.
2. Run **Repair** to execute setup. Local import alone does not install Python
   dependencies or weights.
3. Wait for `Setup complete` and a successful exit. The initial setup downloads
   about **10.09 GB (9.40 GiB)** of model files plus Python dependencies. Keep
   additional disk space for the venv, pip cache and interrupted downloads.
4. In Workflows, add **ACE-Step Text to Music**.

For installation from the repository, use **Models/Extensions → Install from GitHub**:

`https://github.com/DrHepa/modly-acestep-15-extension`

Modly runs setup after cloning the repository. This repository's availability
does not imply inclusion in the separate Modly extension catalog.

### Setup and persistent storage

The actual path is:

`<Modly models_dir>/modly-acestep-15-extension/checkpoints/`

The snapshot contains Turbo, the official VAE, Qwen3-Embedding-0.6B and the
optional 1.7B music planner. All four are provisioned during setup so enabling
the planner later does not cause a generation-time download.

Current Modly does **not** supply `models_dir` in the process/setup JSON. The
adapter therefore resolves it from Modly's settings only when `extensionsDir`
identifies this installed extension. Explicit `MODLY_MODELS_DIR` or `MODELS_DIR`
overrides are also supported. A successful manual setup records its resolved
path inside the venv. Unbound or conflicting settings cause an actionable
failure; the adapter does not invent a sibling models directory.

On Repair/update, each existing file is checked against its pinned size and
SHA-256 or Git-blob identity. Correct files are reused **without an HF request**,
even if the extension directory or venv was replaced. Only missing/corrupt files
are downloaded. Partial files are resumed when the server supports HTTP Range;
verification must pass before atomic replacement. Setup does not delete weights.

Updating the extension's source does not change its pinned weights. A future
explicit change to `weights.lock.json` may require new or changed files; that is
not a redundant download of the same snapshot. Renaming the extension id or
moving `models_dir` without moving its data creates a different storage location.

### Manual setup

Prefer Modly's Repair, which provides its real interpreter/GPU context. The
legacy form is supported for a manual installation (run from this directory):

```bash
python setup.py /absolute/path/to/modly/python /absolute/path/to/modly-acestep-15-extension 0 0
```

Run it with the same Python as the first argument. `0 0` requests CPU when no
accelerator is supplied; use Modly's JSON context for CUDA rather than guessing
a GPU capability. The JSON form includes `python_exe`, `ext_dir`, `gpu_sm`,
`cuda_version`, `accelerator`, `platform`, and `arch`. An explicit `models_dir`
is accepted for manual invocation, but is not claimed as an upstream field.

If automatic storage discovery fails, set `MODELS_DIR` to the exact path shown
in Settings before manual setup. For a relocated/custom Modly configuration,
`MODLY_USER_DATA` can identify the directory containing its `settings.json`.

## Usage

1. Connect a **Text** node containing a musical description, such as
   `Warm acoustic jazz, brushed drums, mellow piano, relaxed evening atmosphere`.
   The description field on the process node is a fallback if no text is connected.
2. For the first test, use **10 seconds**, **Instrumental**, **planner Off**,
   **Automatic device**, **Automatic offload**, seed **42**.
3. Run the workflow. Setup/import/loading/generation errors appear in the logs;
   a successful run returns a WAV file to Modly's audio output.
4. For vocals, switch Instrumental to **Vocals / lyrics**. **Expand the Lyrics
   field before pasting** the multiline lyrics, preserving `[Verse]` / `[Chorus]`
   tags and line breaks. Expanding after a single-line paste does not restore lost
   breaks; literal `\n` separators are also accepted.
5. For a full vocal song, try **planner On** when sufficient memory is available.
   In the demo comparison, this resolved omitted opening lyrics and a long intro
   reported with planner Off. This is one user-verified result, not a universal
   alignment guarantee; the extension default remains Off for the lighter smoke test.

The planner generates music codes and can fill missing musical metadata. This
adapter does not enable automatic lyric writing, caption rewriting, audio input,
cover/repaint, LoRA, training or multiple output batches.

## Parameters

| Parameter | Meaning |
| --- | --- |
| Music description | Connected `input.text` takes priority; 1–4096 characters. |
| Lyrics / Instrumental | Lyrics are used only in vocal mode; instrumental sends `[Instrumental]`. |
| Duration | 10–600 seconds, default 30. Upstream GPU-tier limits are enforced; long audio needs more memory. |
| Seed | `-1` chooses a random seed; 0–4294967295 reproduces a chosen seed subject to hardware/library nondeterminism. |
| BPM | `0` for unspecified, otherwise 30–300. |
| Key / scale | Optional upstream text such as `C major`. |
| Time signature | Unspecified, 2/4, 3/4, 4/4 or 6/8. |
| Vocal language | Upstream language code such as `en`, `es`, `ja`, or `unknown`. |
| 5Hz music planner | Off by default; optional local 1.7B LM using the portable PyTorch backend. |
| Planner temperature | 0.1–2.0; visible only with the planner enabled. |
| Device | Automatic, CUDA or CPU. An unavailable explicitly requested CUDA device is an error. |
| Memory mode | Auto, CPU offload or resident. Offload trades speed for reduced GPU residency. |

Turbo sampling is fixed to its upstream **8-step** setting. No ineffective CFG
or arbitrary model-mode controls are exposed. Runtime defaults come directly
from `manifest.json`.

## Outputs

One **48 kHz stereo, 32-bit float WAV** per run, under:

`<workspaceDir>/Workflows/ACE-Step-1.5/<unique-run-id>/`

A JSON sidecar records requested parameters, actual output duration/seed and
the upstream code revision. The processor returns the existing absolute WAV
path as `done.result.filePath`; intermediate caches live below `tempDir`.
The WAV32 path uses upstream AudioSaver with SoundFile directly, so FFmpeg and
torchcodec are not required for this node.

## Requirements and compatibility

CPython **3.11 and 3.12**, with `venv` and pip available. Native packages use
explicit platform lanes from upstream's dependency policy:

| Target | PyTorch / torchaudio | torchvision | CUDA lane | CPU lane |
| --- | --- | --- | --- | --- |
| Windows x64 | 2.7.1 | 0.22.1 | cu128 | cpu |
| Linux x64 | 2.10.0 | 0.25.0 | cu128 | cpu |
| Linux ARM64 | 2.10.0 | 0.25.0 | cu130 | cpu |

Official wheels were confirmed for all six platform/Python combinations, in
both CUDA and CPU variants. This is package availability, **not** a claim of
completed runtime testing on every target. Linux x64 CPU imports, native
operations, actual upstream orchestration around a controlled inference
boundary, and WAV writing passed on Python 3.11 and 3.12. Linux ARM64 on NVIDIA
GB10 with Python 3.12/cu130 also passed real setup, Repair and a 10-second
instrumental generation through the actual processor. A subsequent 120-second
vocal song with the 1.7B planner enabled was generated in Modly and accepted by
the user after listening. See VALIDATION.md for the scope of each check.

Linux needs a distribution compatible with the selected manylinux wheels
(glibc 2.28 or later). Windows may need the Microsoft Visual C++ runtime.
CUDA requires a compatible NVIDIA driver: this installer deliberately requires
the driver to advertise CUDA 12.8 for x64 or 13.0 for ARM64. It reads
`nvidia-smi`, because the audited Modly host caps its reported value at 12.8.
ARM64 means Linux SBSA/aarch64 with these wheels; this is not a blanket claim
for every Jetson/JetPack version. No local CUDA compiler is required.

This release uses unquantized inference with optional CPU offload. No minimum
VRAM guarantee is made: memory depends on duration, planner and device. On an
8 GB GPU, start with planner Off, CPU offload and a 10-second track. CPU can be
very slow and needs enough system RAM; plan generously, especially with the LM.
Setup tests native imports/kernels **before** starting the model download.

## Limitations

- Real instrumental and planner-enabled vocal inference have been exercised on
  GB10/Linux ARM64/Python 3.12. The vocal demo has user listening acceptance;
  this does not qualify every lyric, duration, voice or musical style. Clean
  Install from GitHub, comprehensive UI lifecycle/cancellation and other
  hardware/ABI targets still require their own acceptance checks.
- No macOS, Windows ARM64, ROCm, MPS, XPU, MLX, vLLM, FlashAttention or
  quantized-model lane is exposed by this release.
- Processes load models anew for each run. The LM tokenizer can take 1–2 minutes
  to initialize; a heartbeat log prevents silent waiting.
- The audited Modly Python process runner's `terminate()` is a no-op. OS
  SIGTERM/interrupts are handled, but Modly's workflow Stop cannot be promised
  to cancel an already-running Python process until the host fixes that path.
- One setup/generation at a time uses this extension's asset lock. No global
  cross-extension weight deduplication or `weights_owner_id` behavior is claimed.
- Setup verifies full hashes. Generation performs local size/existence and
  model-code checks; use Repair for a full integrity verification.
- No model weights are bundled or rehosted in this repository.

## Troubleshooting

| Symptom | Action |
| --- | --- |
| Cannot resolve models_dir | Verify Modly Settings paths; use `MODELS_DIR` or `MODLY_USER_DATA` for manual setup. No weights have been downloaded to a guessed location. |
| Interrupted download | Run Repair. Verified files are retained and `.part` files are resumed when possible. |
| Missing/incomplete weights during generation | Run Repair; the processor intentionally refuses network fallback. |
| CUDA unavailable or unsupported driver | Update the driver or install an explicit CPU lane. Do not replace torch independently of torchaudio/torchvision. |
| `nvidia-cusparselt-cu13 0.8.0 is not supported on this platform` on Linux ARM64 CUDA 13 | Update this extension and run Repair. The official aarch64 wheel has an incorrect internal SBSA tag. Setup validates its identity, ELF architecture and native load in the venv, then corrects only that known tag and its RECORD hash. `pip check` remains mandatory; other dependency/native errors still stop setup. |
| CUDA out of memory | Disable the planner, reduce duration and use CPU offload. Close other GPU jobs. |
| Missing opening lyrics / unexpectedly long instrumental intro | Check vocal mode and preserved lyric line breaks first. With enough memory, try planner On; the published demo used temperature 0.85. Prompt instructions alone do not guarantee lyric order. |
| LoRA / Lightning / bitsandbytes warnings | These optional training packages are intentionally omitted; they are not needed by the exposed node. |
| Long pause in loading/planning | Watch the live heartbeat and stage logs. First initialization is slower than inference. |
| Setup failed | Read the final error and the preceding pip/native logs. It exits nonzero and does not report successful setup. |
| Python version changed | Repair creates the correct venv and preserves the incompatible venv under a uniquely named backup. Weights remain outside both. |

All process stdout is newline-delimited JSON. Python prints and native stdout /
stderr are forwarded as Modly `log` events; progress is monotonic and there is
exactly one terminal success or error. Upstream error text/tracebacks are kept
visible rather than being replaced with a generic success result.

## Development and verification

```bash
python -m unittest discover -s tests -v
python scripts/check_wheels.py
```

In an isolated environment with the dependencies prepared:

```bash
python scripts/ci_prepare.py
python -m acestep_modly.health --accelerator cpu
```

Set `ACESTEP_TEST_UPSTREAM=1` when running unittest to enable the upstream API
integration tests. Those tests replace only the expensive model boundary; their
synthetic audio is not a generated-music example. The supplied CI matrix adds
Windows/Linux/ARM64 × Python 3.11/3.12 CPU checks when pushed to GitHub. See
[GitHub Actions](https://github.com/DrHepa/modly-acestep-15-extension/actions)
for current results; the prepublication test record is in VALIDATION.md.

For a real local acceptance test after setup, run the supplied
`scripts/smoke_generate.py` using the extension's **venv Python**, with
`--workspace` and `--temp` absolute directories. It calls the actual processor
and verifies its terminal protocol and output file; it has no model substitute.

## Upstream and credits

- Extension integration: **DrHepa**.
- Modly: **Lightning Pixel** — [lightningpixel/modly](https://github.com/lightningpixel/modly).
- ACE-Step developers: [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5),
  source pinned to `ca1e85fe9430179831e6bc6be790c332190a3866`.
- Official weights: [ACE-Step/Ace-Step1.5](https://huggingface.co/ACE-Step/Ace-Step1.5/tree/19671f406d603126926c1b7e2adc169acbcade22),
  pinned to `19671f406d603126926c1b7e2adc169acbcade22`.
- Qwen contributors for the text encoder / LM foundations and the authors of
  PyTorch, Transformers, Diffusers and the other runtime dependencies.

`vendor/ace-step-runtime.tar.gz` contains unmodified upstream source and its
license, not checkpoints. `source.lock.json` records its SHA-256. Setup copies
the pinned Turbo model Python files into the local checkpoint directory,
matching upstream's code-sync behavior without a runtime download or rewrite.

## License

The DrHepa adapter is MIT-licensed. Upstream ACE-Step source has an MIT root
license and some files with their own Apache-2.0 headers; those notices are
preserved. The official HF model card declares MIT for its snapshot. Qwen and
other dependencies retain their own licenses; the adapter license does not
relicense them. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and `licenses/`.
