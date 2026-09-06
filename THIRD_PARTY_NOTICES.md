# Third-party notices

The integration files are Copyright (c) 2026 DrHepa, MIT. This does not transfer
authorship of models, libraries or Modly to the extension author.

## Bundled ACE-Step source

Origin: https://github.com/ace-step/ACE-Step-1.5

Commit: `ca1e85fe9430179831e6bc6be790c332190a3866`.

The bundled archive is the upstream `acestep/` tree and root `LICENSE`, produced
with `git archive` at that commit. No model weights are included. The source is
not modified; adapter subclasses and environment configuration live outside it.
Setup extracts this archive and copies the pinned model-side Python files to
the local checkpoint directory, as the upstream initializer normally does.

Root license: MIT, Copyright (c) 2026 ACEStep. Full text is in
`licenses/ACE-Step-MIT.txt` and inside the source archive. Individual source
files including Turbo model definitions also carry Apache License 2.0 notices
(including the upstream spelling “ACESTEO Team”). These headers are retained
unchanged; `licenses/Apache-2.0.txt` supplies the license text.

## Downloaded weights and tokenizers

Source: https://huggingface.co/ACE-Step/Ace-Step1.5

Snapshot: `19671f406d603126926c1b7e2adc169acbcade22`.

The model card declares MIT. Setup downloads only the immutable files listed
in `weights.lock.json` directly from the official repository, including its
README/model card. The text encoder is Qwen3-Embedding-0.6B and the 1.7B music
LM is Qwen-based; Qwen upstream components carry their own Apache-2.0 terms.
Relevant upstream references:

- https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
- https://huggingface.co/Qwen/Qwen3-1.7B

No checkpoints are distributed in the extension or mirrored to another HF
repository. Users remain responsible for the rights in prompts/lyrics and
their use of generated material; the wrapper license does not grant unrelated
content rights.

## Installed dependencies

Dependencies are installed from their original package indices, not embedded
in the archive. Their package license files remain in the venv. Key packages
include PyTorch/torchaudio/torchvision, Transformers, Diffusers, Accelerate,
Hugging Face Hub, SoundFile/libsndfile, NumPy, SciPy, PyWavelets,
pytorch-wavelets and vector-quantize-pytorch. Review their distributed license
notices before redistributing a complete runtime image.

## Host

Modly is authored by Lightning Pixel and its contributors:
https://github.com/lightningpixel/modly . No Modly host source is bundled here.
