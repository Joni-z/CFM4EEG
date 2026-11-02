# PAC representation readout

This module asks whether PAC relationships can be recovered from **final frozen
token embeddings**. It separates measurement accuracy, token readability and
clinical utility. The existing CFC operator and clinical trainer are unchanged.

## What is reused

| Component | Implementation / source |
|---|---|
| Scaling and affine readout | scikit-learn `StandardScaler` and `Ridge` |
| Scores and uncertainty | scikit-learn `r2_score`, `roc_auc_score`; SciPy `bootstrap` |
| Fixed dimensionality reduction | scikit-learn `GaussianRandomProjection` |
| CfFT and WavePatch | Existing repository model constructors and frontends, strict checkpoint loading |
| TFM | Original author factory and `tokenize(...)[0]` at the commit in [sources.json](../upstream/sources.json) |
| TFM preprocessing | Small adapted author STFT routine with MIT notice in `features.py` |
| Synthetic signals | Harmonic signal construction adapted from the existing capability generator; originals untouched |

The [EEG probing paper](https://arxiv.org/html/2605.11410v2), §3.2, provides
methodological precedent. Its public audit repository is not linked in v2;
Appendix D describes anonymous supplementary code and a planned public release.
We do not label this implementation as the authors' code. Analytic curve targets,
pooled unclipped R², validation-MSE selection and RMS detection are our adaptations.
References and versions are in [pac_probe_sources.json](../upstream/pac_probe_sources.json).

## Target and comparison

The generator supplies the normalized phase-conditioned mean amplitude curve,
with its constant baseline subtracted. Exact bin integrals retain modulation
depth. H1, H2, mixed and independent-null families use disjoint base signals
across train/validation/test. These are latent population targets, not estimates
from our operator. Null signals can show accidental dependence over a finite
window. Component power is matched in expectation, not the exact spectrum.

| Row | Features given to the same affine readout |
|---|---|
| CfFT | Final frontend output, before positional encoding / Transformer |
| TFM | Full tokenizer's quantized embeddings, not integer code IDs |
| WavePatch | Trained waveform-patch linear projection |
| Band power | Five fixed physical-band powers; a shortcut control, not a fourth tokenizer |

All neural rows receive the same complete 5-second, 200-Hz signal. First-pass
readout budget is 9 time positions × 64 features, TFM's native output size.
Fixed linear averaging/projection reduces the other outputs; extraction rejects
time upsampling and feature padding. Compression can discard information: the
conclusion is readability under this declared budget. Band power is the declared
low-dimensional control. No probe gradient enters any tokenizer.

Input gain is fixed across rows (`--input-scale 20` by default). CfFT/WavePatch
normalization must be declared from their training recipe; TFM retains the
author's per-channel q95 normalization. Checkpoint hashes, source, endpoint,
input scaling and budget accompany each cache. Predeclare checkpoints before
inspecting scores; the planned primary CfFT row uses TUEV seed 0 validation-best.
Different pretraining sources are recorded, not presented as matched training.

R² uses non-null examples and the standard test-mean total-SSE/total-SST
definition; negative scores remain negative, undefined scores are `null`.
Detection AUROC uses predicted curve RMS over both null and non-null records.
Scaling and ridge fitting use training rows only; validation MSE selects alpha.
Test predictions, per-family scores and paired group-bootstrap error differences
are saved. A high score establishes PAC readability, not all-CFC superiority or
actual downstream use.

## Commands

Run from the repository root, with the CPU development environment or the native
TFM runtime where appropriate. Outputs belong outside the source tree; commands
refuse to overwrite them. The commands below are examples, not completed results.

```bash
export PAC_PROBE_OUT=/path/outside/repository/pac-probe

# Bounded implementation controls; no trained tokenizer or clinical training.
python -m scripts.experiments.pac_token_probe validate --output "$PAC_PROBE_OUT/validation.json"

# Formal sample counts are configurable; use small counts for a smoke first.
python -m scripts.experiments.pac_token_probe generate --output "$PAC_PROBE_OUT/data.npz" --seed 0

python -m scripts.experiments.pac_token_probe extract \
  --dataset "$PAC_PROBE_OUT/data.npz" --row cfft \
  --config configs/model/tuev-cfc-reference.yaml --checkpoint /path/to/cfft/best.pt \
  --normalization div100 --training-channels 16 --device cuda \
  --output "$PAC_PROBE_OUT/cfft.npz"

python -m scripts.experiments.pac_token_probe extract \
  --dataset "$PAC_PROBE_OUT/data.npz" --row wavepatch \
  --config configs/model/tuev-wave-patch.yaml --checkpoint /path/to/wavepatch/best.pt \
  --normalization div100 --training-channels 16 --device cuda \
  --output "$PAC_PROBE_OUT/wavepatch.npz"

python -m scripts.experiments.pac_token_probe extract \
  --dataset "$PAC_PROBE_OUT/data.npz" --row tfm --vendor-root vendor/tfm \
  --checkpoint /path/to/tfm_tokenizer_last.pth --device cuda \
  --output "$PAC_PROBE_OUT/tfm.npz"

python -m scripts.experiments.pac_token_probe extract \
  --dataset "$PAC_PROBE_OUT/data.npz" --row bandpower --output "$PAC_PROBE_OUT/power.npz"

python -m scripts.experiments.pac_token_probe evaluate \
  --dataset "$PAC_PROBE_OUT/data.npz" \
  --cache cfft="$PAC_PROBE_OUT/cfft.npz" --cache tfm="$PAC_PROBE_OUT/tfm.npz" \
  --cache wavepatch="$PAC_PROBE_OUT/wavepatch.npz" --cache power="$PAC_PROBE_OUT/power.npz" \
  --output "$PAC_PROBE_OUT/readout"
```

Project YAML must match the saved architecture/configuration. Supply its actual
training channel count for strict restoration; only the frontend processes the
single-electrode synthetic signal. No signal is duplicated to fill the montage.
The missing-checkpoint path fails; it never constructs a random comparison row.
Use trusted checkpoints: legacy project checkpoints include Python metadata.

## Validation status and scope

`validate` checks exact recovery, invertible coordinates, known H2 loss, added
noise, constant/random controls, 20 label permutations and 18/24/36-bin oracle
recovery. Oracle information is supplied only inside these correctness controls;
it is never an input to the four-row comparison.

This command passing is **implementation readiness**, not completion of the
scientific benchmark validation. Before formal ranking, audit power/spectrum
shortcuts, frequency/noise generalization, the readout budget and actual operator
coverage. The fixed first-pass generator does not establish robustness to every
PAC process or physiological ground truth. Existing operator validation records
remain separate. The clinical oracle experiment (true PAC substituted into the
tokenizer) and AAC/PPC downstream comparisons are not launched by this module.

Tests: `python -m pytest -q tests/test_pac_probe_*.py`. To include the real pinned
TFM implementation, set `TFM_VENDOR_ROOT=/path/to/pinned/TFM-Tokenizer` before
running tests. Temporary random weights in adapter tests are fixtures only.
