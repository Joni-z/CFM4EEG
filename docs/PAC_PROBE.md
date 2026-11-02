# PAC representation readout

The current synthetic candidate is the **relation-context slot**, described
below. It preserves every original content token exactly and adds 5% more token
positions. The earlier global-3%-residual study is retained as exploratory
history; its global perturbation cap did not bound changes within each band.

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

## Full-token residual study (2026-11-03)

The later study uses **all 40,960 original token coordinates**, with no time
pooling or random token compression. It strictly restores the original TUEV
seed-0 checkpoint (`8da3e24c9eed68d67c8a4cc64ecdfc4cb1ab67cf6d9081cc9a285b79d101048f`).
No clinical model, operator implementation or training recipe is changed.

The metric remains the analytic 24-bin curve R²/RMSE, plus curve-RMS detection
AUROC. A fixed-band SciPy SOS/Hilbert/harmonic-regression reference and the CFC
operator both receive waveforms only. On the fresh confirmation test of 2,048
records, their curve R² values are 0.984 and 0.978. The operator's H1/H2 coverage
is 100%. The independent-null generator can produce finite-window apparent
coupling: in an 800-record audit, 36% of null frequency pairs were within one
Fourier resolution bin. Detection AUROC is therefore a secondary measure, not
a requirement of 100% accuracy. Increasing signal context from 5 to 20 seconds
improves the operator's AUROC from 0.932 to 0.991 in a separate 256-record check.
Gain and polarity checks preserve recovery; strong added noise reduces it.

Three frozen residual prototypes preserve the full original output and token
grid. They use the existing operator and seeded, label-independent projections:

| Candidate | Operation | Development R² at 10% perturbation limit |
|---|---|---:|
| Coefficient residual | Add projected H1/H2 real/imaginary coefficients | 0.928 |
| Content residual | Add relation-conditioned original content | 0.472 |
| Rotation tangent | Add a bounded relation-conditioned rotation correction | 0.459 |

Each residual is scaled using training inputs only and capped per record by
RMS(delta)/RMS(original). The coefficient residual reaches development R² 0.793
at a 1% limit and 0.898 at 3%. The 3% version was selected as a small-change
candidate before a fresh generator-seed confirmation (9,103). Confirmation
uses 2,048 training, 512 validation and 2,048 test records; it does not reuse the
development seed (1,103). The perturbation choice is exploratory, not a
preregistered clinical non-inferiority threshold.

| Confirmation row | Curve R² | Detection AUROC |
|---|---:|---:|
| Original final tokens | -0.001 | 0.489 |
| Original + coefficient residual, 3% limit | 0.902 | 0.918 |
| Same residual, relations shifted between samples within each split | -0.001 | 0.488 |

Readouts use train-only scaling, a centered linear or quadratic kernel and
validation-only choice of degree and regularization. All rows retain identical
token dimensionality and fitting budgets. H1/H2/mixed confirmation R² values
for the 3% candidate are 0.903/0.904/0.900. Directly exposing measured coefficients
can make PAC decoding easier by construction. These results establish recovery
on this fixture, not superiority to TFM, clinical gains or retention of the
original downstream accuracy. The external TFM/WavePatch comparison remains
uncompleted. Original tokens may also admit recovery by other readout families.

The experiment code is in `scripts/experiments/pac_metric_calibration.py`,
`pac_metric_stress.py`, `pac_token_candidates.py` and `pac_candidate_search.py`.
`PACResidualCandidates.forward` returns the actual modified token tensor; it
does not expose a separate operator output to the readout. This is a frozen
research prototype, not a new production clinical configuration.

```bash
python -m scripts.experiments.pac_candidate_search \
  --reference /path/to/original-checkpoint-directory \
  --output /path/outside/repo/confirmation \
  --seed 9103 --train 2048 --val 512 --test 2048 \
  --variants coefficient --gains .03 --include-shuffle
```

The reference directory must contain `original-training.yaml` and the original
`original-best.pt`. All outputs refuse to overwrite existing records. Raw
arrays, predictions, selected readout settings and run protocols are retained
outside the repository under `maintenance/autoresearch-pac-token-20261103`.

## Broader audit and selected context candidate

The global 3% residual can change a low-energy band by much more than 3%:
the maximum measured bandwise RMS change was 111.5% in the first confirmation.
It is not the selected preservation design. Band-bounded alternatives were
tested: low-variance placement of coefficients reaches native-token R² 0.863
at a 10% per-band cap, but only 0.177 after the common 576-feature reduction.

The selected alternative leaves the original `[B,C,16,20,128]` content grid
unchanged and appends one relation-context slot per band, yielding
`[B,C,16,21,128]`. Its 128-dimensional relation embeddings are the same fixed,
label-independent lift of measured H1/H2 coefficients mapped to token bands.
Their amplitude scale is calibrated using training inputs only. The last slot
represents the measurement context, not another 0.25 s of EEG. No content
weights or encoder layers are replaced. The tokenizer parameter count remains
9,760; added analysis kernels and projection matrices are buffers. A synthetic
16-channel forward check accepts the new grid through the unchanged original
encoder and head. This is compatibility evidence, not task-performance evidence.

A third generator seed (16,103) confirms the selected design. With all neural
rows reduced to 9 positions x 64 features and the same validation-tuned readout:

| Frozen representation | Curve R² | Detection AUROC |
|---|---:|---:|
| Original CfFT | -0.001 | 0.485 |
| CfFT + relation-context slot | 0.903 | 0.937 |
| Released multi-dataset TFM tokenizer | 0.019 | 0.496 |
| Trained waveform-patch tokenizer | -0.002 | 0.518 |
| Context-slot control with wrong-sample PAC | 0.001 | 0.484 |

H1/H2/mixed R² values are 0.917/0.893/0.900. Native, uncompressed context-token
R² is 0.868. The 576-feature budget and native-coordinate results are different
diagnostics, not interchangeable scores. The released TFM checkpoint strictly
loads into the pinned author factory. Its SHA256 is
`f7929eed0f8275d08a7e18743fdd188c7ba5b4722151333b4d386ca05b84d6d9`.
The TUEV single-dataset release is incompatible with that factory and was not
scored. WavePatch is restored from the checkpoint archive, SHA256
`6bb370405986633aab7d9d2ff750ff95ebc15babaf50888e73dff4cb84049aa8`.
Different tokenizer training histories prevent a causal claim that explicit
modeling universally outperforms implicit modeling.

Four additional predeclared physical-pair fixtures test 3/24, 3/40, 4/32 and
5/40 Hz, retaining the analytic H1/H2/mixed/null targets. The operator's harmonic
coverage is 100% in every fixture. Context-token R² under the common budget is
0.831/0.820/0.852/0.727; corresponding AUROC is 0.767/0.724/0.852/0.769. Narrow
frequency jitter makes population-null detection harder in these records.
All selected pairs remain in the report. The operator itself retains R²
0.967--0.984, separating measurement accuracy from finite-sample token readout.

The first cycle-2 common-budget run mistakenly reshaped band-major caches as
time-major. Those outputs (`common-comparison` and the initial multi-pair
residual readouts) are superseded. `cfft_sequence` now transposes the axes
explicitly; a test checks each physical time slice. Native-token cycle-1
scores were unaffected. Current context results use the corrected conversion,
and two independent constructions give identical predictions.

`pac_residual_placement.py` evaluates the band-bounded and context alternatives;
`pac_tokenizer_comparison.py` restores the four real tokenizer endpoints.
`PACResidualCandidates.context_tokens(x, scale=...)` returns the actual final
token tensor. Its serialization and exact content-prefix preservation are
tested. No downstream training or clinical evaluation was run in this study.
