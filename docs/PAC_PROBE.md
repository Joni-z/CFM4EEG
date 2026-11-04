# PAC representation readout

The current attribution test restores the original **16-band quadrature +
three-axis CfFT**. Four clinical checkpoints have completed frozen synthetic
probes. No model weights or token dimensions were changed. Earlier candidate
screens below remain separate from this attribution test.

This module asks whether PAC relationships can be recovered from **final frozen
token embeddings**. It separates measurement accuracy, token readability and
clinical utility. The existing CFC operator and clinical trainer are unchanged.

## Original three-axis rotation (2026-11-05)

TUEV and CHB-MIT checkpoints, seeds 0 and 2, were fixed before probing. Each
receives the same two synthetic fixtures: analytic PAC curves and four-record
selective-pairing groups. Each fixture has 1,024/256/512 train/validation/test
records. All 40,960 native token coordinates are retained. Each frozen frontend
is evaluated with measured, identity and wrong-record relations; these are
interventions on its existing weights, not separately trained clinical models.

| Checkpoint | Pairing: measured, linear | Identity, linear | Measured, quadratic | Identity, quadratic |
|---|---:|---:|---:|---:|
| TUEV s0 | .646 | .508 | .855 | .898 |
| TUEV s2 | .631 | .523 | .738 | .830 |
| CHB-MIT s0 | .682 | .525 | .951 | .965 |
| CHB-MIT s2 | .645 | .514 | .896 | .916 |

The fixed primary linear readout improves by .107–.156 with measured relations.
Identity is better with the quadratic readout in all four checkpoints; validation
chooses quadratic for both measured and identity. Thus this is evidence of easier
**linear** access to this pairing label, not greater information retention or
readout-independent superiority. Wrong-record linear accuracy is .445–.473.

Full PAC-curve R² from measured tokens remains about −.003 for linear and
quadratic readouts. A separate known-frequency-pair diagnostic of the native
local moment recovers the first-harmonic component with R² .796–.933; full-curve
R² is .402–.471. The pair is selected using generator frequencies only in this
diagnostic, never in token inputs. Neither result establishes general PAC
estimation or clinical benefit. Linear waveform-projection recovery is also
higher for identity (.854–.929) than measured (.099–.171); this is limited-readout
accessibility, not proof that the rotation destroys information.

Physical spectra are matched within each pairing quadruplet. TUEV's div100
preserves that matching. CHB-MIT's sample-specific q95 normalization does not:
a post-normalization two-power quadratic control reaches .756 accuracy
(linear .557). CHB-MIT pairing therefore has a normalization-related shortcut
and must not be described as spectrum-matched at the model input.

The four checkpoints share one synthetic data realization. Bootstrap intervals
resample test quadruplets only for measured versus identity. Wrong-record donors
connect groups, so the original ordinary group intervals for that contrast are
superseded; report only point differences. Source hashes, raw predictions and
the independent audit are retained in maintenance run
`legacy-triaxial-validation-20261105`. Independent token replay matches exactly;
38 focused tests pass. Clinical seed-2 identity controls are configured separately
with unchanged legacy recipes and data; their queued status is not a result.

Reproduction (output must be a new directory):

```bash
python -m scripts.experiments.legacy_rotation_probe \
  --config /path/to/original/config.yaml --checkpoint /path/to/original/best.pt \
  --normalization div100 --counts 1024 256 512 --seed 20261105 \
  --output /path/outside/repository/legacy-rotation-probe
```

Use `q95` for the bound CHB-MIT checkpoints. The runner keeps both physical and
post-normalization power controls and selects readout regularization on validation.

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

## Earlier broader audit and context-slot control

The global 3% residual can change a low-energy band by much more than 3%:
the maximum measured bandwise RMS change was 111.5% in the first confirmation.
It is not the selected preservation design. Band-bounded alternatives were
tested: low-variance placement of coefficients reaches native-token R² 0.863
at a 10% per-band cap, but only 0.177 after the common 576-feature reduction.

The context-slot alternative leaves the original `[B,C,16,20,128]` content grid
unchanged and appends one relation-context slot per band, yielding
`[B,C,16,21,128]`. Its 128-dimensional relation embeddings are the same fixed,
label-independent lift of measured H1/H2 coefficients mapped to token bands.
Their amplitude scale is calibrated using training inputs only. The last slot
represents the measurement context, not another 0.25 s of EEG. No content
weights or encoder layers are replaced. The tokenizer parameter count remains
9,760; added analysis kernels and projection matrices are buffers. A synthetic
16-channel forward check accepts the new grid through the unchanged original
encoder and head. This is compatibility evidence, not task-performance evidence.

A third generator seed (16,103) confirms that context-slot prototype. With all neural
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


## Original-path diagnosis and content-only fusion (2026-11-03)

The original PAC-inspired tokenizer does not promise recovery of every PAC
parameter. Its first angular moment misses a pure second-harmonic dependence;
source-weight normalization and unit-modulus rotation discard some strength
information within that path; multiple source edges become one rotation.
These are conditional path properties, not proofs of information loss from
the entire waveform/token representation. The learned narrow filters can
also attenuate modulation sidebands within an individual band.

A frozen-checkpoint diagnostic uses 1,024 training, 256 validation and 512
test records. Raw local complex moments recover the analytic curve at R²
0.412, versus -0.001 for final content tokens. Frozen identity rotation is
0.001; rotation alone therefore does not explain the failure. The endpoints
have different dimensions and are diagnostic, not a fair model ranking.
Cubic/RBF readouts and all-band token-coherence summaries do not resolve
final-token recovery. This remains a tested readout limitation, not an
impossibility result for all decoders.

Four small content-only transforms were screened: time-local envelope/waveform
gain, signed-carrier binding, already-aligned-carrier binding, and nonnegative
content-magnitude binding. All preserve the old projections/rotation and
16 × 20 × 128 grid. None adds a separate PAC token. Uncalibrated variants fail.
Train-input-only per-band RMS calibration improves the last variant's readout.
It adds a measured-relation-weighted correction proportional to local,
bias-free content magnitude. No signal content means no correction; the
per-band RMS cap controls the change. It exposes measured relations inside
content coordinates and does not validate the old rotation mechanism alone.

The 30% per-band cap was selected using development validation MSE before
confirmation on fresh generator seed 271,103 (1,024/256/1,024). Readout degree
and regularization still use validation only. All rows keep exactly 20 positions.

| Frozen representation | Native curve R² | Common 576-feature R² |
|---|---:|---:|
| Original content tokens | -0.002 | -0.002 |
| Content-magnitude binding, 10% per-band cap | 0.391 | 0.147 |
| Content-magnitude binding, 30% per-band cap | 0.781 | 0.701 |
| Wrong-sample guidance, 30% per-band cap | -0.002 | -0.002 |

The 30% variant's mean whole-record RMS change is 5.92% (maximum 9.67%),
but an individual band can change by 30%; a small whole-record value must
not hide that fact. Native H1/H2/mixed R² values are 0.783/0.778/0.782.
The operator itself reaches curve R² 0.978, with both harmonics supported
in every test record at the fixed 4/32-Hz query. Native detection AUROC is
0.875; common-budget detection AUROC is 0.840. A 2,000-resample paired
bootstrap gives a positive-record MSE reduction of 0.05656, 95% percentile
interval [0.05499, 0.05807], against the original. These are independent
synthetic records, not clinical subjects.

The stronger wrong-guidance control is reconstructed from cached tokens by
the exact affine gain relation and independently checked against actual
inference. Prediction-order checks and separate R²/AUROC recomputation agree
within 1e-12. Strict prototype serialization reload reproduces tokens exactly.
All failed candidates and original results remain outside the source tree
under `maintenance/pac-content-fusion-20261103`. Executed source snapshots
are retained beside the result arrays. The prototype has no new trainable
parameters; its measurement/filter/projection state and train-derived
calibration are saved as buffers. **Clinical accuracy preservation is untested.**

Reproduce with `scripts/experiments/pac_content_diagnosis.py`,
`pac_readout_diagnosis.py`, `pac_structured_readout_diagnosis.py` and
`pac_content_fusion_search.py`. The actual content-only prototype is
`PACContentFusion` in `pac_content_fusion.py`, used through
`fuse(prepared, mode="radial_binding", gain=0.3)` or its `forward` method.
The original neural content parameters remain trainable when the caller
wants to fine-tune; the synthetic study performs inference only. No clinical
trainer, model factory, dataset, checkpoint or paper is modified.

## Single measured-PAC path (2026-11-04)

This screen removes the duplicated old-PAC plus new-PAC path. All three research
variants call `CFCOperator` once, reuse the trained Sinc/content projections and
return the original 16-band × 20-position × 128-feature tensor. They add neither
tokens nor learned parameters. The measured H1/H2 relations act on two groups
of existing complex coordinates. Physical target frequencies map to learned
bands through their actual half-power responses; unsupported bands use
unrotated content. No generator labels or query pair enter token construction.

| Candidate | Sole content transform | Development native R² | Common-budget R² |
|---|---|---:|---:|
| Original CfFT | Original local-PAC rotation | -0.001 | -0.001 |
| A: aligned | Measured relation aligns projected source harmonics; unit rotation | 0.002 | 0.172 |
| B: strength | A with bounded coupling-strength gain | 0.002 | 0.188 |
| C: direct | Bounded complex gain from measured relations, replacing the old rotation | 0.784 | 0.768 |

C uses `content * (1 + 0.5 * relation)`, with relation magnitude capped at one.
The gain lies in [0.5, 1.5] and angle in [-30°, 30°]. No relation-only additive
vector is packed into the token. C does not retain the original source-reference
invariance. A/B retain it for consistent phase/coefficient reference changes.
Their weaker recovery is a readout result, not proof that every decoder fails.

The development fixture uses seed 1103 and 1,024/256/512 train/validation/test
records. Common-budget validation MSE selects C before confirmation. Gains,
readout degree options and regularization grid are fixed; probe settings use
validation only. The common budget remains 9 × 64 features. Native and common
results differ because compression changes the finite-sample readout problem.

| Fresh confirmation, seed 41104; 1,024 test records | Native R² | Common R² | Common detection AUROC |
|---|---:|---:|---:|
| Original CfFT | -0.002 | -0.002 | 0.478 |
| Unrotated original content | -0.002 | -0.002 | 0.488 |
| C: direct | 0.773 | 0.750 | 0.906 |
| C with wrong-record relations | -0.002 | -0.002 | 0.488 |

Wrong relations use a derangement within each split, while current-record
content is held fixed. The operator itself reaches R² 0.978. Removing the
learned content projection bias in a post-screen diagnostic retains common
R² 0.730; a bias-only carrier reaches 0.966 but carries essentially no tested
waveform/power information. Thus a high PAC score alone cannot establish a
useful content representation. The ordinary C transform also decodes fixed
waveform projections at native R² 0.880 versus 0.916 for unrotated content,
and generating band powers at 0.768 versus 0.922. These are limited readout
diagnostics, not signal reconstruction or clinical outcomes.

A separate 3/24-Hz fixture (seed 51104, 1,024 test records) gives C native/common
R² 0.745/0.743, with no retuning. These fixtures cover two physical pair ranges,
not general multi-source EEG. In the reference checkpoint only four learned
target bands meet the physical mapping threshold.

At the end of this synthetic screen, downstream preservation was unknown.
C changes unrotated content by
2.21% mean whole-record RMS, but differs from the original rotated tokens by
141.56%. Keeping the tensor shape and parameter count does not make this a
small perturbation of the trained model. The subsequently authorized clinical
study is recorded below.

Implementation: `scripts/experiments/pac_singlepath.py`; runner:
`pac_singlepath_search.py`. Forty-three relevant tests pass, including exact
equivalence to the earlier probe, single measurement, no old-PAC call, finite
gradients, no-content behavior and serialization. An independent replay
checks 72 reported target/budget metrics and prediction IDs across all three
datasets. Source snapshots,
protocol, predictions and prototype weights are retained outside the repository
under `maintenance/pac-singlepath-20261104`.

## Candidate C downstream and D/E/F screen (2026-11-04)

The user approved C for TUEV and CHB-MIT seed0 from-scratch training on 145.
`tokenizer_mode=cfc_direct` uses the exact screened transform through
`DirectPACFrontend`; both configurations retain the original encoder, loss,
augmentation, optimizer, selection and stopping recipe. Accepted arrays are
preloaded. Clinical execution is frozen at commit `134804c`, under
`/data2/zz5070/CFM4EEG-rebuild-20261021/cfc-direct-20261104` (GPU0 TUEV,
GPU1 CHB-MIT). Smoke gates check real batches, gradients, validation, strict
checkpoint reload and source/data identity before training. Best checkpoints
are saved. No synthetic improvement is treated as a downstream result.

TUEV seed0 completed all 20 epochs: test kappa **0.660108**, balanced accuracy
**0.642785**, weighted F1 **0.826950**. The validation-selected checkpoint is
epoch7 (zero-based), validation kappa 0.603914. All three test metrics were
independently recomputed from 29,421 saved predictions. The original rotating
CfFT seed0 had validation kappa 0.586466 and test kappa 0.693214: improved
validation/PAC readout did not preserve its test kappa. CHB-MIT remains running.

The CHB-MIT original-tokenizer control uses
`configs/model/chbmit-original-control.yaml`, seed0. Its sole computational
configuration difference from C is `tokenizer_mode=pac_interaction`; encoder,
loss, augmentation, schedule, stopping, accepted data and preload are identical.
The release retains C's Python source at `134804c` plus this reviewed config,
under `chbmit-original-control-20261104` on 145. This is the original **two-axis**
model; its historical old-data seed0 test AUC-PR was 0.604655. The older
three-axis range model's 0.696913 belongs to a different encoder. Data and
recipe audit artifacts are under `maintenance/chbmit-control-audit-20261104`.

D/E/F use the same single measured relation, harmonic groups and content grid:

| Candidate | Content transform | Development native/common PAC R² |
|---|---|---:|
| D: strength only | Positive content gain from coupling depth, no angle | 0.001 / -0.001 |
| E: angle only | Unit-normalized C transform, no gain | 0.410 / 0.343 |
| F: envelope only | C transform acts on envelope; waveform contribution passes unchanged | 0.784 / 0.768 |

D lacks preferred-phase information by construction; its curve score does
not establish that strength is useless for classification. F was selected by
common-budget validation MSE. On confirmation seed41104, its native/common
R² is 0.773/0.750; wrong-record guidance gives -0.002/-0.002. C on the same
records is 0.773/0.750. F preserves the separate waveform contribution but
still replaces the original rotation; it has not demonstrated a clinical
advantage and is not submitted downstream. Only readouts are fitted in the synthetic screen; the tokenizer stays frozen. Outputs and source snapshots are under
`maintenance/pac-c-downstream-20261104`; original checkpoint and paper are
unchanged. Forty-six focused tests pass, including exact C prototype/adapter
parity, encoder initialization, gradients, checkpoint reload and F's waveform
bypass. The D/E/F study does not alter the frozen clinical execution source.

## What the downstream discrepancy identifies (2026-11-04)

C is not an isolated estimator replacement. Its global record-level relation
is repeated at all patch positions; the direct transform does not consume the
learned source-phase projections. A CPU backward check confirms no gradient
for that projection in C, versus a nonzero gradient in the original. The
original applies local, feature-dependent unit rotations; C uses bounded
complex gains shared within harmonic groups. Initial physical mapping covers
four of16 content bands. These changes alter time support, active learned
paths, frequency coverage and content transformation together.

Historical same-data controls separate several contributions:

| Comparison | Seed | Full / control primary score | Interpretation |
|---|---:|---:|---|
| TUEV two-axis: measured / identity | 0 | .684601 / .616390 | Rotation helps this run |
| TUEV two-axis: measured / identity | 2 | .713350 / .699077 | Smaller positive rotation effect |
| CHB-MIT two-axis: measured / identity | 0 | .604655 / .664544 | Rotation is not universally beneficial |
| CHB-MIT two-axis: full / envelope-only | 2 | .623756 / .561782 | Waveform content contributes |

These CHB-MIT rows use the historical evaluation composition; see the
[fixed-prediction rescore](AUDIT.md#chb-mit-historical-score-decomposition-2026-11-04).
Earlier raw-band comparisons also show strong performance without PAC, but
vary band count, token count and attention cost together. They support the
frequency-resolved content/encoder combination, not a pure filter effect.

Current C TUEV is a tradeoff: kappa .660108 versus original .693214, balanced
accuracy .642785 versus .612351. The PAC screen used a frozen old checkpoint;
clinical C trained from scratch. Accurate coefficient recovery, final-token
recoverability and clinical discrimination are separate measured outcomes.
Neither a high PAC readout nor a single lower clinical kappa identifies the
benefit of estimator accuracy alone. Detailed matched configurations and
graph checks are in the existing maintenance audit's `mechanism-followup.md`.

### Candidate C, legacy CHB-MIT snapshot evaluation (2026-11-04)

A frozen copy of the current validation-best checkpoint at epoch1 step6600
(natural-grid validation PR .284963) was evaluated without retraining:

| Test cohort | AUC-PR | AUROC | Balanced accuracy |
|---|---:|---:|---:|
| Legacy enriched windows, 17,386 rows /276 positives | .534567 | .882914 | .628506 |
| Natural-grid subset, 17,224 rows /114 positives | .317193 | .824056 | .613918 |

This is an interim checkpoint while clinical training continues. Selection
used the current natural-grid validation set; it was not reselected on legacy
validation or test scores. The exact legacy test arrays were restored and
hash-verified. Strict checkpoint loading, all52 training source hashes, full
prediction/label order and independent metric recomputation pass. FP32 inference
at batch128 took679seconds. Checkpoint SHA starts `d8582906abcdab3f`; full weights,
predictions and receipts are under `maintenance/chbmit-c-eval-legacy-20261104`.
Both ongoing CHB-MIT training jobs and their selected checkpoints are unchanged.
