# Model and tokenizer configurations

## CfFT

CfFT constructs EEG tokens from band content and measured cross-frequency
relationships. A learnable Sinc filterbank and analytic-signal transform provide
band waveform, envelope and phase features. Local phase–amplitude statistics
condition the rotation of envelope and waveform content before encoding.

The local-moment reference, [tuev-scratch.yaml](../configs/model/tuev-scratch.yaml), uses 16 bands,
128-dimensional tokens and six Transformer layers. A content patch contains
50 samples at 200 Hz, or 0.25 seconds. `pac_patch_len` independently controls
the relation-estimation window; the reference sets it to 50 samples as well.
These local moments are computational features, not validated physiological PAC
measurements.

The encoder alternates temporal attention and joint electrode–band row
attention: two attention axes. `arch: triaxial` and the filenames `triaxial.py`
are retained implementation names; the selected `space_over_bands: true` and
`freq_mixer: none` determine the two-axis configuration. They do not imply three
separate attention stages.

Implementation locations:

| Component | Source |
|---|---|
| Model assembly and frontend selection | [build.py](../paclock_bench/models/paclock/build.py) |
| Band features and relation-conditioned tokens | [frontend/triaxial.py](../paclock_bench/models/paclock/frontend/triaxial.py) |
| Encoder and positional embeddings | [triaxial.py](../paclock_bench/models/paclock/triaxial.py) |
| Checkpoint naming and implementation history | [PROVENANCE.md](../paclock_bench/models/paclock/PROVENANCE.md) |

## Waveform-patch control

[wave_patch.py](../paclock_bench/models/wave_patch.py) projects raw waveform
patches linearly and uses the same two-axis encoder. The TUEV configurations
provide CfFT and WavePatch pairs both with and without augmentation:

| Tokenizer | With augmentation | Without augmentation |
|---|---|---|
| CfFT | [tuev-scratch.yaml](../configs/model/tuev-scratch.yaml) | [tuev-scratch-noaug.yaml](../configs/model/tuev-scratch-noaug.yaml) |
| WavePatch | [tuev-wave-patch.yaml](../configs/model/tuev-wave-patch.yaml) | [tuev-wave-patch-noaug.yaml](../configs/model/tuev-wave-patch-noaug.yaml) |

The model factory constructs these project models. Baseline models use the
separate [original-author entrypoints](UPSTREAM.md).

## Explicit phase-amplitude measurement frontend

`tokenizer_mode: measured_pac` selects a separate optional measurement path in
[measurement.py](../paclock_bench/models/paclock/frontend/measurement.py) and its
token adapter in [measured.py](../paclock_bench/models/paclock/frontend/measured.py).
It retains the content projections and two-axis encoder. Reference configs and
their 0.25-second local-moment implementation remain available separately.

The component uses fixed physical analysis anchors initialized from the content
bank: narrow complex phase filters and wider analytic amplitude filters. For
each eligible source/target pair, it fits envelope against sine/cosine of source
phase, removing an intercept and linear time trend. Contexts request at least
three nominal source cycles and one second. A regularized 2-by-2 solve returns
the first-harmonic coefficient, relative modulation depth, baseline, drift,
residual error and phase-condition diagnostics. Validity also checks real
filter support, observed phase rate, cycle coverage and excluded frequencies.
Padding never counts as extra measured data. Invalid estimates have a separate
reason code; they are not interpreted as biological zero coupling.
These statistical contexts are independent of the 0.25-second content tokens.
`pac_patch_len` controls the legacy local-moment path, not the statistical
context of `measured_pac`.
Use `measurement_mode: measured` for guidance and `measurement_mode: identity`
for its unrotated carrier control. The legacy `pac_token_mode` does not select
controls for this frontend.

Analysis anchors are checkpointed and remain fixed; the content Sinc filters
can learn. Their respective envelopes are different quantities. The estimator
does not use labels, batch statistics or additional learned measurement
parameters. It measures first-harmonic phase-amplitude dependence, not all CFC
types or causal interactions. Symmetric two-peak modulation, common transients,
weak-signal uncertainty and missing acquisition context remain limitations.

The adapter transports source phase features with the conjugate measured
coefficient, retaining absolute modulation depth. It bounds their complex sum
by one plus the sum of absolute contributions and adds an identity reference
before unit normalization. This yields identity for missing/zero guidance and
preserves the envelope/waveform pair's squared norm. It does not transport the
complete source waveform. Physical measurability does not imply class relevance.

The TUEV templates explicitly guard its 60 Hz preprocessing notch with a
conservative 56–64 Hz exclusion. New datasets must supply their own sample rate,
passband and excluded intervals. A longer statistical context cannot restore
modulation frequencies already removed by preprocessing.

```bash
PYTHONPATH=. python scripts/experiments/pac_measurement_benchmark.py \
  --split dev --output /path/to/measurement-development
PYTHONPATH=. python scripts/experiments/pac_measurement_benchmark.py \
  --split confirmation --output /path/to/measurement-confirmation
PYTHONPATH=. python scripts/experiments/pac_measurement_stress.py \
  --output /path/to/measurement-stress
PYTHONPATH=. python scripts/experiments/pac_tokenizer_learning.py \
  --output /path/to/spectrum-matched-learning
python -m pytest -q tests/test_pac_measurement.py
```

The recovery benchmark saves every estimate and reports accuracy with validity
coverage. Its short-window and same-context moment baselines share the analysis
filters. The learning check uses the actual tokenizer/encoder, paired initial
weights and positive/negative signals with equal Fourier magnitudes. These
synthetic checks do not establish a clinical advantage. The seed-0 clinical
templates are [measured](../configs/model/tuev-explicit-pac-measured.yaml) and
[identity](../configs/model/tuev-explicit-pac-identity.yaml); selection uses
validation, and the final test is evaluated after training.

For a more selective learning check, `--task cross_pair` constructs quadruplets
`(L,H), (shift L,shift H), (L,shift H), (shift L,H)`. Each individual low/high
waveform appears once in either class, and all four signals have equal Fourier
magnitudes. Only the timing between nonstationary components changes. The
mispaired signals are not asserted to have zero or no delayed CFC. This removes
the within-band phase differences introduced by global phase randomization.
Both arms use the same seed-0 initialization, data, update budget and
validation-best checkpoint selection. It compares the complete guidance branch;
it does not isolate rotation from the source-phase projection, or prove that a
learned model cannot discover the relationship. Grouped uncertainty must resample
whole quadruplets, not their four dependent signals independently.

```bash
PYTHONPATH=. python scripts/experiments/pac_tokenizer_learning.py \
  --task cross_pair --epochs 100 --seed-offset 100000 \
  --output /path/to/selective-pairing-confirmation
python -m pytest -q tests/test_pac_learning_design.py
```

### Validation results

The [validation record](../review/PAC_COMPONENT_VALIDATION_20261028.json) binds
these results to their source versions, configurations, selected checkpoints
and evidence archives. Clinical training used `b8ddd17`; complete-test replay
used `e2cf787`. The component remains an optional frontend.

On synthetic confirmation signals, first-harmonic modulation-depth MAE was
0.00404 and preferred-phase MAE was 0.673 degrees at valid central anchors with
nonzero modulation and no added noise or 20 dB colored noise. Coverage was 100%
across primary central anchors, including zero-modulation controls. Coverage
across all saved positions, including boundaries, noise levels and negative
controls, was 67.9%.

The selective-pairing task above gave 95.31% accuracy with measured guidance
versus 69.79% with identity, using the same seed-0 initialization, 800 updates
per arm and validation-best selection. Its 192 test signals form 48 independent
quadruplets. This evaluates the full guidance branch, including phase features.

The TUEV pair used the accepted shared16 cohort, seed 0, identical augmentation
and 20 complete epochs per arm. Validation kappa selected the checkpoints.

| Frontend setting | Test kappa | Balanced accuracy | Weighted F1 |
|---|---:|---:|---:|
| Identity | 0.6834 | 0.6211 | 0.8350 |
| Measured guidance | 0.6942 | 0.6185 | 0.8423 |

Measured guidance increased kappa by 0.0108 and slightly reduced balanced
accuracy. Each saved checkpoint reproduced all 29,421 test logits exactly with
the replay source. This is one paired seed and supports feasibility; it does not
establish a stable advantage or a ranking against other TUEV protocols.
See the [evidence locations and audit scope](AUDIT.md#component-validation-record).

Estimator references: Penny et al., *Testing for nested oscillation* (2008),
doi:10.1016/j.jneumeth.2008.06.035; van Wijk et al., *Parametric estimation of
cross-frequency coupling* (2015), doi:10.1016/j.jneumeth.2015.01.032;
Dvorak and Fenton, *Toward a proper estimation of phase–amplitude coupling in
neural oscillations* (2014), doi:10.1016/j.jneumeth.2014.01.002.

## Optional copula frontend

The copula pilot separates per-band content from explicitly estimated
dependence, following Sklar's distinction between marginal distributions and a
copula. It does not estimate the full joint EEG distribution or assert that an
encoder cannot learn dependencies from raw signals.

- Retain the 16-band Sinc bank, 0.25-second content patches and two-axis encoder.
  Project log-envelope and waveform patches to 64 coordinates each and
  concatenate them.
- In one-second, non-overlapping relation windows, transform each envelope and
  real/imaginary unit-phase component through its empirical marginal CDF to
  normal scores. Average ranks handle ties. Estimation is per electrode/window.
- Estimate signed envelope correlation and phase/envelope squared multiple
  correlation after shrinking the correlation matrix 5% toward identity. For
  phase/envelope estimation, lower-frequency phase is paired with higher-frequency
  envelope, using the filters' current center frequencies. All band pairs remain
  eligible for content fusion.
- Supply these summaries as per-head biases to one four-head band-attention
  layer whose values carry content. Exclude self-edges from this branch and
  preserve each band's content through a residual with initial learned scale 0.1.
- Stop gradients through relation estimates; filter parameters learn through
  content. No batch pooling, target labels or held-out statistics enter the
  estimator.

These summaries use a Gaussian-copula approximation, not a complete dependence
model or a physiological PAC estimator. This frontend has no preferred-phase
rotation and does not use EOT. The Gaussian approximation, short-window
uncertainty and filter overlap remain limitations; Sklar's theorem does not
establish a predictive advantage.

[frontend/copula.py](../paclock_bench/models/paclock/frontend/copula.py) is selected
only by `tokenizer_mode: copula`. The
[TUEV pilot configuration](../configs/model/tuev-copula-lenovo.yaml) uses seed 0,
the accepted shared16 cohort and validation-best selection. Physical batch 16
with two accumulation steps gives effective batch 32; stochastic augmentation
and dropout draws differ from one physical batch of 32. Its Windows runtime uses
memory-mapped arrays and zero loader workers to avoid spawning dataset copies.
Checkpoints and provenance use the project trainer and external output paths.

Reference: Ince et al., *A statistical framework for neuroimaging data analysis
based on mutual information estimated via a Gaussian copula*, Human Brain Mapping
38:1541–1573 (2017), doi:10.1002/hbm.23471.
