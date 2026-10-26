# Model and tokenizer configurations

## CfFT

CfFT constructs EEG tokens from band content and measured cross-frequency
relationships. A learnable Sinc filterbank and analytic-signal transform provide
band waveform, envelope and phase features. Local phase–amplitude statistics
condition the rotation of envelope and waveform content before encoding.

The reference configurations in [configs/model/](../configs/model/) use 16 bands,
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
