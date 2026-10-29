# Model and tokenizer

CfFT separates per-band content from measured cross-frequency context. The
learnable Sinc bank supplies waveform and envelope features; the independent
[CFC operator](CFC_OPERATOR.md) supplies physically indexed PAC measurements.
The tokenizer determines how these two inputs enter the encoder.

## Tokenizer variants

| Variant | Configuration keys | TUEV configuration | CHB-MIT configuration |
|---|---|---|---|
| CFC residual | `tokenizer_mode: cfc`, `cfc_fusion: residual` | [tuev-cfc-residual.yaml](../configs/model/tuev-cfc-residual.yaml) | [chbmit-cfc-residual.yaml](../configs/model/chbmit-cfc-residual.yaml) |
| Earlier CFC rotation | `tokenizer_mode: cfc`, default `cfc_fusion: rotation` | [tuev-cfc-h2.yaml](../configs/model/tuev-cfc-h2.yaml) | [chbmit-cfc-h2.yaml](../configs/model/chbmit-cfc-h2.yaml) |
| Local-moment reference | `tokenizer_mode: pac_interaction` | [tuev-scratch.yaml](../configs/model/tuev-scratch.yaml) | [chbmit-scratch.yaml](../configs/model/chbmit-scratch.yaml) |

### CFC residual

1. Split EEG into 16 learnable content bands. Project each 0.25-second patch's
   log-envelope and waveform into 64 coordinates each, forming a 128-wide token.
2. Measure PAC on the operator's separate six-source/five-target physical grid.
   Use raw baseline-normalized coefficients (`cfc_stabilized: false`), with
   unsupported or unobservable coefficients masked to zero.
3. For each harmonic, multiply the conjugate coefficient by the matching
   source-phase-power patch projection and sum over sources. Each harmonic has
   its own bias-free value projection.
4. A learned 16-by-10 softmax readout supplies the five-target/two-harmonic
   responses to every content band. Add this residual to the content token
   using a per-band learned gain initialized to one.

The readout is learned conditioning: it does not claim that all 16 content
bands were measured as physical PAC targets. It replaces the earlier sparse
filter-response mapping and mandatory rotation. It does not normalize weak
coefficients to unit strength; zero or unavailable relations give zero residual.
Independent source-phase reference changes cancel between each coefficient and
its matching phase power. The second harmonic is not treated as a unique angle.

Waveform projection retains its zero initialization. Added readout initialization
preserves the common content/encoder/head initialization stream. The TUEV model
has 1,257,430 parameters, including 32,944 added readout/gain parameters.
The measurement operator has no trainable parameters.

Content patches remain 50 samples at 200 Hz. The CFC operator fits a relationship
using the full available record, then evaluates it against local phase features;
its repeated coefficients are not independent local PAC estimates.
`pac_patch_len` controls the older local-moment path, not this measurement context.
`cfc_guidance: identity` disables the residual branch while preserving content.
`cfc_stabilized: true` explicitly opts into uncertainty-based shrinkage.

### Earlier configurations

The local-moment reference measures relationships within the learned content
filterbank and uses them to rotate envelope/waveform features. Its default
0.25-second moments are computational features, not cycle-averaged physiological
PAC measurements. The earlier `cfc` rotation instead uses the physical CFC grid,
filter-response mapping and an identity-anchored bounded reference. These
configurations retain their original source and checkpoint identities.

Two additional experimental frontends remain available:

- `tokenizer_mode: measured_pac` uses the earlier first-harmonic estimator in
  [measurement.py](../paclock_bench/models/paclock/frontend/measurement.py).
  [Measured](../configs/model/tuev-explicit-pac-measured.yaml) and
  [identity](../configs/model/tuev-explicit-pac-identity.yaml) templates use
  `measurement_mode`, not `cfc_guidance`. Its synthetic and clinical checks are
  recorded in [PAC_COMPONENT_VALIDATION_20261028.json](../review/PAC_COMPONENT_VALIDATION_20261028.json).
- `tokenizer_mode: copula` uses rank-based dependence summaries as attention
  biases with residual content fusion. The
  [pilot configuration](../configs/model/tuev-copula-lenovo.yaml) records its
  Windows runtime and batch accumulation. It is separate from the PAC operator.

## Encoder and controls

The reference encoder has six layers, 128 features and two attention axes:
temporal attention and joint electrode–band row attention.
`space_over_bands: true` and `freq_mixer: none` select this configuration.
`arch: triaxial` and the filename `triaxial.py` are retained implementation names.

[wave_patch.py](../paclock_bench/models/wave_patch.py) applies a linear projection
to raw waveform patches and uses the same two-axis encoder. TUEV supplies
[augmented](../configs/model/tuev-wave-patch.yaml) and
[unaugmented](../configs/model/tuev-wave-patch-noaug.yaml) controls, paired with
[augmented](../configs/model/tuev-scratch.yaml) and
[unaugmented](../configs/model/tuev-scratch-noaug.yaml) local-moment references.

## Training

Both residual configurations use seed 0, scratch initialization and
validation-best checkpoint selection. TUEV retains 20 epochs, batch 32,
AdamW at 1e-4, weight decay 0.05, dropout 0.35, augmentation and label smoothing 0.1.
CHB-MIT uses six complete cosine epochs, no early stopping, full validation every
1,000 steps plus epoch end, and focal positive weight 0.75. This CHB recipe differs
from the earlier pilot and is not an isolated structural comparison.

From the repository root, with an allocated GPU and the data/output environment
variables set as described in the [README](../README.md):

```bash
python scripts/rebuild/preflight.py --config configs/model/tuev-cfc-residual.yaml --mode smoke
python -m paclock_bench.training.train --config configs/model/tuev-cfc-residual.yaml --seed 0 --out "$CFM_REBUILD_ROOT/smoke" --smoke-steps 4
python scripts/rebuild/preflight.py --config configs/model/tuev-cfc-residual.yaml --mode train
python -m paclock_bench.training.train --config configs/model/tuev-cfc-residual.yaml --seed 0 --out "$CFM_REBUILD_ROOT/runs"
```

Use the CHB-MIT configuration for that dataset. Keep each output directory unique;
the trainer refuses to overwrite existing checkpoints. Preserve selected weights,
predictions, configuration and provenance. Metric definitions and result checks
are maintained in the [audit guide](AUDIT.md).

## Source map

| Component | Source |
|---|---|
| Model assembly | [build.py](../paclock_bench/models/paclock/build.py) |
| CFC token integration | [frontend/cfc.py](../paclock_bench/models/paclock/frontend/cfc.py) |
| Band content and local-moment reference | [frontend/triaxial.py](../paclock_bench/models/paclock/frontend/triaxial.py) |
| Two-axis encoder | [triaxial.py](../paclock_bench/models/paclock/triaxial.py) |
| Compatibility notes | [PROVENANCE.md](../paclock_bench/models/paclock/PROVENANCE.md) |
