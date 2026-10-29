# Model and tokenizer

CfFT uses a learnable 16-band Sinc bank, envelope/waveform patch projections,
full complex rotation and a six-layer two-axis Transformer. The CFC-reference
variant keeps this model and replaces supported rotation references with the
[new PAC measurement](CFC_OPERATOR.md).

## Configurations

| Tokenizer | TUEV | CHB-MIT |
|---|---|---|
| Original CfFT | [tuev-scratch.yaml](../configs/model/tuev-scratch.yaml) | [chbmit-scratch.yaml](../configs/model/chbmit-scratch.yaml) |
| Original CfFT + new operator | [tuev-cfc-reference.yaml](../configs/model/tuev-cfc-reference.yaml) | [chbmit-cfc-reference.yaml](../configs/model/chbmit-cfc-reference.yaml) |

The second row changes `tokenizer_mode` to `cfc`. It adds no trainable parameters:
the TUEV model remains 1,224,486 parameters. Common content, encoder and head
weights initialize identically to the original configuration.

## Measurement connection

1. Construct the original local band features and rotation references.
2. Run the unchanged dual-harmonic CFC operator on its physical six-source,
   five-target grid.
3. Use its first-harmonic coefficient with the matching measured source-phase
   features. Weight the physical targets by the content filter's power response
   at their centers, retaining correspondences within its half-power passband.
4. Normalize the resulting reference and apply the original full rotation to
   the same envelope/waveform content. If the measurement or correspondence is
   unavailable, retain the original local reference exactly.

The operator still jointly fits both harmonics; the original single-reference
rotation consumes H1. There is no residual branch, learned readout, extra
shrinkage or identity-centered rotation. The physical grid is not relabeled as
a 16-by-16 measurement. Content patches stay 0.25 seconds; measured coefficients
use the full record, while the fallback uses the original local moments.
`pac_token_mode: identity` retains the original unrotated-content control.

The encoder alternates temporal and joint electrode–band attention.
`space_over_bands: true` and `freq_mixer: none` select these two axes;
`arch: triaxial` remains an implementation name.

## Training

Both new configurations copy their dataset's original scratch recipe: seed 0,
batch 32, AdamW at 1e-4, weight decay 0.05, dropout 0.35, cosine schedule,
augmentation and validation-based checkpoint selection. Arrays are preloaded
into RAM as a runtime setting.

TUEV retains 20 epochs and label smoothing 0.1. CHB-MIT retains the original
20-epoch maximum, validation every 200 steps, patience 20 validation evaluations,
and focal alpha 0.25 / gamma 2. Early stopping can end it before 20 epochs.

With an allocated GPU, set `PACLOCK_PROC` to the directory containing `processed/`
and `CFM_REBUILD_ROOT` to an external output directory, then run:

```bash
python scripts/rebuild/preflight.py --config configs/model/tuev-cfc-reference.yaml --mode smoke
python -m paclock_bench.training.train --config configs/model/tuev-cfc-reference.yaml --seed 0 --out "$CFM_REBUILD_ROOT/smoke" --smoke-steps 4
python scripts/rebuild/preflight.py --config configs/model/tuev-cfc-reference.yaml --mode train
python -m paclock_bench.training.train --config configs/model/tuev-cfc-reference.yaml --seed 0 --out "$CFM_REBUILD_ROOT/runs"
```

Use the CHB-MIT configuration for that dataset. Output directories must be new;
selected checkpoints, predictions and provenance are retained. See [AUDIT.md](AUDIT.md).

## Controls and sources

[WavePatch](../paclock_bench/models/wave_patch.py) uses raw waveform patches and
the same encoder. Its [augmented](../configs/model/tuev-wave-patch.yaml) and
[unaugmented](../configs/model/tuev-wave-patch-noaug.yaml) configurations accompany
the original CfFT controls. The earlier `measured_pac` and copula experiments
remain separate frontends.

| Component | Source |
|---|---|
| Model assembly | [build.py](../paclock_bench/models/paclock/build.py) |
| New operator connection | [frontend/cfc.py](../paclock_bench/models/paclock/frontend/cfc.py) |
| Original content and rotation | [frontend/triaxial.py](../paclock_bench/models/paclock/frontend/triaxial.py) |
| Transformer | [triaxial.py](../paclock_bench/models/paclock/triaxial.py) |

The discontinued rotation-adapter and residual-readout pilots are preserved in
[cfc-rotation-source-20261029](https://github.com/Joni-z/CFM4EEG/tree/cfc-rotation-source-20261029)
and [cfc-residual-source-20261030](https://github.com/Joni-z/CFM4EEG/tree/cfc-residual-source-20261030).
Their configurations and checkpoints use those releases.
