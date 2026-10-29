# CFM4EEG / CfFT

CfFT constructs EEG tokens from band content and explicitly measured
cross-frequency relationships. A learnable filterbank provides waveform and
envelope features; a separate CFC operator measures phase–amplitude dependence.
The residual tokenizer uses these measurements to condition all content bands
before a compact Transformer encoder.

This repository contains the model, clinical preprocessing and baseline
reproduction code. Baselines call the original authors' implementations through
explicit data/runtime bridges. The package name `paclock_bench` is retained for
import and checkpoint compatibility.

## Guides

| Guide | Contents |
|---|---|
| [Model](docs/MODEL.md) | Tokenizer variants, encoder, configurations and training commands |
| [CFC operator](docs/CFC_OPERATOR.md) | Measurement algorithm, API, masks, calibration and validation records |
| [Data](docs/DATA.md) | Cohorts, preprocessing, units and dataset acceptance |
| [Baselines](docs/UPSTREAM.md) | Pinned author sources, native recipes and execution commands |
| [Audit](docs/AUDIT.md) | Smoke checks, checkpoint selection, metrics and evidence |

## Layout

| Path | Contents |
|---|---|
| `paclock_bench/models/` | CfFT tokenizer, CFC operator, encoder and waveform control |
| `paclock_bench/data/`, `paclock_bench/training/` | Array loading, losses, metrics and project trainer |
| `preprocessing/`, `data_reference/` | Clinical preparation and cohort/hash references |
| `configs/model/`, `configs/datasets/` | Project model variants and dataset definitions |
| `configs/baselines/` | Supported native baseline recipes |
| `configs/deferred/`, `configs/planned/` | Conditional recipes and blocked adaptations |
| `scripts/rebuild/`, `scripts/experiments/` | Native baseline bridges and synthetic CFC validation |
| `review/` | Reviewed configuration hashes and immutable validation records |
| `upstream/sources.json` | Author repositories, pinned commits and entry files |
| [environments/](environments/README.md) | Recorded model-specific runtimes |
| `tests/`, `slurm/rebuild/` | CPU checks and scheduler templates |

Author checkouts and released-weight caches belong in ignored `vendor/`.
Datasets, run outputs and trained checkpoints stay outside the source tree.

## Run

Set `PACLOCK_PROC` to the directory containing `processed/` and
`CFM_REBUILD_ROOT` to a separate output directory. Use allocated GPUs for both
smoke tests and training.

- **CfFT:** follow the [model commands](docs/MODEL.md#training).
- **Baselines:** restore the pinned vendor checkout and follow its
  [native entrypoint](docs/UPSTREAM.md#execution). The project trainer is not a
  fallback for native baseline training.

Every formal run requires accepted data, a reviewed configuration and a matching
GPU smoke with checkpoint reload. See the [audit guide](docs/AUDIT.md).

```bash
python -m pytest -q
python -m tests.check_preprocessing
```

CPU checks do not replace model-specific GPU smoke tests.
