# CFM4EEG / CfFT

CfFT is an EEG tokenizer that represents band content using measured cross-frequency relationships. A learnable filterbank separates the signal into bands; local phase–amplitude statistics guide the transformation of envelope and waveform features before Transformer encoding.

This repository contains the model, clinical EEG preprocessing, and baseline reproduction code. Baselines call the original authors’ model and training code; dataset and runtime adaptations stay in small, explicit entrypoints. The Python package name `paclock_bench` is retained so existing checkpoint keys and imports remain usable.

## Documentation

| Guide | Contents |
|---|---|
| [Model](docs/MODEL.md) | CfFT, explicit PAC measurement, the waveform-patch control and optional frontends |
| [Data](docs/DATA.md) | Preprocessing, cohorts, units and accepted dataset versions |
| [Baselines](docs/UPSTREAM.md) | Original sources, recipes, weight loading and execution commands |
| [Audit](docs/AUDIT.md) | Smoke checks, checkpoint selection, metrics and result recording |

## Layout

| Path | Contents |
|---|---|
| `paclock_bench/models/paclock/` | CfFT tokenizer and encoder |
| `paclock_bench/data/`, `paclock_bench/training/` | Array contracts, metrics and CfFT training |
| `preprocessing/` | Clinical readers and dataset preparation |
| `configs/datasets/` | Dataset definitions |
| `configs/baselines/<model>/` | Supported original-code baseline configurations |
| `configs/model/` | CfFT frontend variants and waveform-patch control configurations |
| `configs/deferred/` | Additional recipes with explicit stage or exposure checks |
| `configs/planned/` | Unfinished adaptations; explicitly blocked from launch |
| `scripts/rebuild/` | Original-code baseline entrypoints and input bridges |
| `scripts/experiments/` | Synthetic PAC recovery and tokenizer-learning checks |
| `upstream/sources.json` | Author repositories, exact commits and original entry files |
| [environments/](environments/README.md) | Recorded execution environments and compatibility notes |
| `tests/` | CPU checks for data contracts, selection and numerical behavior |

Pinned author checkouts and released-weight caches live in ignored `vendor/`. Dataset caches, trained checkpoints and run outputs live outside the source tree.

## Reproduce a baseline

1. Read [baseline sources and recipes](docs/UPSTREAM.md). Restore the pinned checkout into `vendor/` and use its recorded environment.
2. Read [dataset contracts](docs/DATA.md). Set `PACLOCK_PROC` to the parent of `processed/`, and `CFM_REBUILD_ROOT` to a separate output directory.
3. Choose the configuration under `configs/baselines/<model>/`. Run its actual GPU smoke and checkpoint reload before formal training.

For entries with the common config interface, for example BIOT:

```bash
export PACLOCK_PROC=/path/to/data
export CFM_REBUILD_ROOT=/path/to/outputs
python scripts/rebuild/launch.py configs/baselines/biot/tuev-scratch.yaml 0 smoke
python scripts/rebuild/launch.py configs/baselines/biot/tuev-scratch.yaml 0 train
```

Use allocated GPUs for smoke and training. Other entrypoints, released-weight initialization and multi-stage TFM execution are covered in the [baseline guide](docs/UPSTREAM.md#execution).

## Audit and checks

[Audit guide](docs/AUDIT.md) describes the path from author commit to input manifest, configuration, smoke receipt and selected checkpoint. Configurations record deviations; a shared-cohort adaptation is not an exact reproduction of a paper’s original data protocol.

```bash
python -m pytest -q
python -m tests.check_preprocessing
```

CPU checks do not replace model-specific GPU smoke tests.
