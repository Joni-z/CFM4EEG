# CFM4EEG / CfFT

CfFT is an EEG tokenizer that represents band content using measured cross-frequency relationships. A learnable filterbank separates the signal into bands; local phase–amplitude statistics guide the transformation of envelope and waveform features before Transformer encoding.

This repository contains the model, clinical EEG preprocessing, and baseline reproduction code. Baselines call the original authors’ model and training code; dataset and runtime adaptations stay in small, explicit entrypoints. The Python package name `paclock_bench` is retained so existing checkpoint keys and imports remain usable.

The [copula-guided tokenizer](docs/COPULA-TOKENIZER.md) is an explicit alternative
frontend: empirical marginal transforms provide dependence summaries that guide
residual fusion of band content, followed by the same two-axis encoder.

## Layout

| Path | Contents |
|---|---|
| `paclock_bench/models/paclock/` | CfFT tokenizer and encoder |
| `paclock_bench/data/`, `training/` | Array contracts, metrics and CfFT training |
| `preprocessing/` | Clinical readers and dataset preparation |
| `configs/datasets/` | Dataset definitions |
| `configs/baselines/<model>/` | Supported original-code baseline configurations |
| `configs/model/` | CfFT and the explicit waveform-patch control configurations |
| `configs/deferred/` | Native pretraining/pretrained recipes, kept separate from scratch work |
| `configs/planned/` | Unfinished adaptations; explicitly blocked from launch |
| `scripts/rebuild/` | Original-code baseline entrypoints and input bridges |
| `upstream/sources.json` | Author repositories, exact commits and original entry files |
| `environments/` | Package versions observed in the actual execution environments |
| `tests/` | CPU checks for data contracts, selection and numerical behavior |

Upstream checkouts live in ignored `vendor/`; data, checkpoints and run outputs live outside the source tree. Historical experiment material is stored separately; it is not another importable implementation.

## Reproduce a baseline

1. Read [original sources and adaptations](docs/UPSTREAM.md). Restore the pinned checkout into `vendor/` and use its recorded environment.
2. Read [dataset contracts](docs/DATA.md). Set `PACLOCK_PROC` to the parent of `processed/`, and `CFM_REBUILD_ROOT` to a separate output directory.
3. Choose the configuration under `configs/baselines/<model>/`. Run its actual GPU smoke and checkpoint reload before formal training.

For entries with the common config interface, for example BIOT:

```bash
export PACLOCK_PROC=/path/to/data
export CFM_REBUILD_ROOT=/path/to/outputs
python scripts/rebuild/launch.py configs/baselines/biot/tuev-scratch.yaml 0 smoke
python scripts/rebuild/launch.py configs/baselines/biot/tuev-scratch.yaml 0 train
```

Use the appropriate baseline environment and allocated GPUs. Conformer, LaBraM, native23 EEGPT and TFM have explicit commands in [UPSTREAM.md](docs/UPSTREAM.md). The dispatcher never substitutes the project’s generic trainer for a missing baseline implementation.

## Audit and checks

[Audit guide](docs/AUDIT.md) describes the path from author commit to input manifest, configuration, smoke receipt and selected checkpoint. Configurations record deviations; a shared-cohort adaptation is not an exact reproduction of a paper’s original data protocol.

```bash
python -m pytest -q
python -m tests.check_preprocessing
```

CPU checks do not replace model-specific GPU smoke tests. Existing jobs execute frozen releases; changes to this working tree require a fresh matching smoke before a new formal run.
