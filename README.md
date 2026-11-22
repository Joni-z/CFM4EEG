# CFM4EEG

This project studies what an EEG tokenizer should expose to a clinical classifier. It follows two questions.

| Research question | Implementation and evidence | Research materials |
|---|---|---|
| **Cross-frequency coupling.** Can we measure phase–amplitude coupling (PAC), make it readable from EEG tokens, and determine whether a classifier uses it? | [Original CfFT and model variants](docs/MODEL.md) · [standalone CFC operator](docs/CFC_OPERATOR.md) · [measurement, readout, Erase and Closure](experiments/audits/README.md) | [PAC manuscript and editable figures](papers/pac/README.md) · [presentation](presentations/pac.pptx) |
| **Waveform and envelope.** What does a compact encoder gain from seeing both the signed waveform and its amplitude over time? | [Matched content controls](docs/DEMODULATION_CONTROLS.md) · [TUEV and CHB-MIT evidence](papers/demodulation/evidence/README.md) | [Demodulation manuscript](papers/demodulation/README.md) · [presentation](presentations/demodulation.pptx) |

The original CfFT computes local phase–amplitude moments from learned bands. The standalone CFC operator has its own measurement experiments. The demodulation model reads an electrode × band × time grid of waveform and envelope features.

## Find an experiment

- [Browse results by dataset](experiments/README.md) or [download the workbook](experiments/Experiment-Records.xlsx).
- [Browse experiments by research question](experiments/audits/README.md), [inspect model versions](experiments/models.csv), or [look up checkpoint records](experiments/checkpoints.csv).
- [Compare data protocols](docs/DATA.md), including the [two CHB-MIT test grids](docs/CHBMIT_SCORE_AUDIT.md).
- [Find earlier implementations and manuscripts](docs/PROJECT_LAYOUT.md) and [both editable talks](presentations/README.md).

## Run the code

The [model guide](docs/MODEL.md#training) describes the CfFT trainer. The [baseline guide](docs/UPSTREAM.md#execution) lists each author's entrypoint, the [pinned sources](upstream/sources.json) and the required [environments](environments/README.md). Set `PACLOCK_PROC` to the parent of `processed/` and `CFM_REBUILD_ROOT` to the run-output directory. The [audit guide](docs/AUDIT.md) explains data acceptance, configuration review and checkpoint checks.

```bash
python -m pytest -q
python -m tests.check_preprocessing
```

Use `main` for code, configurations, documentation, experiment records, manuscripts and talks. The Python package retains the name `paclock_bench` for existing imports and checkpoints. Raw EEG, weights and machine-specific environments are stored separately; see the [artifact guide](docs/PROJECT_LAYOUT.md).
