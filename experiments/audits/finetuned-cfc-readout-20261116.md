# Official-pretrained, task-fine-tuned CFC readout

Status: 40/40 verified scores. All four original checkpoints use seed 0. The neural models remain frozen; only the existing linear ridge probes are fitted.

The readout protocol, test windows, PAC/AAC targets and pooling match the preceding [comparison](cfc-stage-comparison-20261116.md). CBraMod/LaBraM here are task-fine-tuned checkpoints, while their rows in the preceding comparison use released general pretraining weights. LaBraM retains its recorded fresh bipolar spatial slots; no input adaptation is changed.

## tokenizer

| Dataset | Model | PAC delta–beta | PAC theta–gamma | PAC alpha–gamma | AAC theta–gamma | AAC delta–gamma |
|---|---|---:|---:|---:|---:|---:|
| tuev | cbramod | 0.007 | 0.000 | 0.101 | 0.180 | 0.167 |
| tuev | labram | 0.087 | 0.011 | 0.025 | 0.081 | 0.006 |
| chbmit | cbramod | 0.144 | 0.000 | 0.000 | 0.038 | 0.026 |
| chbmit | labram | 0.114 | 0.058 | 0.000 | 0.089 | 0.051 |

## encoder_peak

| Dataset | Model | PAC delta–beta | PAC theta–gamma | PAC alpha–gamma | AAC theta–gamma | AAC delta–gamma |
|---|---|---:|---:|---:|---:|---:|
| tuev | cbramod | 0.469 | 0.295 | 0.333 | 0.540 | 0.505 |
| tuev | labram | 0.312 | 0.206 | 0.226 | 0.431 | 0.383 |
| chbmit | cbramod | 0.275 | 0.181 | 0.000 | 0.348 | 0.184 |
| chbmit | labram | 0.183 | 0.086 | 0.000 | 0.097 | 0.049 |

Four checkpoint binary hashes were verified against their private Hugging Face archive. The original training records verify official pretrained initialization; the complete fine-tuned states load strictly. No EEG training or additional clinical evaluation is performed. Existing comparison scores remain unchanged.

[Full precision and bindings](finetuned-cfc-readout-20261116.json) · [CSV](finetuned-cfc-readout-20261116.csv)
