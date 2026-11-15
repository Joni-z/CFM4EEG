# CFC readout comparison

All scores are R², rounded to three decimals. PAC cells show tokenizer → validation-selected encoder layer; AAC columns show encoder scores only. Bold marks the highest positive value for each endpoint within each dataset. Negative dimension-wise R² is clipped before averaging.

TFM/CBraMod/LaBraM use released checkpoints; CfFT uses the fixed project scratch checkpoints. Demod-WE uses the demodulation manuscript’s fixed seed-2 W+E three-axis checkpoints. This table summarizes existing checkpoint evaluations.

## TUEV

| Model | PAC δ–β | PAC θ–γ | PAC α–γ | AAC θ–γ | AAC δ–γ |
|---|---:|---:|---:|---:|---:|
| TFM | **0.421** → 0.537 | **0.292** → 0.333 | **0.333** → 0.342 | 0.531 | 0.527 |
| CBraMod | 0.151 → 0.383 | 0.000 → **0.379** | 0.122 → 0.385 | 0.526 | 0.508 |
| LaBraM | 0.083 → 0.283 | 0.005 → 0.362 | 0.027 → 0.305 | 0.520 | 0.420 |
| CfFT | 0.391 → 0.589 | 0.209 → 0.378 | 0.211 → **0.414** | 0.455 | 0.452 |
| Demod-WE | 0.150 → **0.709** | 0.055 → 0.262 | 0.100 → 0.382 | **0.613** | **0.559** |

## CHB-MIT

| Model | PAC δ–β | PAC θ–γ | PAC α–γ | AAC θ–γ | AAC δ–γ |
|---|---:|---:|---:|---:|---:|
| TFM | 0.159 → 0.215 | **0.159** → 0.167 | 0.000 → 0.000 | 0.314 | **0.276** |
| CBraMod | 0.108 → 0.306 | 0.037 → 0.134 | 0.000 → 0.000 | **0.346** | 0.235 |
| LaBraM | 0.097 → 0.212 | 0.034 → 0.251 | 0.000 → 0.000 | 0.272 | 0.135 |
| CfFT | **0.262** → **0.329** | 0.158 → **0.282** | 0.000 → 0.022 | 0.230 | 0.143 |
| Demod-WE | 0.104 → 0.158 | 0.074 → 0.176 | 0.000 → **0.037** | 0.232 | 0.186 |

Sources: [encoder comparison](cfc-released-40-20261115.csv), [tokenizer PAC comparison](cfc-tokenizer-pac-20261116.csv), [Demod-WE readout](demod-cfc-readout-20261116.csv).
