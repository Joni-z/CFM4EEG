# Experiments by research question

| Question | Evidence | What was compared |
|---|---|---|
| Can the original CfFT operator recover known coupling? | [Synthetic measurement](pac-measurement/README.md) | The original filterbank and local moment on signals with a prescribed coupling relationship, before token construction. |
| Does CfFT make PAC readable from its tokens? | [Two-axis tokenizer ablations](pac-token-readout-twoaxis-20261113.json) | Four token constructions on TUEV, seed 2, with a common reader and the same three PAC readout targets. |
| What can we read from other models? | [Released and fine-tuned checkpoints](cfc-stage-comparison-20261116.md) · [fine-tuned scores](finetuned-cfc-readout-20261116.md) | Tokenizer and encoder endpoints, with checkpoint initialization and task fine-tuning recorded separately. |
| Does removing readable PAC change prediction? | [Frozen-model erasure](cfc-erase-20261122.md) | Original predictions, a PAC eraser, a shuffled-target eraser and rank-matched random erasers. |
| How much can PAC predict by itself? | [PAC-only Closure](cfc-pac-closure-20261122.md) | The same classifier fitted to 18 PAC summaries or random features, compared with the full models on the same test windows. |
| Which checkpoint reports the main CfFT model on TUEV? | [Main checkpoint and reproduction](tuev-main-checkpoint-20261123.md) | The two-axis measured seed-2 checkpoint (κ 0.7134), its two sibling seeds, a re-evaluation with current code and a training-code equivalence check. |
| Does PAC add to band power, and more than AAC? | [Feature-level increments](pac-power-increment-20261123.md) | Power, PAC and AAC summaries from raw EEG, alone and combined, with dimension-matched Gaussian controls and the Closure classifier. |
| Does CfFT need PAC specifically, or would AAC do? | [AAC-guided CfFT](cfft-aac-control-20261123.md) | The same model with the PAC moment replaced by an envelope correlation; same parameters, initialization and recipe. TUEV pairs two Torch runs; TUSZ, IIIC, ISRUC and CHB-MIT pair a Torch AAC run with the existing PAC record, and each row also lists the strongest baseline. |
| What do waveform and envelope contribute? | [Matched content controls](../../docs/DEMODULATION_CONTROLS.md) · [manuscript evidence](../../papers/demodulation/evidence/README.md) | Change the token content while holding the reader fixed. |
| Which CHB-MIT test windows were used? | [Protocol and score audit](../../docs/CHBMIT_SCORE_AUDIT.md) | Legacy seizure-enriched windows and the natural test grid have separate comparisons. |

The TUEV main-model result is the seed-2 checkpoint above. The TUEV tokenizer-ablation table uses seed 2. The CfFT TUEV erasure and Closure comparison uses seed 0. Follow the checkpoint and cohort in each report when combining results.
