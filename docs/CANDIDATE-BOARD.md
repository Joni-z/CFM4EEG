# Candidate board

One row per live hypothesis. Claim an axis by adding a row before submitting. Status is one of
proposed / screening / passed-screen / replicating / adopted / closed.

| id | axis | hypothesis | owner | status | gate doc | outcome |
|---|---|---|---|---|---|---|
| local_residual | A mixing | zero-init shared projection of the filtered band waveform added to the rotation coupling token; init reproduces rot2 exactly incl. RNG | F | screening | docs/ROT2-RESIDUAL-GATE-2026-09-14.md | seed-0 joint gate FAILED on TUEV/Sleep-EDF (TUEV bal.acc -0.018; Sleep-EDF kappa +0.006 < +0.01). CHB-MIT + ISRUC arms in flight. **Verified 2026-09-14 by A: both TUEV arms selected epoch-0 checkpoints, so the TUEV clause compared two one-epoch models; residual won all four test comparisons. Do not record as closed. See docs/N-BANDS-FINDING-2026-09-15.md, section 6.** |
| coupling_self | B geometry | include the diagonal Z_jj (within-band phase-amplitude asymmetry = waveform shape) in the aligned-phase sum; keeps n_b rows; also makes phase-reference invariance hold for every band | A | closed (paused) | — | seed 0: TUSZ 0.721 (best TUSZ on record), TUEV 0.707, CHB-MIT 0.618, IIIC 0.482. Parked while axis A is live — same problem. |
| d_axis_aug | D recipe | augmentation is admissible in a single shipped model; the TUEV cost is attributable to the coupling-hostile transforms (flip, frequency mask) | A | screening | docs/D-AXIS-GATE-2026-09-14.md | **CLOSED 2026-09-15.** augsafe moved the peak only to evaluation 1 (val kappa 0.6667 vs 0.6339, a real but small lift). The premise that flip and frequency mask are coupling-hostile is contradicted by `tuev-crofremo_n5`, the best TUEV run on record, which uses all five transforms. |
| d_axis_wd | D recipe | the recipe never trains on TUEV: all 3 seeds select the epoch-0 checkpoint. weight_decay is 1e-5 against LaBraM/CBraMod 5e-2 | A | screening | docs/EPOCH0-SELECTION-AUDIT-2026-09-14.md | **CLOSED 2026-09-15. Not the lever.** weight_decay 5e-2 at 8 bands still peaks at evaluation 0 (val kappa 0.6253 vs control 0.6339). Independently confirmed by the archived `tuev-crofremo_s4`, which carries wd 5e-2, dropout 0.35 and full augmentation at 8 bands and also peaks at 0. |
| n_bands_16 | B geometry | 16 aligned-phase bands instead of 8 is what lets the model train past the first epoch on TUEV. Every 16-band run peaks at evaluation 8-13; every 8-band run peaks at 0-2 regardless of weight decay, dropout or augmentation; 24 bands peaks at 0 again. Correction 2026-09-15: s4 is not a matched n5 control (coupling_self differs). Later peak alone is not a performance win. | A | proposed | docs/N-BANDS-FINDING-2026-09-15.md | TUEV seed 0 only: kappa 0.7391 vs 0.7352 (inside seed noise, not a win) and balanced accuracy 0.6926 vs 0.5678 (far outside it). Never run on any other corpus. |
| ptS_init | E init | pretrained init on the weak columns, same architecture, only the init row changes | unowned | proposed | — | Siena 0.456 +- 0.041 vs scratch 0.170 +- 0.086 (3v3). Ten of twelve cells n=1. |
| multiscale_rotation | C scale | multi-scale `pac_patch_len` has only ever been run with `product`; untested with `rotation` | unowned | proposed | — | low priority: both existing multi-scale runs underperformed their single-scale siblings |

## F follow-up, 2026-09-15 (user authorized)

The five older AMD residual/routing jobs were stopped at the user's request.
The pending Torch routing duplicates and their handover controller are closed.
Axes A/B are now tested jointly under the existing n5 regularization recipe;
the former axis-A in-flight restriction is superseded by this explicit request.

| id | axis | hypothesis | owner | status | gate doc | outcome |
|---|---|---|---|---|---|---|
| nb16_n5_followup | A/B | At 16 bands, compare pure coupling, waveform residual and separate waveform rows under one n5 recipe | F | closed | docs/NB16-FOLLOWUP-GATE-2026-09-15.md | Superseded by user structural-priority correction before full training; only hardware smoke 419224 ran. |
| modulation_carrier | A mixing / token construction | Preserve waveform coordinates inside the PAC rotation: analytic-amplitude quadrature versus full waveform carrier; two tasks each | F | screening | docs/MODULATION-CARRIER-GATE-2026-09-15.md | GPU smoke 419229 passed; four trainers RUNNING in packed AMD job 419232, source 53077a2. |

| residual16_backfill | A mixing | Retest per-band waveform residual at 16 bands under n5 recipe | F | admitted | docs/RESIDUAL16-GATE-2026-09-15.md | Reuse free slots in 419232; CHB first, TUEV next free slot; seed0 only. |

## Model identity registry (2026-09-15)

Use [MODEL-VERSIONS.md](MODEL-VERSIONS.md) for all reports. Paper duplex = Duplex-F192 (cf2_v1d192); earlier duplex = Duplex-A128 (paclock_duplex). New candidates must register a stable alias, immutable run ID, structural delta, gate and budget before launch. Never use bare duplex or omit seed count.
