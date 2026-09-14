# Candidate board

One row per live hypothesis. Claim an axis by adding a row before submitting. Status is one of
proposed / screening / passed-screen / replicating / adopted / closed.

| id | axis | hypothesis | owner | status | gate doc | outcome |
|---|---|---|---|---|---|---|
| local_residual | A mixing | zero-init shared projection of the filtered band waveform added to the rotation coupling token; init reproduces rot2 exactly incl. RNG | F | screening | docs/ROT2-RESIDUAL-GATE-2026-09-14.md | seed-0 joint gate FAILED on TUEV/Sleep-EDF (TUEV bal.acc -0.018; Sleep-EDF kappa +0.006 < +0.01). CHB-MIT + ISRUC arms in flight. |
| coupling_self | B geometry | include the diagonal Z_jj (within-band phase-amplitude asymmetry = waveform shape) in the aligned-phase sum; keeps n_b rows; also makes phase-reference invariance hold for every band | A | closed (paused) | — | seed 0: TUSZ 0.721 (best TUSZ on record), TUEV 0.707, CHB-MIT 0.618, IIIC 0.482. Parked while axis A is live — same problem. |
| d_axis_aug | D recipe | augmentation is admissible in a single shipped model; the TUEV cost is architecture-dependent and attributable to the coupling-hostile transforms (flip, frequency mask) | A | screening | docs/D-AXIS-GATE-2026-09-14.md | 4 runs submitted 2026-09-14 on mi2101x |
| ptS_init | E init | pretrained init on the weak columns, same architecture, only the init row changes | unowned | proposed | — | Siena 0.456 +- 0.041 vs scratch 0.170 +- 0.086 (3v3). Ten of twelve cells n=1. |
| multiscale_rotation | C scale | multi-scale `pac_patch_len` has only ever been run with `product`; untested with `rotation` | unowned | proposed | — | low priority: both existing multi-scale runs underperformed their single-scale siblings |
