# Measurements used in this draft

`results.json` contains the completed test records cited in the manuscript,
per-class metrics recomputed from saved TUEV predictions, and frozen-token
recovery scores. Original result and data-manifest SHA-256 hashes bind each
classification row. Model weights, full predictions and clinical datasets are
not copied into this paper directory.

## Classification records

| Key | Recorded run | Seed | Main metric |
|---|---|---:|---:|
| `tuev_full` | `tuev-legacy-triaxial-identity-20261105` | 2 | κ 0.7111908486 |
| `chbmit_full` | `chbmit-legacy-triaxial-identity-20261105` | 2 | PR-AUC 0.6994134955 |
| `tuev_raw` | `tuev-wave_patch_threeaxis-20261105` | 2 | κ 0.5758048478 |

Both reference runs use unrotated waveform/envelope tokens and the same
three-axis reader. The TUEV raw control uses 16 learned rows, not physical
frequency bands. It matches the reader and training recipe but has a larger
frontend parameter count. The comparison therefore tests the entire frontend.

The current data are the 16-bipolar-channel protocols documented in Appendix A.
CHB-MIT includes seizure-window enrichment in all splits; do not replace its
17,386-window test set with natural-grid metrics or pool it with them.
Native 23-channel baseline results do not fill these table entries.

TUEV full/raw labels were verified identical in order; kappa, balanced accuracy
and weighted F1 were independently recomputed and matched the stored records
within 1e-12. Per-class values use those same saved predictions. No seed averaging
or inferred standard deviations are used.

## Recovery records and figures

`recovery-tuev.npz` and `recovery-chbmit.npz` contain small derived example arrays
from the completed frozen-token recovery check. `make_figures.py` uses the first
held-out example/channel, band index 11, and the middle of three selected patches.
Aggregate scores in `results.json` use all sampled held-out tokens, not only the
plotted example. Reconstruction uses actual identity-model checkpoints, not
rotated-token variants.

The implementation is [`demodulation_reconstruction.py`](../../../scripts/experiments/demodulation_reconstruction.py).
The check has no classifier updates. Full-rank 64×50 projection matrices make
local recovery algebraically possible; recovery establishes preservation, not
predictive necessity.

## Empty table cells

Bandwise waveform-only and envelope-only controls require final test artifacts.
The reference scores must not be combined with historical rotated ablations to
fill them. The raw CHB-MIT result and protocol-matched baseline results are also
left empty until validated. This directory contains only finished measurements.
