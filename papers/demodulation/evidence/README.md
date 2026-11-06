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
| `tuev_waveform` | `tuev-demodulation-waveform-20261106` | 2 | κ 0.4983532624 |
| `tuev_envelope` | `tuev-demodulation-envelope-20261106` | 2 | κ 0.6416508750 |
| `chbmit_raw` | `chbmit-wave_patch_threeaxis-20261105` | 2 | PR-AUC 0.6418777629 |

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
within 1e-12. Per-class values use those same saved predictions. These reference/control entries are individual runs; no inferred standard
deviations are used.

The new TUEV waveform-only and envelope-only arms retain both 64-coordinate
projections, their initialization and all 1,620,774 parameters. The input type
of the projection blocks is the only changed model field. Both finish 20 epochs
and select validation evaluation 1, matching the paired reference. New
test metrics were replayed from stored logits and labels with zero numerical
difference. Each entry records final result, prediction, checkpoint and
data-manifest hashes. The CHB raw learned-row arm finishes six epochs by patience,
selecting evaluation 246; the test cohort contains 17,386 windows.

## Recovery records and figures

`recovery-example.json` contains only the four plotted 50-sample traces
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

The CHB-MIT envelope-only arm and TUEV mean-envelope/bandmix follow-ups have
no completed final test record in this revision. Their cells remain empty.
Historical rotated ablations are not used to fill them. The CHB-MIT
waveform-only cell is also empty; no such matched run is recorded. The
CHB-MIT scratch EEGPT and REVE main-table cells have no compatible
completed run and remain empty. Unrecorded native TUEV parameter counts
are also left unreported. This directory contains only finished measurements.

## Historical main-table baselines

`baseline-results.json` preserves 52 completed run records, grouped into 20
model/task/initialization cells. Each multi-run cell reports all three seeds
from one canonical recipe using sample SD (ddof=1). TUEV LaBraM and EEGPT
use the verified shared16 native-engine seed0 runs. No archived wrong-recipe,
wrong-batch or stale variant is merged into those families, and no seed is
removed for a low score or first-evaluation peak.

Every included metric was matched against its original document in the
project's historical ledger. Each public entry retains its record ID,
original artifact hash, recipe identity, seed, actual test metrics and the
recorded training settings. Native LaBraM metric aliases are mapped to the
shared headers. The original 26,067-window LaBraM results and natural-grid
CHB-MIT results are not used in the current common-protocol comparison.

The manuscript distinguishes pretrained and scratch initialization and
documents model-specific adaptations in Appendix D. Parameter counts are
reported separately for TUEV and CHB-MIT; other-head sizes are not transferred
to unrecorded native entries.
