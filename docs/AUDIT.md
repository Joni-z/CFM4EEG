# Experiment audit

For the model-specific CHB-MIT enriched/natural-grid score review, see
[CHB-MIT score audit](CHBMIT_SCORE_AUDIT.md). It separates numerical integrity,
checkpoint rescoring, completed training and comparison eligibility.

## Follow one execution

Check each model/dataset/seed in this order:

1. [Original source](../upstream/sources.json): author repository, pinned commit
   and main/engine files.
2. Configuration and [entrypoint](UPSTREAM.md): task, scratch/pretrained setting,
   input shape, optimizer, budget and declared adaptations. A planned config is
   not executed evidence.
3. [Dataset contract](DATA.md): exact cohort, units, manifest, acceptance receipt
   and required build/transfer receipts.
4. GPU smoke: real optimizer updates, finite values, checkpoint reload and an
   identity bound to the executed source/config/data/weights.
5. Formal execution: recorded CLI/environment, validation selection, stopping
   reason and retained checkpoint actually used for final testing.
6. Numerical verification: complete test coverage, label order, prediction
   finiteness and independent recomputation of the displayed metrics.

CfFT and WavePatch use the project trainer. Baselines use their explicit author
entrypoints; missing code is an error. A first-evaluation validation peak is a
recorded diagnostic, not automatic grounds to reject a native-selected checkpoint.

Shared-array smokes use training/validation data without evaluating the test set.
Reading file metadata to check an acceptance receipt is distinct from opening
held-out arrays. The legacy native23 EEGPT reader opens its native test payload
during smoke; record that exception explicitly, rather than claiming isolation.
See the [EEGPT interface](UPSTREAM.md#eegpt).

## Selection and common metrics

Retain the author's selection rule and disclose differences between paper text
and released code. BIOT/FFCL's released default selects a terminal checkpoint.
Its original result and a separately archived validation-best reevaluation are
different records; neither silently replaces the other. The archive monitors
validation kappa for six-class tasks and validation AUROC for binary tasks.
Never choose checkpoints or omit a completed result according to test scores.

The tracking convention uses trapezoidal AUC-PR and a fixed 0.5 binary threshold.
Some BIOT/FFCL/LaBraM utilities call average precision `pr_auc`; BIOT/FFCL also use
validation-derived thresholds. Keep native values and recompute common metrics
from the same predictions. Do not treat AP and trapezoidal AUC-PR as identical.

A native full-test confusion matrix permits exact recomputation of multiclass
kappa, balanced accuracy and weighted F1, with its marginals checked against test
class counts. It does not establish prediction order or permit AUC calculation.
Record this audit scope separately from a raw-checkpoint GPU replay or an audit
of saved per-example predictions.

Numerical correctness and comparison eligibility are separate: shared16 and
native23 TUEV have different cohorts. Recording both protocols does not authorize
a same-test ranking. Training, runtime, preprocessing and initialization
adaptations remain part of the comparison record.

## Record a completed result

### CHB-MIT recipe comparison (2026-11-04)

The original-control run changes only C's tokenizer back to `pac_interaction`.
Its optimizer, focal loss, augmentation and validation selection match the
historical two-axis CfFT recipe. The corrected evaluation set differs as
documented in [Data](DATA.md); old/new PR-AUC is not a paired comparison.

| Active recipe | Input scale | Loss | Selection / budget |
|---|---|---|---|
| CfFT original / C | Per-window/channel Q95 | Focal alpha .25, gamma 2 | PR-AUC every200steps; patience20evaluations; up to20epochs |
| Native CBraMod scratch | Restored physical microvolts /100 | BCE | AUROC; full50epochs |
| Native TFM single-dataset | Author Q95 | Focal alpha .8, gamma .7 | AUROC; intended50epochs |

CBraMod's checked corrected-test PR-AUC is .569932. The TFM run was interrupted
at zero-based epoch42; its separately evaluated validation-best epoch4 gives
PR-AUC .266357. Preserve that interruption status. These results use the common
corrected cohort, not the TFM paper's original preprocessing. Recipe differences
are candidate explanations, not isolated effects established by these runs.
Execution receipts and source comparisons are retained outside the repository
under `maintenance/chbmit-control-audit-20261104`.

### CHB-MIT historical-score decomposition (2026-11-04)

The saved predictions permit a direct evaluation-composition check. Removing
the 162 seizure-centered extra positive test windows, without changing weights,
scores or thresholds, gives:

| Historical seed0 model | Original test AUC-PR | Natural-grid subset AUC-PR | Difference |
|---|---:|---:|---:|
| Two-axis measured CfFT | .604625 | .403814 | -.200811 |
| Three-axis range CfFT | .696899 | .485518 | -.211380 |

The retained 17,224 rows have114 positives; removed rows are all positive.
The new-to-old row mapping reproduces every accepted test label. Independent
recomputation agrees within1e-12. Archived logits are float16, explaining the
roughly1e-5 differences from the original full-precision result files. These
are fixed historical predictions rescored on a subset, not new training or a
checkpoint selected on corrected validation data. AUROC also falls from about
.913 to .884: the change includes positive-example composition, not just class
prevalence. The90 extra validation positives can additionally affect checkpoint
selection; that effect has not been isolated.

All683 effective raw EDF hashes match. After matching records, all training
label sequences match. Across185 sampled complete signal windows, maximum
relative RMS difference is1.46e-6. Training row order did change: legacy workers
assembled records in completion order; the rebuild sorts files. Thus identical
seeds do not imply identical minibatches.

The archived `twoaxis-runtime` model source matches the current original path.
A same-input CPU single-update check found identical initialization, logits,
loss, gradients, AdamW updates and learning rate. This excludes a change in the
tested core calculation, not all CUDA/runtime or full-training differences.
The historical successful two-axis run also predicted all negatives at epoch0
step5000 (validation PR .062058), before reaching its best checkpoint at epoch4
step5000. Current early validation must not be compared with that final test.

Evidence: the maintenance audit's `data-causal-recomputed.json`, row maps,
`evaluation-independent-check.json`, `training-parity.json` and
`stage-comparison.json`. Original results and running releases are preserved.

### Legacy content, rotation and encoder attribution (2026-11-05)

The reference here is the 16-band `quad_range` quadrature model, not the earlier
8-band duplex family. The archived ledger reconstructs 74 paired comparisons
from 96 endpoints; independent recomputation agrees. These are historical
protocol results, not a new comparison with corrected-data native baselines.

CHB-MIT seed0 gives a useful encoder-by-guidance comparison:

| Encoder / historical AUC-PR | Identity, no rotation | Measured rotation |
|---|---:|---:|
| Joint electrode-band two-axis | .664544 | .604655 |
| Separate electrode/band three-axis | .671299 | .696913 |

The three-minus-two difference is .006755 without rotation and .092258 with
rotation. This suggests a representation/encoder interaction; one seed and
changed capacity/initialization do not establish its cause. TUEV's interaction
has the opposite sign: three-axis identity .729197 exceeds measured .713880,
while two-axis measured .684601 exceeds identity .616390. Own-phase and
weights-only are other rotations, not substitutes for the identity control.

Keeping waveform content matters strongly in the three-axis CHB-MIT seed2
comparison (.709807 versus envelope-only .543221); the corresponding TUEV
difference is small (.715456 versus .713653). Older raw-band models also improve
with eight bands over one on all three shared CHB-MIT and TUSZ seeds, but band
count changes tokens and compute. The old flat comparison additionally changes
frontend electrode mixing, positional paths, normalization and capacity; it does
not isolate attention factorization.

Both axis variants retain cross-band communication. Three-axis separates
electrode and band attention; two-axis attends jointly over electrode×band.
The reference frequency mixer is ordinary attention and ignores the explicit
coupling argument. Guidance acts in the frontend before this learned mixing.
Identity bypasses the learned source-phase path. A conditional unit rotation
preserves projected content norm outside its numerical clamp, not all original
signal information or all PAC coefficients.

Saved seed0 predictions locate the CHB-MIT gain more narrowly. Historical PR-AUC
on chb23 is .509141→.807399 for two→three axes, but on chb24 it is
.669842→.624787. The same signs hold on the natural-grid subset
(.284291→.604144 and .476238→.409370). These subject effects are not an additive
decomposition of pooled PR-AUC. All 31 record-label hashes and both prediction
label arrays were checked; independent threshold integration reproduces metrics
within 2.22e-16. The pooled natural-grid axis advantage remains .081705, but the
historical validation ranking favors two-axis in both shared seeds.

The supported account is useful frequency-resolved content with a compatible
encoder, plus task-dependent phase guidance. Accurate PAC estimation has not
been identified as the clinical cause. The earlier original-token PAC readout
used a TUEV two-axis checkpoint, not this CHB-MIT three-axis checkpoint. Keep
measurement, token recovery and clinical claims bound to their actual models.

Evidence and full scope: external
`maintenance/legacy-triaxial-attribution-20261105/{report.md,contrasts.json,
prediction-diagnostics.json,review/EXPERIMENT_AUDIT.md}`. The independent review
passes arithmetic but rates causal attribution provisional. No training or
paper edit was performed for this audit.

### Result entry

After the checks pass, enter the three metrics into the matching dataset sheet
of the existing tracking workbook and update its overview state. Leave
pending/failed/unverified cells blank. Back up the workbook, check dataset/model/
setting headers and read back the saved cells. Retain the audit scope, source,
checkpoint hash and comparison limitations with the result record.

A completed process is not approval by itself. Preserve checkpoints, native
outputs and validation histories, including negative results. Do not relabel
partial or time-budget-truncated training as a completed native budget.

## Source and runtime records

Current code belongs here; execution receipts, result collection, plotting,
workbook operations and scheduler incidents belong with external run records.
Authoritative runtime inventories are in [environments/](../environments/README.md).
Use `python -m pytest -q` and `python -m tests.check_preprocessing` for CPU checks;
GPU and full-dataset checks require the corresponding compute allocation.

Existing executions retain their immutable source releases. A changed
source/config/data/weight binding requires a new matching smoke before formal
execution. Copy current source, restore pinned vendors and use external data and
output roots when creating a release; an old smoke cannot approve changed files.

## Component validation record

The optional first-harmonic `measured_pac` frontend has its own
[validation record](../review/PAC_COMPONENT_VALIDATION_20261028.json): raw-signal
recovery, selective cross-band pairing and one paired seed-0 TUEV experiment.
Clinical training used `b8ddd17`; full-test replay used `e2cf787`. Its result,
source hashes and checkpoint bindings are separate from the dual-harmonic CFC
operator and later tokenizer integrations.

The JSON record gives evidence archive names, SHA-256 values and path conventions.
Full predictions, validation histories and selected checkpoints remain outside
source control. Operator-only validation records are indexed in the
[CFC guide](CFC_OPERATOR.md#validation-and-reproduction).

An identity control disables the complete guidance branch, including its phase
projection. Fixed-checkpoint coefficient interventions measure reliance on a
trained construction; they do not replace retrained mechanism controls.
