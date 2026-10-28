# Experiment audit

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

Historical cleanup records are external, relative to the repository root:
`../history/repository-cleanup-20261022/` retains prior files and experiment
records; `../maintenance/repository-cleanup-20261022/` holds the independently
SHA-verified snapshot and path mapping. These are provenance records, not another
active implementation.

## Component validation record

[PAC_COMPONENT_VALIDATION_20261028.json](../review/PAC_COMPONENT_VALIDATION_20261028.json)
records the completed `measured_pac` measurement and tokenizer checks. It covers
raw-signal recovery, selective cross-band pairing and the full seed-0 TUEV pair.
The record belongs to this optional frontend; local-moment reference results
retain their existing model identity.

The complete evidence is outside Git in the workspace directory
`../maintenance/autoresearch-pac-20261028/`, relative to the checkout. It includes
`report.md`, per-case estimates, all attempted settings, raw predictions,
validation histories, selected checkpoints and figures. The verified mirror is
on 145. The portable snapshot is `component-validation-e2cf787.tar.gz`; its
SHA256 and paths within the archive are in the repository record.

Clinical training source `b8ddd17` and checkpoint-replay source `e2cf787` have
distinct roles. Both complete checkpoints were replayed on all 29,421 test
windows, with identical label order and zero logit differences. The clinical
archive's 69 evidence files were hashed individually; the full snapshot has
444 verified files. Source and configuration bindings are retained in each run's provenance.
The recorded configuration paths are portable templates; their environment-based
data roots differ from the executed absolute paths, whose hashes are recorded
separately.

This is an internal shared16 comparison, separate from native23 or published
scores. Identity disables the complete guidance branch, including its phase
projection. Fixed-checkpoint coefficient interventions diagnose reliance on the
measured values and remain separate from retrained mechanism controls.
