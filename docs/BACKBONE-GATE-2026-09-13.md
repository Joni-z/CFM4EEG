# RMSNorm and GEGLU backbone screen

Status: all four seed-0 jobs 417388–417391 completed 20 epochs normally
with Slurm exit 0:0. The predeclared validation-metric gate passed. Selected
checkpoint class review and a bounded second-seed replication follow below.
The joint augmentation screen did not deliver a
TUEV kappa gain, so its automatic second-seed expansion remains held.

The next hypothesis concerns the encoder, while retaining the coupling/local
token construction. REVE reports an EEG ablation favoring RMSNorm plus GEGLU
over LayerNorm/GELU combinations on its tested tasks; this motivates a transfer
check, not an expectation that its gains will reproduce here. See
[REVE v1, Appendix D, Table 20](https://arxiv.org/html/2510.21585v1#A4.T20).

`model_kwargs.block_variant: rms_geglu` replaces the four pre-normalizations
in each tri-axial block with RMSNorm (epsilon 1e-5), and the 2D GELU FFN with
GEGLU at hidden width ceil(4D/3). At D=192, both FFNs have the same number of
matrix weights; normalization and bias counts differ slightly. The tokenizer,
positions, axis attentions, width, depth and final readout are unchanged.
This is one combined backbone change, not separate causal ablations of norm
and activation. It is not claimed as the paper's novel contribution.

The default `legacy` branch preserves the original constructor order, state
keys and arithmetic. Unknown variants fail explicitly. Scratch configurations
are used; compatibility with old pretraining checkpoints is not established.

## Verification before training

`smoke/smoke_rms_geglu.py` runs only through `slurm/smoke_gpu.slurm`. It checks
the actual full model against committed legacy source, exact initialization
and evaluation logits, and bounded float32 optimizer-step agreement including
an old-versus-old repeat. It also checks that the new module is reached through
the real builder, handles flat inputs, reloads a strict checkpoint, and has
finite nonzero normalization/FFN gradients on real training batches.

The first smoke, 417375, exited after 46 seconds at an overly strict bitwise
optimizer check: one of 3200 elements differed by 5.82e-11. Its log is retained.
Initialization and forward equality remain bitwise requirements; the retry
records both legacy self-repeat and new-legacy optimizer differences with
rtol 1e-6 and atol 1e-8. No production training setting was changed for this
verification. Retry 417377 passed the functional checks but failed the budget
check because a stochastic frequency-augmentation path first compiled after
the initial two batches. Its complete receipt is retained rather than dropped.

Only training arrays are opened by this combined smoke. Timing excludes two
warmup steps after explicitly warming every augmentation path, and each
projected training duration must fit under 80% of its
unchanged cap. The projection excludes data loading, evaluation and saving;
the receipt records source/config hashes and the actual hardware/runtime.

Smoke 417383 completed successfully in 1:26 on mi2101x. Initialization and
evaluation remained bit-exact; both the old-versus-old and new-legacy optimizer
comparisons had maximum absolute differences of 5.82e-11. All four real-batch
cases passed. TUEV legacy/new projected training times are 3.693/3.982 hours;
Sleep-EDF times are 2.662/2.851 hours. The new TUEV model has 3,612,886
parameters versus 3,616,726, with peak allocated memory 12.06 GiB versus
10.25 GiB. Timing includes all twelve post-warmup observations, with the same
observed augmentation sequence for each pair. The receipt is
`results/audits/rms-geglu-smoke-20260913.json`. The single-configuration smoke
also warms every augmentation path before its existing budget check.

## Bounded comparison

Four admitted seed-0 configurations are under `configs/backbone_gate/`:
TUEV and Sleep-EDF, each with legacy and RMSNorm/GEGLU. Both arms use joint
128/32/32 coupling/band/broadband content and the same augmentation list.
Within each corpus the only substantive change is the backbone variant.
These distinct run identities do not reuse the held augmentation-only plan.

Admission source is `3049b8375951a963293f4284b4128d85e0beb356`.
The process and individual-smoke receipts are recorded in
`results/audits/rms-geglu-admission-20260913.json`:

| Corpus / arm | Job | Node | Trainer PID |
| --- | --- | --- | --- |
| TUEV legacy | 417388 | k006-004-v3 | 1749039 |
| TUEV RMSNorm/GEGLU | 417389 | k006-004-v4 | 769216 |
| Sleep-EDF legacy | 417390 | k006-004-v5 | 3415815 |
| Sleep-EDF RMSNorm/GEGLU | 417391 | k006-004-v6 | 1734259 |

All four use CPython 3.9.21, Torch 2.7.1+rocm6.3, ROCm 6.3 and the
rocBLAS override. VM v6 has a different OS Python build; byte-identical OS
images or interpreter executables are not claimed. This receipt establishes
successful admission, not model performance.

TUEV keeps 20 epochs and a five-hour training cap within a six-hour allocation;
Sleep-EDF keeps 20 epochs and a four-hour cap within a five-hour allocation.
Use one mi2101x node per configuration, with a further real-batch budget smoke
before each trainer starts. Maximum requested wall time totals 22 single-card
node hours, not a verified conversion to the site's balance units. Do not
shorten epochs or extend caps to force a result into the complete-run table.

Before reading any new training result, the budget-advancement rule is fixed:
all arms must complete normally; TUEV selected-validation kappa must improve
by at least .02 with no balanced-accuracy decline, and Sleep-EDF kappa and
balanced accuracy must not decline. Class recalls at the same selected
checkpoint require explicit review. A failure prevents automatic expansion,
not permanent elimination of a model family based on one seed.

If this screen is promising, a second matched seed and further corpora are
required. It cannot establish a final candidate on ten-plus datasets, causal
physiology, a pretraining benefit or SOTA. No new pretraining is admitted.


## Completed seed-0 gate and replication

The authoritative completion receipt is
`results/audits/rms-geglu-completion-20260913.json`. At their kappa-selected
checkpoints, TUEV validation kappa improves from 0.555408 to 0.613908 and
balanced accuracy from 0.526702 to 0.625214. Sleep-EDF validation kappa improves
from 0.685853 to 0.692185 and balanced accuracy from 0.671420 to 0.683072.
All four training runs completed 20 epochs within their original caps,
using 14.244 training hours in total across single-card nodes.

The automatically reported test outcomes are retained, not used to change
the predeclared validation rule. TUEV test kappa improves by 0.086599, whereas
Sleep-EDF test kappa declines by 0.030120. This validation/test disagreement
precludes describing the package as an established generalization improvement.
Neither a final candidate nor a cross-protocol SOTA result is established.

Validation-only job 417667 reopens the exact selected checkpoints to check
class recalls and reproduce the stored selection metrics. It performs no
training and does not open the test split.

The next comparison is fixed in
`results/audits/rms-geglu-seed1-plan-20260913.json`: three new seed-1 runs,
namely TUEV RMSNorm/GEGLU and both Sleep-EDF arms. The completed TUEV legacy
control `tuev-factorized_f4_aug_confirm:1` is reused under its original identity.
Its substantive configuration, cohort and loss match; the shorter old cap was
nonbinding. Source review confirms unchanged PAC frontend, augmentation,
data and loss, with the later trainer change only recording selected metrics;
legacy backbone equivalence was already checked on GPU. The old control used
mi2104x and is explicitly historical: simultaneous or byte-identical training
runtime is not claimed. No result is renamed or copied into a missing seed cell.

Maximum requested replication wall time is 16 single-card node hours. Keep
20 epochs and the same five/four-hour training caps. Individual real-batch
smokes must pass before trainers start. Second-seed validation kappa and
balanced accuracy must not decline on either corpus; the two-seed mean TUEV
kappa gain must remain at least 0.02, with explicit class review. Failure holds
automatic expansion rather than permanently eliminating a family. Further
corpora, a third seed and pretraining require a new evidence-based review.

The selected-checkpoint review passed for all five checkpoints, including the
historical TUEV seed-1 control. Its receipt is
`results/audits/rms-geglu-selected-validation-20260913.json`. TUEV improvement
is concentrated in GPED (12/291 to 242/291 correct); SPSW remains weak
(0/119 to 4/119), while PLED and EYEM recall decline. No class newly falls to
zero recall. This supports only bounded replication, with particular attention
to whether the gain survives the stronger historical seed-1 control.

Second-seed admissions are now running: TUEV RMSNorm/GEGLU 417675 on
k006-004-v4, Sleep-EDF legacy 417676 on k006-004-v5 and Sleep-EDF
RMSNorm/GEGLU 417677 on k006-004-v6. All individual smokes passed and actual
UID/configuration/seed/cgroup/GPU-selector checks verified the trainers. The
admission source is `05c1b58b1af8b418e3d74c281e4890056ea3c34c`; see
`results/audits/rms-geglu-seed1-admission-20260913.json`. Pure training
projections are 3.990, 2.746 and 2.889 hours, excluding I/O and evaluation.

Initial attempts 417671–417673 exited before training because the old wrapper
hardcoded seed 0; output guards preserved existing results. The wrapper now
reads the YAML seed for both preflight and training, matching the smoke. Six
local regression tests passed. The failed attempts and fix are retained in
`results/audits/rms-geglu-seed1-launch-fix-20260913.json`. No model or training
recipe changed during this correction.
