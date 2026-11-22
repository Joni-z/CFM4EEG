# Quad-Range host replacement — 2026-09-16

## Scope

User authorized BIOT and CBraMod host replacements, excluding CHB-MIT.
User clarified: ONLY TUEV and IIIC, four runs total. Broad-panel jobs
422025 and 422026 were cancelled immediately upon clarification. Their partial
logs are retained; no broad-panel result enters this experiment.
All core arms use seed 0. No new native or own-phase host arms are launched.
This evaluates encoder architecture reuse under supervised scratch training,
not pretrained-weight compatibility or pretrained tokenizer transfer.

## Core protocol

Four runs: BIOT/CBraMod × TUEV/IIIC. Exact legacy host recipes and data roots
are inherited from configs/quad_hosts in CFM4EEG-transfer-pretrain-20260915.
Only the frontend frequency initialization is changed to [0.5,75] Hz.
The range frontend has 16 bands, width128, patch50, quadrature content.
The 16 bands are concatenated, four successive patches averaged to each
native host window and projected to the native feature width.
BIOT retains hop100; CBraMod retains hop200. All native encoder, positional
modules and classifier initialized weights are verified equal under seed0.
No pretrained weights are loaded. Preserve the host optimizer and loss.

## Checks before training

GPU-only real native-loader batch checks, token-grid and native-tail parity,
nonzero waveform and encoder gradients, finite loss, actual Range cutoffs,
and checkpoint reload into a newly constructed model. Smoke training weights
are discarded. Existing output artifacts forbid a new launch.

## Interpretation

Historical native results can be reused only with compatible input cohorts,
preprocessing and training protocols; record seed differences explicitly.
Do not label different-seed comparisons paired. No generic SOTA claim.
The broad panel needs per-dataset sampling rate and sequence-head checks;
old adapter hardcoded 200Hz and flat samples, so copying its YAMLs blindly
would not be valid for Sleep-EDF or sequence ISRUC.

## Execution update

Core job 422019 retains CBraMod TUEV/IIIC. BIOT constructor argument forwarding
failed the initial smoke before training and was corrected in e5dd28b.
BIOT retries run on independent single-card mi2101x allocations with 12h
scheduler time and 10.5h internal training caps (previous cap 22.5h); epoch
limits and early stopping are unchanged. If a run hits its runtime budget,
it is explicitly budget-limited, not a completed epoch schedule.
No native control or own-phase host arm is added. All four are seed0.
