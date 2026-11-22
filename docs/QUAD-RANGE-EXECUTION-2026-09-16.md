# Quad-Range: frozen candidate and launch ledger (2026-09-16)

## Scope and seed assignment

User selected Quad-Range as the candidate. No A128/old-Quad training is launched.
Main seeds are explicitly assigned before these runs: CHB-MIT 0, TUSZ 0,
IIIC 0, TUAB 2, TUAR 2, ADFD 1, CAUEEG 2, TUEP 1.
Existing TUEV s0 and ongoing Sleep-EDF s1 remain separate recorded results.
These main runs are exploratory development evidence, not seed averages.
UPDATED by user: controls use seed 2; the full reference is reused from the main table (TUEV s0, CHB-MIT s0). Extra full s2 runs are cancelled. These comparisons are UNPAIRED and contain seed variation; do not label differences as isolated structural effects.

## Fixed design

16 bands, width 128, six tri-axial blocks, quadrature content (a+i*w)*u_hat.
Range initialization is [0.5,75] Hz at 200 Hz; Sleep-EDF uses [0.5,49] Hz.
Cutoffs remain learnable. Original task-specific training recipes are retained.
Complete config sources and destinations are in range-final-manifest-20260916.json.
No split, metric, checkpoint-selection, label or data-preprocessing changes.
Checkpoints are selected by validation; preserve best.pt and all result files.

## Main results (8 new runs)

| Dataset | Seed | Site | Initial status |
|---|---:|---|---|
| CHB-MIT | 0 | 145 | prepared |
| TUSZ | 0 | AMD | prepared |
| IIIC | 0 | AMD | prepared |
| TUAB | 2 | AMD | prepared |
| TUAR | 2 | 145 | prepared |
| ADFD | 1 | 145 | prepared |
| CAUEEG | 2 | 145 | prepared |
| TUEP | 1 | AMD | prepared |

## Mechanism comparisons (6 retained runs; 2 full-reference runs cancelled)

TUEV on AMD and CHB-MIT on 145: own, weight, envelope, seed 2. Full rows reuse main seed-0 results.

| Variant | Intervention | Matched properties |
|---|---|---|
| full | measured alpha and preferred-phase alignment | reference |
| own | use each band's own projected phase | same parameters, initial state, grid, recipe |
| weight | retain measured alpha; set eta=1 | same parameters, initial state, grid, recipe |
| envelope | fixed zero mask on projected waveform w | same allocated parameters and initialization; waveform parameters have zero gradient and are inactive |

The envelope control is not claimed to have the same effective trainable capacity.
The masking leaves the complex token grid and all other initialized weights identical.
Actual parameter counts and active waveform parameters are recorded by GPU contract.
No PAC-as-feature, shuffled-Z, pretraining or new host jobs are included in this launch.

## Admission and resource policy

- Isolated code checkout; original running Sleep-EDF is untouched.
- Contract verifies full-model compatibility with commit 837dcf9, matched state,
  own/weight/measured formulas, envelope masking, finite gradients, strict checkpoint reload.
- Every run uses actual-data smoke before fresh training; smoke weights are discarded.
- Timing excludes first steps and warms each augmentation path; over-budget checks fail closed.
- 145 checks compute applications across all users, takes shared locks and rechecks vacancy.
  Never evict the existing VLLM process or the ongoing Sleep-EDF run.
- AMD has two four-config packs: TUEV controls and main TUSZ/IIIC/TUAB/TUEP.
  All compute, including smoke, runs inside allocations. No login-node torch work.
- Stop markers: STOP_ALL_RANGE in this checkout, or STOP in an individual run directory.
- Every config has a unique output key; existing outputs prohibit launch.

## Interpretation

Report every completed paired result, including reversals and diagnostic warnings.
A single-seed improvement is descriptive and does not establish statistical reliability.
Intervals spanning zero do not establish equivalence. No outcome-dependent seed replacement.

## Live records

Per-run receipts: results/*-pipeline.json and results/shared145-*.json.
Per-run contracts/smokes: results/*-contract.json and results/*-smoke.json.
AMD pack receipts: logs/*-pack.json. Training outputs: runs/<name>/seedN/.
An updated submitted-job ledger is appended after scheduler confirmation.

## Launch and admission update

Code launch commit: 37cf9e2. AMD jobs 421933 (TUEV controls) and 421934 (main pack).
145 wrapper PIDs initially 569316..569323; per-run receipts are authoritative.
All 16 structural contracts passed. Actual-data timing caught budget overruns
before training for CHB-MIT, TUSZ and TUAB. No numerical/model failure occurred.
145 has NO scheduler time limit: CHB-MIT's inherited internal max_hours=22
was raised to 60 for all retained CHB-MIT variants, preserving epochs, early
stopping, optimizer, data and seed. The guard estimates ~33h for ALL 20 epochs;
early stopping may finish much sooner and is unchanged. No reduction of data.
The complete Range s2 runs are cancelled at user request; retain any partial logs.
MI350 submission was rejected: the submission filter caps this partition at 4h.
TUSZ/TUAB were instead submitted to Torch with a 48h scheduler limit and 46h
internal budget. Only runtime caps change; maximum epochs, optimizer, data,
seed, validation frequency and early stopping are retained. Existing historical
cap differences must not be omitted when comparing runs.

## Confirmed job ledger (latest check)

| Run | Site / job | State |
|---|---|---|
| CHB-MIT full s0 | 145 / wrapper 581582 | training |
| CHB-MIT own s2 | 145 / wrapper 581583 | training |
| CHB-MIT weight s2 | 145 / wrapper 581584 | training |
| CHB-MIT envelope s2 | 145 / wrapper 581585 | waiting for vacant GPU |
| TUAR full s2 | 145 / wrapper 569321 | training |
| ADFD full s1 | 145 / wrapper 569322 | training |
| CAUEEG full s2 | 145 / wrapper 569323 | training |
| TUEV own/weight/envelope s2 | AMD 421933 | all training |
| IIIC full s0, TUEP full s1 | AMD 421934 | both training |
| TUSZ full s0 | Torch 17894559 | pending QOSMaxGRESPerUser |
| TUAB full s2 | Torch 17894560 | pending QOSMaxGRESPerUser |
| TUEV full s2 | AMD 421933 individual process | operator stopped; preserve partial checkpoint |
| CHB-MIT full s2 | 145 | cancelled after budget smoke; no training |

There are 14 retained new runs: 11 training, 1 waiting on 145, 2 pending on
Torch at this snapshot. The existing Sleep-EDF s1 continues independently.
AMD packs have vacant slots after the full-s2 cancellation and large-dataset
reassignment; do not claim they remain fully utilized. Avoid disrupting the
remaining live runs merely to repack them. Future packs should group similar
measured durations and validate partition submission limits first.

Full s0 references and s2 controls are different-seed comparisons, not paired
structural-effect estimates. No paper numbers have been updated by this launch.
