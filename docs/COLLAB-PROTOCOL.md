# Two-agent exploration protocol (2026-09-14)

Two agents work this repo on the same cluster account: **A** (Claude, checkout `PACLock`) and
**F** (Codex, checkout `PACLock-factorized`, branch `codex/factorized-tokenizer-20260913`).
`PACLock/runs` is the superset of both agents' results (1029 vs 970 dirs, none unique to F), so
results pool automatically; nothing else does. This file is the missing part.

We are in EXPLORATION. There is no final model. The split below is by SEARCH AXIS, not by role,
because the failure this protocol exists to prevent already happened once: A built self-inclusive
coupling and F built local_residual in the same week, for the same problem.

## The problem both agents are solving
Event-morphology corpora want coupling-only tokens (TUEV: coupling-only 0.733 +- 0.016 vs duplex
0.690 +- 0.036 vs waveform-only 0.536 +- 0.041, 3 seeds each). Seizure/artifact/sleep corpora want
within-band waveform content (CHB-MIT: duplex 0.698 +- 0.055 vs coupling-only 0.506 +- 0.086).
Every fixed mixture loses one side. A model that resolves this is the deliverable.

## Axis ownership
| axis | question | owner |
|---|---|---|
| A. mixing | how and where waveform content re-enters a coupling-first grid: residual, gate, routing | **F** (local_residual in flight) |
| B. coupling geometry | self-inclusive diagonal Z_jj, rotation vs product, band count | unowned — do not start while axis A is live, it solves the same problem |
| C. estimation scale | multi-scale `pac_patch_len` | unowned, LOW priority: both multi-scale runs underperformed their single-scale siblings (TUEV [50,100,200] 0.692 vs 0.733; ISRUC [200,600,1200] 0.690 vs 0.702), confounded by product-vs-rotation. The documented "halving the window buys +0.050" gain is already banked — every config is at 50. |
| D. recipe x architecture | is augmentation admissible in a single shipped model | **A** (screen running, `docs/D-AXIS-GATE-2026-09-14.md`) |
| E. initialisation | pretrained init (ptS) on weak columns | unowned. Only 3-seed-vs-3-seed mover of a weak column in the archive: Siena 0.456 +- 0.041 vs 0.170 +- 0.086. Ten of twelve cells are n=1. |

Claim an unowned axis by adding a row to `docs/CANDIDATE-BOARD.md` before submitting anything.

## One screen protocol, so results are comparable across agents
Today the two agents measure differently — F gates on validation-selected kappa plus balanced accuracy
on TUEV/Sleep-EDF; A has been reading test means on four corpora. The same candidate can pass one and
fail the other. From now on every screen:
1. Writes a dated `docs/<NAME>-GATE-<date>.md` BEFORE submitting, containing: hypothesis, the single
   config field that changes, the control (with its seed count, from disk — never re-run a control that
   exists), predeclared numeric thresholds, and the budget cap.
2. Runs seed 0 only, at most 4 runs, on the four screen corpora or a justified subset:
   TUEV (needs coupling-only) | CHB-MIT (needs waveform) | TUSZ (our strongest column) | ISRUC or
   Sleep-EDF (a weak column).
3. Selects checkpoints on validation. Test numbers are descriptive.
4. Promotes a pass to a paired seed-1/2 replication ONLY. Never straight to a 12-corpus campaign.
5. Records the outcome in the same file, including failures, and updates the board.

## Sharing what is expensive, not what is cheap
- **Numbers come from `scripts/gen_tables.py` output (`results/cells_*.json`, `results/tables/*.tex`).**
  Whoever lands runs regenerates and commits. Neither agent re-aggregates `runs/*/seed*/result.json`
  from scratch — A burned a full workflow doing exactly that for a table F already had.
- **Verify each other's conclusions; do not re-derive them.** Reading a claim and checking its numbers
  is cheap; repeating the analysis is not. Every gate outcome and every audit gets one adversarial
  read by the other agent before it reaches the paper.
- **Known-wrong facts are corrected in place, with the date.** Recent examples: TUAR's best baseline is
  ffcl 0.7025 +- 0.0122, not cbramod_pretrained; the CBraMod additive CHB-MIT delta is -0.028 paired
  over 3 seeds, not +0.124 (seed 0 only); the b2 pretraining has been dead since 2026-09-12
  (TIMEOUT, then NODE_FAIL, then cancelled at 140k/219k).

## Compute discipline — the shared allocation is the binding constraint
Balance 2026-09-14: 1836 of 2250 used, ~414 left. Recent peak burn 62 units/day exhausts it around
2026-09-20, before the full-paper deadline.
- Job names are prefixed by owner: `A_*` and `F_*`. Check `squeue -u yifanwang` before submitting.
- Partitions our allocation may use: **mi2101x** (1 GPU MI210, 16 cores, billing weight 0.01/node-hour) and
  **mi2104x** (4 GPU, 128 cores, 0.04/node-hour). `mi3501x` (MI355X, weight 0.0125, 7 nodes idle) looks
  attractive and is NOT authorized for this account — the submit filter rejects it. Checked 2026-09-14.
- A screen is at most 4 runs. Prefer mi2101x / mi3501x single-GPU nodes (16-24 cores) over a packed
  mi2104x node (128 cores) — roughly half the billing for the same four runs, and 20 nodes sit idle.
- A cancelled job is still billed for the time it ran; a PENDING job cancelled before it starts is free.
- No seed campaigns above 6 runs without the principal investigator saying so.

## Start here

`docs/N-BANDS-FINDING-2026-09-15.md` is the current state of the exploration, written to be read by
the other agent without opening anything else. It supersedes the cross-corpus part of
`docs/EPOCH0-SELECTION-AUDIT-2026-09-14.md`, which is retracted in its section 5.

## Paper
One agent holds the pen at a time; the other proposes edits through this file. Push only through
`./push_overleaf.sh` on `mbp` (it refuses to push if the build fails). `abstract.tex` in that
directory is an orphan — the live abstract is inline in `iclr2027_conference.tex`.
The author block in that file carries real names: never attach the source as supplementary material.
