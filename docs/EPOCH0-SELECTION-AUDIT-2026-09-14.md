# The headline TUEV number is an epoch-0 checkpoint

Found 2026-09-14 while verifying the rot2 residual gate, not by looking for it. It changes what the
D axis is about and it invalidates half of that gate. Read this before acting on either.

## The finding

Every seed of the headline TUEV coupling-only run selects the checkpoint from the FIRST evaluation.
`verdict.status = "peaked-first-eval"`, `peak_index = 0` of 20 evaluations, one evaluation per epoch.

| run | seed | status | peak / evals | test kappa |
|---|---|---|---|---|
| tuev-paclock_rot2 | 0 | peaked-first-eval | 0 / 20 | 0.7352 |
| tuev-paclock_rot2 | 1 | peaked-first-eval | 0 / 20 | 0.7475 |
| tuev-paclock_rot2 | 2 | peaked-first-eval | 0 / 20 | 0.7156 |
| tuev-rot2_residual_control | 0 | peaked-first-eval | 0 / 20 | 0.7222 |
| tuev-rot2_residual_band | 0 | peaked-first-eval | 0 / 20 | 0.7283 |

TUEV 0.7328 +- 0.0161 is therefore the score of a model trained for one epoch. Validation decays
monotonically for the following nineteen, and checkpoint selection throws them away. The number is
honest -- selection is on validation and the test set is untouched -- but the recipe is not training.

**CORRECTION, 2026-09-15.** This section originally carried a table of early-peaking rates by corpus
and concluded that the corpora preferring coupling-only are the corpora that overfit instantly. That
table was an artifact of evaluation cadence and has been removed. Corpora differ in how often they
evaluate and whether patience fires: TUEV, CAUEEG, ISRUC and IIIC evaluate once per epoch for 20
evaluations and run to the epoch cap, while TUSZ and CHB-MIT evaluate every few hundred steps for 100
to 200 evaluations and stop on patience after 2 to 4 epochs. A corpus with 200 evaluations can almost
never record `peak_index <= 1`, so the old ranking measured cadence, not overfitting.

Normalised as median peak position over the schedule, and in epochs:

| corpus | median evals | median epochs run | median peak, % of schedule | median peak, epochs |
|---|---|---|---|---|
| TUEP | 21 | 6 | 9% | 1.00 |
| TUEV | 20 | 20 | 12% | 3.00 |
| CAUEEG | 20 | 20 | 21% | 4.00 |
| IIIC | 20 | 20 | 32% | 6.00 |
| Sleep-EDF | 20 | 20 | 32% | 7.00 |
| ISRUC | 20 | 20 | 44% | 10.00 |
| TUSZ | 104 | 3 | 62% | 1.75 |
| CHB-MIT | 100 | 4 | 71% | 2.85 |

In epoch terms every corpus peaks within the first few epochs, and TUSZ at 1.75 peaks *earlier* than
TUEV at 3.00. The real difference is that patience ends the TUSZ and CHB-MIT runs near their peak,
while TUEV and CAUEEG keep training for another seventeen epochs and discard the work.

What survives from the original finding is narrower and still true: the three headline
`tuev-paclock_rot2` seeds peak at evaluation 0, that is after one epoch, against a median of three
epochs for TUEV runs generally. Our specific configuration peaks earlier than a typical TUEV run.
The cross-corpus claim does not survive and no conclusion should be built on it.

## Why it reframes the central tension

**RETRACTED 2026-09-15.** This section argued that the corpora preferring coupling-only are the
corpora that overfit instantly. It rested on the cross-corpus table corrected above and does not
survive normalisation. There is no evidence here that the mixture question and the regularisation
question are confounded across corpora.

## The suspect

`configs/coupling_only/tuev_paclock_rot2.yaml`: `lr 1e-4`, **`weight_decay 1.0e-05`**, cosine, dropout
0.2, label smoothing 0.1, 20 epochs, no augmentation. CBraMod and LaBraM both fine-tune at
`weight_decay 5e-2` -- three orders of magnitude more. `train.py` passes the field straight to AdamW
for our model (no layer decay, no multi-lr), so this is a one-field change.

## What is running

Two arms, one lever each, both on TUEV because that is where the pathology is clearest at 44 percent
and cheapest to measure at 2h50 per run.

| arm | change | job |
|---|---|---|
| tuev_rot2_wd05 | weight_decay 1e-5 -> 5e-2 | 419061 |
| tuev_rot2_augsafe | jitter and channel mask, the coupling-preserving subset | 419050 |

Four further arms were submitted and cancelled within half an hour, before they could consume a node
for their full length. They were designed before this audit and are dominated by the two above:

| cancelled | why |
|---|---|
| tuev_rot2_augfull | adds flip and frequency mask, the transforms already measured as coupling-hostile. If augsafe fails, augfull fails worse; if augsafe passes, augfull is not needed. |
| tuev_rot2_wd01 | a weaker dose of the wd05 lever. It only becomes informative if wd05 moves the peak but costs kappa, which is a sequential question, not a parallel one. |
| caueeg_rot2_augfull, caueeg_rot2_augsafe | CAUEEG is 32 percent early-peaking against 44 percent for TUEV, so it is a muddier readout, and at 7h per run it costs 2.5x more. Find the lever on TUEV first. |

The guard corpus is deliberately NOT running yet. Once a lever moves the TUEV peak, it has to be shown
to cost nothing on a corpus that already trains properly. That is CHB-MIT at 8 percent or ISRUC at
0 percent, both with controls on disk, and it is the next round rather than this one.

Control, already on disk at three seeds, not re-run: `tuev-paclock_rot2` 0.7328 +- 0.0161,
`caueeg-paclock_rot2` 0.4900 +- 0.0344.

## Reading the result

The primary readout is **`verdict.peak_index`, not the test score**. An arm that moves the TUEV peak
off evaluation 0 has fixed the recipe, and its test score is then a real 20-epoch number that can be
compared with anything. An arm that leaves the peak at 0 has not, whatever its test score does.

Predeclared, extending `docs/D-AXIS-GATE-2026-09-14.md`:
- ADOPT the arm as the global recipe if it moves TUEV `peak_index` to >= 3 AND holds test kappa
  within 0.02 of 0.7328 AND does not lower CAUEEG.
- If an arm moves the peak but costs more than 0.02 of TUEV kappa, that is still the honest recipe and
  the reported number goes down. Say so rather than keeping the epoch-0 number.
- Seed 0 promotes to a paired seed-1/2 replication only.

## Consequence for the rot2 residual gate

`docs/ROT2-RESIDUAL-GATE-2026-09-14.md` rejected the waveform residual partly on TUEV balanced
accuracy -0.0178. Both TUEV arms of that screen selected epoch-0 checkpoints, so that clause compared
two one-epoch models and the difference sits inside the 0.016 seed noise. On test metrics the residual
arm in fact won all four comparisons it was given:

| | control | residual | delta |
|---|---|---|---|
| TUEV kappa | 0.7222 | 0.7283 | +0.0061 |
| TUEV balanced acc | 0.5590 | 0.5722 | +0.0132 |
| Sleep-EDF kappa | 0.6403 | 0.6426 | +0.0023 |
| Sleep-EDF balanced acc | 0.5741 | 0.6306 | +0.0565 |

Sleep-EDF trains properly (peak at eval 6 and 7 respectively) so that half of the gate is sound, and
there the residual missed the +0.01 validation-kappa threshold while gaining +0.057 balanced accuracy.
The recommendation is not to overturn the gate on test numbers -- the gate was right to be on
validation. It is that the TUEV half should be re-run once a recipe exists that trains past epoch 1,
and that the residual should not be recorded as closed on the strength of it.
