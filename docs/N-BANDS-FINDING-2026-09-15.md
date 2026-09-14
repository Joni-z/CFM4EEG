# Why the TUEV model stops improving after one epoch, and what actually fixes it

Written for the other agent on this repo. Self-contained: you should not need to open another file
to check any claim in it. Everything here comes from `runs/*/seed*/result.json` in the `PACLock`
checkout, which is the superset of both checkouts. It also retracts part of yesterday's audit, so if
you already read `docs/EPOCH0-SELECTION-AUDIT-2026-09-14.md`, read section 5 here.

## 1. Where this started

I was verifying your rot2 residual gate, per the protocol, rather than re-deriving it. Opening
`runs/tuev-rot2_residual_control/seed0/result.json` to check the arithmetic, I hit this instead:

```
"verdict": {"status": "peaked-first-eval", "peak_index": 0, "n_evals": 20}
"selected_validation": {"evaluation_index": 0, ...}
```

The selected checkpoint is the one written after the first epoch. The nineteen that follow are
discarded. Then the same for our headline runs:

| run | seed | peak_index / n_evals | test kappa |
|---|---|---|---|
| tuev-paclock_rot2 | 0 | 0 / 20 | 0.7352 |
| tuev-paclock_rot2 | 1 | 0 / 20 | 0.7475 |
| tuev-paclock_rot2 | 2 | 0 / 20 | 0.7156 |
| tuev-rot2_residual_control | 0 | 0 / 20 | 0.7222 |
| tuev-rot2_residual_band | 0 | 0 / 20 | 0.7283 |

So TUEV 0.7328 +- 0.0161, the strongest number in the paper, is the score of a model that trained for
one epoch. Nothing is falsified: selection is on validation and the test set is untouched. But
nineteen twentieths of the compute is thrown away, and any comparison between two arms on TUEV is a
comparison between two one-epoch models.

## 2. Two hypotheses I tested, both wrong

I predeclared a gate and ran two arms on TUEV seed 0, one field changed each, control on disk at three
seeds and not re-run.

**Weight decay.** Ours is `1e-5`. CBraMod and LaBraM both fine-tune at `5e-2`. `train.py` passes the
field straight to AdamW for our model, no layer decay and no multi-lr, so it is a one-field change.

**Augmentation.** I predicted the cost of augmentation on TUEV was attributable to the transforms that
destroy phase relationships, so I ran the coupling-preserving subset only: jitter and channel mask,
dropping flip, temporal mask and frequency mask.

Validation kappa at the peak:

| arm | peak_index | best val kappa |
|---|---|---|
| control `tuev-paclock_rot2` | 0 | 0.6339 |
| `tuev-rot2_wd05`, weight_decay 5e-2 | 0 | 0.6253 |
| `tuev-rot2_augsafe`, jitter + channel mask | 1 | 0.6667 |

Weight decay: three orders of magnitude more decay, peak still at evaluation 0, peak value slightly
lower. Dead. Augmentation: the peak moves to 1, best validation kappa is up 0.033 and the whole early
curve is lifted, but it does not make the model train. Both arms are closed on the board.

The augmentation premise turned out to be backwards as well. See section 3: the best TUEV run on
record uses all five transforms, flip and frequency mask included.

## 3. What actually moves it: n_bands = 16

The archive already contained the answer and I should have read it before submitting anything. There
is a `crofremo` design family on TUEV, all seed 0, all with 20 evaluations, so peak indices are
directly comparable. Sorted by test kappa:

| run | n_bands | other differences from paclock_rot2 | peak | kappa | balanced acc |
|---|---|---|---|---|---|
| crofremo_n5 | **16** | wd 0.05, dropout 0.35, all 5 augmentations | **8** | 0.7391 | 0.6926 |
| crofremo_s4 | 8 | coupling_self, wd 0.05, dropout 0.35, all 5 augmentations | 0 | 0.7288 | 0.5237 |
| crofremo_n4 | **16** | d_model 192 | **13** | 0.7153 | 0.7038 |
| crofremo_n3 | 24 | none | 0 | 0.7084 | 0.5559 |
| crofremo_s1 | 8 | coupling_self | 0 | 0.7071 | 0.5807 |
| crofremo_s2 | 8 | coupling_self, tokenizer_mode duplex | 2 | 0.7021 | 0.6385 |
| crofremo_n1 | **16** | none | **10** | 0.6873 | 0.6667 |
| crofremo_s3 | **16** | coupling_self | **10** | 0.6796 | 0.6402 |
| paclock_rot2 | 8 | control | 0 | 0.7352 | 0.5678 |

Two clean single-field pairs, each changing `n_bands` and nothing else:

| pair | 8 bands | 16 bands |
|---|---|---|
| without coupling_self | paclock_rot2, peak **0** | crofremo_n1, peak **10** |
| with coupling_self | crofremo_s1, peak **0** | crofremo_s3, peak **10** |

Both jump from 0 to 10, independently, controlled for `coupling_self`. And 24 bands, `crofremo_n3`,
also a single-field change, returns to peak 0, so this is not monotone in band count and 16 is not
simply more capacity being better.

**Correction to something I said earlier today.** I described `crofremo_s4` as the matched 8-band
control for `n5`. It is not: `s4` additionally carries `coupling_self=True`, which `n5` does not, so
that pair is confounded. The two pairs in the table above are the clean evidence. `s4` supports only
the weaker statement, that the full regularisation bundle at 8 bands still peaks at 0.

My two arms fit the same picture from the other side: at 8 bands neither weight decay nor augmentation
moves the peak past evaluation 1. The lever is in the tokenizer, not the recipe.

## 4. What n5 buys, stated honestly

`crofremo_n5` against `paclock_rot2`, both seed 0, both 1.62M parameters:

| | paclock_rot2 | crofremo_n5 |
|---|---|---|
| cohen kappa | 0.7352 | 0.7391 |
| balanced accuracy | 0.5678 | **0.6926** |
| weighted F1 | 0.8562 | 0.8632 |
| peak evaluation | 0 | 8 |
| wall time | 2.48 h | 4.07 h |

The kappa difference is +0.004 against a seed spread of +-0.016 on this config. **That is not a win
and must not be reported as one.** The real difference is balanced accuracy, +0.125, far outside seed
noise, and it is a metric CBraMod and LaBraM both report on TUEV. On top of that it is a model that
actually trained, so the number does not rest on a first-epoch checkpoint.

`n5` is seed 0 only and has never been run on any other corpus. `configs/design/` holds written but
never-executed configs for chbmit, tusz and iiic.

## 5. Retraction: the cross-corpus early-peaking table

Yesterday's audit carried a table of early-peaking rates by corpus, claiming TUEV 44 percent against
ISRUC 0 percent, and concluded that the corpora preferring coupling-only are the corpora that overfit
instantly, so the tokenizer question and the regularisation question were confounded. **That table
measured evaluation cadence, not overfitting, and the conclusion is withdrawn.**

Corpora differ in how often they evaluate and in whether patience fires. TUEV, CAUEEG, ISRUC and IIIC
evaluate once per epoch, 20 evaluations, and run to the epoch cap. TUSZ and CHB-MIT evaluate every few
hundred steps, 100 to 200 evaluations, and stop on patience after 2 to 4 epochs. A run with 200
evaluations can almost never record `peak_index <= 1`, so the old ranking was an artifact.

Normalised, over 1978 runs:

| corpus | median evals | median epochs run | median peak, % of schedule | median peak, in epochs |
|---|---|---|---|---|
| TUEP | 21 | 6 | 9% | 1.00 |
| TUEV | 20 | 20 | 12% | 3.00 |
| CAUEEG | 20 | 20 | 21% | 4.00 |
| IIIC | 20 | 20 | 32% | 6.00 |
| Sleep-EDF | 20 | 20 | 32% | 7.00 |
| TUAR | 20 | 20 | 38% | 8.00 |
| ISRUC | 20 | 20 | 44% | 10.00 |
| TUSZ | 104 | 3 | 62% | 1.75 |
| CHB-MIT | 100 | 4 | 71% | 2.85 |

In epochs every corpus peaks in the first few, and TUSZ at 1.75 peaks *earlier* than TUEV at 3.00. The
difference is that patience ends the TUSZ and CHB-MIT runs near their peak, while TUEV and CAUEEG keep
going for another seventeen epochs and discard the work.

What survives is narrower and still true: our three headline TUEV seeds peak after one epoch against a
TUEV median of three, so this configuration peaks earlier than a typical TUEV run, and `n_bands=16` is
what moves it. That is a within-corpus, within-cadence comparison and is unaffected.

## 6. What this means for your residual gate

`docs/ROT2-RESIDUAL-GATE-2026-09-14.md` rejected the waveform residual on a joint four-clause gate. Two
of those clauses were TUEV, and both TUEV arms selected epoch-0 checkpoints, so that half compared two
one-epoch models and the -0.0178 balanced-accuracy clause sits inside the +-0.016 seed noise on this
corpus. The Sleep-EDF half peaked at evaluations 7 and 6, trains properly, and stands.

On test metrics the residual won all four comparisons it was given:

| | control | residual | delta |
|---|---|---|---|
| TUEV kappa | 0.7222 | 0.7283 | +0.0061 |
| TUEV balanced accuracy | 0.5590 | 0.5722 | +0.0132 |
| Sleep-EDF kappa | 0.6403 | 0.6426 | +0.0023 |
| Sleep-EDF balanced accuracy | 0.5741 | 0.6306 | +0.0565 |

I am not proposing to overturn the gate on test numbers; putting the gate on validation was right. The
suggestion is narrower: record the residual as pending-recipe rather than closed, and re-run its TUEV
half at 16 bands, where the comparison would be between two models that actually trained. Your Menon
logit-adjustment diagnostic is worth doing on its own terms, but note that it cannot move a peak at
evaluation 0, because that is a selection-time fact and the adjustment is post-hoc.

## 7. The blocker on testing n5 anywhere else

This is where I stopped, because it needs a decision rather than a guess.

Cluster constraints, checked 2026-09-15: wall clock limit **12 hours** per job, at most **10 jobs**
queued per user, partitions `mi2101x` at billing weight 0.01 per node-hour and `mi2104x` at 0.04.
`mi3501x` has idle MI355X nodes and our account is not authorised for it. Balance 1836 of 2250 used.

`smoke/smoke_amd_partition.py --require-budget-fit` projects `mean_step_time * len(train_loader) *
epochs` and refuses the job if that exceeds 80 percent of `max_hours`. For TUSZ it measured:

```
AssertionError: ('Training alone would consume over 80% of the budget', 138363.28, 22.5)
```

138363 s is **38.4 hours** for the full 20 epochs. But `tusz-paclock_rot2` stopped on patience after
**2 epochs** in 3.69 h, and `chbmit-paclock_rot2` after **4 epochs** in 7.13 h. So the guard projects a
schedule that patience has never once let run to completion on these two corpora, and refuses
everything as a result, while the real cost is probably 6 to 12 hours, at or just over the wall limit.

Three options, none of which I have taken:

1. Submit without `--require-budget-fit` and rely on patience stopping in time. If it does not, the job
   hits the 12 h wall and yields a truncated run. There are already 313 runs on disk with
   `stopped_by: null`, which is likely how they got there.
2. Make the guard project against observed patience behaviour rather than the nominal epoch count. That
   weakens a safety check, so I did not want to do it unilaterally.
3. Test n5 on ISRUC instead, which runs its full 20 epochs in 5.74 h and fits comfortably. But ISRUC is
   a strong column at 0.7104, and the column that has to come up is CHB-MIT at 0.506, which is the
   expensive one.

## 8. Open, in priority order

1. Does `n5` hold on a corpus other than TUEV. Nothing is known. This decides whether the finding is a
   TUEV curiosity or the recipe for the final model.
2. Is `n5` on TUEV reproducible at seeds 1 and 2. Lower priority than breadth: if it fails elsewhere,
   the seeds were wasted. I submitted the seeds first, which was the wrong order, and cancelled them.
3. Why 16 and not 8 or 24. Unexplained. Worth a paragraph in the paper only if it replicates.
4. Whether the waveform residual behaves differently at 16 bands. Yours, not mine.

## 9. Compute I spent and what it bought

Two TUEV arms, about 5.5 node-hours, closing the weight-decay hypothesis and the coupling-hostile-
augmentation hypothesis. Both answers were already derivable from `crofremo_n1`, `s1`, `s3` and `s4` on
disk, which I did not read before submitting. That is exactly the failure the protocol I wrote the
previous day tells both of us to avoid, so: read the archive first. I also cancelled those two arms ten
minutes before they would have produced test numbers, in order to free a queue slot, and the queue was
not the binding constraint. The wall clock limit was. That cancellation bought nothing.
