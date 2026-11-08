# CHB-MIT score audit: historical CfFT and three-axis E+W

2026-11-08. **Disposition: qualified use, not an unconditional benchmark pass.**
The retained predictions reproduce the recorded scores. No evidence was found
in the inspected paths of labels being supplied as model inputs, fabricated
ground truth, or score self-normalization. The enriched and natural-grid
evaluations remain different tasks, and the available natural-grid scores
below are checkpoint rescoring rather than completed new-protocol training.

## Bound model identities and scores

All numbers are trapezoidal PR-AUC, not Cohen's kappa or average precision.

| Model | Seed | Original enriched result | Natural-grid subset rescore |
|---|---:|---:|---:|
| Historical rotating CfFT, two-axis | 0 | 0.6046553742 | 0.4038136797 |
| Historical rotating Quad-Range, three-axis | 0 | 0.6969134781 | 0.4855182416 |
| Unrotated three-axis E+W | 2 | 0.6994134955 | 0.4882846033 |

These are different configurations. The two historical rotating models also
have recorded seed-2 enriched results of 0.6237556105 and 0.7098068743,
respectively; the table does not silently substitute those seeds. E-only is
another configuration: its 0.7436105115 / 0.5346450541 results must not be
assigned to E+W.

The historical rotating models retain float16 logits. Replaying those stored
logits gives enriched scores 0.6046248136 / 0.6968987246, within approximately
1e-5 to 3e-5 of the original float32 result. Their natural-grid scores use the
same stored predictions and a fixed row map. The original result records
remain unchanged.

## What was verified

- The corrected train/validation/test labels exactly match the mapped old
  labels. Training retains 316,205 windows and 2,883 positives. Validation
  changes from 21,184 / 150 positives to 21,094 / 60; test changes from
  17,386 / 276 to 17,224 / 114. All removed validation/test rows are positive.
- The extra windows come from seizure-centered sampling. Their labels agree
  with the retained interval annotations in the reviewed construction; this
  does not make the enriched test representative of the natural window grid.
  Both prevalence and the positive examples being evaluated change.
- The reviewed subject/record allocation found no cross-split subject overlap
  or duplicate raw-record hash evidence. Existing numerical signal checks cover
  185 windows, not every signal value in the corrected arrays.
- The inspected trainers select checkpoints on validation PR-AUC. The model
  receives X, while the clinical labels enter the loss and metric calculation.
  Evaluation uses eval mode, restores the selected weights, and saves test
  predictions.
- E+W's final checkpoint and the earlier snapshot have the same SHA256:
  `8bc51f9e14c13e45c28ebf1c37004ce256b94942ea35cabfda592f279d79e31f`.
  Its selected validation tag is `epoch 2 step 6200`, index 130, matching the
  final validation argmax. Its final saved predictions reproduce all reviewed
  test metrics within 1.11e-16. Thus 0.4882846033 belongs to the final selected
  checkpoint, despite originating in an evaluation taken during training.
- E+W's AP changes from 0.7017071444 to 0.4941559149. The PR-AUC drop is not
  explained by renaming AP or by trapezoidal interpolation alone.

The fixed natural-grid subset preserves prediction identity and isolates the
test-composition change. It does not reselect checkpoints using the corrected
validation set or run the model again on every corrected signal array.

## Limits on interpretation

1. **Enriched scores are conditional on that declared task.** They can be kept
   and compared with results on the same windows, with training adaptations
   disclosed. They cannot be ranked against natural-grid scores or published
   results using a different cohort and metric convention.
2. **Full corrected-protocol training is not established.** The inspected
   original CfFT corrected-data run is operator-stopped, has a checkpoint, and
   has no final result/test artifact. The E+W model above trained and selected
   on the enriched protocol. Neither rescore establishes a completed pipeline
   trained and selected under corrected validation.
3. **Reporting and adaptive development need qualification.** Historical
   `epoch0_peak_check` passed test metrics into a diagnostic rejection rule
   (`test AUROC <= 0.51`). Using that rule to omit completed low-score runs
   would bias a summary. This audit did not establish which rows were actually
   omitted, and the audited high-score runs do not trigger that threshold.
   E+W also had an interim test evaluation. Validation-based checkpoint code
   does not establish that later research choices were blind to test feedback.
4. **The evidence is limited in size.** Test data come from two subjects, and
   the E+W comparison shown has one seed. Related windows are not independent
   patient replicates. Neither broad stability nor a causal benefit of the
   encoder, rotation or paired content follows from this table alone.

The strongest justified conclusion is that these scores are traceable and
numerically reproducible within their recorded protocols. They do not support
a claim that these models lead the corrected natural-grid benchmark.

## Evidence and review scope

- [Numerical replay and label-map checks](../experiments/audits/chbmit-20261108/numeric-replay.json)
- [Fresh review and input hashes](../experiments/audits/chbmit-20261108/fresh-review.json)
- [Current corrected-control artifact status](../experiments/audits/chbmit-20261108/natural-original-control-status.json)
- [Existing content-control records](DEMODULATION_CONTROLS.md)
- [All CHB-MIT experiment records](../experiments/datasets/chbmit.md)

The semantic review used a fresh native reviewer context, requested as
`gpt-6-astra / ultra`. It is a same-family provisional review; the tool did not
return independent backend identity attestation. Deterministic file/hash and
prediction checks are reported separately from that semantic judgment.

No training, GPU inference, checkpoint reselection, manuscript edits or changes
to original result numbers were performed in this audit. Raw snapshots,
mapping arrays and source comparisons are retained locally under
`.local/workspace/maintenance/chbmit-four-cells-audit-20261108/`.
