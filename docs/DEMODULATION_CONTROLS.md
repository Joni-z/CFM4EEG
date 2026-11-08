# Frozen content recovery and matched controls

This study tests the unrotated 16-band, width-128, depth-6 reference. It
reuses its completed seed-2 checkpoints. No main-table or pretrained runs are
launched. All new downstream controls use the same seed, reader, optimizer,
augmentation, validation selection and stopping recipe as their reference.

## Method check

`scripts/experiments/demodulation_reconstruction.py` freezes the checkpoint,
extracts tokens before positional encoding, and fits shared linear readouts
for band-limited waveform and log1p envelope. The decoder uses 128 training
windows; ridge strength is selected on 32 validation windows. Evaluation uses
64 held-out test windows, four fixed electrodes, all 16 bands, and three fixed
interior patches per window. No clinical labels enter the readout.

Each projection maps 50 samples to 64 coordinates. Both trained projections
have rank 50, so a known-weight inverse is also reported. Near-perfect recovery
establishes preservation of these frontend inputs; it does not establish their
clinical value or show that another tokenizer discards them.

| Dataset | Waveform R² | log1p envelope R² | Waveform NRMSE |
|---|---:|---:|---:|
| TUEV | 0.9999998973 | 0.9999999999998 | 0.0003205 |
| CHB-MIT | 0.9999999824 | 0.9999999999997 | 0.0001326 |

The script records checkpoint and manifest hashes, exact sampled indices,
decoder selection errors, per-band test errors, projection ranks, and example
traces. Use `--no-plot` on the unchanged training environment and render saved
examples on a plotting host.

## Downstream questions

| Dataset | Setting | Question |
|---|---|---|
| TUEV | `content_view: waveform` | Does explicit envelope content help beyond filtered waveform? |
| TUEV | `content_view: envelope` | How much does envelope content explain without waveform? |
| CHB-MIT | `content_view: envelope` | Does keeping waveform complement the envelope view? |

The first batch used GPU0/1/4 plus the existing raw-CHB task on GPU2. Both
TUEV content arms and the raw-CHB task have completed; CHB-MIT envelope-only
also completed on November 7. The original bandmix process was stopped under the user's
four-card limit, with all existing artifacts retained. Its separately named
successor and the patch-mean-envelope control were subsequently authorized.

The two content controls fill **both** coordinate blocks with independent
projections of the same input type. They keep all projection parameters and
the reference's zero initialization of the second projection. The old
`waveform_content: false` zero-mask and `waveform_modulation: carrier`
initialization are not used for these controls.

Band mixing is a fixed dense orthogonal transform along the band axis, before
positional encoding. It preserves the token count, total squared norm and
invertibility. It changes the meaning of each row from a physical band to a
mixture. It does not test whether the three-axis encoder is universally best.

The default frontend retains its state keys and exact outputs against the
frozen reference source. New settings are accepted only with unrotated full
quadrature tokens and the attention frequency reader, without auxiliary loss.

## Completed downstream results

All rows below use seed 2. The three physical-band content settings use the
same 1,620,774 parameters, training recipe, initialization and 20-epoch budget;
only `content_view` changes. Both new arms completed all 20 epochs, selected by
validation kappa. They and the reference selected evaluation index 1. Scores
were recalculated from saved final test logits and labels, with zero numeric
difference for the new content arms.

| TUEV content | Kappa | Balanced accuracy | Weighted F1 |
|---|---:|---:|---:|
| Band-limited waveform in both coordinate blocks | 0.4983532624 | 0.5103231058 | 0.7348056416 |
| log1p envelope in both coordinate blocks | 0.6416508750 | 0.5770917221 | 0.8122308543 |
| Paired log1p envelope and waveform, reused reference | 0.7111908486 | 0.6335895541 | 0.8494470647 |

The shared TUEV test cohort contains 29,421 windows. These are one-seed
content comparisons, not estimates of across-seed variability. They support
complementarity in this setting; they do not isolate local pairing or prove
that envelope trajectories outperform patch-level amplitude summaries.

The separate CHB-MIT raw learned-row three-axis control also completed:
PR-AUC 0.6418777629, AP 0.6424818500, AUROC 0.9277181325, balanced accuracy
0.5380142556. The reused paired reference gives PR-AUC 0.6994134955. Both use
the 17,386-window enriched test cohort. This raw/full comparison changes the
whole frontend, not just envelope content. The raw model stopped by patience
after six epochs. Final prediction files and best checkpoints are preserved;
metric replay reproduced all recorded test scores.

## Follow-up controls

- `tuev-demodulation-patchmean.yaml`: keep the original full-window Hilbert
  computation, waveform path, token dimensions and projections. Average
  **log1p envelope** within each 50-sample patch, broadcast that scalar across
  the patch, then apply the original envelope projection. It tests the value
  of within-patch envelope variation beyond a local amplitude summary.
- `tuev-demodulation-bandmix-restart.yaml`: the same fixed orthogonal
  band-basis control, under a fresh run name. The stopped original directory
  is retained and is not overwritten or relabeled as complete.

Both retain seed 2 and change one model field relative to the reference.
Separate source is frozen at `demodulation-followups-20261106/source/` on145;
`preflight.json` records exact default parity and identical initialization.
Both require actual GPU training/reload smokes before formal training. The
follow-up allocation is GPU0/1, with the existing CHB envelope run onGPU4.
Both smokes passed ten train steps, validation and strict reload without test
evaluation. Formal training was verified on145: patchmean PID2475314/GPU0
and bandmix successor PID2475318/GPU1. Excluding the first smoke step, mean
training-step times were 0.3070s and 0.3148s, respectively. New jobs have no
final test scores at launch.

Both follow-ups subsequently completed all20epochs and final tests:

| Supplementary TUEV input | Kappa | Balanced accuracy | Weighted F1 |
|---|---:|---:|---:|
| Waveform + patch-mean log1p envelope | 0.6953713258 | 0.6124834297 | 0.8415543419 |
| Paired content in a mixed band basis | 0.5853923586 | 0.5295831644 | 0.7881733120 |

The user then narrowed priority to core content evidence. The patchmean
process had already exited. A cooperative stop request for bandmix arrived
during its final test pass; it completed normally (`stopped_by: epochs`),
not as a canceled run. Its pre-request best checkpoint was separately
preserved. Both final tests were replayed from stored logits with zero
metric difference. At that November 6 check, GPU0/1 were released and CHB
envelope PID762197 onGPU4 was the only remaining task from this study. No further exploration
of these variants is scheduled. Their manuscript results remain in the
appendix, outside the main content table.

## CHB-MIT envelope-only result, November 7

The last content arm completed six epochs and stopped by patience. Its
validation-selected checkpoint is epoch 4, step 9400, evaluation 246. It is not
an interim evaluation or a wall-clock-truncated result. Saved test logits
reproduce every recorded metric exactly. Best checkpoint SHA256:
`98f535b00cd51812c9d1090b3ae324379706a909e6ff528b34a135e9fd7a5d19`.

| CHB-MIT content, seed 2 | PR-AUC | Average precision | AUROC | Balanced accuracy |
|---|---:|---:|---:|---:|
| Envelope only | 0.7436105115 | 0.7448687723 | 0.9244401105 | 0.6157959156 |
| Paired envelope/waveform reference | 0.6994134955 | 0.7017071444 | 0.9178836429 | 0.6303178919 |

Both have 1,620,258 parameters, the same input contract, class counts and
17,386-window enriched test protocol. The only changed model field is
content_view. Envelope-only is higher by 0.0441970161 PR-AUC and 0.0065564675 AUROC;
the paired reference is higher by 0.0145219763 balanced accuracy. TUEV supports
a paired-content advantage; the CHB primary metric does not. Do not describe
waveform complementarity as established across both tasks from these runs.
All comparisons here use one shared seed.

## Frozen E-only checkpoint on corrected test data, November 8

The completed E-only seed-2 checkpoint was evaluated without training on the
actual accepted natural-grid test arrays: 17,224 windows, 114 positives,
16 bipolar channels, 200 Hz, 10 seconds, stored Q95 normalization. The model
and checkpoint selection remain those of the legacy-validation run; there
is no validation reselection or threshold tuning in this evaluation.

| Test cohort, same checkpoint | PR-AUC | Average precision | AUROC | Balanced accuracy |
|---|---:|---:|---:|---:|
| Original enriched test, 17,386 windows | 0.7436105115 | 0.7448687723 | 0.9244401105 | 0.6157959156 |
| Actual corrected natural grid, 17,224 windows | 0.5346450541 | 0.5408366754 | 0.9016554390 | 0.5963451147 |

The original saved predictions restricted to the same natural-grid windows
give PR-AUC 0.5346450608, essentially the same result. The maximum absolute
logit difference between new-array inference and mapped old predictions is
1.6689301e-6. Thus the approximately 0.209 PR-AUC difference is accounted for
by test composition, rather than by a new training run. Different cohorts
must not be pooled into one ranking.

All 54 frozen source hashes, the checkpoint hash and all corrected test
artifact hashes were checked. A 32-window old-input parity check matched
saved logits within 1.1920929e-7. Independent replay of the new prediction
file reproduces every test metric exactly. Both training and capacity runs
remain stopped; GPU4 was used only for 91.83 seconds of inference and released.

The separate evaluation output on145 is
`/data2/zz5070/CFM4EEG-rebuild-20261021/eonly-natural-grid-eval-20261108/`.
Its `evaluate.py`, `result.json`, `test_scores.npz`, selection receipt and
local independent check preserve the evaluation. The checkpoint SHA remains
`98f535b00cd51812c9d1090b3ae324379706a909e6ff528b34a135e9fd7a5d19`;
the corrected manifest SHA is
`f914c074e9f7ab165fed5b27919cbbe9dbd12c5a3e0faf36756bb2bd87672567`.
This is a frozen-checkpoint re-evaluation, not a model trained and selected
under the corrected protocol. The manuscript and Overleaf were not changed.

## Core evidence and minimal capacity comparison

The main comparison concerns local waveform/envelope content. The completed
six-block TUEV comparison is W-only0.498353, E-only0.641651 and paired0.711191.
The CHB-MIT envelope-only arm has completed with the reverse primary-metric
ordering shown above. Mean-envelope and bandmix are supplementary; no further
variants of either are scheduled.

Two additional TUEV configs reduce the reader from six blocks to two:
`tuev-capacity2-paired.yaml` and `tuev-capacity2-waveform.yaml`. They retain
16bands,width128,seed2,20epochs,learning rate,augmentation and validation
selection. The pair differs only in content_view and uses identical parameter
initialization and parameter count. Existing six-block results are reused.
This tests whether paired content retains its advantage with a smaller
encoder; it does not assume that the capacity trend will hold.

The frozen release is `demodulation-capacity-20261106/` on145. GPU0/1 are
assigned only after idle-card checks and actual training/validation/reload
smokes. CHB envelope was left onGPU4. No model-size sweep, baseline rerun or
pretraining task is part of this pass.

The user subsequently requested discussing the plan before running it. The
agent had already launched the two capacity arms and issued cooperative stop
requests for both; retain their outputs, without restart or further training
until the user authorizes after discussion. These configs remain a proposed
comparison rather than an approved active plan. The pre-existing CHB envelope
run is retained.
Both capacity processes have now exited with `operator_stop`, `stop_file`;
each saved `best.pt`, and neither evaluated the test set. As checked on
November 7, no project GPU process or training controller from this study is
active on145. No new experiments or restarts are authorized before discussion.

## Locations and execution

145 release root:
`/data2/zz5070/CFM4EEG-rebuild-20261021/demodulation-controls-20261106/`

- `source-method-v2/`: source used for completed recovery analysis.
- `method-{tuev,chbmit}/`: recovery protocol, scores and example arrays.
- `source-controls/`: frozen source and four original prepared configs.
- `preflight.json`: exact initialization/parameter parity and source comparison.
- `smoke/`: ten-step train/validation and checkpoint-reload checks, no test.
- `runs/`: downstream outputs; completion requires final `result.json`.
- `logs/`: separate smoke and full-run logs.

Data are reused from `legacy-triaxial-validation-20261105/data/`, with no
second normalization. CHB-MIT uses the recorded 17,386-window enriched test
protocol; these controls are not natural-grid main-table results.

Example invocation from `source-controls/`:

```bash
export LEGACY_VALIDATION_DATA=/data2/zz5070/CFM4EEG-rebuild-20261021/legacy-triaxial-validation-20261105/data
CUDA_VISIBLE_DEVICES=0 /data2/zz5070/CFM4EEG-rebuild-20261021/venv/bin/python -u \
  -m paclock_bench.training.train \
  --config configs/model/tuev-demodulation-waveform.yaml --seed 2 \
  --out /data2/zz5070/CFM4EEG-rebuild-20261021/demodulation-controls-20261106/runs
```

Run the matching ten-step smoke in a distinct output root before full training.
Retain all checkpoints and report every arm, including contrary results.
