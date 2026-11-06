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
remains in progress. The original bandmix process was stopped under the user's
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
