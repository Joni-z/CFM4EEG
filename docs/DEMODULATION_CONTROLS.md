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

Three new controls run on GPU0/1/4. Together with the existing raw-CHB task on
GPU2, this uses four GPUs. The prepared TUEV `band_mix_seed: 2` control is
deferred under the user's total-four-card limit; its smoke passed, but its
full-run process was stopped and all existing artifacts retained.

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

## Locations and execution

145 release root:
`/data2/zz5070/CFM4EEG-rebuild-20261021/demodulation-controls-20261106/`

- `source-method-v2/`: source used for completed recovery analysis.
- `method-{tuev,chbmit}/`: recovery protocol, scores and example arrays.
- `source-controls/`: frozen source and four prepared configs (three active).
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
