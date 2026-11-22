# CfFT geometric content fusion: TUEV pilot

## Question
Does transporting source-band content along measured phase relations improve the existing two-axis tokenizer? This is a new experiment, distinct from the Procrustes interpretation of the existing tokenizer in the manuscript.

## Frozen design
- Same learned 16-band sinc frontend, Hilbert transform, patch and relation windows (50 samples), 128-wide six-layer two-axis Transformer.
- Original target-content rotation remains intact.
- Source content q_i = envelope feature + i * waveform feature.
- Shared gate: LayerNorm(concatenated envelope and waveform) -> Linear(128,64) -> sigmoid. The gate scales q_i, so the full complex projected content, rather than only a content-derived scalar gate, is transported.
- Orient source content with normalized projected source phase; use the existing measured conjugate relation angle and relative magnitude to align and aggregate eligible sources i<j.
- No source/fallback relation: zero message. The original path retains its existing behavior.
- Interleave real/imaginary message components, apply bias-free 128-to-128 projection, and add a bounded per-band residual (0.1*tanh(scale), initialized to 0.01).
- Added module creation preserves the original model initialization RNG state.
- This is feature transport, not raw-waveform time alignment. Phase-reference invariance holds for analytic coordinate changes with content held fixed, not arbitrary EEG perturbations.

## Compute and protocol
- 145 TUEV is incomplete: only a partial train_signals.npy and labels/manifest, no val/test signals. No other user's jobs stopped.
- AMD runtime: `/home1/yifanwang/CFM4EEG-geo-fusion-20261016`.
- Smoke: 464413, completed successfully on MI210 single-card partition `mi2101x`.
- Training: 464424, `cfft-geo-tuev-s0`, seed 0, 20 epochs, patience 20, original augmentation and AdamW recipe, max_hours 0, 12-hour Slurm limit.
- Train/validation/test windows: 68,445 / 15,487 / 29,421, shape 16 x 1,000.
- Validation Cohen kappa selects checkpoint; test runs only at training completion.
- All improving checkpoints are retained by the existing training loop.
- Trainable parameters: original 1,224,486; transport 1,249,398 (+24,912).
- Smoke warm-up-excluded median step: 0.3268 s, batch 32; peak allocated memory 12.15 GiB. Historical full baseline ~4.05 hours; expected new run ~4-5 hours, subject to evaluation and system throughput.

## Comparison
The stored, configuration-matched seed-0 baseline is:
`/home1/yifanwang/CFM4EEG-twoaxis-20260917/runs/tuev-range_twoaxis_measured_20260917/seed0/result.json`
- test kappa 0.684600667
- balanced accuracy 0.636579850
- weighted F1 0.835754959
- validation best kappa 0.625956863, evaluation index 7, all 20 epochs completed.
Use this raw matching run for the pilot comparison. Do not compare a new single seed with a manuscript aggregate as if paired.

## Checks passed
Shared initialization equality, zero-residual exact original logits, directed source/target loop equivalence, conjugate angle sign, source-content influence with fixed relations, analytic-reference invariance, finite zero/near-zero behavior, nonzero finite branch gradients, full data-array payload sizes, real-batch optimizer steps, strict checkpoint roundtrip, equal parameter counts of prepared controls.

## Controls prepared, not submitted
`no_alignment` removes relation angles only from the new residual branch. `ordinary` uses identical gates, projection, residual coefficients and relation-magnitude weights, but directly aggregates content without phase orientation/alignment. Both retain the original tokenizer path. The user requested a single-seed first pilot; no 3-seed or additional full-control runs have been submitted.

## Artifacts
- `configs/geometric/tuev_aligned_s0.yaml`
- `results/experiment-receipt.json`: source/config/data-manifest hashes, baseline pointer and job IDs.
- `results/geo-smoke.json`: checks and timing.
- `runs/tuev-cfft_geo_transport_20261016/seed0/`: checkpoint history, progress and eventual result.
- `logs/cfft-geo-tuev-s0-464424.out`
