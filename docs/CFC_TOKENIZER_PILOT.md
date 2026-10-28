# Dual-harmonic CFC tokenizer pilot

`CFCFrontend` connects the frozen `cfc-ready-source-20261029` PAC measurement
component to the existing 16-band quadrature tokenizer and two-axis encoder.
These are new experiments; earlier CfFT benchmark results do not evaluate this
component. Start with seed0 on accepted TUEV and rebuilt CHB-MIT arrays.

## Interface

1. Measure PAC on raw input using the fixed6-source by5-target physical grid.
   Estimate both harmonics over the full5/10-second record, separately from the
   0.25-second token patches. Physical validity and uncertainty remain those of
   the frozen operator. There are no learned measurement parameters.
2. Use masked, baseline-normalized coefficients. The pilot explicitly selects
   the tested optional stabilized view; it does not change the raw estimator.
3. Project each source phase power through the shared bias-free phase patch
   projection. Multiply harmonic k's projected phase by the conjugate of its
   matching measured coefficient and sum sources/harmonics per physical target.
   Both harmonics retain their own phase convention; no harmonic2 angle is
   treated as a unique preferred phase.
4. Map the5 physical targets to the16 content bands using the actual learned
   Sinc responses at the target frequencies. Retain absolute response strength
   relative to each filter's peak and cap row mass at1. Do not force low-frequency
   or out-of-band rows to receive strong relationships by row normalization.
5. Form an identity-anchored bounded reference, normalize that reference, and
   rotate the existing envelope/waveform coordinates. Unavailable or zero
   relations give identity. Do not unit-normalize individual weak PAC estimates.

The diagnostic strength returned to the encoder retains the physical6x5 axes.
Only the two-axis encoder (`freq_mixer: none`) without auxiliary reconstruction
is supported; it does not interpret this tensor as a16x16 content relation.

## Runs

Configs: `configs/model/{tuev,chbmit}-cfc-h2.yaml`. Both keep the corresponding
`*-scratch.yaml` data, training, encoder, loss and augmentation recipe. Loading
arrays into available host RAM changes I/O only. No test score selects configs
or checkpoints. The training script saves validation-best weights including
the operator's non-tensor configuration state.

```sh
export PACLOCK_PROC=/data2/zz5070/CFM4EEG-rebuild-20261021/data
python -m paclock_bench.training.train --config configs/model/tuev-cfc-h2.yaml \
  --seed 0 --smoke-steps 4 --out /path/to/pilot/smoke
python -m paclock_bench.training.train --config configs/model/tuev-cfc-h2.yaml \
  --seed 0 --out /path/to/pilot/runs
```

Run each dataset on its own GPU with explicit `CUDA_VISIBLE_DEVICES` binding.
Smoke must verify finite forward/backward, optimizer updates, validation and
strict checkpoint restoration before long training. Preserve the frozen source,
config, accepted data identity, smoke receipt and logs for each launch.
