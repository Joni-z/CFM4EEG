# CFC residual tokenizer pilot

This pilot keeps the frozen dual-harmonic CFC measurement operator and the
six-layer two-axis encoder. It changes how measured relationships enter tokens.
The previous rotation pilot remains available through `*-cfc-h2.yaml`.

## Token construction

1. Compute each of16 learned content bands' envelope and waveform patch features.
   Preserve their real/imaginary concatenation as the direct content path.
2. The CFC operator fits one record-wide relationship on its physical6x5grid.
   Use the raw baseline-normalized complex coefficients (`cfc_stabilized:false`).
   Unsupported or unobservable coefficients are zero. The estimator, physical
   masks and uncertainty diagnostics are unchanged.
3. For each harmonic separately, multiply the conjugate coefficient by the
   matching source-phase-power patch projection and sum across sources.
   This gives five target response channels per harmonic, invariant to an
   independent change of each source's phase reference.
4. Two independent bias-free linear maps read the first and second harmonic
   responses. A learned16x10 softmax readout supplies PAC context to EVERY
   content band. This is semantic conditioning, not a claim that a low-frequency
   content band was itself measured as a high-frequency PAC target.
5. Add the relation residual to content, with a trainable per-band gain initially1.
   There is no compulsory rotation, no identity-centered angular restriction,
   and no normalization that turns weak coefficients into unit-strength signals.
   All-zero relations yield exactly the content path. The waveform projection
   retains its established zero initialization; new readout initialization does
   not advance the encoder/head random stream.

Token shape remains[B,electrode,16,patch,128]. Harmonics remain separate until
their learned readouts; the record-wide coefficient is evaluated against local
phase features, not relabeled as a locally estimated coefficient. This pilot
makes no claim to solve optimal relation-window selection.

## Training

Both pilots use seed0, accepted datasets, scratch initialization and
validation-only checkpoint selection. Test predictions and best.pt are retained.

- TUEV: unchanged20epochs, batch32, AdamW1e-4, WD0.05, dropout0.35, augmentation,
  label smoothing0.1 and full validation once per epoch.
- CHB-MIT: fixed6epochs, cosine over all6, patience0, full validation every1000steps
  plus epoch end; focal positive weight0.75 instead of0.25. This is an explicit
  recipe adjustment together with the new tokenizer, not an isolated structural
  ablation. The fixed binary threshold remains0.5; PR-AUC is trapezoidal and AP
  is reported separately. No class resampling or test-based threshold tuning.

The earlier20-validation patience could stop CHB after2epochs while cosine still
retained97.55% of its initial learning rate. Less frequent validation also avoids
scoring more validation windows than training windows several times over.

## Verification before full training

Tests cover all-band relation influence, independent source-reference changes,
first/second harmonic contributions, exactly-zero fallback, gradients, strict
checkpoint loading, and backwards compatibility of the old rotation pilot.
Real training inputs must additionally show finite relation/content RMS ratios,
nonzero relation gradients and correctly restored checkpoints. Smoke scores are
not clinical performance evidence.
