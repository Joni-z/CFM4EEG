# Copula-guided tokenizer

This pilot separates per-band content from explicitly estimated dependence,
motivated by Sklar's distinction between marginal distributions and a copula.
It does not estimate the full joint EEG distribution or assert that an encoder
cannot learn dependencies from raw signals.

## Representation

- Keep the learnable 16-band Sinc bank, 0.25-second content patches and the
  existing two-axis Transformer. Project log-envelope and waveform patches to
  64 coordinates each, then concatenate them.
- Use one-second, non-overlapping relation windows. Within each electrode and
  window, map each envelope and each real/imaginary unit-phase component through
  its empirical marginal CDF to normal scores. Use average ranks for ties.
- Estimate signed envelope correlation and phase/envelope multiple correlation
  after shrinking the correlation matrix 5% toward identity. The latter uses
  phase components from a lower-center-frequency band and the envelope of a
  higher-center-frequency band. Determine eligibility from the current filters,
  not their initial indices. All band pairs remain eligible for content fusion.
- Supply these two measured summaries as per-head biases to one four-head band
  attention layer. Its value vectors carry actual content. Exclude self-edges
  from the fusion branch and retain each band's own content through a residual
  whose learned scale starts at 0.1.
- Relation estimates are stop-gradient and computed separately per example,
  electrode and window. Filter parameters learn through the content branch.
  No batch pooling, target labels or held-out statistics enter the estimator.

AAC is signed; phase/envelope dependence is bounded squared multiple
correlation. These summaries correspond to a Gaussian-copula approximation,
not a complete characterization of dependence or a validated physiological PAC
measurement. The method has no preferred-phase rotation and does not use EOT.
Short-window uncertainty, filter overlap and the Gaussian approximation remain
limitations; Sklar's theorem does not establish a predictive advantage.

## Implementation and pilot

The complete new frontend is
`paclock_bench/models/paclock/frontend/copula.py`. Its builder selects it only
when `tokenizer_mode: copula`; older configurations retain their own frontend.
`configs/model/tuev-copula-lenovo.yaml` trains the TUEV pilot from scratch with
seed 0, the accepted shared16 cohort and validation-best checkpoint selection.
Physical batch 8 and four accumulation steps give effective batch 32; stochastic
augmentation/dropout draws differ from a single physical batch of 32.

Windows uses memory-mapped arrays and zero DataLoader workers to avoid spawning
copies of the dataset on a 16-GB machine. Checkpoints and source provenance are
written by the normal project trainer. Runtime records and data copies live
outside this source tree.

Reference: Ince et al., *A statistical framework for neuroimaging data analysis
based on mutual information estimated via a Gaussian copula*, Human Brain
Mapping 38:1541–1573 (2017), doi:10.1002/hbm.23471.
