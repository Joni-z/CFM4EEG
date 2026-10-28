# CFC operator: PAC measurement

`PACCurveOperator` measures phase–amplitude dependence from raw EEG. Its output
is a harmonic modulation curve, with physical-support masks and a predictive
score. It is independent of the tokenizer and has no trainable parameters.

```python
from paclock_bench.models.paclock.frontend.measurement_curve import PACCurveOperator

operator = PACCurveOperator()  # 200 Hz; input [batch, electrode, samples]
measurement = operator(x)
coefficients = measurement.coef
curve = measurement.modulation(source=2, target=2, patch=10)
```

## Computation

1. Fixed source search cells have nominal centers 2, 3, 4, 5, 6 and 8 Hz;
   target amplitude filters have centers 18, 24, 32, 40 and 48 Hz.
2. Estimate a source spectral peak from the raw signal inside each disjoint
   cell, using a Hann-tapered periodogram and sub-bin interpolation. No generated
   truth, labels or other batch examples enter this step.
3. Apply a DC-corrected complex Morlet phase filter centered on the estimated
   frequency (`phase_cycles=9`). Extract target envelopes with separate broad
   complex FIR filters. Use linear convolution and discard unreliable edges.
4. Regress target envelope on time, cosine/sine of source phase, and cosine/sine
   of twice source phase. Only supported harmonics enter the regression.
   Source-amplitude nuisance terms are optional and disabled by default.
5. Return complex cosine/sine coefficients and a positive fitted baseline.
   Their normalized Fourier sum describes envelope modulation as a function of
   source phase. Second-harmonic phase is periodic modulo pi.
6. Estimate predictive gain by fitting alternating complete phase cycles and
   predicting the other cycles, relative to the nuisance-only model. This score
   is not a p-value. Detection thresholds require separate calibration.

Eligibility checks use the worst source-frequency support in each search cell,
the actual amplitude-filter response, a 56–64 Hz notch exclusion by default,
phase coverage, and regression conditioning. First-harmonic eligibility does not
imply second-harmonic eligibility. Some low-frequency pairs are unavailable in
short records.

The main output shape is `[B,C,P,source,target,harmonic]`; the default has two
harmonic slots and `P=T//50`. `valid` must accompany coefficients. With global
context, the same estimate supplies multiple output anchors; these are not
independent observations.

## Context and configuration

`local=False` uses the whole available record after edge masking. `local=True`
also fits source-frequency-dependent local windows, regularized toward the
whole-record estimate. A valid local fit may be used even if the whole-record
fit is unavailable. `local_valid` distinguishes these estimates from global
fallbacks.

Checkpoint state includes the physical filters, support table and constructor
configuration. Loading into an incompatible constructor raises an error.
Sampling rates other than 200 Hz require explicit passband and notch settings
within Nyquist. For example, at 100 Hz use `passband=(0.5,49)` and the actual
dataset's excluded intervals.

## Reproduce synthetic checks

From the repository root, use separate output directories:

```bash
python -m pytest tests/test_pac_curve.py tests/test_pac_measurement.py -q
python -m scripts.experiments.cfc_operator_benchmark \
  --split dev --cases 96 --harmonics 1 --confidence-folds cycles \
  --output /path/to/outputs/first-harmonic
python -m scripts.experiments.cfc_operator_benchmark \
  --split dev --cases 96 --refine-phase --phase-cycles 9 \
  --confidence-folds cycles --output /path/to/outputs/pac-curve
```

The benchmark supplies continuously sampled unknown source/carrier frequencies.
Ground truth chooses only the predeclared nearest output for scoring; it is not
passed to the operator. All invalid positive records incur an error penalty.
The condition named `offgrid` is an independent 20 dB repeat, not a separate
on-grid/off-grid comparison.

The [validation record](../review/CFC_OPERATOR_VALIDATION_20261028.json) contains
the frozen source, independent signal splits, paired error differences,
record-level detection results, and NumPy replay audit. The baseline is the
existing first-harmonic regression evaluated with the same whole-record
context and physical grid. It is not a new run of the historical classifier.

The release confirmation has 5,500 records per arm. Across its 3,500 positive
records, curve RMSE is 0.09963 for the baseline and 0.02881 for the new operator.
The paired reduction interval is [0.06972,0.07190], bootstrapped by independent
record within condition. Using thresholds calibrated separately at nominal 5%
record-level false positives, observed rates are 3.7% / 4.6% and sensitivities
87.6% / 100%. These apply to the declared synthetic conditions, including
modulation depths at least 0.2 and 10/20 dB noise, not arbitrary EEG recordings.

Nonlinear-envelope and simultaneous-two-carrier challenges reduce RMSE from
0.07777 to 0.03906 and 0.05586 to 0.04408. A separate 200-record, 10-second
time-varying-phase challenge reduces coefficient error from 0.29441 to 0.05029
on local-valid anchors; local coverage is 68.94%.

Non-sinusoidal oscillations and common transients remain interpretation stress
cases. Their statistical dependencies must not be relabeled independent nulls.
This validation does not identify physiological interaction or establish
clinical prediction gains.

## Tokenizer boundary

This standalone operator is not selected by an existing `tokenizer_mode`.
`tokenizer_mode: measured_pac` still uses the earlier first-harmonic operator in
`measurement.py`. Its clinical results and checkpoints retain their original
meaning. Connecting the curve operator to a tokenizer is a separate experiment.

## Statistical sources

- Penny et al., *Testing for nested oscillation*, 2008,
  doi:10.1016/j.jneumeth.2008.06.035.
- van Wijk et al., *Parametric estimation of cross-frequency coupling*, 2015,
  doi:10.1016/j.jneumeth.2015.01.032.
- Nadalin et al., *A statistical framework to assess cross-frequency coupling
  while accounting for confounding analysis effects*, 2019,
  doi:10.7554/eLife.44287.

The implementation uses a linear ridge objective; it does not reproduce a gamma
GLM likelihood, an information-theoretic PAC estimator, or a causal test.
