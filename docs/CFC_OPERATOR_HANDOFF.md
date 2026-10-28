# CFC operator: PAC measurement handoff

The component is ready for the step before tokenizer integration. It measures
PAC on a fixed physical frequency grid and returns the measured relation,
its observable support and a cycle-block uncertainty diagnostic. It has zero
trainable parameters. No tokenizer, encoder or downstream training is selected
or run by this interface.

## API

```python
import torch
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator

operator = CFCOperator(block_cycles=1, shrink=False)
x = torch.randn(2, 16, 2000)  # [batch, electrode, samples], 200 Hz, 10 seconds
measurement = operator(x)
raw = measurement.relation_features(stabilized=False)
stable = measurement.relation_features(stabilized=True)
grid = operator.physical_grid()
```

`shrink=False` keeps `measurement.coef` equal to the original harmonic-regression
estimate. `raw_coef` always preserves that estimate. The explicit stabilized
view multiplies each complex coefficient by the uncertainty-derived weight;
it changes magnitudes, while retaining their phase directions.

| Output | Shape | Meaning |
|---|---|---|
| `coef`, `raw_coef` | B,C,P,6,5,2 | Complex coefficients in input-amplitude units |
| `baseline` | B,C,P,6,5 | Positive fitted amplitude baseline |
| `raw['features']` | B,C,P,6,5,2,2 | Dimensionless coefficients; last axis is real/imaginary |
| `harmonic_valid` | B,C,P,6,5,2 | Physical support and usable fit, per harmonic |
| `coefficient_variance` | B,C,P,6,5,2 | Cycle-block trace variance in amplitude units squared |
| `uncertainty_valid` | B,C,P,6,5 | Enough usable blocks and an available covariance diagnostic |
| `reliability` | B,C,P,6,5,2 | Bounded shrinkage weight, not a PAC probability |
| `phase_valid` | B,C,6,L | Temporal source-phase guard and amplitude mask |
| `source_frequency` | B,C,6 | Signal-estimated physical source frequency |

P is `L // patch_len`. The default source anchors are 2,3,4,5,6,8 Hz and target
anchors are 18,24,32,40,48 Hz. `physical_grid()` returns their fixed coordinates,
source search cells and harmonic support. They are separate from any learned
content filterbank.

The default measurement uses the full available record. Its repeated P outputs
are the same context estimate, not P independent local measurements. A separate
`PACCurveOperator(local=True)` remains available for the previously validated
local/global-fallback mode; cycle uncertainty currently supports global mode
without source-amplitude nuisance covariates and rejects incompatible options.

## Algorithm

1. Estimate one source spectral peak inside each fixed search cell from raw
   samples, then apply its DC-corrected complex Morlet phase filter.
2. Extract target envelopes with separate FIR filters broad enough to preserve
   the supported modulation sidebands. Remove filter-edge samples.
3. Fit the target amplitude to an intercept, time trend and cosine/sine terms
   of source phase and twice source phase. Remove nuisance terms before the
   ridge solve. Return each harmonic's complex cosine/sine coefficient.
4. Check actual filter response, excluded intervals, observed cycles, phase
   coverage, conditioning and finite inputs. First-harmonic support does not
   imply second-harmonic support.
5. Sum residualized regression scores within observed cycle blocks and form a
   cluster-sandwich covariance. At least four blocks with at least one observed
   cycle each are required. The two-coordinate variance trace is retained for
   each harmonic.
6. Optionally stabilize each coefficient with `r = power / (power + variance)`.
   Where uncertainty is unavailable, keep the raw estimate and mark the
   uncertainty mask false. Normalize by the fitted baseline and carry masks
   alongside the real/imaginary features.

The covariance is a diagnostic rather than a calibrated confidence interval.
The shrinkage rule is a bias–variance choice with a zero-centered prior; it is
not uniformly better on all PAC shapes or carrier processes.

## Coordinate and unit checks

If the source phase reference is shifted by delta, harmonic k's coefficient
rotates by k*delta. The predicted modulation curve remains unchanged. Each
harmonic covariance trace and shrinkage weight remain unchanged under that
coordinate rotation.

If signal amplitude is multiplied by a positive constant, coefficients and
baseline scale by that constant and covariance scales by its square. The
normalized relation features and shrinkage weights remain unchanged, away
from numerical floors and with unchanged validity masks. These statements
refer to coordinate/unit changes, not arbitrary perturbations of raw EEG.

## Calibration is separate from token features

```python
from paclock_bench.models.paclock.frontend.pac_calibration import PACCalibration

# One maximum score per independent null record, from matching reference inputs.
cal = PACCalibration.fit(reference_scores, operator, channels=1, samples=2000,
                         alpha=0.05, confidence=0.95)
cal.save('pac-calibration.json')
```

The marginal rule uses rank `ceil((n+1)*(1-alpha))`. The optional tolerance rule
uses a binomial order-statistic bound. Strict `score > threshold` handles ties
conservatively. If the required rank exceeds n, detection abstains.

These guarantees require exchangeable null records; the tolerance interpretation
uses iid calibration records. Reference scores must be representative of the
intended null distribution. A matched signature alone does not establish that
assumption for clinical EEG. One record is the calibration unit; patches,
electrodes and overlapping cycles are not independent calibration examples.

Calibration is bound to score source, physical filter state, constructor
configuration, Torch runtime/device type, channel count and record length.
Incompatible application raises an error. A one-channel, 10-second synthetic
calibration must not silently be reused for a 23-channel, 5-second clinical
record. Continuous relation features do not require significance thresholding.

## Checkpoint and device contract

```python
package = {'constructor': operator.constructor_config(),
           'state_dict': operator.state_dict()}
torch.save(package, 'cfc-operator.pt')
saved = torch.load('cfc-operator.pt', weights_only=True, map_location='cpu')
restored = CFCOperator(**saved['constructor'])
restored.load_state_dict(saved['state_dict'])
```

An incompatible constructor is rejected. Fixed filters and measurement settings
are checkpointed. Eager CPU and CUDA are tested; float64 regression is required
and MPS is rejected. Physical buffer precision is preserved under a parent's
half/bfloat16 cast; complex filter components remain intact. Filtering/regression
retain input gradients, while spectral
peak selection, masks and internal predictive-score calculations are not fully
differentiable. Raw EEG inputs normally need no gradients during training.

For 100 Hz input, explicitly set `patch_len=25`, a passband below 50 Hz and valid
excluded intervals, e.g. `passband=(0.5,49), excluded_intervals=()`. Use the actual
dataset passband rather than treating that example as a preprocessing recipe.

## Before connecting a tokenizer

- Use raw EEG with known sampling rate and passband; keep learned content
  filtering separate.
- Preserve real/imaginary order, both harmonics, physical coordinates and masks.
- Distinguish an unobservable relation from a measured small relation. A zero
  invalid entry is an unavailable measurement, not evidence of absent PAC.
- Do not normalize weak coefficients to unit magnitude. Do not interpret the
  two-harmonic curve as a unique preferred angle; harmonic2 is periodic modulo pi.
- The 6x5 measurement grid is not a 16x16 content-band matrix. The next tokenizer
  must define how it consumes these physical-grid relations; replacing an old
  N-by-N tensor by matching dimensions would be incorrect.

## Evidence and positioning

The readiness evidence covers oracle error decomposition, coherent and stochastic
carriers, strength/phase/shape/noise sweeps, frequency/record-length coverage,
independent calibration, local fallback, serialization, heterogeneous batches,
invalid input and actual EEG-window CPU/CUDA forward/backward checks. See
`review/CFC_OPERATOR_READY_20261029.json` and the
[evidence tag](https://github.com/Joni-z/CFM4EEG/tree/cfc-ready-evidence-20261029).

Stabilization reduces weak/coherent error but slightly worsens some strong mixed
cases, especially stochastic carriers. Very weak coupling and unsupported
low-frequency/short-record pairs remain limitations. This evidence establishes
measurement behavior, not universal estimation superiority, neural causality,
or downstream predictive benefits.

Harmonic PAC regression, cluster covariance, shrinkage and rank calibration have
prior foundations. The component's concrete contribution is their explicit,
physically indexed measurement interface with support/uncertainty information
for the subsequent tokenizer. Standalone estimator originality is not established.

Sources: [Penny et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18674562/),
[Kramer and Eden, 2013](https://pubmed.ncbi.nlm.nih.gov/24012829/),
[Perley and Coleman, 2024](https://pubmed.ncbi.nlm.nih.gov/38841426/),
[Dvorak and Fenton, 2014](https://pubmed.ncbi.nlm.nih.gov/24447842/),
[Cameron and Miller, 2015](https://escholarship.org/uc/item/1jq5d0pq),
[Shafer and Vovk, 2008](https://jmlr.org/papers/v9/shafer08a.html),
[Wilks, 1941](https://doi.org/10.1214/aoms/1177731788).
