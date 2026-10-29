# CFC operator: PAC measurement

`CFCOperator` measures first/second-harmonic phase–amplitude dependence from raw
EEG. It returns coefficients, physical support, validity and uncertainty without
trainable parameters. Token construction is a separate step, described in the
[model guide](MODEL.md).

## API and coordinates

```python
import torch
from paclock_bench.models.paclock.frontend.measurement_uncertain import CFCOperator

operator = CFCOperator(sample_rate=200, patch_len=50, block_cycles=1, shrink=False)
x = torch.randn(2, 16, 2000)  # batch, electrode, samples
measurement = operator(x)
raw = measurement.relation_features(stabilized=False)
stable = measurement.relation_features(stabilized=True)
grid = operator.physical_grid()
```

The six source search cells are anchored at 2, 3, 4, 5, 6 and 8 Hz; five target
amplitude filters are centered at 18, 24, 32, 40 and 48 Hz. The source frequency
is refined within its fixed search cell. These physical coordinates are
independent of a tokenizer's learned content bands.

Let `B,C,P,S,T,K` denote batch, electrode, output patch, source, target and
harmonic, with `P=samples//patch_len`, `S=6`, `T=5`, `K=2` by default.

| Field | Shape | Meaning |
|---|---|---|
| `raw_coef`, `coef` | B,C,P,S,T,K | Harmonic regression coefficients in input-amplitude units |
| `baseline` | B,C,P,S,T | Positive fitted amplitude baseline |
| `raw['features']` | B,C,P,S,T,K,2 | Baseline-normalized real/imaginary coefficients |
| `harmonic_valid` | B,C,P,S,T,K | Physical support combined with usable-fit validity |
| `coefficient_variance` | B,C,P,S,T,K | Cycle-block covariance trace in squared amplitude units |
| `uncertainty_valid` | B,C,P,S,T | Enough usable blocks for the covariance diagnostic |
| `reliability` | B,C,P,S,T,K | Optional shrinkage weight, not a PAC probability |
| `source_phase`, `phase_valid` | B,C,S,samples | Source unit phase and temporal mask |
| `source_frequency` | B,C,S | Signal-estimated source frequency |

`shrink=False` preserves the raw estimate in `coef`. The stabilized feature view
explicitly multiplies raw coefficients by reliability; both views normalize by
the fitted baseline. Unsupported/unobservable entries have zero features and a
false validity mask. They are unavailable measurements, not evidence of no PAC.

## Estimation

1. Estimate a spectral peak in each source cell using a Hann-tapered periodogram
   and sub-bin interpolation; apply its DC-corrected complex Morlet phase filter.
2. Extract target envelopes using broad analytic FIR filters that retain the
   supported modulation sidebands. Discard unreliable filter-edge samples.
3. Regress each envelope on an intercept, time trend and supported cosine/sine
   terms of source phase and twice source phase. Fit harmonic coefficients by
   detrended ridge regression.
4. Check physical filter response, excluded frequencies, observed cycles,
   phase coverage, conditioning and finite inputs. H1 support does not imply H2
   support; short records can leave low-frequency pairs unobservable.
5. Aggregate regression scores in observed-cycle blocks for a cluster-sandwich
   covariance diagnostic. At least four usable blocks are required.
6. Optionally shrink by `power/(power+variance)`. Where covariance is unavailable,
   retain the raw coefficient and mark uncertainty unavailable.

The default fits the full available record and repeats its coefficients across
output patches. Patches are not independent measurements. The separate
`PACCurveOperator(local=True)` in
[measurement_curve.py](../paclock_bench/models/paclock/frontend/measurement_curve.py)
provides source-dependent local fits with global fallback. `CFCOperator` adds
uncertainty to its global mode; it rejects local/amplitude-covariate options.
The base curve operator has its own output API, including `modulation(...)`.

Under an independent source-phase reference shift, harmonic k's coefficient
rotates by k times that shift; its predicted curve is unchanged. Under positive
signal rescaling, coefficients/baseline scale together and normalized features
remain unchanged, away from numerical floors and validity changes. These are
coordinate/unit properties, not invariance to arbitrary EEG perturbations.
Do not turn weak coefficients into unit-length vectors or interpret H2 as a
unique preferred phase.

## Calibration and interpretation

```python
from paclock_bench.models.paclock.frontend.pac_calibration import PACCalibration

# One maximum score per independent, matching null record.
calibration = PACCalibration.fit(reference_scores, operator, channels=1,
                                samples=2000, alpha=0.05, confidence=0.95)
calibration.save('pac-calibration.json')
```

Rank calibration uses independent null records with a strict score threshold.
Its marginal guarantee requires exchangeability; the optional tolerance rule
requires iid calibration records. Code, physical filters, constructor settings,
runtime/device, channel count and record length are bound into its signature.
A synthetic calibration is not automatically valid for clinical EEG.
Continuous token features do not require significance thresholding.

Cycle covariance is a finite-record diagnostic, not a calibrated confidence
interval. `predictive_gain` uses alternating cycles within the filtered record;
it is an internal consistency score, not a p-value or independent-trial
validation. Shrinkage is a bias–variance choice and can worsen strong mixed cases.
Non-sinusoidal waves, common transients and overlapping filters also limit
physiological interpretation. Measurement validity alone does not establish
neural causality or clinical prediction gains.

## Checkpoints and devices

```python
package = {'constructor': operator.constructor_config(),
           'state_dict': operator.state_dict()}
torch.save(package, 'cfc-operator.pt')
saved = torch.load('cfc-operator.pt', weights_only=True, map_location='cpu')
restored = CFCOperator(**saved['constructor'])
restored.load_state_dict(saved['state_dict'])
```

Physical filters and settings are checkpointed; incompatible constructors fail.
Eager CPU/CUDA and float64 regression are supported; MPS is rejected. Filtering
and regression retain input gradients, but peak selection, masks and predictive
scores are not fully differentiable. Parent half/bfloat16 casting preserves
physical-buffer precision. Raw training inputs normally need no gradients.

Use the actual sampling rate, passband and excluded intervals. At 100 Hz, for
example, `patch_len=25`, `passband=(0.5,49)` and `excluded_intervals=()` satisfy
Nyquist, but these example settings do not define a dataset preprocessing recipe.
Default 200 Hz settings exclude 56–64 Hz. Longer context cannot restore sidebands
removed by acquisition or preprocessing.

## Validation and reproduction

Each record below retains its own source/configuration, scope and evidence tags;
results from different generators, noise definitions or operator revisions must
not be pooled. Clinical/tokenizer evidence is listed separately in [AUDIT.md](AUDIT.md#component-validation-record).

| Record | Scope |
|---|---|
| [Operator validation](../review/CFC_OPERATOR_VALIDATION_20261028.json) | Recovery, record-level detection and independent NumPy replay |
| [Factorial controls](../review/CFC_OPERATOR_CONTROLS_20261028.json) | Frequency refinement and H1/H2 contributions |
| [Wider capabilities](../review/CFC_OPERATOR_CAPABILITIES_20261029.json) | Frequency/length coverage, target-band noise, nulls and drifting phase |
| [Interface readiness](../review/CFC_OPERATOR_READY_20261029.json) | Raw/stabilized views, support, uncertainty, calibration and serialization |

These compare declared synthetic conditions and internal controls; they do not
establish superiority over mature PAC estimators. The adaptive candidate remains
an archived experiment, not the default operator. Full evidence is accessible
through each record's source/evidence tags, including
[cfc-ready-source-20261029](https://github.com/Joni-z/CFM4EEG/tree/cfc-ready-source-20261029)
and [cfc-ready-evidence-20261029](https://github.com/Joni-z/CFM4EEG/tree/cfc-ready-evidence-20261029).

```bash
python -m scripts.experiments.cfc_operator_reproduce summarize --root /path/to/extracted-evidence
python -m scripts.experiments.cfc_operator_reproduce run --root /path/to/new-outputs --cases 500 --calibration-cases 200
python -m scripts.experiments.cfc_operator_numpy_audit --output /path/to/numpy-audit.json
python -m scripts.experiments.cfc_operator_capabilities --help
python -m scripts.experiments.cfc_operator_capabilities_report --help
```

Use the recorded source tag to replay an exact older experiment. The current
capability evaluator retains the non-adaptive two-arm comparison.

Statistical foundations: [Penny et al., 2008](https://pubmed.ncbi.nlm.nih.gov/18674562/),
[van Wijk et al., 2015](https://doi.org/10.1016/j.jneumeth.2015.01.032),
[Nadalin et al., 2019](https://doi.org/10.7554/eLife.44287),
[Dvorak and Fenton, 2014](https://pubmed.ncbi.nlm.nih.gov/24447842/),
[Cameron and Miller, 2015](https://escholarship.org/uc/item/1jq5d0pq),
[Shafer and Vovk, 2008](https://jmlr.org/papers/v9/shafer08a.html).
