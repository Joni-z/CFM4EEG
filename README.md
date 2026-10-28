# CFC operator v2: reproducible evidence

Frozen operator source: [b2e9e80](https://github.com/Joni-z/CFM4EEG/commit/b2e9e80c48eee5d93e0d15d22fc665772276f2e1), tag `cfc-operator-v2-source-20261028`.
The operator, primary benchmark, and original tests are byte-identical to published main `5d5839c`.

This evidence tree contains synthetic per-record estimates, operator settings/state, numerical audits, original execution scripts, portable reproduction scripts, and checksums. It is kept in its own evidence tag, outside the main code branch. It contains no patient recordings or trained clinical model checkpoint.

## Download and recompute

Download the GitHub archive for tag `cfc-operator-v2-evidence-20261028` and extract it. From a checkout of current main:

```bash
python -m scripts.experiments.cfc_operator_reproduce summarize \
  --root /path/to/extracted-evidence --summary-output /path/to/recomputed.json
python -m scripts.experiments.cfc_operator_numpy_audit \
  --output /path/to/numpy-audit.json
python -m scripts.experiments.cfc_operator_challenges generalization \
  --output /path/to/generalization.json --cases 200
python -m scripts.experiments.cfc_operator_challenges time-varying \
  --output /path/to/time-varying.json --cases 200
```

The summary command verifies this manifest before recomputation and leaves downloaded records unchanged. To regenerate all primary records, calibrations and factorial arms:

```bash
python -m scripts.experiments.cfc_operator_reproduce run \
  --root /path/to/new-outputs --cases 500 --calibration-cases 200
```

`reproduction_scripts/` pins the portable scripts also published on main. `original_scripts/` preserves the originally executed workspace-bound scripts for provenance. Raw signal arrays are regenerated deterministically from the pinned generator, record seed and settings; they are not supplied to the estimator as truth.

## Interpretation

The original aggregate reduction is 71.1%, with 95.2% of absolute gain coming from double/mixed-harmonic cases. The other five conditions improve 16.3%. This is a curve-recovery improvement including a larger phase-function class, not a 71% same-order estimation gain.

Same-q9, same-context controls:

| Harmonics | Fixed centers | Refined centers |
|---|---:|---:|
| First | 0.105757 | 0.094645 |
| First and second | 0.051304 | 0.028812 |

The added factorial controls use previously exposed release signals. The operator is frozen and these comparisons did not select a new candidate.

The two defined independent null types calibrate nominal 5% record-level error. Transient and nonsinusoidal records are separate interpretation stress cases, not assumed zero-dependence nulls. Cycle splitting shares filtering and frequency estimation; its score is internal. Independent-record calibration is the separate statistical check.

Frequency/length coverage and challenging conditional amplitude/phase effects are not comprehensively established. The new operator is standalone; this evidence does not establish tokenizer or clinical prediction improvements.
