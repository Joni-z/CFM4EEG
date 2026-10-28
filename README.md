# CFC operator readiness evidence — 2026-10-29

This archive records PAC measurement behavior and engineering checks before
connecting the component to a tokenizer. It contains synthetic per-record
measurements, complete development/confirmation logs, plots, diagnostic
reports, calibration scores, and fixed non-trained operator state.

## Source and execution identity

- Clean interface source: `cfc-ready-source-20261029` on Joni-z/CFM4EEG.
- Full development history: `cfc-ready-development-20261029`.
- Numerical confirmation executed from `769b44b97367c237a4690602857cb8bd6f35bc88`,
  an ancestor retained by the development tag. `frozen-dev.json` records drivers.
- `frozen-algorithm-equivalence.json` checks that final interface-only changes
  did not alter the confirmed core regression, source analysis, or covariance.
- Early development used the byte-exact `run_candidate_initial.py`; confirmation
  used `run_candidate.py`. Both are retained with their recorded hashes.
- The local companion originally ran with an inline seed override. Its original
  manifest remains intact; `reproduction.json` and `run_local_companion.py`
  preserve the actual seed 36000000 rather than rewriting execution provenance.

## Reading order

1. `report.md`: measured findings and limits.
2. `ready-summary.json`: numerical values; `ready-validation.pdf`: overview.
3. `results.tsv`: all keep/discard/crash/confirmation entries.
4. Per-run `manifest.json`, `records.json[.gz]`, `summary.json`.
5. `completion-audit.json`, `final-tests.log`, engineering JSON reports.

Original absolute paths are retained as provenance, not portable commands.
There are 10,960 distinct synthetic raw-input hashes across the six
confirmation/reference collections counted by `independence-audit.json`.
Paired output variants do not add independent records. Oracle diagnostic arms
are separate, and do not provide ground truth to the candidate.

## Reproduce

Clone Joni-z/CFM4EEG and fetch the tags above. Use Python 3.12, NumPy 2.5.1,
SciPy 1.18.0, and Torch 2.14.1 for the recorded Mac CPU environment.
Checkout the numerical execution commit in a separate worktree, then run:

```sh
PYTHONPATH=/path/to/execution-source python run_candidate.py --help
PYTHONPATH=/path/to/execution-source python carrier_check.py --help
PYTHONPATH=/path/to/execution-source python calibrate_records.py --help
```

Exact arguments/configurations and streams are recorded per run. CPU thread
count changes runtime, not the protocol. For the final interface, use the clean
source tag and follow `docs/CFC_OPERATOR_HANDOFF.md`. A copy of this archive can
recompute the numerical summary/plot with:

```sh
PYTHONPATH=/path/to/clean-source python summarize_ready.py
```

Run that in a disposable archive copy because it writes summary/plot files.
`audit_completion.py` additionally assumes the recorded local repository path
for Git checks; adapt that path when reproducing elsewhere.

## Calibration and feature meanings

`calibration-final-meta-10s/` contains the final source/state/runtime-bound
calibration objects. Thresholds/ranks use exactly the same 2,000 reference scores
as their originals; only identity binding changed, not statistical selection.
These objects apply to the specified synthetic one-channel 10-second null
mixture, under exchangeability/iid assumptions. They are not universal clinical
EEG significance thresholds. Continuous features need no threshold gate.

Raw harmonic coefficients remain the default. Stabilization helps weak
coherent modulation but increases bias for some strong stochastic mixed cases.
Covariance is a diagnostic, not a calibrated confidence interval. Local mode
with fallback remains separate from the global uncertainty wrapper. Unsupported
pairs/harmonics remain visible through masks.

## Real EEG and source-tree boundary

No licensed EEG arrays, labels, trained downstream weights, or actual
`engineering-inputs.npz` are included. Engineering JSONs record finite-value,
gradient, shape, CPU/CUDA and resource observations plus an input hash only.
The included `.pt` stores fixed physical filters and zero learned parameters.
No tokenizer/encoder implementation or downstream training was changed here.

`SHA256SUMS` covers every evidence file at publication (excluding itself).
Remote publication verification is recorded outside this immutable archive.
