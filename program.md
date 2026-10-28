# Task-specific autoresearch: PAC capabilities, 2026-10-29

Upstream autoresearch README/program read; unchanged. This task adapts the fixed-evaluator/logging protocol to CPU PAC measurement, not LM training.

## Authorized scope
Validate the independent PAC operator's specific capabilities and novelty boundaries. No requirement to beat traditional PAC estimators; no downstream model training. Use a detached worktree.

## Fixed evaluator
New cfc_operator_capabilities.py evaluates only raw inputs, with fixed physical bank and unchanged old release. Test generator and scoring before freezing; record all evaluator revisions. Generator truth never configures the operator. Record target-band, source-band and total SNR. All invalid estimates remain with zero-modulation fallback and separate coverage. Record-wise predictive thresholds calibrated on independent records. No physiological causality claim.

## Run order and budget
1. Freeze and run current H1 and H2 baseline on development signals.
2. Inspect recovery and full-frequency/length coverage.
3. At most two single-mechanism development candidates only if justified by a concrete measured failure; five-minute CPU evaluation budget per candidate.
4. Freeze chosen configuration before confirmation. If confirmation informs a redesign, it becomes exposed development evidence and requires new confirm seeds.
5. Preserve every run; log keep/discard/crash/confirm in results.tsv.

## Current work bounds
Complete one finite capability-validation round, report all positive and negative findings, then return results to the user. Do not silently expand into a PAC-estimator ranking benchmark. Historical release and upstream files are immutable.

## Scientific target
Recover strength, preferred phase and curve shape; report all 30 source/target cells at 5/10/20 seconds, detection FPR/sensitivity, and local estimates including global fallback. A passing synthetic check is capability evidence, not proof of algorithmic novelty.
