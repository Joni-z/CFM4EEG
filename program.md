# Task-specific autoresearch: CFC ready for tokenizer handoff

Upstream /Users/mr.z/autoresearch README.md and program.md read; unchanged. Adapted task is CPU PAC measurement, not LM pretraining.

User goal: complete the independent operator until the step immediately before putting it into a tokenizer run. Use requirements R1–R7 in task_plan.md as the completion contract; do not shrink it to a test-only patch.

Editable: isolated operator implementation, operator diagnostics/tests and readiness documentation. Do not change existing tokenizer behavior or train downstream models. No forced traditional-estimator competition.

Baseline first. Fixed development inputs and metrics, max five-minute CPU runtime per candidate. Keep/discard/crash decisions require joint recovery, physical coverage, calibration and cost evidence; no metric cherry-picking. Keep old signals and artifacts immutable. Ground-truth phase/envelope may be used only in explicitly named diagnostic oracle arms, never in the raw-input candidate.

Before each independent confirmation, freeze operator/evaluator/configs and new random streams. Any redesign after seeing confirmation makes that split exposed; use a new confirmation stream. Record raw hashes, config/source hashes, all invalid and fallback cases, runtime and input units.

Do not claim new PAC-estimation theory merely from Fourier regression, cross-validation, noise-moment correction or rank calibration; state reused methods and the actual method combination and intended interface.

Continue until R1–R7 are verified or a genuine external blocker occurs. The stopping point is a ready and audited measurement component, not tokenizer training.
