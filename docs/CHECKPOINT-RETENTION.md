# Checkpoint retention policy — effective 2026-09-17

User instruction: do not delete experiment checkpoints from now on.

- Keep weights for all completed, early-stopped, failed and cancelled experiments,
  including old architectures and controls; a weak score is not grounds for deletion.
- Preserve result.json, configuration, seed and selection metadata alongside weights.
- Never use checkpoint deletion to free disk without a new explicit user instruction.
- Existing selection checkpoints are snapshotted to `checkpoints/retained-*`.
- Newly started trainers retain every newly selected best in `checkpoints/best-*`;
  `best.pt` remains the latest validation-selected checkpoint used by evaluation.
  Hard links avoid duplicate storage until best.pt is atomically replaced.
- Live trainers already imported the previous code. They continue saving best.pt;
  today's snapshot protects the current version, but this does not retroactively
  guarantee every intervening best from those processes is archived.
- This rule concerns experiment weights, not synthetic smoke-test scratch artifacts.
- An archive/header or CRC check does not certify model-code compatibility or
  reproduce the test metrics. Those are separate release checks.
- External rented-compute seed weights have not been located or audited here.

Machine-readable audit: `checkpoint-audit-20260917.json` in this directory.
Snapshot script checks full ZIP CRC and SHA256 for completed runs; live snapshots
have a separate retained path and are not described as final release weights.
