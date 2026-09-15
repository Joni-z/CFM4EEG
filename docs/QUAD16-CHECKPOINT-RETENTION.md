# Quad16 checkpoint retention for release

User requires retaining the best seed checkpoint for eventual open-source release.

- Retain every Quad16 seed's validation-selected best.pt until the complete seed comparison and release archive are verified. Do not remove these files in storage cleanup. Checkpoints are small relative to datasets.
- Current trainer defaults run_monitor=True; RunControl.record atomically writes best.pt at each validation improvement. It retains the selected weights on normal completion and cooperative stop. Existing running jobs require no restart.
- Future Quad16 configs must not disable run_monitor. Keep best.pt, result.json, progress.json, exact configuration and source revision together.
- Select a release seed per dataset using the same primary validation metric and protocol; tie-break by lower seed. Do not choose from test scores. Only completed eligible runs enter final selection; identify budget-censored or flagged runs separately.
- Paper reports all seeds with mean and standard deviation. A selected release checkpoint is not a replacement for this aggregate.
- These are downstream task checkpoints, not evidence of a single universally pretrained FM checkpoint. Pretrained weights, when available, need a separate release entry.
- Before public upload, archive the source revision, preprocessing/split manifest and checkpoint SHA256, check loading and inference, and verify release permissions. No public upload has been performed by this retention task.

Initial off-cluster backup: /Users/mr.z/PACLock-monitor/quad16-release-backup, grouped by source host and worktree, containing existing best.pt plus available result/progress JSON. Later finishing seeds remain on their source host until included in the next archive. Do not claim this initial copy is continuous backup.
