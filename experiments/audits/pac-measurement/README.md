# Original CfFT: known-PAC measurement

[Results and interpretation](analysis-report.md) · [exact conditions](conditions.csv) · [uncertainty](stats-appendix.md) · [numeric verification](independent-check.json)

Measurement-only CPU assay of the original two-axis measured seed-2 checkpoint at 0.25 s. Stops at Z before token construction; no model changes, training or probe.

Run from the repository root with the existing CPU environment:

```bash
.local/workspace/venv/bin/python scripts/experiments/pac_original_measurement.py \
  --checkpoint .local/legacy/checkpoints/objects/1fb2b132f935d890373f38f47ee2b5e99c796b3d5d9b76773c74b94b3e8b1a3c.pt \
  --output .local/workspace/maintenance/pac-measurement-original-20261118/reproduction
```

The output path must be new. Full per-recording outputs remain under ignored `.local/`; their hash and complete generator settings are published here. Synthetic measurement scores stay separate from clinical benchmark scores.
