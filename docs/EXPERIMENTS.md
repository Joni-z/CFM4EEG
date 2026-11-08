# Experiment records

The canonical ledger is now versioned in this repository:

- **[Browse by dataset](../experiments/README.md)** directly on GitHub.
- **[Download the complete workbook](../experiments/Experiment-Records.xlsx)** for filtering across datasets, model versions, protocols and seeds.
- [Full run records](../experiments/runs.csv), [model versions](../experiments/models.csv), [operator/readout studies](../experiments/studies.csv), [checkpoint locations](../experiments/checkpoints.csv), and [records requiring review](../experiments/conflicts.csv).

The Mac mini copy is `/Users/mr.z/CFM4EEG/experiments/`. The workbook and dataset pages come from the same CSV records; they are not independently maintained result tables. `record_id` identifies an entry, `source_ref` links its original record, and `model_version` / `recipe_id` identify the architecture and training recipe. Historical source paths remain as recorded even when their local copies have moved.

Check `protocol`, `setting`, `data_binding` and `metric_convention` before comparing scores. CHB-MIT enriched and natural-grid results are separate. Average precision is not trapezoidal PR-AUC. Running records have no test scores. A completed record is not an automatic numerical certification, and historical checkpoint locations do not imply current online availability.

The index combines archived and collected metadata. Rebuilding it does not contact clusters or refresh live job status. Collection coverage and source hashes are in [summary.json](../experiments/summary.json).

After collecting new metadata on the Mac mini:

```bash
cd /Users/mr.z/CFM4EEG
python3 scripts/experiments/build_ledger.py
.local/workspace/venv/bin/python scripts/experiments/render_workbook.py
python3 scripts/experiments/publish_records.py
```

The builder needs the retained `.local/workspace/maintenance/` collections and `.local/legacy/experiments/` archive. The workbook exporter requires openpyxl; the webpage exporter uses the Python standard library. These scripts do not train models, alter original metrics or choose best seeds.

Current targeted study: [frozen W/E recovery and matched demodulation controls](DEMODULATION_CONTROLS.md). See [project layout](PROJECT_LAYOUT.md) for source, paper and local archive locations.
