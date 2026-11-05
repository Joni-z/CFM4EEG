# Experiment records

Current targeted study: [frozen W/E recovery and matched demodulation
controls](DEMODULATION_CONTROLS.md). Its Method analysis and downstream controls
have separate outputs; no main-table jobs are launched by this study.

The consolidated local archive is:

```text
/Users/mr.z/CFM4EEG-rebuild-20261021/experiments/
```

Open `Experiment-Records.xlsx` for filtering by dataset, model version, seed,
training status and evaluation protocol. It contains six sheets: clinical
records, model versions, operator/readout studies, checkpoint locations,
records requiring review, and a short guide. Metrics display three decimals
while retaining their original values.

| File | Purpose |
|---|---|
| `runs.csv` | Clinical result records, checkpoint evaluations and latest collected training progress |
| `models.csv` | Configuration-based versions: actual tokenizer, encoding axes, physical bands, content rows and dimensions |
| `studies.csv` | Synthetic/operator trials, frozen-token readouts and linked experiment analyses |
| `checkpoints.csv` | Recorded host/path, available SHA256 and archived object location |
| `conflicts.csv` | Multiple results for matching settings/seeds and historical conflicts, with no score-based winner selection |
| `source-records.jsonl` | Original result documents and source copies, including diagnostic revisions |
| `summary.json`, `README.md` | Input hashes, coverage, collection dates and usage |

Filter `dataset`, then `model_display`, `encoder_axes` and `seed`. Follow
`source_ref` to the original record, `model_version` to its structure, and
`checkpoint_ref` to its weights. `recipe_id` separates training recipes.
Check `protocol`, `data_binding` and `metric_convention` before comparing rows.
Running records have empty test-score cells. Interim and subset evaluations
have distinct record types.

Old `raw` band tokens still use sinc filtering. WavePatch encodes unfiltered
waveform patches. Its three-axis control uses learned content rows rather than
physical frequency bands. A grid constructor named `triaxial` alone does not
establish three-axis attention; inspect `space_over_bands` and `freq_mixer`.

Identical training/config/test documents with different diagnostic verdicts
share one table row; every original document and ID remains in the source
archive. Conflicting scores remain separate. Missing fields stay unknown.
The inventory is not a new numerical certification of all historical runs.
Historical checkpoint locations do not establish current online availability.

The bounded 145 metadata collection and existing local/historical archives are
combined. The workbook records each collection's cutoff; rebuilding alone does
not query clusters. Torch could not be refreshed during the initial collection.

To rebuild after collecting new metadata:

```bash
cd /Users/mr.z/CFM4EEG-rebuild-20261021
python3 maintenance/experiment-ledger-20261105/build_ledger.py
venv/bin/python maintenance/experiment-ledger-20261105/render_workbook.py
```

Data and run outputs remain outside this source repository. Original results,
frozen execution releases, trained weights, the manuscript and the Numbers
tracking workbook are preserved.
