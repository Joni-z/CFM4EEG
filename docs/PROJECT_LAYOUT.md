# Project locations

The active project is **`/Users/mr.z/CFM4EEG`** on the Mac mini.
This directory itself is the Git checkout for
[Joni-z/CFM4EEG](https://github.com/Joni-z/CFM4EEG), branch `main`.

## On GitHub

| Entry | Contents |
|---|---|
| [Experiment records](../experiments/README.md) | Dataset pages, complete workbook, original-precision CSVs and source records |
| [Model guide](MODEL.md) | Model implementation, configurations and training commands |
| [Baseline guide](UPSTREAM.md) | Author sources and reproduction entrypoints |
| [Data guide](DATA.md) | Dataset protocols and preprocessing |
| [Demodulation paper](../papers/demodulation/README.md) | Repository manuscript, figures and compiled PDF |
| [Audit guide](AUDIT.md) | Completed checks and interpretation boundaries |

The repository paper is distinct from independent MBP or Overleaf edits;
those edits enter GitHub only after an explicit manuscript sync. Directory
consolidation does not merge competing manuscripts or change their content.

## Local files retained under the same root

Large artifacts and older working copies are excluded from Git:

| Local path under `/Users/mr.z/CFM4EEG/` | Contents |
|---|---|
| `.local/workspace/` | Previous rebuild workspace: data caches, environments, collected evidence and execution snapshots |
| `.local/workspace/maintenance/` | Existing task notes, native-run audit receipts and metadata collections |
| `.local/legacy/` | Historical CFM4EEG archive: older code, experiments, papers and checkpoint backups |
| `.local/cvpr-preview/` | Earlier CVPR layout previews and compilation outputs |
| `.local/legacy-links/` | Retained compatibility link objects removed from the home directory listing |

The old `CFM4EEG-rebuild-20261021/repo` is now the project root. Its experiment
ledger moved from outside Git into `experiments/`. The old top-level CFM4EEG
and PACLock links are no longer daily entry points.

Migration records are retained locally at
`.local/workspace/maintenance/project-consolidation-20261108/`: `move-map.json`
maps old paths to new paths, and `migration-verification.json` records the
file-preservation checks. Original result documents retain their recorded
historical paths; the move map locates their current local copies.

Raw EEG, prediction arrays, checkpoint binaries and environments remain local
or in their existing external archives. GitHub publishes the corresponding
metadata and checkpoint locations, not these large artifacts.
