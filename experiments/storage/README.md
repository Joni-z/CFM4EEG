# Mac mini storage cleanup

2026-11-10. This records deletion of unused local caches, not deletion of
experimental results or a change to training recipes.

| Removed category | Files | Allocated space |
|---|---:|---:|
| Old dataset signal-array staging copies | 21 | 75.71 GiB |
| Extracted token/feature matrices from older analysis tasks | 102 | 28.00 GiB |
| Total | 123 | 103.71 GiB |

The dataset copies were under
`.local/legacy/workspaces/PACLock-monitor/quad16-breadth/staging/` for ADFD,
CAUEEG, CHB-MIT, ISRUC, Siena, Sleep-EDF and TUAR. Only `*_signals.npy`
payloads were removed; all 21 label arrays and seven original manifests remain.
Each affected dataset directory contains a small local removal marker.

The feature caches came from the six older analysis directories enumerated in
the deletion manifest. Their source, checkpoint inputs, result summaries,
prediction artifacts and small fixtures were retained. Reusing the deleted
matrices now requires re-extraction; deletion does not change their recorded
experimental measurements.

## Preservation checks

- All 21,041 protected result, prediction, checkpoint, paper and metadata files
  retained their SHA256 values after deletion.
- No checkpoint or result/prediction artifact was deleted.
- The current source checkout, original manuscript and cluster files were not
  changed. No training was launched or stopped.
- Exact current-user open-file checks found no candidate payload in use before
  deletion. Each path was also checked against its saved inode, size, mtime and
  allowed category before unlinking.
- Free available space increased from about 12.31 GiB to 116.01 GiB at the
  immediate post-cleanup check. Available capacity can subsequently change.

## Recoverability and limits

The three CHB-MIT signal files have a current copy on145 at
`/data2/zz5070/CFM4EEG-rebuild-20261021/legacy-triaxial-validation-20261105/data/chbmit/`.
Full SHA256 and file-size comparisons matched every byte of the local copies
before deletion. This is the old enriched dataset, distinct from the corrected
natural-grid copy.

AMD and Torch authentication were unavailable during this pass. The remaining
six obsolete staging datasets were deleted under the user's authorization;
their original transfer/preprocessing sources and local metadata remain, but
this cleanup does **not** certify a currently accessible remote backup for
each one. Restoring them may require reconnecting to the original data host
or rerunning preprocessing.

## Records

- [Exact deletion manifest](20261110/cleanup-manifest.json)
- [Deletion and preservation summary](20261110/cleanup-result.json)
- [Full CHB-MIT remote/local checksum comparison](20261110/chbmit-backup-verified.json)

The complete protected-file inventory and execution log remain locally at
`/Users/mr.z/CFM4EEG/.local/workspace/maintenance/storage-cleanup-20261110/`.

## 145 checkpoint archive (2026-11-21)

The [private Hugging Face archive](https://huggingface.co/JoniZz/CFM4EEG-checkpoint-archive/tree/cdb6374ac043679d8005e7f69f6d0c811f2b622d)
contains the 98 distinct checkpoint objects from this inventory, including 56 new
uploads. The archive index maps original paths to SHA-256 objects. An archived
checkpoint was downloaded on 145 and verified by SHA-256 before removing 76 cold
local copies. This freed at least 583,819,264 bytes on 145. The weights needed
for current readouts and Erase, vendor weights, and EEG datasets remain local.

The exact offload receipt on 145 is
`/data2/zz5070/.maintenance/storage-145-compact-20261121/released.jsonl`.
Restore an offloaded path on 145 with the existing archive client:

```sh
/data2/zz5070/.maintenance/hf-checkpoint-archive-20261026/client-env/bin/python \
  /data2/zz5070/.maintenance/storage-145-compact-20261121/restore.py \
  "$original_path"
```
