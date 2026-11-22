# Repository guide

Develop in the `main` branch of [Joni-z/CFM4EEG](https://github.com/Joni-z/CFM4EEG). On the Mac mini, the checkout is `/Users/mr.z/CFM4EEG`.

| Content | Location |
|---|---|
| EEG models, tokenizers and representation probes | [Python package](../paclock_bench/) · [model guide](MODEL.md) · [PAC probes](PAC_PROBE.md) |
| Clinical preprocessing and configurations | [preprocessing](../preprocessing/) · [dataset definitions](../configs/datasets/) · [protocols](DATA.md) |
| Baseline reproduction | [author sources](../upstream/sources.json) · [training guide](UPSTREAM.md) · [environment definitions](../environments/) |
| Experiments | [ledger and workbook](../experiments/README.md) · [results by question](../experiments/audits/README.md) · [archived weight references](../experiments/checkpoints.csv) |
| Manuscripts | [PAC](../papers/pac/README.md) · [demodulation](../papers/demodulation/README.md) |
| Talks | [Editable decks and Google Slides](../presentations/README.md) |

`scripts/rebuild/` contains the native baseline launchers referenced by the reviewed training configurations. `slurm/rebuild/` contains their scheduler entrypoints. Both are executable components of this checkout.

## Historical sources

Earlier versions live at fixed Git revisions. Open one of the links below to read the code, recipe or manuscript as it existed for that study.

| Study | Source |
|---|---|
| PAC control experiments | [Archived model code](https://github.com/Joni-z/CFM4EEG/tree/5d47754bba1cd36106cfa17853128a67d9fddbc8/paclock_bench/models) · [configurations](https://github.com/Joni-z/CFM4EEG/tree/5d47754bba1cd36106cfa17853128a67d9fddbc8/configs) |
| Two-axis CfFT | [Archived model code](https://github.com/Joni-z/CFM4EEG/tree/1fe7ecb81ee26cf7ec092488e441a2293e55ee22/paclock_bench/models) · [configurations](https://github.com/Joni-z/CFM4EEG/tree/1fe7ecb81ee26cf7ec092488e441a2293e55ee22/configs) |
| Geometric fusion on TUEV | [Fusion module](https://github.com/Joni-z/CFM4EEG/blob/8ec35d20375bcc8f8959fa97e0341355f65461b9/paclock_bench/models/paclock/frontend/geometric_fusion.py) · [control recipes](https://github.com/Joni-z/CFM4EEG/tree/8ec35d20375bcc8f8959fa97e0341355f65461b9/configs/geometric) · [original seed-0 test result](../experiments/originals/tuev/geometric-fusion/seed0/result.json) |
| Earlier PAC manuscripts | [ICLR-format draft](https://github.com/Joni-z/CFM4EEG/tree/08d94356e6b74036111b307d355c44eef2134686/papers/pac) · [subsequent two-column draft](https://github.com/Joni-z/CFM4EEG/tree/16d1e4a39092abe228e18dc8b12b643d9518ad39/papers/pac) |
| Earlier demodulation manuscript | [Section-based draft and supporting evidence](https://github.com/Joni-z/CFM4EEG/tree/16d1e4a39092abe228e18dc8b12b643d9518ad39/papers/demodulation) |

[Early presentation files](https://github.com/Joni-z/CFM4EEG/tree/c03305b0741244a41f710078846fe9706ca5b796/presentations), [PAC figure-analysis sources](https://github.com/Joni-z/CFM4EEG/tree/c03305b0741244a41f710078846fe9706ca5b796/papers/pac/figure-analysis) and [research notes](https://github.com/Joni-z/CFM4EEG/blob/c03305b0741244a41f710078846fe9706ca5b796/docs/POLYSPECTRAL-POSITIONING-20261009.md) are preserved alongside the earlier manuscripts.

The historical code revisions overlay the preserved runtime source files on their archived project base. The manuscript snapshots retain the source text. Use the experiment ledger for measured scores.

## Local and remote artifacts

Large EEG arrays, weights, cached environments, frozen execution directories and private operational notes stay in their original storage. [Checkpoint records](../experiments/checkpoints.csv) identify recorded locations; [storage receipts](../experiments/storage/README.md) document transfers and checksum checks. Dataset access restrictions continue to apply.

On the Mac mini these files sit under the ignored `.local/` directory. `.local/legacy/` holds the historical archive, and `.local/workspace/` holds the data and receipts used to build the ledger. Its `maintenance/project-consolidation-20261108/move-map.json` resolves pre-migration paths. Run and checkpoint records retain the original paths for provenance.
