# 实验记录

在 GitHub 上直接打开下面的数据集页面；每条记录保留模型、seed、协议、训练状态和指标。

**[下载完整实验工作簿](Experiment-Records.xlsx)** · [模型版本](models.csv) · [算子与表征实验](studies.csv) · [checkpoint 索引](checkpoints.csv) · [待核对记录](conflicts.csv)

本次索引生成时间：`2026-11-10T14:59:55.722217+08:00`。共 **2684** 条运行/评估记录、**268** 个模型版本。生成索引不代表重新查询了集群任务，采集范围见 [summary.json](summary.json)。

| 数据集 | 记录数 | 下载 |
|---|---:|---|
| [TUEV](datasets/tuev.md) | 447 | [CSV](datasets/tuev.csv) |
| [CHB-MIT](datasets/chbmit.md) | 233 | [CSV](datasets/chbmit.csv) |
| [IIIC](datasets/iiic.md) | 206 | [CSV](datasets/iiic.csv) |
| [TUSZ](datasets/tusz.md) | 211 | [CSV](datasets/tusz.csv) |
| [TUAB](datasets/tuab.md) | 123 | [CSV](datasets/tuab.csv) |
| [ISRUC](datasets/isruc.md) | 164 | [CSV](datasets/isruc.csv) |
| [Sleep-EDF](datasets/sleepedf.md) | 130 | [CSV](datasets/sleepedf.csv) |
| [ADFD](datasets/adfd.md) | 111 | [CSV](datasets/adfd.csv) |
| [TUAR](datasets/tuar.md) | 93 | [CSV](datasets/tuar.csv) |
| [TUEP](datasets/tuep.md) | 109 | [CSV](datasets/tuep.csv) |
| [CAUEEG](datasets/caueeg.md) | 100 | [CSV](datasets/caueeg.csv) |
| [apava](datasets/apava.md) | 1 | [CSV](datasets/apava.csv) |
| [bci_iv_2a](datasets/bci_iv_2a.md) | 215 | [CSV](datasets/bci_iv_2a.csv) |
| [eegmat](datasets/eegmat.md) | 16 | [CSV](datasets/eegmat.csv) |
| [faced](datasets/faced.md) | 174 | [CSV](datasets/faced.csv) |
| [mumtaz](datasets/mumtaz.md) | 52 | [CSV](datasets/mumtaz.csv) |
| [physionet_mi](datasets/physionet_mi.md) | 198 | [CSV](datasets/physionet_mi.csv) |
| [siena](datasets/siena.md) | 95 | [CSV](datasets/siena.csv) |
| [tusz_type](datasets/tusz_type.md) | 6 | [CSV](datasets/tusz_type.csv) |

## 查阅方式

[CHB-MIT 新旧协议分数审计](../docs/CHBMIT_SCORE_AUDIT.md) 单独说明旧 CfFT 与三轴 E+W 的数值来源和比较边界。

1. 按数据集进入，再对照 `protocol`、`setting`、`recipe_id` 和 seed。
2. 用 `record_id` 在完整工作簿或 [runs.csv](runs.csv) 中查询；`source_ref` 指向原始记录，`checkpoint_ref` 指向权重记录。
3. `completed_record` 只说明存在完成记录；历史结果不因收入本表就自动通过新的数值审计。
4. `verified_completion_receipt` 是已保存的审核凭据补录，见 [receipts](receipts/README.md)，不是新训练。
5. CHB-MIT 的 `legacy_enriched_17386` 与 `natural_grid_17224` 分开；AP 与梯形 PR-AUC 分列。

## 完整文件

| 文件 | 内容 |
|---|---|
| [runs.csv](runs.csv) | 全部字段和原始精度；文件较大，网页阅读请用上面的分数据集页面 |
| [source-records.jsonl](source-records.jsonl) | 原始结果文档及来源副本索引 |
| [summary.json](summary.json) | 收集范围、源文件哈希及数量 |
| [publication-check.json](publication-check.json) | 本次网页导出和记录完整性检查 |

## 更新

收集新结果元数据后，在项目根目录运行：

```bash
python3 scripts/experiments/build_ledger.py
.local/workspace/venv/bin/python scripts/experiments/render_workbook.py
python3 scripts/experiments/publish_records.py
```

前两步需要 Mac mini 上的本地历史记录/采集文件和 openpyxl；网页导出只需 Python 标准库。这些命令不启动训练、不自动连接集群、不重选 checkpoint。
