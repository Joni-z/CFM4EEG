#!/usr/bin/env python3
"""Build GitHub-readable views of the existing CSV experiment ledger.

No training, metric recomputation, aggregation, or best-run selection.
Full-precision CSVs and original source documents remain unchanged.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS = ROOT / "experiments"
NAMES = {
    "tuev": "TUEV", "chbmit": "CHB-MIT", "iiic": "IIIC", "tusz": "TUSZ",
    "tuab": "TUAB", "isruc": "ISRUC", "sleepedf": "Sleep-EDF", "adfd": "ADFD",
    "tuar": "TUAR", "tuep": "TUEP", "caueeg": "CAUEEG",
}
FIELDS = [
    "record_id", "run_name", "model_display", "model_version", "setting", "seed",
    "status", "evidence_type", "protocol", "test_windows", "primary_metric",
    "primary_value", "cohen_kappa", "pr_auc", "average_precision", "auroc",
    "balanced_acc", "weighted_f1", "encoder_axes", "physical_bands", "recipe_id",
    "data_binding", "checkpoint_sha256", "source_ref", "checkpoint_ref",
]


def read_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def escape(value):
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def score(value):
    return f"{float(value):.6f}" if value else ""


def main():
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in RECORDS.iterdir() if p.is_file() and p.name != "README.md"}
    rows = read_rows(RECORDS / "runs.csv")
    assert len({r["record_id"] for r in rows}) == len(rows)
    counts = Counter(r["dataset"] for r in rows)
    summary = json.loads((RECORDS / "summary.json").read_text())
    assert summary["clinical_records"] == len(rows)
    folder = RECORDS / "datasets"
    folder.mkdir(exist_ok=True)
    dataset_links = []
    recovered_ids = set()
    order = [*NAMES, *sorted(set(counts) - set(NAMES))]
    for dataset in order:
        if dataset not in counts:
            continue
        selected = [r for r in rows if r["dataset"] == dataset]
        selected.sort(key=lambda r: (r["protocol"], r["model_display"], r["setting"],
                                     r["run_name"], r["seed"], r["record_id"]))
        csv_path = folder / f"{dataset}.csv"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({key: r.get(key, "") for key in FIELDS} for r in selected)
        exported = read_rows(csv_path)
        assert exported == [{key: r.get(key, "") for key in FIELDS} for r in selected]
        recovered_ids.update(r["record_id"] for r in exported)
        title = NAMES.get(dataset, dataset)
        protocols = Counter(r["protocol"] for r in selected)
        text = [f"# {title} 实验记录", "", "[返回总入口](../README.md)", "",
                f"共 {len(selected)} 条记录。[完整精度及来源 CSV]({dataset}.csv)。",
                "同一表中的数据协议、模型版本和训练配置可能不同，请先核对协议；本页不选择最佳 seed、不合并跨协议成绩。",
                "空白表示原记录缺失；没有测试结果的训练记录保留空白。显示六位小数，CSV 保留原始精度。", ""]
        for protocol, count in sorted(protocols.items()):
            text += [f"## {escape(protocol)}（{count} 条）", "",
                     "| 记录 ID | 模型 / run | 初始化 | seed | 状态 / 类型 | κ | PR-AUC | AP | AUROC | Bal. acc. | wF1 |",
                     "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
            for r in selected:
                if r["protocol"] != protocol:
                    continue
                cells = [f'`{r["record_id"]}`',
                         f'{escape(r["model_display"])}<br>{escape(r["run_name"])}',
                         escape(r["setting"]), escape(r["seed"]),
                         f'{escape(r["status"])}<br>{escape(r["evidence_type"])}',
                         *[score(r[key]) for key in ["cohen_kappa", "pr_auc", "average_precision",
                                                    "auroc", "balanced_acc", "weighted_f1"]]]
                text.append("| " + " | ".join(cells) + " |")
            text.append("")
        (folder / f"{dataset}.md").write_text("\n".join(text))
        dataset_links.append(f"| [{title}](datasets/{dataset}.md) | {len(selected)} | [CSV](datasets/{dataset}.csv) |")
    assert recovered_ids == {r["record_id"] for r in rows}
    readme = ["# 实验记录", "",
              "在 GitHub 上直接打开下面的数据集页面；每条记录保留模型、seed、协议、训练状态和指标。", "",
              "**[下载完整实验工作簿](Experiment-Records.xlsx)** · "
              "[模型版本](models.csv) · [算子与表征实验](studies.csv) · "
              "[checkpoint 索引](checkpoints.csv) · [待核对记录](conflicts.csv)", "",
              f'本次索引生成时间：`{summary["updated_at"]}`。共 **{len(rows)}** 条运行/评估记录、'
              f'**{summary["model_versions"]}** 个模型版本。生成索引不代表重新查询了集群任务，采集范围见 [summary.json](summary.json)。', "",
              "| 数据集 | 记录数 | 下载 |", "|---|---:|---|", *dataset_links, "",
              "## 查阅方式", "",
              "1. 按数据集进入，再对照 `protocol`、`setting`、`recipe_id` 和 seed。",
              "2. 用 `record_id` 在完整工作簿或 [runs.csv](runs.csv) 中查询；`source_ref` 指向原始记录，`checkpoint_ref` 指向权重记录。",
              "3. `completed_record` 只说明存在完成记录；历史结果不因收入本表就自动通过新的数值审计。",
              "4. `verified_completion_receipt` 是已保存的审核凭据补录，见 [receipts](receipts/README.md)，不是新训练。",
              "5. CHB-MIT 的 `legacy_enriched_17386` 与 `natural_grid_17224` 分开；AP 与梯形 PR-AUC 分列。", "",
              "## 完整文件", "",
              "| 文件 | 内容 |", "|---|---|",
              "| [runs.csv](runs.csv) | 全部字段和原始精度；文件较大，网页阅读请用上面的分数据集页面 |",
              "| [source-records.jsonl](source-records.jsonl) | 原始结果文档及来源副本索引 |",
              "| [summary.json](summary.json) | 收集范围、源文件哈希及数量 |",
              "| [publication-check.json](publication-check.json) | 本次网页导出和记录完整性检查 |", "",
              "## 更新", "",
              "收集新结果元数据后，在项目根目录运行：", "", "```bash",
              "python3 scripts/experiments/build_ledger.py",
              ".local/workspace/venv/bin/python scripts/experiments/render_workbook.py",
              "python3 scripts/experiments/publish_records.py", "```", "",
              "前两步需要 Mac mini 上的本地历史记录/采集文件和 openpyxl；网页导出只需 Python 标准库。"
              "这些命令不启动训练、不自动连接集群、不重选 checkpoint。", ""]
    (RECORDS / "README.md").write_text("\n".join(readme))
    for name, digest in before.items():
        assert hashlib.sha256((RECORDS / name).read_bytes()).hexdigest() == digest, name
    check = {"status": "PASS", "clinical_records": len(rows), "dataset_pages": len(counts),
             "every_record_exported_once": True, "compact_csv_values_exact": True,
             "source_files_unchanged_by_web_export": True,
             "scope": "Record export verification; not a re-audit of every historical training run."}
    (RECORDS / "publication-check.json").write_text(json.dumps(check, indent=2) + "\n")
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
