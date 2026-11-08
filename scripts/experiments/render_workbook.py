#!/usr/bin/env python3
"""Render the experiment ledger CSVs as a static, auditable Excel workbook.

Usage (from any directory):
  venv/bin/python maintenance/experiment-ledger-20261105/render_workbook.py
  venv/bin/python maintenance/experiment-ledger-20261105/render_workbook.py --verify-only

CSV files remain authoritative. The renderer preserves source order, every column,
and full numeric precision available in Excel; it performs no result selection or
scientific aggregation. Numeric comments retain the exact source CSV spelling.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter


REPO = Path(__file__).resolve().parents[2]
SHEETS = (
    ("runs.csv", "实验总表"),
    ("models.csv", "模型版本"),
    ("studies.csv", "算子与表征"),
    ("checkpoints.csv", "检查点"),
    ("conflicts.csv", "待核对"),
)
FONT = "Arial"
HEADER_FILL = PatternFill("solid", fgColor="354757")
ALT_FILL = PatternFill("solid", fgColor="F4F6F8")
GRAY_FILL = PatternFill("solid", fgColor="E7E9EB")
COMPLETE_FILL = PatternFill("solid", fgColor="EDF1EE")
GRID = Border(bottom=Side(style="hair", color="E2E6EA"))
NUMERIC = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
INTEGER = re.compile(r"^[+-]?\d+$")
METRIC = re.compile(
    r"(?:auc|aupr|auroc|prauc|f1|accuracy|balanced_acc|precision|recall|"
    r"kappa|mcc|sensitivity|specificity|loss|rmse|mae|mse|r2|r_squared|"
    r"pearson|spearman|score|metric)", re.I
)
TEXT_TOKEN = re.compile(
    r"(?:^|_)(?:id|seed|hash|sha\d*|version|path|ref|source|config|"
    r"checkpoint|name|label|protocol|split|dataset|task|status|state|kind|"
    r"type|mode|family|head)(?:$|_)", re.I
)
HIDDEN_TOKEN = re.compile(
    r"(?:source_ref|source_path|config_path|config_ref|checkpoint_path|"
    r"checkpoint_ref|artifact_path|output_dir|source_files|source_refs|"
    r"provenance_json|metrics_json|evidence_path|evidence_ref)", re.I
)
INCOMPLETE = re.compile(
    r"(?:incomplete|running|stopped|stop_requested|partial|pending|queued|"
    r"failed|aborted|interrupted|terminated|not_run|missing|unverified|"
    r"待核对|未完成|运行中|已停止|中断|失败)", re.I
)
COMPLETE = re.compile(r"^(?:completed?|finished|succeeded|success|done|已完成)$", re.I)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    csv.field_size_limit(100_000_000)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = list(reader.fieldnames or [])
        if not headers or any(not key for key in headers) or len(set(headers)) != len(headers):
            raise ValueError(f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    for index, row in enumerate(rows, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {index} has an inconsistent column count: {path}")
    return headers, rows


def excel_columns(headers: list[str], rows: list[dict[str, str]]) -> tuple[list[str], list[dict[str, str]], dict[str, int]]:
    """Split long source fields into hidden continuation columns, without adding rows."""
    parts = {header: max(1, max((math.ceil(len(row[header]) / 32000) for row in rows), default=1)) for header in headers}
    expanded_headers: list[str] = []
    for header in headers:
        expanded_headers.append(header)
        for part in range(2, parts[header] + 1):
            continuation = f"{header}__part_{part}"
            if continuation in headers:
                raise ValueError(f"Continuation header collides with a source header: {continuation}")
            expanded_headers.append(continuation)
    expanded_rows: list[dict[str, str]] = []
    for row in rows:
        expanded: dict[str, str] = {}
        for header in headers:
            raw = row[header]
            expanded[header] = raw[:32000] if parts[header] > 1 else raw
            for part in range(2, parts[header] + 1):
                expanded[f"{header}__part_{part}"] = raw[(part - 1) * 32000:part * 32000]
        expanded_rows.append(expanded)
    return expanded_headers, expanded_rows, {header: count for header, count in parts.items() if count > 1}


def is_metric(header: str) -> bool:
    # Metadata such as metric_name / metric_source remains text.
    return bool(METRIC.search(header)) and not bool(TEXT_TOKEN.search(header))


def as_cell_value(header: str, raw: str) -> Any:
    if raw == "":
        return None
    if TEXT_TOKEN.search(header):
        return raw
    stripped = raw.strip()
    if not NUMERIC.fullmatch(stripped):
        return raw
    if INTEGER.fullmatch(stripped):
        digits = stripped.lstrip("+-")
        # Excel cannot faithfully store long integers, and leading zeros are data.
        if len(digits) > 15 or (len(digits) > 1 and digits.startswith("0")):
            return raw
        return int(stripped)
    value = float(stripped)
    return value if math.isfinite(value) else raw


def write_value(cell: Any, value: Any) -> None:
    if isinstance(value, str) and len(value) > 32767:
        raise ValueError(f"Excel text limit exceeded at {cell.coordinate}")
    cell.value = value
    # An evidence string may begin '=' or resemble #N/A; never create a formula/error.
    if isinstance(value, str):
        cell.data_type = "s"


def column_width(header: str, values: list[str]) -> float:
    key = header.lower()
    if HIDDEN_TOKEN.search(key):
        return 34
    if is_metric(key):
        return max(13, min(22, len(header) + 2))
    if any(part in key for part in ("note", "reason", "description", "limitation", "comment")):
        return 46
    if any(part in key for part in ("status", "state", "protocol")):
        return 24
    if any(part in key for part in ("hash", "sha")):
        return 22
    sample_length = max((min(len(value), 50) for value in values[:250]), default=0)
    return max(14, min(34, max(len(header) + 2, sample_length + 2)))


def style_table(sheet: Any, headers: list[str], rows: list[dict[str, str]]) -> None:
    sheet.sheet_view.showGridLines = False
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.orientation = "landscape"
    sheet.page_setup.paperSize = sheet.PAPERSIZE_A3
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.print_title_rows = "1:1"
    sheet.auto_filter.ref = sheet.dimensions
    leading_ids = 0
    for header in headers[:3]:
        if re.search(r"(?:^|_)id$", header, re.I):
            leading_ids += 1
        else:
            break
    sheet.freeze_panes = f"{get_column_letter(max(1, leading_ids) + 1)}2"
    sheet.row_dimensions[1].height = 32
    for index, header in enumerate(headers, 1):
        cell = sheet.cell(1, index)
        cell.font = Font(name=FONT, size=10, bold=True, color="FFFFFF")
        cell.fill = HEADER_FILL
        cell.alignment = Alignment(vertical="center", wrap_text=True)
        letter = get_column_letter(index)
        sheet.column_dimensions[letter].width = column_width(header, [row[header] for row in rows])
        if HIDDEN_TOKEN.search(header) or "__part_" in header:
            sheet.column_dimensions[letter].hidden = True
            cell.comment = Comment("完整保留的技术来源列；取消隐藏即可查看。权威原文仍在同名 CSV 列中。", "Codex")
    for row_index, row in enumerate(rows, 2):
        sheet.row_dimensions[row_index].height = 32
        for column_index, header in enumerate(headers, 1):
            cell = sheet.cell(row_index, column_index)
            cell.font = Font(name=FONT, size=10, color="263440")
            cell.border = GRID
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            if row_index % 2 == 0:
                cell.fill = ALT_FILL
            if isinstance(cell.value, (int, float)):
                cell.number_format = "0.000" if is_metric(header) or isinstance(cell.value, float) else "0"
                cell.alignment = Alignment(vertical="top", horizontal="right")
                if is_metric(header):
                    cell.comment = Comment(f"原始 CSV 数值：{row[header]}\n显示格式为三位小数，未按显示位数取整。", "Codex")
            else:
                cell.number_format = "@"
            if re.search(r"(?:status|state)", header, re.I) and isinstance(cell.value, str):
                if INCOMPLETE.search(cell.value):
                    cell.fill = GRAY_FILL
                    cell.font = Font(name=FONT, size=10, color="5F666D")
                elif COMPLETE.fullmatch(cell.value.strip()):
                    cell.fill = COMPLETE_FILL
            if isinstance(cell.value, str) and cell.value.startswith(("https://", "http://")):
                cell.hyperlink = cell.value


def add_readme(workbook: Workbook, input_dir: Path, source_data: dict[str, Any], summary: Any) -> None:
    sheet = workbook.create_sheet("说明")
    rows: list[tuple[str, str]] = [
        ("用途", "实验记录 CSV 的只读式结果视图；用于检索和核对，CSV 为权威记录。"),
        ("输入目录", str(input_dir)),
        ("生成时间", datetime.now().astimezone().isoformat(timespec="seconds")),
        ("来源记录更新时间", str(summary.get("updated_at", "未记录"))),
        ("远端采集截止", str(summary.get("remote_collection", "未记录"))),
        ("输入文件数", str(summary.get("input_files", "未记录"))),
        ("范围", "保留输入行顺序与全部列；不挑选最佳 seed，不跨协议汇总，不补造实验条目。"),
        ("数值", "指标显示三位小数；数值单元格未按显示位数取整，批注保留原始 CSV 数值文本。"),
        ("文本", "ID、seed、hash、协议及来源字段按文本保留；长技术路径列可取消隐藏查看。"),
        ("超长原文", "超过 Excel 单格长度的原文按 32000 字符分入同名隐藏续列；按 __part_2、__part_3 顺序连接即可恢复，CSV 原文不变。"),
        ("状态", "未完成、运行中、停止及待核对状态使用灰色；完成状态使用浅灰绿色。"),
        ("源文件边界", "生成过程只读取 CSV/JSON；不修改论文、Numbers、代码、模型权重或实验配置。"),
        ("完整来源与哈希", "全部输入路径及 input_hashes 保留在同目录 summary.json 中。"),
    ]
    for filename, title in SHEETS:
        headers, records = source_data[filename]
        rows.append((f"{title} / {filename}", f"{len(records)} 条记录，{len(headers)} 个原始字段"))
    for key, label in (("historical_unique_results", "历史唯一结果"), ("trial_rows", "试验行数"), ("conflict_groups", "待核对分组数"), ("by_status", "临床记录状态"), ("by_dataset", "临床记录数据集"), ("parse_errors", "解析错误"), ("limits", "来源限制")):
        if key in summary:
            value = summary[key]
            text = json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else str(value)
            rows.append((label, text))
    sheet.append(["项目", "内容"])
    for key, value in rows:
        write_value(sheet.cell(sheet.max_row + 1, 1), key)
        write_value(sheet.cell(sheet.max_row, 2), value)
    style_table(sheet, ["项目", "内容"], [{"项目": key, "内容": value} for key, value in rows])
    sheet.column_dimensions["A"].width = 40
    sheet.column_dimensions["B"].width = 110
    sheet.freeze_panes = "B2"
    for row in range(2, sheet.max_row + 1):
        sheet.row_dimensions[row].height = min(90, max(30, 16 * (len(str(sheet.cell(row, 2).value)) // 90 + 1)))


def verify(output: Path, source_data: dict[str, Any]) -> dict[str, Any]:
    workbook = load_workbook(output, data_only=False)
    expected_titles = [title for _, title in SHEETS] + ["说明"]
    if workbook.sheetnames != expected_titles:
        raise AssertionError(f"Unexpected sheets: {workbook.sheetnames}")
    counts: dict[str, int] = {}
    numeric_cells = 0
    metric_cells = 0
    text_ids = 0
    continuation_columns: dict[str, int] = {}
    oversized_source_cells = 0
    for filename, title in SHEETS:
        source_headers, source_records = source_data[filename]
        headers, records, parts = excel_columns(source_headers, source_records)
        sheet = workbook[title]
        if sheet.max_row != len(records) + 1 or sheet.max_column != len(headers):
            raise AssertionError(f"Row/column count mismatch: {title}")
        if [cell.value for cell in sheet[1]] != headers:
            raise AssertionError(f"Header mismatch: {title}")
        for row_index, record in enumerate(records, 2):
            for column_index, header in enumerate(headers, 1):
                cell = sheet.cell(row_index, column_index)
                expected = as_cell_value(header, record[header])
                if isinstance(expected, (int, float)):
                    if not isinstance(cell.value, (int, float)) or not math.isclose(cell.value, expected, rel_tol=1e-14, abs_tol=1e-15):
                        raise AssertionError(f"Numeric precision mismatch: {title}!{cell.coordinate}: {expected!r} -> {cell.value!r}")
                    numeric_cells += 1
                    if is_metric(header):
                        if cell.number_format != "0.000" or not cell.comment or record[header] not in cell.comment.text:
                            raise AssertionError(f"Metric display/raw-value mismatch: {title}!{cell.coordinate}")
                        metric_cells += 1
                elif cell.value != expected:
                    raise AssertionError(f"Text mismatch: {title}!{cell.coordinate}: {expected!r} -> {cell.value!r}")
                if TEXT_TOKEN.search(header) and expected is not None:
                    if cell.data_type != "s":
                        raise AssertionError(f"Identifier became numeric: {title}!{cell.coordinate}")
                    text_ids += 1
        column_positions = {header: index for index, header in enumerate(headers, 1)}
        for row_index, record in enumerate(source_records, 2):
            for header, count in parts.items():
                restored = "".join(str(sheet.cell(row_index, column_positions[header if part == 1 else f"{header}__part_{part}"]).value or "") for part in range(1, count + 1))
                if restored != record[header]:
                    raise AssertionError(f"Long-text reconstruction mismatch: {title}:{row_index}:{header}")
                if len(record[header]) > 32767:
                    oversized_source_cells += 1
        continuation_columns[title] = len(headers) - len(source_headers)
        counts[title] = len(records)
    for sheet in workbook:
        for row in sheet:
            for cell in row:
                if cell.data_type in {"f", "e"}:
                    raise AssertionError(f"Formula/error cell found: {sheet.title}!{cell.coordinate}")
                if cell.value is not None and (cell.font.name != FONT or cell.font.sz != 10):
                    raise AssertionError(f"Unexpected font: {sheet.title}!{cell.coordinate}")
    workbook.close()
    return {
        "output": str(output), "sheet_rows": counts, "numeric_cells_verified": numeric_cells,
        "metric_precision_cells_verified": metric_cells, "text_identifier_cells_verified": text_ids,
        "oversized_source_cells_reconstructed": oversized_source_cells,
        "continuation_columns": continuation_columns,
        "formula_cells": 0, "error_cells": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=REPO / "experiments")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    input_dir = args.input_dir.resolve()
    output = (args.output or input_dir / "Experiment-Records.xlsx").resolve()
    source_paths = [input_dir / name for name, _ in SHEETS] + [input_dir / "summary.json"]
    before = {str(path): digest(path) for path in source_paths}
    source_data = {name: read_csv(input_dir / name) for name, _ in SHEETS}
    summary = json.loads((input_dir / "summary.json").read_text(encoding="utf-8"))
    if not args.verify_only:
        workbook = Workbook()
        workbook.remove(workbook.active)
        workbook.properties.title = "CFM4EEG 实验记录"
        workbook.properties.subject = "Materialized experiment ledger; authoritative data retained in CSV"
        workbook.properties.creator = "Codex Scholar"
        workbook.properties.description = "Static CSV export. No best-seed selection, cross-protocol aggregation, or formulas."
        for filename, title in SHEETS:
            source_headers, source_records = source_data[filename]
            headers, records, parts = excel_columns(source_headers, source_records)
            sheet = workbook.create_sheet(title)
            sheet.append(headers)
            for row_index, record in enumerate(records, 2):
                for column_index, header in enumerate(headers, 1):
                    write_value(sheet.cell(row_index, column_index), as_cell_value(header, record[header]))
            style_table(sheet, headers, records)
            for column_index, header in enumerate(headers, 1):
                if header in parts:
                    sheet.cell(1, column_index).comment = Comment(f"原 CSV 超长字段；完整原文分入本列及 {parts[header] - 1} 个同名隐藏续列，按顺序连接可恢复。", "Codex")
        add_readme(workbook, input_dir, source_data, summary)
        workbook.save(output)
        workbook.close()
    result = verify(output, source_data)
    after = {str(path): digest(path) for path in source_paths}
    if before != after:
        raise AssertionError("Source CSV/JSON changed during rendering or verification")
    result["input_files_unchanged"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
