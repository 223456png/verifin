"""Phase 6 — 复杂表格解析测试（tools/table_parser.py：旋转表头/水平多级/跨页拼接）。"""

from __future__ import annotations

import pytest

from verifin.tools.evidence import parse_markdown_tables
from verifin.tools.table_parser import (
    extract_from_tables_advanced,
    merge_table_fragments,
)

_TRANSPOSED = """
| Year | Revenue | Gross margin |
|---|---|---|
| 2023 | 11,800 | 40% |
| 2024 | 12,400 | 42% |
"""

# 旋转表头 1：指标在顶行、年份在第一列 → (metric, period, value)
def test_transposed_table_extraction() -> None:
    hit = extract_from_tables_advanced(_TRANSPOSED)
    assert hit is not None
    metric, period, value, unit = hit
    assert metric == "revenue"
    assert period == "2024"
    assert value == pytest.approx(12400.0)


# 旋转表头 2：指标同义词（Turnover→revenue）+ FY 期间列
def test_transposed_with_synonym_and_fy() -> None:
    table = """
| Year | Turnover |
|---|---|
| FY2023 | 10,000 |
| FY2024 | 12,400 |
"""
    hit = extract_from_tables_advanced(table)
    assert hit is not None
    metric, period, value, _ = hit
    assert metric == "revenue"
    assert period == "FY2024"
    assert value == pytest.approx(12400.0)


# 水平多级表头：(metric × year) 交叉索引 → 定位 (Revenue, 2024) 单元格
def test_horizontal_multilevel_header() -> None:
    table = """
| Item | Revenue | Revenue | Cost | Cost |
|  | 2023 | 2024 | 2023 | 2024 |
|---|---|---|---|---|
| NovaTech | 11,800 | 12,400 | 7,000 | 7,400 |
"""
    hit = extract_from_tables_advanced(table)
    assert hit is not None
    metric, period, value, _ = hit
    assert metric == "revenue"
    assert period == "2024"
    assert value == pytest.approx(12400.0)


# 跨页拼接：两页拼接去重（续页表头归一 + 连续重复行剔除）
def test_cross_page_fragments_merged() -> None:
    page1 = parse_markdown_tables("| Item | FY2024 |\n|---|---|\n| Revenue | 12,400 |\n")
    page2 = parse_markdown_tables(
        "| Item (Cont.) | FY2024 |\n|---|---|\n| Revenue | 12,400 |\n| Cost | 7,400 |\n"
    )
    merged = merge_table_fragments(page1 + page2)
    assert merged is not None
    assert merged.rows == [["Revenue", "12,400"], ["Cost", "7,400"]]
    # 表头取归一后的第一组
    assert merged.headers[0][0] == "Item"
