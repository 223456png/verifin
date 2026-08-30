"""Phase 5 — 表格感知证据提取测试（tools/evidence.py 表格适配器）。"""

from __future__ import annotations

import pytest

from verifin.schemas import Evidence
from verifin.tools.evidence import EvidenceExtractor, parse_markdown_tables

EXTRACTOR = EvidenceExtractor()

_SIMPLE_TABLE = """
| Metric | 2023 | 2024 |
|---|---|---|
| Revenue | 11,800 | 12,400 |
| Gross margin | 40% | 42% |
"""

# 简单列表/单表头：从 3 列表格提取 2024 revenue
def test_extract_simple_table_revenue() -> None:
    evidence = EXTRACTOR.extract(
        {"chunk_id": "t1", "doc_id": "nova-fy2024", "doc_name": "nova-fy2024.pdf",
         "content": _SIMPLE_TABLE, "metadata": {}}
    )
    assert isinstance(evidence, Evidence)
    assert evidence.metric == "revenue"
    assert evidence.period == "2024"
    assert evidence.value == pytest.approx(12400.0)


# 多级表头（组头 FY2024 + 明细年份行）：仍能定位 2024 列
def test_extract_multilevel_header_table() -> None:
    table = """
| Metrics | FY2024 | FY2024 |
| Metric | 2023 | 2024 |
|---|---|---|
| Revenue | 11,800 | 12,400 |
"""
    evidence = EXTRACTOR.extract(
        {"chunk_id": "t2", "doc_id": "nova-fy2024", "doc_name": "nova-fy2024.pdf",
         "content": table, "metadata": {}}
    )
    assert evidence.metric == "revenue"
    assert evidence.period == "2024"
    assert evidence.value == pytest.approx(12400.0)


# 合并单元格（首列空白承接上一行标签）：cell() 应按列向上承接
def test_merged_cell_carry_forward() -> None:
    table = """
| Item | 2023 | 2024 |
|---|---|---|
| Revenue | 11,800 | 12,400 |
|   | 6,000 | 6,300 |
"""
    tables = parse_markdown_tables(table)
    assert len(tables) == 1
    spec = tables[0]
    # 续行首列空白 → 向上承接 "Revenue"
    assert spec.cell(1, 0) == "Revenue"
    assert spec.cell(1, 2) == "6,300"
    # 提取器只认显式标签行：取上一行（显式 Revenue 行）的 2024 值
    evidence = EXTRACTOR.extract(
        {"chunk_id": "t3", "doc_id": "nova-fy2024", "doc_name": "nova-fy2024.pdf",
         "content": table, "metadata": {}}
    )
    assert evidence.value == pytest.approx(12400.0)