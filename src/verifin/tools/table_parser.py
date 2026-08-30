"""复杂表格解析扩展（不依赖 LangGraph；复用 evidence 的 TableSpec/parse_tables）。

新增能力（Phase 6，docs/designs/2026-08-29-phase-6 §D7）：
- **旋转表头**：指标在顶行、年份在第一列 → (metric, period) → value；
- **水平多级表头**：指标行 × 年份行交叉索引 → (metric, period) → value；
- **跨页拼接**：续页表头归一（Item (Cont.) ≡ Item）+ 剔除重复表头行与完全重复行。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from verifin.tools.evidence import TableSpec, _match_metric, parse_tables
from verifin.tools.unit_parser import parse_number_with_unit

_YEAR_RE = re.compile(r"20\d{2}")
_CONTINUATION_RE = re.compile(r"\(?(?:cont\.?|continued|续)\)?", re.IGNORECASE)


def extract_from_tables_advanced(content: str) -> Optional[Tuple[str, str, float, Optional[str]]]:
    """从旋转表头 / 水平多级表头的表格中提取 (metric, period, value, unit)。

    标准结构（行标签指标）由 evidence 的 ``_extract_from_tables`` 优先处理；
    本函数覆盖旋转表头（指标在列头、年份在首列）与水平多级表头（指标行 × 年份行）。
    """
    for table in parse_tables(content):
        hit = _extract_transposed(table)
        if hit is not None:
            return hit
        hit = _extract_horizontal_multilevel(table)
        if hit is not None:
            return hit
    return None


def _extract_transposed(table: TableSpec) -> Optional[Tuple[str, str, float, Optional[str]]]:
    """旋转表头：表头行含指标关键词，首列数据行为年份（2024/FY2024）。"""
    if not table.headers or not table.rows:
        return None
    header = table.headers[-1]  # 最明细的表头行
    metric_column: Optional[int] = None
    metric: Optional[str] = None
    for col, cell in enumerate(header[1:], start=1):
        matched = _match_metric(cell)
        if matched:
            metric_column = col
            metric = matched
            break
    if metric_column is None or metric is None:
        return None
    # 选最新年份行
    best: Optional[Tuple[int, str, float, Optional[str]]] = None
    for row_index, cells in enumerate(table.rows):
        if not cells:
            continue
        year_match = _YEAR_RE.search(cells[0])
        if not year_match:
            continue
        cell = table.cell(row_index, metric_column)
        parsed = parse_number_with_unit(cell)
        if parsed is None:
            continue
        value, unit = parsed
        candidate = (int(year_match.group(0)), cells[0].strip(), value, unit)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        return None
    return metric, best[1], best[2], best[3]


def _extract_horizontal_multilevel(table: TableSpec) -> Optional[Tuple[str, str, float, Optional[str]]]:
    """水平多级表头：表头行 1 = 指标、表头行 2 = 年份，(metric, year) 交叉定位。

    形如::

        | Item | Revenue | Revenue | Cost | Cost |
        |      | 2023    | 2024    | 2023 | 2024 |
        | NovaTech | 11,800 | 12,400 | ... |
    """
    if len(table.headers) < 2 or not table.rows:
        return None
    metric_row = table.headers[-2]
    year_row = table.headers[-1]
    # 指标列集合
    metric_columns = {
        col: _match_metric(cell)
        for col, cell in enumerate(metric_row)
        if _match_metric(cell)
    }
    if not metric_columns:
        return None
    # 每个指标列找最新年份列
    best_per_metric = {}
    for col, metric in metric_columns.items():
        year_match = _YEAR_RE.search(year_row[col]) if col < len(year_row) else None
        if not year_match:
            continue
        year = int(year_match.group(0))
        current = best_per_metric.get(metric)
        if current is None or year > current[0]:
            best_per_metric[metric] = (year, col)
    if not best_per_metric:
        return None
    # 选择年份最新的 (metric, col) 组合
    metric, (year, col) = max(
        best_per_metric.items(), key=lambda item: item[1][0]
    )
    parsed = parse_number_with_unit(table.cell(0, col))
    if parsed is None:
        return None
    value, unit = parsed
    period = year_row[col].strip() if col < len(year_row) else str(year)
    return metric, period, value, unit


def _normalize_header_cell(cell: str) -> str:
    return _CONTINUATION_RE.sub("", cell.strip().lower()).strip()


def _headers_compatible(left: TableSpec, right: TableSpec) -> bool:
    if not left.headers or not right.headers:
        return False
    left_first = left.headers[0]
    right_first = right.headers[0]
    if len(left_first) != len(right_first):
        return False
    return _normalize_header_cell(left_first[0]) == _normalize_header_cell(right_first[0])


def _row_equals_header(row: List[str], headers: List[List[str]]) -> bool:
    return any(
        len(row) == len(header) and all(
            _normalize_header_cell(a) == _normalize_header_cell(b)
            for a, b in zip(row, header)
        )
        for header in headers
    )


def merge_table_fragments(tables: List[TableSpec]) -> Optional[TableSpec]:
    """跨页拼接：连续片段表头兼容（首列标签归一）→ 合并数据行。

    - 剔除与任一片段表头重复的行（续页重复表头）；
    - 剔除与上一行完全重复的行（续页重复首行）。
    """
    if not tables:
        return None
    merged_headers = tables[0].headers
    merged_rows: List[List[str]] = []
    previous_row: Optional[List[str]] = None
    for table in tables:
        if not _headers_compatible(tables[0], table):
            continue
        for row in table.rows:
            if _row_equals_header(row, table.headers):
                continue
            if previous_row is not None and row == previous_row:
                continue  # 去重：与上一行完全一致（续页重复）
            merged_rows.append(row)
            previous_row = row
    return TableSpec(headers=merged_headers, rows=merged_rows)