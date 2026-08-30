"""规则证据抽取器：从文档 chunk 中抽取四要素 + 数值（不依赖 LangGraph）。

Phase 4 使用确定性规则；``llm_mode`` 预留 Phase 7 升级为 LLM 抽取。
Phase 5 增加表格感知：markdown 表 / 简单 HTML 表 → 结构化解单（TableSpec），
供 metric/period/value 从表格行标签 + 年份列定位（支持多级表头与合并单元格承接）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from verifin.schemas import Evidence
from verifin.tools.unit_parser import parse_number_with_unit

# 指标关键词词典（canonical key -> 同义表达列表，全小写以便大小写不敏感匹配）
METRIC_KEYWORDS: Dict[str, List[str]] = {
    "revenue": ["revenue", "sales", "turnover", "income"],
    "operating_margin": ["operating margin", "operating income", "ebit margin", "return on sales"],
    "gross_margin": ["gross margin", "gross profit margin", "gross profit"],
    "net_margin": ["net margin", "net profit margin", "net income margin"],
    "EBITDA": ["ebitda", "earnings before interest", "ebit"],
    "cost": ["cost of goods sold", "cost of goods", "cogs", "cost", "expense"],
    "profit": ["net profit", "net income", "profit"],
}

_PERIOD_RE = re.compile(r"\b(20\d{2}|FY\s?20\d{2}|Q[1-4]\s?20\d{2})\b")
# 单位可选；% 无词边界要求，词单位（million/billion/M/B）要求词边界
_VALUE_RE = re.compile(r"\$?\s*(\d[\d,]*\.?\d*)\s*(%|(?:million|billion|[MB])\b)?", re.IGNORECASE)
_YEAR_VALUE_RE = re.compile(r"\b20\d{2}\b")
# Phase 10.1：短语/行标签中的年份 token（"due in 2018" → 行 "2018"）
# 与 "after <year>" → thereafter 行（"due after 2020" → 行 "2021 - thereafter"）
_PHRASE_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")
_AFTER_YEAR_RE = re.compile(r"\bafter\s+((?:19|20)\d{2})\b", re.IGNORECASE)
# Phase 10.1：脚注引用标记（"( 1 ) ( 2 )"）——非财经数值
_FOOTNOTE_MARKER_RE = re.compile(r"\(\s*\d{1,2}\s*\)")
# Phase 8：年份锚定抽取的最大字符窗口（数值与年份提及距离超过该值不视为同上下文）
_YEAR_ANCHOR_WINDOW = 200
# Phase 9：数值出现在年份**之前**的距离惩罚（财报语序 "in 2015 ... $5829" 年份先行）
_PRE_YEAR_PENALTY = 100
_DEFINITION_RE = re.compile(
    r"\b(?:calculated|defined|computed)\s+as\s+([^.;]+)", re.IGNORECASE
)
_ENTITY_RE = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:\s+(?:[A-Z][A-Za-z0-9]*|&)){0,3}\b")

# 句子首词/虚词（首字母大写）不作为实体候选
_ENTITY_STOPWORDS = {
    "what", "who", "which", "when", "where", "how", "tell", "is", "are", "was",
    "were", "do", "does", "did", "in", "the", "a", "an", "for", "of", "on",
    "and", "or", "its", "their", "this", "that", "per", "as", "by", "with",
    "to", "from", "at", "not", "no", "us", "gaap", "ifrs", "usd", "eoy",
}

# 指标同义词按长度降序（最长优先，避免 revenue 的 "income" 抢在 "net income" 前）
_SORTED_METRIC_TERMS: List[tuple] = sorted(
    ((canonical, synonym.lower()) for canonical, synonyms in METRIC_KEYWORDS.items() for synonym in synonyms),
    key=lambda pair: len(pair[1]),
    reverse=True,
)

_DOC_NAME_ENTITY_RE = re.compile(r"^([A-Za-z0-9]+)(?:[-_](?:fy|FY|20\d{2}|filing|report))")


def extract_entity_candidates(text: str) -> List[str]:
    """从文本中提取首字母大写短语候选实体（过滤问句首词/年份类 token）。

    供证据抽取与 claim 抽取复用。
    """
    candidates: List[str] = []
    for raw in _ENTITY_RE.findall(text):
        tokens = raw.split()
        # 剥离前导虚词（如 "The NovaTech Group" → "NovaTech Group"）
        while tokens and tokens[0].lower() in _ENTITY_STOPWORDS:
            tokens = tokens[1:]
        candidate = " ".join(tokens).strip()
        if not candidate:
            continue
        # 单候选且为虚词/年份形态（FY2024、Q2、2024）→ 丢弃
        if candidate.lower() in _ENTITY_STOPWORDS or re.fullmatch(
            r"(?:FY\s?20\d{2}|Q[1-4](?:\s?20\d{2})?|20\d{2})", candidate, re.IGNORECASE
        ):
            continue
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def _match_metric(text: str) -> Optional[str]:
    """按最长同义词优先在文本中扫描指标，返回 canonical key；无命中返回 None。"""
    lowered = text.lower()
    for canonical, synonym in _SORTED_METRIC_TERMS:
        if re.search(r"\b" + re.escape(synonym) + r"\b", lowered):
            return canonical
    return None


@dataclass
class TableSpec:
    """解析后的表格结构。

    Attributes:
        headers: 表头行列表（多级表头为多行，每行是单元格列表）。
        rows: 数据行列表（原始单元格；空白单元格通过 :meth:`cell` 按列向上承接）。
    """

    headers: List[List[str]] = field(default_factory=list)
    rows: List[List[str]] = field(default_factory=list)

    def cell(self, row: int, col: int) -> str:
        """取 (row, col) 单元格；空白时向上承接（合并单元格语义）。"""
        for r in range(row, -1, -1):
            if r >= len(self.rows):
                continue
            cells = self.rows[r]
            value = cells[col].strip() if col < len(cells) else ""
            if value:
                return value
        return ""


_SEPARATOR_CELL_RE = re.compile(r"^:?-{2,}:?$")


def _is_separator_row(cells: List[str]) -> bool:
    return bool(cells) and all(_SEPARATOR_CELL_RE.match(cell) for cell in cells)


def _pipe_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_markdown_tables(text: str) -> List[TableSpec]:
    """解析 markdown 管道表格；分隔行之前的连续行为表头（支持多级表头）。"""
    tables: List[TableSpec] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        block: List[str] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index])
            index += 1
        rows = [_pipe_row(line) for line in block]
        separator_index = next(
            (i for i, cells in enumerate(rows) if _is_separator_row(cells)), None
        )
        if separator_index is None:
            # 无分隔行（loader 序化的 FinQA 表格无 |---|）：首行作表头，
            # 其余全部为数据行——不得额外丢一行（Phase 9 修正：此前
            # rows[2:] 会把首条数据行当分隔行丢掉，如 "2014 net revenue"）
            tables.append(
                TableSpec(headers=rows[:1], rows=rows[1:])
            )
        else:
            tables.append(
                TableSpec(
                    headers=rows[:separator_index] or rows[:1],
                    rows=rows[separator_index + 1:],
                )
            )
    return tables


_HTML_ROW_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_CELL_RE = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_HTML_TAG_RE = re.compile(r"<[^>]+>")


def parse_simple_html_tables(text: str) -> List[TableSpec]:
    """解析简单 HTML 表格（<table> 内的 <tr>/<td>/<th> 行；不支持跨列属性）。"""
    tables: List[TableSpec] = []
    headers: List[List[str]] = []
    rows: List[List[str]] = []
    for row in _HTML_ROW_RE.findall(text):
        cells = [_HTML_TAG_RE.sub("", cell).strip() for cell in _HTML_CELL_RE.findall(row)]
        if not cells:
            continue
        # 简单策略：<tr> 行内有 <th> 视为表头行，否则为数据行
        if re.search(r"<th", row, re.IGNORECASE):
            headers.append(cells)
        else:
            rows.append(cells)
    if headers or rows:
        tables.append(
            TableSpec(
                headers=headers if headers else rows[:1],
                rows=rows if headers else rows[1:],
            )
        )
    return tables


def parse_tables(text: str) -> List[TableSpec]:
    """聚合解析 markdown + HTML 表格。"""
    return parse_markdown_tables(text) + parse_simple_html_tables(text)


_YEAR_IN_CELL_RE = re.compile(r"20\d{2}")
# FinQA 括号负数惯例："(32)" → -32
_PAREN_NEG_RE = re.compile(r"^\(\s*([\d,.]+)\s*\)$")


def _parse_cell_value(cell: str):
    """表格单元格 → (value, unit)；括号负数 ``(32)`` 解析为 -32。"""
    stripped = cell.strip()
    match = _PAREN_NEG_RE.match(stripped)
    if match:
        return parse_number_with_unit("-" + match.group(1))
    return parse_number_with_unit(stripped)


def _table_column_years(table: TableSpec) -> Dict[int, int]:
    """从表头行聚合每列年份；明细行优先（多级表头：组行 FY2024 不应覆盖明细 2023/2024）。"""
    years: Dict[int, int] = {}
    for header_row in reversed(table.headers):
        for col, cell in enumerate(header_row):
            if col in years:
                continue
            match = _YEAR_IN_CELL_RE.search(cell)
            if match:
                years[col] = int(match.group(0))
    return years


def _extract_from_tables(content: str) -> Optional[Tuple[str, int, float, Optional[str]]]:
    """从表格中提取 (metric, period_year, value, unit)。

    策略：数据行首列显式标签命中指标词典 → 取该行「最新年份列」的数值。
    多级表头：年份可出现在任一表头行。合并单元格续行（首列空白）不作为新候选。
    """
    for table in parse_tables(content):
        if not table.rows:
            continue
        column_years = _table_column_years(table)
        if not column_years:
            continue
        latest_col = max(column_years, key=column_years.get)
        period_year = column_years[latest_col]
        for row_index, cells in enumerate(table.rows):
            if not cells or not cells[0].strip():
                continue  # 合并续行：继承标签不作为独立候选
            metric = _match_metric(cells[0])
            if not metric:
                continue
            parsed = parse_number_with_unit(table.cell(row_index, latest_col))
            if parsed is None:
                continue
            value, unit = parsed
            return metric, period_year, value, unit
    return None


class EvidenceExtractor:
    """把检索 chunk（dict）抽取为结构化 Evidence（规则模式）。"""

    def __init__(self, llm_mode: bool = False) -> None:
        self.llm_mode = llm_mode  # Phase 7 可升级为 LLM 抽取

    # ---- Phase 8（真实数据适配）：位置感知的锚定抽取 ----
    # 真实财报 chunk 内含大量数字（页码/年份/无关指标），「取首个数字」噪声极高。
    # 改为最近邻锚定：数值取离「指标同义词提及」最近的数字（无指标信号时退
    # 到离年份最近的数字），期间/指标同步锚定到该数值所在的上下文。

    @staticmethod
    def _metric_spans(content: str) -> List[Tuple[int, int, str]]:
        """全部指标同义词提及的 (start, end, canonical) 位置列表（按位置排序）。"""
        lowered = content.lower()
        spans: List[Tuple[int, int, str]] = []
        for canonical, synonym in _SORTED_METRIC_TERMS:
            for match in re.finditer(r"\b" + re.escape(synonym) + r"\b", lowered):
                spans.append((match.start(), match.end(), canonical))
        # 同起点命中长短两词（"ebit" vs "ebit margin"）时，更长（更特异）者优先
        return sorted(spans, key=lambda span: (span[0], -span[1]))

    @staticmethod
    def _year_spans(content: str) -> List[Tuple[int, int, str]]:
        """全部 4 位年份的 (start, end, year) 位置列表。"""
        return [
            (match.start(), match.end(), str(match.group(0)))
            for match in _YEAR_VALUE_RE.finditer(content)
        ]

    @staticmethod
    def _number_spans(content: str) -> List[Tuple[int, int, float, Optional[str]]]:
        """全部数值候选项的 (start, end, value, unit) 位置列表（含 $/%/词单位）。"""
        year_starts = {match.start() for match in _YEAR_VALUE_RE.finditer(content)}
        spans: List[Tuple[int, int, float, Optional[str]]] = []
        for match in _VALUE_RE.finditer(content):
            # match 含 \$?\s* 前缀，match.start() 可能落在前导空白上——
            # 先定位数字首个字符的真实位置再做排除判断
            leading = len(match.group(0)) - len(match.group(0).lstrip("$ "))
            digit_start = match.start() + leading
            if digit_start in year_starts:
                continue  # 4 位年份被误抽为数值 → 剔除
            if digit_start > 0 and content[digit_start - 1].isalnum():
                continue  # Q2/条款编号等字母紧邻的数字不是财经数值
            try:
                value = float(match.group(1).replace(",", ""))
            except ValueError:
                continue
            # 报表编号（"Form 10-K"/"10-Q"/"S-1"）：数字后紧跟 -字母
            # 是 SEC 文件代码而非财经数值（曾把 "Form 10-K" 抽成 10.0
            # 造成合并不冲突的假 major conflict）
            num_end = match.start(1) + len(match.group(1))
            if (
                num_end < len(content)
                and content[num_end] == "-"
                and num_end + 1 < len(content)
                and content[num_end + 1].isalpha()
            ):
                continue
            # 千分位年份形态（"2,015"）躲过 \b20\d{2}\b 边界检查 →
            # 无单位且落在年份区间的整数视为年份干扰剔除
            if value == int(value) and 1800 < value < 2200 and not match.group(2):
                continue
            unit_raw = (match.group(2) or "").lower()
            unit = "%" if unit_raw == "%" else unit_raw or None
            spans.append((match.start(), match.end(), value, unit))
        return spans

    @classmethod
    def _anchored_value(
        cls, content: str, year_filter: Optional[str] = None, prefer: str = "metric"
    ) -> Tuple[Optional[float], Optional[str], Optional[str], Optional[str]]:
        """锚定抽取 (value, unit, period, metric)：
        - ``prefer="metric"``（默认）：数值取离「指标同义词提及」最近的数字，
          年份距离作次优键；无指标信号时退到年份最近邻；
        - ``prefer="year"``（计算流）：``year_filter`` 指定年份，取该年份
          ``_YEAR_ANCHOR_WINDOW`` 字符窗口内最近的数值（两侧均可，支持
          "… $X in 2005 …" 与 "… 2005 … $X …" 两种语序）；
        - period/metric 取被选数值最近邻的年份/指标 canonical。
        """
        numbers = cls._number_spans(content)
        if not numbers:
            return None, None, None, None
        metric_spans = cls._metric_spans(content)
        year_spans = cls._year_spans(content)
        if year_filter is not None:
            year_spans = [span for span in year_spans if span[2] == year_filter]

        def distance(num_start: int, num_end: int, span_start: int, span_end: int) -> int:
            if num_end <= span_start:
                return span_start - num_end
            if span_end <= num_start:
                return num_start - span_end
            return 0

        best: Optional[Tuple[int, ...]] = None
        best_num: Optional[Tuple[int, int, float, Optional[str]]] = None
        for number in numbers:
            num_start, num_end = number[0], number[1]
            metric_dist = min(
                (distance(num_start, num_end, start, end) for start, end, _ in metric_spans),
                default=None,
            )
            year_dist = min(
                (distance(num_start, num_end, start, end) for start, end, _ in year_spans),
                default=None,
            )
            if prefer == "year":
                if year_dist is None or year_dist > _YEAR_ANCHOR_WINDOW:
                    continue
                key = (year_dist, metric_dist if metric_dist is not None else 10**9, num_start)
            elif metric_dist is None and year_dist is None:
                key = (2, num_start, 0)
            elif metric_dist is None:
                key = (1, year_dist if year_dist is not None else 10**9, 0)
            elif year_dist is None:
                key = (0, metric_dist, 10**9)
            else:
                key = (0, metric_dist, year_dist)
            if best is None or key < best:
                best = key
                best_num = number
        if best_num is None:  # pragma: no cover - numbers 非空必有候选
            return None, None, None, None
        value, unit = best_num[2], best_num[3]
        # period：离选中数值最近的年份（year 模式必为 year_filter）
        period: Optional[str] = None
        if year_spans:
            period = min(
                year_spans,
                key=lambda span: distance(best_num[0], best_num[1], span[0], span[1]),
            )[2]
        # metric：标题指标习惯先于数值出现（"... operating margin, calculated
        # as ..., was 15%"）——取文中首个指标提及；距离太远（>400 字符，
        # 属于别的段落）则退到离选中数值最近的提及。
        metric: Optional[str] = None
        if metric_spans:
            nearest_metric = min(
                metric_spans,
                key=lambda span: distance(best_num[0], best_num[1], span[0], span[1]),
            )
            first_metric = metric_spans[0]
            if distance(best_num[0], best_num[1], first_metric[0], first_metric[1]) <= 400:
                metric = first_metric[2]
            else:
                metric = nearest_metric[2]
        return value, unit, period, metric

    @classmethod
    def collect_table_year_values(
        cls, content: str, year: str, spec_metric: Optional[str] = None,
        query: Optional[str] = None, max_n: int = 6
    ) -> List[dict]:
        """表格感知年份候选（Phase 9 计算流）。

        两种表结构（FinQA 均常见）：
        - **水平表**：表头列含年份（``| | 2015 | 2014 |``），行标签为指标名，
          数值按「行 × 年份列」对齐；
        - **垂直表**：行标签自带年份（``| 2014 net revenue | $5,735 |``），
          数值在该行的后续列。

        优先级（anchor_score 越小越可信）：
        0 = 行标签命中指标词典且与程序规格指标一致；
        0.5 = 行标签与用户问题的实词重叠 ≥50%（如 "Standby letters of credit"
        对 "growth rate in the balance of standby letters of credit"）——
        指标词典外的行标签靠问题词重叠锚定；
        1 = 行标签命中指标词典（未匹配规格）；
        8 = 无信号。
        """
        query_tokens = cls._content_tokens(query or "")
        out: List[dict] = []
        for table in parse_tables(content):
            column_years = _table_column_years(table)
            cols = [col for col, yr in column_years.items() if str(yr) == str(year)]
            for row_index, cells in enumerate(table.rows):
                if not cells or not cells[0].strip():
                    continue  # 合并续行
                label = cells[0].strip()
                row_metric = _match_metric(label)
                metric_hit = bool(
                    row_metric
                    and spec_metric
                    and row_metric.lower() == str(spec_metric).lower()
                )
                # 行标签与问题实词重叠（指标词典外的语义锚定）。
                # 双向取大：长行标签（"cash flows provided by ( used in )
                # operating activities including discontinued operations"）的
                # 标签覆盖率被冗余词稀释，但问题侧覆盖率（question tokens
                # 被标签覆盖的比例）不受影响——两个方向都强才是语义锚定
                label_tokens = cls._content_tokens(label)
                shared = label_tokens & query_tokens
                # 单 token 命中不启用问题方向覆盖率（"credit"/"balance" 等
                # 泛化词单hits噪声大）；标签方向覆盖率照常
                query_coverage = (
                    len(shared) / len(query_tokens)
                    if query_tokens and len(shared) >= 2 else 0.0
                )
                overlap = max(
                    len(shared) / len(label_tokens) if label_tokens else 0.0,
                    query_coverage,
                )

                def _score() -> float:
                    if metric_hit:
                        return 0.0
                    # 严格过半：恰好 0.5 的短标签（如 "Balance at December 31,
                    # 2006" 对含 balance 的增长问题）多为泛化词假阳性
                    if overlap > 0.5:
                        return 0.5
                    return 1.0 if row_metric else 8.0

                def _append(value: float, unit: Optional[str]) -> None:
                    out.append({
                        "value": value,
                        "unit": unit,
                        "period": str(year),
                        "row_label": label,
                        "metric": row_metric,
                        "anchor_score": _score(),
                    })

                # 水平表：列年份对齐
                for col in cols:
                    parsed = _parse_cell_value(table.cell(row_index, col))
                    if parsed is not None:
                        _append(parsed[0], parsed[1])
                # 垂直表：行标签自带年份（"2014 net revenue"）
                if _YEAR_VALUE_RE.search(label) and str(year) in label:
                    for col in range(1, len(cells)):
                        parsed = _parse_cell_value(cells[col])
                        if parsed is not None:
                            _append(parsed[0], parsed[1])
        out.sort(key=lambda item: item["anchor_score"])
        return out[:max_n]

    _QUERY_STOPWORDS = {
        "the", "of", "in", "to", "from", "and", "a", "an", "was", "is", "are",
        "what", "how", "much", "did", "do", "for", "by", "with", "at", "on",
        "its", "their", "than", "then", "that", "this", "per", "as", "be",
        "were", "due", "related",
    }

    @classmethod
    def _content_tokens(cls, text: str) -> set:
        """小写实词 token 集合（去停用词；供行标签 × 问题词重叠）。"""
        return {
            token for token in re.findall(r"[a-z]+", (text or "").lower())
            if token not in cls._QUERY_STOPWORDS
        }

    @classmethod
    def _phrase_token_set(cls, phrase: str) -> tuple:
        """短语扩展 token 集（Phase 10.1）：实词 + 年份 token。

        - ``"due in 2018"`` → ``{due(去停用词后无), y2018}``，可命中
          纯年份行标签 ``2018``（FinQA 到期表行标签即年份）；
        - ``"due after 2020"`` → 年份上界语义：排除 ≤2020 的年份 token、
          补 ``thereafter`` token，命中 ``2021 - thereafter`` 行。
        返回 ``(word_tokens, year_tokens)``。
        """
        text = (phrase or "").lower()
        after_match = _AFTER_YEAR_RE.search(text)
        words = cls._content_tokens(text)
        years = {f"y{m.group(1)}" for m in _PHRASE_YEAR_RE.finditer(text)}
        if after_match:
            bound = int(after_match.group(1))
            years = {y for y in years if int(y[1:]) > bound}
            words = (words - {"after"}) | {"thereafter"}
        return words, years

    @classmethod
    def collect_phrase_values(
        cls, content: str, phrase: str, year: Optional[str] = None,
        max_n: int = 6,
    ) -> List[dict]:
        """短语锚定表格数值候选（Phase 10 比率计算流）。

        "what percentage of X are Y" 类比率题常无年份、指标词典也不覆盖
        （"leased facilities" / "total purchase price"），操作数取
        「行标签 × 短语实词重叠」锚定的表格数值：

        - **水平表**：``year`` 给定且表头列含该年份 → 仅取该列；否则取
          每行全部数值列（各列均成候选，含 total 列，组合枚举择优）；
        - **垂直表**：``year`` 给定 → 行标签含该年份的行；否则全部行，
          取行内数值单元格；
        - anchor_score = ``1 - 双向词重叠率``（行标签与短语实词的双向
          覆盖率取大，越大越可信）；返回值带 ``row_label`` / ``column``
          （LLM program 生成的候选上下文也复用本字段）；
        - **年份 token 匹配**（Phase 10.1）：短语含年份（"due in 2018"）或
          "after <year>"（→ ``thereafter`` 行）时，纯年份行标签可命中——
          FinQA 到期/承诺表的行标签即年份；
        - **正文句级扫描**（Phase 10.1）：分子常在脚注正文（"42749 shares
          were repurchased in open-market transactions"）而非表格——
          非表格行与短语词重叠 ≥ 阈值时，行内数值成为候选
          （``row_label`` = 行首片段，供分母 total 行偏好等下游语义）。
        """
        phrase_words, phrase_years = cls._phrase_token_set(phrase or "")
        phrase_all = phrase_words | phrase_years
        if not phrase_all:
            return []
        out: List[dict] = []

        def _row_shared(label: str) -> Optional[tuple]:
            """行标签 × 短语的共享 token 与双向重叠率（无重叠返回 None）。"""
            label_words = cls._content_tokens(label)
            label_years = {
                f"y{m.group(1)}" for m in _PHRASE_YEAR_RE.finditer(label)
            }
            label_all = label_words | label_years
            if not label_all:
                return None
            shared = label_all & phrase_all
            if not shared:
                return None
            overlap = max(
                len(shared) / len(label_all), len(shared) / len(phrase_all),
            )
            return shared, overlap

        for table in parse_tables(content):
            column_years = _table_column_years(table)
            header_row = table.headers[-1] if table.headers else []
            cols = (
                [col for col, yr in column_years.items() if str(yr) == str(year)]
                if year else []
            )
            for row_index, cells in enumerate(table.rows):
                if not cells or not cells[0].strip():
                    continue
                label = cells[0].strip()
                matched = _row_shared(label)
                if matched is None:
                    continue
                shared, overlap = matched
                # 单 token 命中多为泛化词（"total"/"current"）假阳性 → 加惩罚
                anchor = 1.0 - overlap + (0.25 if len(shared) < 2 else 0.0)

                def _append(value: float, unit: Optional[str], column: str) -> None:
                    out.append({
                        "value": value,
                        "unit": unit,
                        "period": str(year or ""),
                        "row_label": label,
                        "column": column,
                        "anchor_score": round(anchor, 3),
                    })

                # 水平表：年份列对齐（有年份列时）或全数值列
                if cols:
                    for col in cols:
                        parsed = _parse_cell_value(table.cell(row_index, col))
                        if parsed is not None:
                            _append(
                                parsed[0], parsed[1],
                                header_row[col].strip() if col < len(header_row) else "",
                            )
                else:
                    # 垂直表行标签自带年份：year 给定时须命中
                    if year and not (_YEAR_VALUE_RE.search(label) and str(year) in label):
                        continue
                    for col in range(1, len(cells)):
                        parsed = _parse_cell_value(cells[col])
                        if parsed is not None:
                            _append(
                                parsed[0], parsed[1],
                                header_row[col].strip() if col < len(header_row) else "",
                            )

        # 正文句级扫描（Phase 10.1）：分子常在脚注正文而非表格
        # （"42749 shares were repurchased in open-market transactions"）。
        # 非表格行（非 ``|`` 开头）与短语词重叠达阈值时，行内数值成候选。
        for line in (content or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("|"):
                continue  # 表格行 / 空行
            line_words = cls._content_tokens(stripped)
            line_years = {
                f"y{m.group(1)}" for m in _PHRASE_YEAR_RE.finditer(stripped)
            }
            line_all = line_words | line_years
            if not line_all:
                continue
            shared = line_all & phrase_all
            if not shared:
                continue
            overlap = len(shared) / max(len(phrase_all), 1)
            # 阈值：短语覆盖 ≥ 1/3 或 ≥2 实词命中（防单泛化词行假阳性）
            if overlap < 0.3 and len(shared & phrase_words) < 2:
                continue
            anchor = 1.0 - min(overlap, 1.0) + (0.25 if len(shared) < 2 else 0.0)
            row_label = stripped[:40]
            # 剥离脚注标记（"( 1 ) ( 2 )"）：FinQA 正文行首的括号小编号
            # 是表格脚注引用而非财经数值（"42749 shares were repurchased
            # ..." 前的 "( 1 )" 会以 1.0 污染分子候选且锚分并列优先）
            numeric_line = _FOOTNOTE_MARKER_RE.sub(" ", stripped)
            for num_start, num_end, value, unit in cls._number_spans(numeric_line):
                out.append({
                    "value": value,
                    "unit": unit,
                    "period": str(year or ""),
                    "row_label": row_label,
                    "column": "",
                    "anchor_score": round(anchor + 0.1, 3),  # 正文来源轻微惩罚
                    "source": "text",
                })

        out.sort(key=lambda item: item["anchor_score"])
        # 同表同值去重（多级表头承接产生的重复）
        seen, deduped = set(), []
        for item in out:
            key = (item["row_label"], item["value"], item["column"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped[:max_n]

    @classmethod
    def collect_total_values(cls, content: str, max_n: int = 4) -> List[dict]:
        """表格 total 行数值候选（Phase 10.1 比率流分母回退）。

        "what percentage of X" 的 X 几乎总是某张表的合计行（"total" /
        "total xxx" / "xxx total"），但 X 短语与行标签常零词重叠
        （"future minimum rental payments" vs 行 "total"）——短语锚定
        扫不到任何分母候选时，以本回退补充。anchor_score 固定 0.75：
        低于真实短语命中（≤0.5），高于泛化兜底（1.25）。
        """
        out: List[dict] = []
        for table in parse_tables(content):
            header_row = table.headers[-1] if table.headers else []
            for row_index, cells in enumerate(table.rows):
                if not cells or not cells[0].strip():
                    continue
                label = cells[0].strip()
                if not re.search(r"^total\b|\btotal[:]?$", label, re.IGNORECASE):
                    continue
                for col in range(1, len(cells)):
                    parsed = _parse_cell_value(cells[col])
                    if parsed is None:
                        continue
                    out.append({
                        "value": parsed[0],
                        "unit": parsed[1],
                        "period": "",
                        "row_label": label,
                        "column": (
                            header_row[col].strip() if col < len(header_row) else ""
                        ),
                        "anchor_score": 0.75,
                        "source": "total_row",
                    })
        return out[:max_n]

    @classmethod
    def collect_year_values(
        cls, content: str, year: str, max_n: int = 3
    ) -> List[dict]:
        """年份窗口内的数值候选列表（Phase 9 计算流多候选枚举）。

        返回 ``[{"value", "unit", "period", "anchor_score"}]``，按到该年份
        提及的字符距离升序（越近越可能是「该年份的指标值」），至多 ``max_n``
        个。无该年份提及或窗口内无数值时返回空列表。
        """
        year_spans = [span for span in cls._year_spans(content) if span[2] == str(year)]
        if not year_spans:
            return []
        collected: List[Tuple[int, float, Optional[str]]] = []
        for num_start, num_end, value, unit in cls._number_spans(content):
            dist = min(
                (
                    # 数值在年份**之前**：财报语序几乎总是 "in 2015 ... $5829"
                    # （年份先行），年份前的数值通常属于上一行/其他年份 → 加惩罚
                    span[0] - num_end + _PRE_YEAR_PENALTY
                    if num_end <= span[0]
                    else (num_start - span[1] if span[1] <= num_start else 0)
                )
                for span in year_spans
            )
            if dist > _YEAR_ANCHOR_WINDOW:
                continue
            collected.append((dist, value, unit))
        collected.sort(key=lambda item: item[0])
        return [
            {"value": value, "unit": unit, "period": str(year), "anchor_score": float(dist)}
            for dist, value, unit in collected[:max_n]
        ]

    # Phase 9.7：实体锚定扫描（跨实体比较 / argmax 接力计算流）
    _ENTITY_ANCHOR_WINDOW = 160

    @classmethod
    def collect_entity_values(
        cls, content: str, entity: str, year: Optional[str] = None,
        spec_metric: Optional[str] = None, max_n: int = 4,
    ) -> List[dict]:
        """实体锚定数值候选（Phase 9.7 跨实体计算流）。

        数值取「实体提及 + 指标同义词 + 年份」三信号最近邻的数字：

        - 实体必须以词边界出现在 content 中（大小写不敏感），否则返回空
          （该 chunk 不属于此实体）；
        - ``spec_metric`` 给定 → 指标 span 限定为该 canonical 的提及，
          chunk 未提及目标指标时返回空（防跨指标串值，如 gross margin
          问题误收同实体的 revenue 数值）；
        - ``year`` 给定 → 数值须落在该年份 ``_ENTITY_ANCHOR_WINDOW``
          字符窗口内（chunk 无该年份提及 → 空）。

        返回 ``[{"value", "unit", "period", "anchor_score"}]`` 按可信度
        升序：anchor = 实体距离/100 +（指标未命中 +1.0）。
        """
        entity_spans = [
            (match.start(), match.end())
            for match in re.finditer(
                r"\b" + re.escape(entity) + r"\b", content or "", re.IGNORECASE
            )
        ]
        if not entity_spans:
            return []

        metric_spans = cls._metric_spans(content)
        if spec_metric:
            filtered = [s for s in metric_spans if s[2] == str(spec_metric).lower()]
            if not filtered:
                return []  # chunk 未提及目标指标 → 不产出候选（防跨指标串值）
            metric_spans = filtered
        year_spans = cls._year_spans(content)
        if year:
            year_spans = [s for s in year_spans if s[2] == str(year)]

        def _distance(num_start: int, num_end: int, spans) -> Optional[int]:
            best: Optional[int] = None
            for start, end, *_ in spans:
                if num_end <= start:
                    dist = start - num_end
                elif end <= num_start:
                    dist = num_start - end
                else:
                    dist = 0
                if best is None or dist < best:
                    best = dist
            return best

        items: List[dict] = []
        for num_start, num_end, value, unit in cls._number_spans(content):
            entity_dist = _distance(num_start, num_end, entity_spans)
            if entity_dist is None or entity_dist > cls._ENTITY_ANCHOR_WINDOW:
                continue
            metric_dist = _distance(num_start, num_end, metric_spans)
            year_dist = _distance(num_start, num_end, year_spans)
            if year and (year_dist is None or year_dist > cls._ENTITY_ANCHOR_WINDOW):
                continue  # 数值不在目标年份的上下文窗口内
            anchor = entity_dist / 100.0
            anchor += 0.0 if metric_dist is not None else 1.0
            items.append({
                "value": value,
                "unit": unit,
                "period": str(year) if year else None,
                "anchor_score": round(anchor, 3),
            })
        items.sort(key=lambda item: item["anchor_score"])
        return items[:max_n]

    def extract(self, chunk: dict) -> Evidence:
        """规则抽取：Entity（内容 → doc_name 回退）/ Period / Metric / Value / Definition。

        表格感知（Phase 5）：chunk 含表格时，metric/period/value 优先取
        「行标签命中指标词典 → 最新年份列」的结构化单元格。
        Phase 8：文本路径改为锚定最近邻抽取（指标提及/年份为锚）。
        """
        content = str(chunk.get("content") or "")
        metadata = chunk.get("metadata") or {} if isinstance(chunk.get("metadata"), dict) else {}
        doc_id = str(metadata.get("doc_id") or chunk.get("doc_id") or "")
        doc_name = str(chunk.get("doc_name") or metadata.get("doc_name") or doc_id or "")

        entity = self._extract_entity(content, doc_name)
        value, unit, period, metric = self._anchored_value(content, year_filter=None)
        if value is None:
            period = self._extract_period(content, doc_name)
            metric = _match_metric(content)
        definition = self._extract_definition(content)

        table_hit = _extract_from_tables(content)
        if table_hit is None:
            # Phase 6：标准结构未命中 → 旋转表头 / 水平多级表头（延迟导入避免环）
            from verifin.tools.table_parser import extract_from_tables_advanced

            advanced = extract_from_tables_advanced(content)
            if advanced is not None:
                metric_name, period_text, advanced_value, advanced_unit = advanced
                metric = metric_name or metric
                period = period_text
                value = advanced_value
                unit = advanced_unit if advanced_unit is not None else unit
        if table_hit is not None:
            table_metric, table_year, table_value, table_unit = table_hit
            metric = table_metric or metric
            period = str(table_year)
            value = table_value
            unit = table_unit if table_unit is not None else unit

        extracted = [entity, period, metric, value]
        confidence = sum(1 for item in extracted if item is not None) / 4.0

        return Evidence(
            chunk_id=str(chunk.get("chunk_id") or ""),
            doc_id=doc_id,
            doc_name=doc_name,
            content=content,
            page_num=metadata.get("page_num") if isinstance(metadata.get("page_num"), int) else None,
            section_title=metadata.get("section_title") or chunk.get("section_title"),
            entity=entity,
            period=period,
            metric=metric,
            definition=definition,
            value=value,
            unit=unit,
            extraction_confidence=round(confidence, 2),
        )

    @staticmethod
    def _extract_entity(content: str, doc_name: str) -> Optional[str]:
        candidates = extract_entity_candidates(content)
        if candidates:
            return candidates[0]
        # 回退：doc_name 前缀解析（novatech-fy2024.pdf → Novatech）
        match = _DOC_NAME_ENTITY_RE.search(doc_name)
        if match:
            token = match.group(1)
            if not token.isdigit():
                return token.title()
        return None

    @staticmethod
    def _extract_period(content: str, doc_name: str) -> Optional[str]:
        combined = f"{content} {doc_name}"
        match = _PERIOD_RE.search(combined)
        return match.group(0) if match else None

    @staticmethod
    def _extract_definition(content: str) -> Optional[str]:
        match = _DEFINITION_RE.search(content)
        return " ".join(match.group(1).strip().lower().split()) if match else None

    @staticmethod
    def _extract_value(content: str) -> tuple:
        # 先剔除年份，避免 "in 2024 ..." 被误抽为数值
        text = _YEAR_VALUE_RE.sub(" ", content)
        match = _VALUE_RE.search(text)
        if not match:
            return None, None
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            return None, None
        unit_raw = (match.group(2) or "").lower()
        unit = "%" if unit_raw == "%" else unit_raw or None
        return value, unit


def extract_evidence(chunk: dict) -> dict:
    """Tool 包装：chunk dict → Evidence dict（JSON 可序列化）。"""
    from dataclasses import asdict

    return asdict(EvidenceExtractor().extract(chunk))