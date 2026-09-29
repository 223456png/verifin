# -*- coding: utf-8 -*-
"""会话入口的查询归一化：中文/中英混合/问法变体 → 检索系统能懂的形态。

只在 API 会话入口（``verifin.api``）调用，benchmark 与测试直跑 graph，
不经此层——评测口径零影响。

背景：规则 planner 的抽取链路按英文书写（实体首字母大写正则、英文指标
词典、``\\b2024\\b`` 词边界年份），中文问法三处全断：实体抓不到、指标
归 None、``2023年`` 的"年"字紧贴数字破坏词边界。本模块把中文指标词映射
为 canonical 英文、剥离中文年份后缀、过滤中文虚词，并在可选的已知实体
表内做大小写修正（novatech → NovaTech），使全小写英文问法也能命中。
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Tuple

# 中文指标词 → canonical 英文（长词优先替换，营业收入必须先于收入）
_ZH_METRIC_TERMS: List[Tuple[str, str]] = [
    ("营业收入", "revenue"),
    ("营业利润率", "operating margin"),
    ("营业利润", "operating income"),
    ("净利润", "net income"),
    ("净利率", "net margin"),
    ("毛利率", "gross margin"),
    ("毛利润", "gross profit"),
    ("净收入", "net income"),
    ("销售额", "sales"),
    ("营收", "revenue"),
    ("收入", "revenue"),
    ("营业额", "revenue"),
    ("毛利", "gross profit"),
    ("成本", "cost"),
    ("费用", "expense"),
    ("增长率", "growth"),
    ("增速", "growth"),
    ("增幅", "growth"),
    ("增长", "growth"),
]

# 中文问句/连接虚词（归一化后剥离，保留字母数字与标点语义）
_ZH_FILLER_RE = re.compile(
    r"[的了吗呢吧请问啊呀]|是多少|多少|什么是|怎么样|如何|以及|还有|和|与"
    r"|是多少|请|告诉我|查询|查一下|看看|看下"
)

# 年份紧贴中文量词/字符："2023年" / "2023 财年" → " 2023 "
_ZH_YEAR_RE = re.compile(r"(20\d{2})\s*(?:财)?年")
_YEAR_PREFIX_ZH_RE = re.compile(r"年\s*(20\d{2})")


def _known_entity_pattern(entities: Iterable[str]) -> List[str]:
    """已知实体表 → 去重、按长度降序（长名优先防子串遮蔽）。"""
    names = {e.strip() for e in entities if e and len(e.strip()) >= 2}
    return sorted(names, key=len, reverse=True)


def normalize_query(
    query: str,
    known_entities: Optional[Iterable[str]] = None,
) -> str:
    """归一化用户问题；输出空串时调用方应回退原文本。

    Args:
        query: 原始用户输入（中/英/混合均可）。
        known_entities: 语料内已知实体拼写表（大小写修正用，如
            ``["NovaTech", "Orion"]``）。demo 合成语料注入四公司名；
            预构建索引路径可传 doc 实体高频表，无则跳过修正。
    """
    if not query or not query.strip():
        return query

    s = unicodedata.normalize("NFKC", query)  # 全角数字/字母/标点 → 半角

    # 中文年份形态先行（保住 \b 边界）：2023年 / 2023 财年 / 年2023
    s = _ZH_YEAR_RE.sub(r" \1 ", s)
    s = _YEAR_PREFIX_ZH_RE.sub(r" \1 ", s)

    # 中文指标 → canonical 英文（长词优先）
    for zh, en in _ZH_METRIC_TERMS:
        if zh in s:
            s = s.replace(zh, f" {en} ")

    # 中文虚词/问句词 → 空格
    s = _ZH_FILLER_RE.sub(" ", s)

    # 已知实体大小写修正（novatech → 语料原拼写 NovaTech；长名优先）
    for ent in _known_entity_pattern(known_entities or []):
        s = re.sub(
            rf"(?<![A-Za-z0-9]){re.escape(ent)}(?![A-Za-z0-9])",
            ent, s, flags=re.IGNORECASE,
        )

    # 空白归一（多空格合一，保留首尾干净）
    s = " ".join(s.split())
    return s or query
