"""数字单位智能归一化（不依赖 LangGraph）。

约定：可比数值统一折算到基准单位 **百万（million)**；
百分比（"%"）与绝对金额不可跨量纲换算，原样保留供上层判异。
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# 各单位 → million 的换算系数（百分号独立处理，不进此表）
_FACTOR_TO_MILLION = {
    "": 1.0, "million": 1.0, "m": 1.0, "millions": 1.0,
    "thousand": 0.001, "k": 0.001,
    "billion": 1000.0, "b": 1000.0, "billions": 1000.0,
    "万": 0.01, "亿": 100.0,
}
_PERCENT_SIGNS = {"%", "percent", "pct"}
_NUMERIC_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def _canonical_unit(unit: Optional[str]) -> str:
    """归一单位写法（大小写、复数、常见别名）。"""
    if unit is None:
        return ""
    lowered = str(unit).strip().lower()
    aliases = {
        "mio": "million", "mn": "million", "mil": "million", "mm": "million",
        "bn": "billion",
        "yuan": "", "usd": "", "dollars": "", "$": "",
    }
    return aliases.get(lowered, lowered)


def normalize_to_base(value: float, unit: Optional[str]) -> Optional[Tuple[float, str]]:
    """把 (value, unit) 折算为基准形态 (value_in_millions, "million")。

    - 百分比原样返回 ``(value, "%")``；
    - 未知单位返回 None（表示不可比较）。
    """
    canonical = _canonical_unit(unit)
    if canonical in _PERCENT_SIGNS:
        return float(value), "%"
    factor = _FACTOR_TO_MILLION.get(canonical)
    if factor is None:
        return None
    return float(value) * factor, "million"


def convert_unit(value: float, from_unit: str, to_unit: str) -> float:
    """单位换算：convert_unit(12.4, "billion", "million") == 12400.0。

    百分比（%）与其他单位不可换算，抛出 ValueError。
    """
    from_canonical = _canonical_unit(from_unit)
    to_canonical = _canonical_unit(to_unit)
    if from_canonical in _PERCENT_SIGNS or to_canonical in _PERCENT_SIGNS:
        raise ValueError("percent unit cannot be converted to absolute units")
    from_factor = _FACTOR_TO_MILLION.get(from_canonical)
    to_factor = _FACTOR_TO_MILLION.get(to_canonical)
    if from_factor is None:
        raise ValueError(f"unknown unit: {from_unit!r}")
    if to_factor is None:
        raise ValueError(f"unknown unit: {to_unit!r}")
    return float(value) * from_factor / to_factor


def parse_number_with_unit(text: str) -> Optional[Tuple[float, Optional[str]]]:
    """从片段文本解析「数值 + 可选单位」（供表格单元格复用）。"""
    match = _NUMERIC_RE.search(text)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    rest = text[match.end():].strip()
    unit: Optional[str] = None
    for _token in _PERCENT_SIGNS:
        if rest.startswith("%"):
            unit = "%"
            break
    if unit is None:
        word = re.match(r"[A-Za-z\u4e00-\u9fff]+", rest)
        if word:
            candidate = _canonical_unit(word.group(0))
            if candidate in _FACTOR_TO_MILLION or candidate in _PERCENT_SIGNS:
                unit = candidate
    return value, unit
