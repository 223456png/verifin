"""上下文感知的隐含指标继承（不依赖 LangGraph）。

规则由 :file:`verifin/inheritance_rules.json` 驱动（随包分发，可被主提示词/配置覆盖）：
- 显式指标优先（问题文本本身含词典指标 → 直接返回）；
- 否则若问题匹配「赚钱类」模式：
    - 历史提及 growth → ``growth_rate``；
    - 历史提及 revenue → ``profit``（净利财务语义）；
    - 无历史 → 规则 fallback（profit）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

_RULES_PATH = Path(__file__).resolve().parent.parent / "inheritance_rules.json"
_rules_cache: Optional[dict] = None


def load_inheritance_rules(force_reload: bool = False) -> dict:
    """加载继承规则 JSON（进程内缓存；force_reload 强制重读）。"""
    global _rules_cache
    if _rules_cache is None or force_reload:
        with open(_RULES_PATH, encoding="utf-8") as handle:
            _rules_cache = json.load(handle)
    return _rules_cache


def _is_earn_query(query: str, rules: dict) -> bool:
    return any(
        re.search(pattern, query, re.IGNORECASE)
        for pattern in rules.get("earn_patterns", [])
    )


def infer_metric(query: str, history: Optional[List[str]] = None) -> Optional[str]:
    """推断查询的隐含指标（canonical key）；无法推断返回 None。

    Args:
        query: 当前用户问题。
        history: 之前轮次的用户消息列表（不含当前问题）。
    """
    from verifin.tools.verifier import _normalize_metric  # 局部导入避免循环依赖

    explicit = _normalize_metric(query or "")
    if explicit:
        return explicit

    rules = load_inheritance_rules()
    if not _is_earn_query(query or "", rules):
        return None

    history_text = " ".join(history or []).lower()
    for rule in rules.get("inheritance", []):
        if any(keyword in history_text for keyword in rule.get("history_has", [])):
            return rule.get("metric")
    return rules.get("fallback_metric")