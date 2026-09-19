"""偏好抽取：从用户指令中提取显式偏好（规则驱动，不依赖 LangGraph）。

规则定义于 :file:`verifin/config/preference_rules.json`（随包分发）：
- 显式指令按正则匹配（"只看年报" → source=annual_report、"合并口径" → consolidation=True）；
- 同一键后出现的规则覆盖先出现的（自然 dict 赋值顺序）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional

_RULES_PATH = Path(__file__).resolve().parent.parent / "config" / "preference_rules.json"
_rules_cache: Optional[dict] = None


def load_preference_rules(force_reload: bool = False) -> dict:
    """加载偏好规则 JSON（进程内缓存）。"""
    global _rules_cache
    if _rules_cache is None or force_reload:
        with open(_RULES_PATH, encoding="utf-8") as handle:
            _rules_cache = json.load(handle)
    return _rules_cache


def extract_preferences(text: str) -> Dict[str, Any]:
    """从用户指令文本抽取显式偏好 dict（如 {"source": "annual_report"}）。"""
    rules = load_preference_rules()
    preferences: Dict[str, Any] = {}
    for rule in rules.get("explicit", []):
        match = re.search(rule["pattern"], text, re.IGNORECASE)
        if not match:
            continue
        if rule.get("match_group"):
            preferences[rule["preference"]] = match.group(rule["match_group"])
        else:
            preferences[rule["preference"]] = rule["value"]
    return preferences
