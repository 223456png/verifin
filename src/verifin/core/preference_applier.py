"""偏好注入：Planner 规划前把偏好并入 claim 与检索查询（不依赖 LangGraph）。

流程（优先级：显式指令 > 显式 claim > 隐式推断 > 系统默认）：
1. 显式指令（preference_extractor 输出）写入 DialogState，来源 explicit；
2. claim 空字段从 DialogState 偏好继承；
3. 显式 claim 字段覆盖旧偏好（update_claim），隐式规则提升（promote_implicit）；
4. 产出检索子任务查询（含合并口径 / 来源偏好 token）与 applied 记录。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from verifin.core.dialog_state import (
    DialogState,
    inherit_preferences,
    promote_implicit,
    update_claim,
)
from verifin.core.preference_extractor import load_preference_rules

_EXPLICIT_FIELDS = {
    "source": "source_preference",
    "consolidation": "consolidation_preference",
    "period": "period_preference",
}


def apply_preferences(
    dialog: DialogState,
    claim: dict,
    explicit_prefs: Optional[Dict[str, Any]] = None,
) -> tuple:
    """把显式指令与历史偏好并入 claim。

    Returns:
        (更新后的 DialogState, 补齐后的 claim, applied 记录 dict)
        applied: {field: {"value": ...}} 或 {field: {"value":..., "source":..., "override": bool}}
    """
    applied: Dict[str, dict] = {}

    # 1. 显式指令 → 偏好覆盖（来源 explicit）
    for key, value in (explicit_prefs or {}).items():
        field_name = _EXPLICIT_FIELDS.get(key)
        if field_name is None:
            continue
        applied[key] = {
            "value": value,
            "source": "explicit",
            "override": getattr(dialog, field_name) is not None
            and getattr(dialog, field_name) != value,
        }
        setattr(dialog, field_name, value)
        dialog.preference_source[key] = "explicit"

    # 2. 空字段从偏好继承
    claim, inherited = inherit_preferences(dialog, claim)
    applied.update(inherited)

    # 3. 轮次更新 + 隐式提升
    dialog = update_claim(dialog, claim)
    dialog = promote_implicit(dialog, load_preference_rules().get("implicit", {}))
    return dialog, claim, applied


def build_retrieval_query(claim: dict, dialog: DialogState) -> str:
    """构造细分检索子任务；有偏好时附加 consolidated / 来源 token。

    无偏好输出与 Phase 5 一致：``retrieve {entity} {metric} {period} more specifically``。
    """
    parts = [
        part
        for part in (
            claim.get("entity"),
            (claim.get("metric") or "").replace("_", " "),
            claim.get("period"),
        )
        if part
    ]
    if dialog.consolidation_preference:
        parts.append("consolidated")
    if dialog.source_preference:
        parts.append(str(dialog.source_preference))
    return f"retrieve {' '.join(parts)} more specifically"