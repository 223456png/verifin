"""会话级对话状态管理（不依赖 LangGraph）。

维护多轮对话的偏好记忆（实体/指标/期间/来源/口径），支持：
- **继承**：新 claim 空字段从偏好补齐；
- **覆盖**：显式字段覆盖旧偏好并记录来源（explicit）；
- **隐式提升**：连续同实体 / 单一指标 → 偏好来源 implicit；
- **序列化**：to_dict/from_dict 供 JSON 调试、回放与 checkpoint 持久化。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class DialogState:
    """一轮到多轮对话的累积状态（新对话重置，不跨会话持久化）。"""

    session_id: str = ""
    entity_preference: Optional[str] = None
    metric_preference: Optional[str] = None
    period_preference: Optional[str] = None
    source_preference: Optional[str] = None
    consolidation_preference: Optional[bool] = None
    last_claim: dict = field(default_factory=dict)
    turn_count: int = 0
    # 偏好来源追溯：key ∈ {entity, metric, period, source, consolidation}，value ∈ {explicit, implicit}
    preference_source: Dict[str, str] = field(default_factory=dict)
    # 隐式推断计数器（连续同实体 / 单一指标）
    consecutive_entity: Optional[str] = None
    consecutive_entity_count: int = 0
    metrics_seen: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[dict]) -> "DialogState":
        data = data or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})


_PREFERENCE_FIELDS = ("entity", "metric", "period")


def inherit_preferences(dialog: DialogState, claim: dict) -> tuple:
    """把 claim 的空字段从偏好补齐；返回 (补齐后的 claim, applied 记录)。

    applied: {field: {"value": ..., "source": preference_source 或 "implicit"}}
    """
    claim = dict(claim)
    applied: Dict[str, dict] = {}
    for field_name in _PREFERENCE_FIELDS:
        if claim.get(field_name):
            continue
        preference = getattr(dialog, f"{field_name}_preference")
        if preference is None:
            continue
        claim[field_name] = preference
        applied[field_name] = {
            "value": preference,
            "source": dialog.preference_source.get(field_name, "implicit"),
        }
    return claim, applied


def update_claim(dialog: DialogState, claim: dict) -> DialogState:
    """多轮更新：显式字段覆盖偏好（来源 explicit）、推进轮次与隐式计数器。"""
    dialog.turn_count += 1
    dialog.last_claim = dict(claim)

    # 显式字段 → 覆盖偏好并记录来源（优先级：显式指令 > 显式 claim > 隐式 > 默认）
    for field_name in _PREFERENCE_FIELDS:
        value = claim.get(field_name)
        if not value:
            continue
        setattr(dialog, f"{field_name}_preference", value)
        dialog.preference_source[field_name] = "explicit"

    # 隐式计数：连续同实体
    entity = claim.get("entity")
    if entity:
        if entity == dialog.consecutive_entity:
            dialog.consecutive_entity_count += 1
        else:
            dialog.consecutive_entity = entity
            dialog.consecutive_entity_count = 1
    metric = claim.get("metric")
    if metric and metric not in dialog.metrics_seen:
        dialog.metrics_seen.append(metric)
    return dialog


def promote_implicit(dialog: DialogState, rules: dict) -> DialogState:
    """隐式提升：达到规则阈值且未被显式声明过的偏好 → 来源 implicit。

    rules: {"consecutive_entity_turns": int, "min_metric_turns": int}
    """
    entity_turns = int(rules.get("consecutive_entity_turns", 3))
    metric_turns = int(rules.get("min_metric_turns", 2))

    if (
        dialog.entity_preference
        and dialog.consecutive_entity_count >= entity_turns
        and dialog.preference_source.get("entity") != "explicit"
    ):
        dialog.preference_source["entity"] = "implicit"

    metrics = [
        metric for metric in dialog.metrics_seen
        if metric and metric != dialog.metric_preference
    ]
    if (
        dialog.metric_preference
        and not metrics
        and len(dialog.metrics_seen) >= metric_turns
        and dialog.preference_source.get("metric") != "explicit"
    ):
        dialog.preference_source["metric"] = "implicit"
    return dialog
