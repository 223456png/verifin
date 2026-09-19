"""Phase 6 — 会话状态管理测试（core/dialog_state.py）。"""

from __future__ import annotations

from verifin.core.dialog_state import (
    DialogState,
    inherit_preferences,
    promote_implicit,
    update_claim,
)

_RULES = {"consecutive_entity_turns": 3, "min_metric_turns": 2}


def _dialog() -> DialogState:
    return DialogState(
        entity_preference="NovaTech",
        metric_preference="revenue",
        period_preference="FY2024",
    )


# 多轮继承 1：entity 为空 → 从偏好继承；显式字段保留
def test_inherit_entity_fills_empty() -> None:
    dialog = _dialog()
    claim = {"entity": None, "period": None, "metric": "operating_margin"}
    filled, applied = inherit_preferences(dialog, claim)
    assert filled["entity"] == "NovaTech"
    assert filled["period"] == "FY2024"
    assert filled["metric"] == "operating_margin"  # 显式不变
    assert applied["entity"]["value"] == "NovaTech"


# 多轮继承 2：metric/period 为空 → 从偏好继承
def test_inherit_metric_and_period() -> None:
    dialog = _dialog()
    filled, _ = inherit_preferences(
        dialog, {"entity": "NovaTech", "metric": None, "period": None}
    )
    assert filled["metric"] == "revenue"
    assert filled["period"] == "FY2024"


# 多轮继承 3：无偏好可继承 → 保持 None，applied 为空
def test_inherit_without_preferences() -> None:
    dialog = DialogState()
    filled, applied = inherit_preferences(
        dialog, {"entity": None, "metric": None, "period": None}
    )
    assert filled == {"entity": None, "metric": None, "period": None}
    assert applied == {}


# 覆盖：新 claim 显式给出 → 覆盖旧偏好并记录来源 explicit
def test_explicit_claim_overrides_preference() -> None:
    dialog = _dialog()
    dialog = update_claim(dialog, {"entity": "Helios", "metric": "cost", "period": "2024"})
    assert dialog.entity_preference == "Helios"
    assert dialog.metric_preference == "cost"
    assert dialog.preference_source["entity"] == "explicit"
    assert dialog.turn_count == 1


# 隐式提升：连续同实体/单一指标 → 来源 implicit（未被显式声明时）
def test_implicit_promotions() -> None:
    dialog = DialogState(
        entity_preference="NovaTech",
        metric_preference="revenue",
        consecutive_entity="NovaTech",
        consecutive_entity_count=3,
        metrics_seen=["revenue", "revenue"],
    )
    dialog = promote_implicit(dialog, _RULES)
    assert dialog.preference_source["entity"] == "implicit"
    assert dialog.preference_source["metric"] == "implicit"


# 序列化往返一致（JSON 调试/回放）
def test_dialog_state_roundtrip() -> None:
    dialog = _dialog()
    dialog = update_claim(dialog, {"entity": "NovaTech", "metric": "revenue", "period": "FY2024"})
    restored = DialogState.from_dict(dialog.to_dict())
    assert restored.entity_preference == dialog.entity_preference
    assert restored.metric_preference == dialog.metric_preference
    assert restored.turn_count == dialog.turn_count
    assert restored.preference_source == dialog.preference_source
