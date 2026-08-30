"""Phase 6 — 偏好注入测试（core/preference_applier.py + 节点集成）。"""

from __future__ import annotations

from verifin.core.nodes import planner_node
from verifin.core.preference_applier import build_retrieval_query
from verifin.core.state import AgentState
from verifin.tools.verifier import resolve_conflicts


# Planner 注入：检索子任务拼接偏好 token（合并口径 + 来源）
def test_build_retrieval_query_with_preferences() -> None:
    from verifin.core.dialog_state import DialogState

    dialog = DialogState(
        source_preference="annual_report", consolidation_preference=True
    )
    query = build_retrieval_query(
        {"entity": "NovaTech", "metric": "revenue", "period": "2024"}, dialog
    )
    assert "NovaTech" in query and "revenue" in query and "2024" in query
    assert "consolidated" in query and "annual_report" in query
    # 无偏好时输出与 Phase 5 一致
    plain = build_retrieval_query(
        {"entity": "NovaTech", "metric": "revenue", "period": "2024"}, DialogState()
    )
    assert plain == "retrieve NovaTech revenue 2024 more specifically"


# Planner hook 记录 preference_applied；无实体问题从偏好继承
def test_planner_records_preference_applied() -> None:
    state = AgentState(
        messages=[{"role": "user", "content": "What was its revenue in 2024?"}],
        dialog_state={
            "entity_preference": "NovaTech",
            "source_preference": "annual_report",
            "consolidation_preference": True,
            "preference_source": {"entity": "explicit"},
        },
    )
    out = planner_node(state)
    assert out["claims"]["What was its revenue in 2024?"]["entity"] == "NovaTech"
    refined = out["sub_tasks"][-1]
    assert "NovaTech" in refined and "annual_report" in refined and "consolidated" in refined
    applied = out["hooks"][-1]["preference_applied"]
    assert applied["entity"]["value"] == "NovaTech"


# Verifier 注入：偏好来源可信度 +0.2，conflict 输出 preference_note
def test_conflict_arbitration_preference_bonus() -> None:
    results = [
        {"chunk_id": "filing", "value": 12400.0, "unit": "million",
         "period": "2024", "source_type": "filing"},
        {"chunk_id": "news", "value": 12410.0, "unit": "million",
         "period": "2024", "source_type": "news"},
    ]
    baseline = resolve_conflicts(results)
    preferred = resolve_conflicts(
        results, preferences={"source_preference": "news"}
    )
    assert baseline["reconciled_value"] is not None
    assert preferred["reconciled_value"] is not None
    # 偏好新闻稿 → 加权均值偏向新闻稿数值（高于基线）
    assert preferred["reconciled_value"] > baseline["reconciled_value"]
    assert "新闻稿" in preferred.get("preference_note", "")