"""Phase 5 — 上下文依赖的指标继承测试（tools/context.py + inheritance_rules.json）。"""

from __future__ import annotations

from verifin.core.nodes import planner_node
from verifin.core.state import AgentState
from verifin.tools.context import infer_metric, load_inheritance_rules


# 历史提及 revenue → "赚了多少钱" 继承为 profit（net income 财务语义）
def test_infer_metric_from_revenue_history() -> None:
    metric = infer_metric(
        "How much did NovaTech earn last year?",
        history=["What was NovaTech revenue in 2024?"],
    )
    assert metric == "profit"


# 历史提及 growth → 当前指标映射为 growth_rate
def test_infer_metric_from_growth_history() -> None:
    metric = infer_metric(
        "How much did NovaTech earn last year?",
        history=["How fast did NovaTech grow in 2024?"],
    )
    assert metric == "growth_rate"


# 显式指标优先：问题本身含 revenue，历史语境不覆盖
def test_explicit_metric_wins() -> None:
    metric = infer_metric(
        "What was NovaTech revenue in 2024?",
        history=["How fast did NovaTech grow in 2023?"],
    )
    assert metric == "revenue"


# 规则由 JSON 驱动：加载规则文件 + 无历史时的 fallback
def test_rules_loaded_from_json() -> None:
    rules = load_inheritance_rules()
    assert isinstance(rules, dict)
    assert "fallback_metric" in rules and "inheritance" in rules
    assert infer_metric("How much did NovaTech earn last year?") == rules["fallback_metric"]
    # 非赚钱类问题无隐含指标
    assert infer_metric("Tell me about NovaTech.") is None


# Planner 集成：隐含指标写入 state.claims，供 verifier 校验使用
def test_planner_writes_inherited_claim() -> None:
    state = AgentState(
        messages=[
            {"role": "user", "content": "What was NovaTech revenue in 2024?"},
            {"role": "user", "content": "How much did NovaTech earn last year?"},
        ]
    )
    out = planner_node(state)
    current_query = "How much did NovaTech earn last year?"
    assert out["claims"][current_query]["metric"] == "profit"
    assert out["claims"][current_query]["entity"] == "NovaTech"
    assert out["sub_tasks"][0] == current_query