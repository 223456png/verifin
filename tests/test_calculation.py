"""Phase 8 — 数值计算接入主循环测试（planner 检测 / calculator 节点 / 图路由 / 端到端）。"""

from __future__ import annotations

import pytest

from verifin.core.graph import build_agent_graph
from verifin.core.nodes import (
    _detect_calculation,
    calculator_node,
    planner_node,
)
from verifin.core.runner import AgentRunner
from verifin.core.state import AgentState
from verifin.tools.registry import ToolRegistry, register_builtin_tools

GROWTH_QUERY = "What was the percentage growth in NovaTech revenue from 2023 to 2024?"


@pytest.fixture(autouse=True)
def _clean_registry():
    ToolRegistry().reset()
    yield
    ToolRegistry().reset()


def _stub_state(query: str, **overrides) -> AgentState:
    return AgentState(
        messages=[{"role": "user", "content": query}], **overrides
    )


def _chunk(content: str, doc_id: str) -> dict:
    return {"chunk_id": doc_id, "content": content, "metadata": {"doc_id": doc_id}}


def _runner(responses: list[list[dict]]) -> AgentRunner:
    """注册真实内置工具 + stub retrieve（顺序返回；耗尽重复最后一个）。"""
    calls = [0]
    register_builtin_tools()

    def fake(query: str, top_k: int = 10) -> list[dict]:
        index = min(calls[0], len(responses) - 1)
        calls[0] += 1
        return responses[index]

    ToolRegistry().register("retrieve", fake, "stub retrieve", {"type": "object"})
    return AgentRunner(build_agent_graph())


def _path(final: dict) -> list[str]:
    return [hook["node"] for hook in final["hooks"]]


# 1. 增长类问题（growth 关键词 + 双年份）→ planner 触发计算并拆基期/比较期子任务
def test_planner_detects_growth_query() -> None:
    out = planner_node(_stub_state(GROWTH_QUERY))
    assert out["calculation_requested"] is True
    assert out["calculation_spec"]["kind"] == "growth_pct"
    assert out["calculation_spec"]["base_period"] == "2023"
    assert out["calculation_spec"]["target_period"] == "2024"
    assert out["calculation_spec"]["metric"] == "revenue"
    assert len(out["sub_tasks"]) == 2
    assert all("Novatech".lower() in task.lower() for task in out["sub_tasks"])
    assert any("2023" in task for task in out["sub_tasks"])
    assert any("2024" in task for task in out["sub_tasks"])


# 2. 百分比问题（percentage 关键词）→ 同样触发；无关键词/单年份 → 不触发（防误伤）
def test_planner_detects_percentage_query() -> None:
    out = planner_node(_stub_state(
        "What is the percentage change in Orion revenue between 2022 and 2023?"
    ))
    assert out["calculation_requested"] is True
    assert out["calculation_spec"]["base_period"] == "2022"
    # 防误伤：无计算关键词的普通查询
    assert planner_node(_stub_state(
        "What was NovaTech revenue in 2024?"
    ))["calculation_requested"] is False
    # Phase 9：单年份 + 净变动关键词 → difference 模板（基期推断为上一年，
    # FinQA 高频模式 "net change in X during 2015" = subtract(cur, prev)）
    single_year_change = planner_node(_stub_state(
        "How much did revenue change in 2024?"
    ))
    assert single_year_change["calculation_requested"] is True
    assert single_year_change["calculation_spec"]["kind"] == "difference"
    assert single_year_change["calculation_spec"]["base_period"] == "2023"
    assert single_year_change["calculation_spec"]["target_period"] == "2024"
    # growth 无双年份仍不触发（"growth in 2024" 常指描述性表述而非计算）
    assert planner_node(_stub_state(
        "What was NovaTech growth in 2024?"
    ))["calculation_requested"] is False
    assert _detect_calculation("plain revenue question") is None


# 3. calculator 节点：从通过证据取基期/比较期数值 → calc_expression 求增长率
def test_calculation_node_returns_value() -> None:
    register_builtin_tools()
    state = _stub_state(
        GROWTH_QUERY,
        calculation_requested=True,
        calculation_spec={"kind": "growth_pct", "base_period": "2023", "target_period": "2024"},
        verify_flags={
            "base_task": {
                "decision": "ACCEPT",
                "results": [{"chunk_id": "a", "passed": True, "value": 10000.0,
                             "unit": "million", "period": "FY2023"}],
            },
            "target_task": {
                "decision": "ACCEPT",
                "results": [{"chunk_id": "b", "passed": True, "value": 12000.0,
                             "unit": "million", "period": "FY2024"}],
            },
        },
    )
    out = calculator_node(state)
    result = out["calculation_result"]
    assert result["value"] == pytest.approx(20.0)      # 百分比刻度
    assert result["fraction"] == pytest.approx(0.2)    # 比率刻度（FinQA 裸数值 GT）
    assert result["difference"] == pytest.approx(2000.0)  # 差值刻度
    assert result["unit"] == "%"
    assert "10000" in result["expression"] and "12000" in result["expression"]
    assert result["evidence"] == ["a", "b"]
    assert out["next_step"] == "synthesizer"
    # 缺少比较期数值 → error 降级（不抛出）
    missing = _stub_state(
        GROWTH_QUERY,
        calculation_spec={"kind": "growth_pct", "base_period": "2023", "target_period": "2024"},
        verify_flags={"base_task": {"decision": "ACCEPT", "results": [
            {"chunk_id": "a", "passed": True, "value": 10000.0, "unit": "million", "period": "FY2023"},
        ]}},
    )
    degraded = calculator_node(missing)
    assert degraded["calculation_result"]["value"] is None
    assert "missing" in degraded["calculation_result"]["error"]


# 4. 图路由：有计算需求时路径包含 calculator
def test_graph_routes_to_calculator() -> None:
    runner = _runner([
        [_chunk("NovaTech revenue in 2023 was $10,000 million.", "base")],
        [_chunk("NovaTech revenue in 2024 was $12,000 million.", "target")],
    ])
    final = runner.run(GROWTH_QUERY)
    path = _path(final)
    assert "calculator" in path, f"计算问题应路由到 calculator: {path}"
    assert final["calculation_result"]["value"] == pytest.approx(20.0)
    answers = [m for m in final["messages"] if m.get("role") == "assistant"]
    assert "Calculated result" in answers[-1]["content"]


# 5. 图路由：无计算需求时跳过 calculator（Phase 3 路径不变）
def test_graph_skips_calculator_when_not_needed() -> None:
    runner = _runner([
        [_chunk("NovaTech revenue in 2024 was $12,000 million.", "good")],
    ])
    final = runner.run("What was NovaTech revenue in 2024?")
    assert "calculator" not in _path(final)
    assert final["calculation_requested"] is False


# 6. 端到端：检索(两年) → 校验 → 计算 → 合成，数值答案 20.0%
def test_end_to_end_with_calculation() -> None:
    runner = _runner([
        [_chunk("NovaTech revenue in 2023 was $10,000 million.", "base")],
        [_chunk("NovaTech revenue in 2024 was $12,000 million.", "target")],
    ])
    final = runner.run(GROWTH_QUERY)
    path = _path(final)
    assert path.count("retriever") == 2 and path.count("verifier") == 2
    assert path[-1] == "synthesizer" and path[-2] == "calculator"
    # 工具族轨迹包含 calc_expression（registry 历史经 calculator 调用）
    result_value = final["calculation_result"]["value"]
    assert result_value == pytest.approx(20.0)
    answer = [m for m in final["messages"] if m.get("role") == "assistant"][-1]["content"]
    assert "20.0%" in answer
    assert final["next_step"] == "end"