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
    # Phase 12 行为变更（D5）：单年份 growth 触发 difference 模板
    # （"growth in 2024" = 2024 vs 2023，与 FinQA gold program
    # subtract(cur, prev), divide(#0, prev) 同构）；无年份 growth 不触发
    growth_1yr = planner_node(_stub_state(
        "What was NovaTech growth in 2024?"
    ))
    assert growth_1yr["calculation_requested"] is True
    assert growth_1yr["calculation_spec"]["kind"] == "difference"
    assert growth_1yr["calculation_spec"]["base_period"] == "2023"
    assert planner_node(_stub_state(
        "What was NovaTech growth?"
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

# ---------------------------------------------------------------------------
# Phase 12.2 跨文档污染守卫（_ratio_candidates）
# [72] 实测：CME 题（outstanding options / plans approved by security holders）
# 检索召回 GPN 同构股权计划表 → 分母锚到异公司值 766801，
# 正确分母是分子同表 total 行 1217121。
# ---------------------------------------------------------------------------

_CME_TABLE = (
    "Table data:\n"
    "| Plan Category | Number of Securities |\n"
    "| equity compensation plans approved by security holders | 1211143 |\n"
    "| equity compensation plans not approved by security holders | 5978 |\n"
    "| total | 1217121 |\n"
)
_GPN_TABLE = (
    "number of securities to be issued upon exercise of "
    "outstanding options warrants and rights 766801\n"
    "Table data:\n"
    "| equity compensation plans approved by security holders | 766801 |\n"
    "| total | 900000 |\n"
)


def _ratio_state() -> dict:
    return {
        "messages": [{
            "role": "user",
            "content": "what percentage of the outstanding options were from "
                       "plans approved by security holders?",
        }],
        "retrieved_docs": [
            {"chunk_id": "cme-1", "content": _CME_TABLE,
             "metadata": {"doc_id": "CME/2010/page_123.pdf"}},
            {"chunk_id": "gpn-1", "content": _GPN_TABLE,
             "metadata": {"doc_id": "GPN/2014/page_92.pdf"}},
        ],
    }


def _ratio_spec() -> "SimpleNamespace":  # noqa: F821
    from types import SimpleNamespace

    return SimpleNamespace(
        kind="ratio",
        numerator="plans approved by security holders",
        denominator="outstanding options",
        base_period=None,
    )


def test_ratio_cross_doc_guard_all_foreign(monkeypatch) -> None:
    """分母候选全部异文档 → 清空，total 回退只注入分子同文档合计行。"""
    from verifin.core import nodes as nodes_mod
    from verifin.core.nodes import _ratio_candidates

    # 屏蔽父文档补全：守卫单测聚焦检索 chunk 直扫路径
    monkeypatch.setattr(nodes_mod, "_call_tool", lambda name, **kw: [])

    out = _ratio_candidates(_ratio_state(), _ratio_spec())
    num_values = [c.value for c in out["num"]]
    den_values = [c.value for c in out["den"]]

    assert 1211143 in num_values            # 分子：CME 行标签命中
    assert 766801 not in den_values         # 跨文档分母被守卫剔除
    assert 1217121 in den_values            # 同文档 total 行回退注入
    for c in out["den"]:
        assert c.chunk_id == "cme-1"        # 分母全部来自分子同文档


def test_ratio_cross_doc_guard_mixed(monkeypatch) -> None:
    """分母候选混合（同文档 + 异文档）→ 只保留同文档候选。"""
    from types import SimpleNamespace

    from verifin.core import nodes as nodes_mod
    from verifin.core.nodes import _ratio_candidates

    monkeypatch.setattr(nodes_mod, "_call_tool", lambda name, **kw: [])

    state = _ratio_state()
    # CME 侧补一个弱分母命中（同文档 prose 值），模拟混合桶
    state["retrieved_docs"][0] = dict(state["retrieved_docs"][0])
    state["retrieved_docs"][0]["content"] = (
        "a total of 42456 outstanding options were outstanding under all plans\n"
        + _CME_TABLE
    )

    out = _ratio_candidates(state, _ratio_spec())
    den_values = [c.value for c in out["den"]]

    assert 42456 in den_values              # 同文档候选保留
    assert 766801 not in den_values         # 异文档候选剔除
    for c in out["den"]:
        assert c.chunk_id == "cme-1"
