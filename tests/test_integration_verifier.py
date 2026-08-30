"""Phase 4 — Agent × 真实 Verifier 集成测试（行为级）。"""

from __future__ import annotations

import pytest

from verifin.core.graph import build_agent_graph
from verifin.core.runner import AgentRunner
from verifin.tools.registry import ToolRegistry, register_builtin_tools

QUERY = "What was NovaTech revenue in 2024?"


@pytest.fixture(autouse=True)
def _clean_registry():
    ToolRegistry().reset()
    yield
    ToolRegistry().reset()


def _register_stub_retrieve(responses: list[list[dict]]):
    """注册真实内置工具（含真实 Verifier/Calculator），仅 stub retrieve。"""

    def fake(query: str, top_k: int = 10) -> list[dict]:
        index = min(calls[0], len(responses) - 1)
        calls[0] += 1
        return responses[index]

    calls = [0]
    register_builtin_tools()
    ToolRegistry().register("retrieve", fake, "stub retrieve", {"type": "object"})
    return calls


def _node_path(final: dict) -> list[str]:
    return [hook["node"] for hook in final["hooks"]]


def _decision(final: dict) -> str:
    flags: dict = final.get("verify_flags") or {}
    return next(
        (flag.get("decision", "REJECT") for flag in flags.values() if isinstance(flag, dict)),
        "REJECT",
    )


# 9.4.1 Agent 端到端：真实 Verifier 校验通过 → ACCEPT → 无 replan
def test_agent_with_real_verifier() -> None:
    _register_stub_retrieve([
        [{"chunk_id": "d1", "content": "NovaTech revenue in 2024 was $12,400 million.",
          "metadata": {"doc_id": "nova-fy2024"}}],
    ])
    runner = AgentRunner(build_agent_graph())
    final = runner.run(QUERY)
    assert final["next_step"] == "end"
    assert "replanner" not in _node_path(final)
    assert _decision(final) == "ACCEPT"
    flags: dict = final["verify_flags"]
    # 每项通过子任务均记录完整校验结构（claim/results/decision）
    for flag in flags.values():
        assert {"claim", "results", "passed_count", "total_count", "decision"} <= set(flag)


# 9.4.2 Verifier REJECT（指标错配：gross margin ≠ revenue）→ 触发 Replanner → 二次检索成功
def test_agent_with_replan_triggered() -> None:
    _register_stub_retrieve([
        [{"chunk_id": "bad", "content": "NovaTech gross margin was 40% in 2024.",
          "metadata": {"doc_id": "nova-fy2024"}}],
        [{"chunk_id": "good", "content": "NovaTech revenue in 2024 was $12,400 million.",
          "metadata": {"doc_id": "nova-fy2024"}}],
    ])
    runner = AgentRunner(build_agent_graph())
    final = runner.run(QUERY)
    path = _node_path(final)
    assert "replanner" in path, f"指标错配应触发 replan: {path}"
    assert final["retry_count"] == 1
    assert final["next_step"] == "end"
    # 重规划后的定向子任务应包含 claim 要素（entity + metric + period）
    refined_task = final["sub_tasks"][-1]
    assert "NovaTech" in refined_task and "revenue" in refined_task and "2024" in refined_task


# 9.4.3 Synthesizer 仅绑定 ACCEPT 证据（PARTIAL 场景：混入不匹配文档）
def test_agent_synthesizer_only_accepted() -> None:
    _register_stub_retrieve([
        [
            {"chunk_id": "good", "content": "NovaTech revenue in 2024 was $12,400 million.",
             "metadata": {"doc_id": "nova-fy2024"}},
            {"chunk_id": "bad", "content": "Helios Energy revenue in 2024 was $500 million.",
             "metadata": {"doc_id": "helion-fy2024"}},
        ],
    ])
    runner = AgentRunner(build_agent_graph())
    final = runner.run(QUERY)
    assert final["next_step"] == "end"
    answer = [m for m in final["messages"] if m.get("role") == "assistant"][-1]["content"]
    assert "NovaTech" in answer
    assert "Helios" not in answer, f"未通过证据不得进入答案:\n{answer}"