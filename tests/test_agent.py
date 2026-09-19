"""Phase 3 — ReAct Agent Core（LangGraph 状态机）规格测试。

设计要点（docs/designs/2026-08-29-phase-3-verifin-agent-core-design.md）：
- 领域 Tool 不依赖 LangGraph：测试经 ToolRegistry 注入 stub 工具，不耦合真实索引/LLM；
- 动态路径是确定性证明：同一问题第二次运行携带 verify_flags passed 上下文 → 缩短路径；
- hooks 记录节点访问轨迹，node 序列即执行路径。
"""

from __future__ import annotations

import pytest

from verifin.core.graph import build_agent_graph
from verifin.core.nodes import (
    planner_node,
    replanner_node,
    retriever_node,
    synthesizer_node,
    verifier_node,
)
from verifin.core.runner import AgentRunner
from verifin.core.state import AgentState
from verifin.schemas import ToolCall, ToolResult
from verifin.tools.registry import ToolRegistry, register_builtin_tools

QUERY = "What was NovaTech revenue?"


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个测试前重置注册表单例，避免跨测试污染。"""
    ToolRegistry().reset()
    yield
    ToolRegistry().reset()


def _register_stub_retrieve(responses: list[list[dict]]):
    """注册真实内置工具（含真实 Verifier/Calculator），仅 stub retrieve；耗尽后重复最后一个。"""

    def fake(query: str, top_k: int = 10) -> list[dict]:
        index = min(calls[0], len(responses) - 1)
        calls[0] += 1
        return responses[index]

    calls = [0]
    register_builtin_tools()
    ToolRegistry().register("retrieve", fake, "stub retrieve", {"type": "object"})
    return calls


def _stub_state(query: str = QUERY, **overrides) -> AgentState:
    kwargs = {"messages": [{"role": "user", "content": query}], **overrides}
    return AgentState(**kwargs)


def _node_path(final: dict) -> list[str]:
    return [hook["node"] for hook in final["hooks"]]


# 7.1 多次获取 Registry 实例 → 断言是同一个对象
def test_tool_registry_singleton() -> None:
    assert ToolRegistry() is ToolRegistry()


# 7.2 注册一个测试工具 → 执行 → 断言结果正确
def test_tool_registration_and_execution() -> None:
    registry = ToolRegistry()
    registry.register(
        "add", lambda a, b: a + b, "add two numbers",
        {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}},
    )
    result = registry.execute(ToolCall(tool_name="add", args={"a": 1, "b": 2}, call_id="c1"))
    assert isinstance(result, ToolResult)
    assert result.success and result.output == 3
    assert result.duration_ms >= 0
    # 未注册工具 → success=False 且带 error
    missing = registry.execute(ToolCall(tool_name="missing", args={}, call_id="c2"))
    assert missing.success is False and "not found" in (missing.error or "")
    # 异常工具 → success=False 且不抛出
    registry.register("boom", lambda: 1 / 0, "boom", {"type": "object"})
    boom = registry.execute(ToolCall(tool_name="boom", args={}, call_id="c3"))
    assert boom.success is False and boom.error
    # 描述列表供 LLM function calling
    names = [tool["name"] for tool in registry.list_tools()]
    assert {"add", "boom"} <= set(names)


# 7.3 输入问题 → planner_node → 断言 sub_tasks 非空
def test_planner_node_creates_sub_tasks() -> None:
    state = _stub_state("What was NovaTech revenue in 2024?")
    out = planner_node(state)
    assert out["sub_tasks"], "sub_tasks 不应为空"
    assert out["sub_tasks"][0] == "What was NovaTech revenue in 2024?"
    assert out["next_step"] == "retriever"
    assert out["current_sub_task_index"] == 0
    # 含指标词 + 年份 → 追加细分任务；纯实体问题 → 仅原查询
    assert len(out["sub_tasks"]) == 2
    assert len(planner_node(_stub_state("Tell me about NovaTech"))["sub_tasks"]) == 1


# 7.4 verifier 路由：ACCEPT → synthesizer；REJECT → replanner
def test_verifier_routes_correctly() -> None:
    # ACCEPT：文档包含查询中的大写实体词
    accept_state = _stub_state(
        sub_tasks=[QUERY],
        retrieved_docs=[{"chunk_id": "d1", "content": "NovaTech revenue grew 5%.", "metadata": {"doc_id": "nova"}}],
    )
    accept = verifier_node(accept_state)
    assert accept["next_step"] == "synthesizer"
    assert accept["verify_flags"][QUERY]["passed"] is True

    # REJECT：文档不含实体词，且无既有 passed 标记
    reject_state = _stub_state(
        sub_tasks=[QUERY],
        retrieved_docs=[{"chunk_id": "d2", "content": "Helios Energy capex plan.", "metadata": {"doc_id": "other"}}],
    )
    reject = verifier_node(reject_state)
    assert reject["next_step"] == "replanner"
    assert reject["verify_flags"][QUERY]["passed"] is False


# 7.5 retry_count == max_retries → replanner_node → next_step == "end"
def test_replanner_retry_limit() -> None:
    state = _stub_state(sub_tasks=[QUERY], retry_count=3, max_retries=3)
    out = replanner_node(state)
    assert out["next_step"] == "end"
    # 未达上限：追加子任务、计数 +1、回检索
    state2 = _stub_state(sub_tasks=[QUERY], retry_count=0)
    out2 = replanner_node(state2)
    assert out2["next_step"] == "retriever"
    assert out2["retry_count"] == 1
    assert len(out2["sub_tasks"]) == 2


# 7.6 完整运行（检索 → 验证通过 → 合成）→ 最终状态正确
def test_agent_full_run_success() -> None:
    _register_stub_retrieve([
        [{"chunk_id": "d1", "content": "NovaTech revenue in 2024 was $12,400 million.", "metadata": {"doc_id": "nova-fy2024"}}],
    ])
    runner = AgentRunner(build_agent_graph())
    final = runner.run("What was NovaTech revenue in 2024?")
    assert final["next_step"] == "end"
    path = _node_path(final)
    assert "planner" in path and "synthesizer" in path
    assert "replanner" not in path
    # 多子任务回环：2 个子任务都要经过 retriever/verifier
    assert path.count("retriever") == 2 and path.count("verifier") == 2
    answers = [m for m in final["messages"] if m.get("role") == "assistant"]
    assert answers and "NovaTech" in answers[-1]["content"]


# 7.7 需要重规划的完整运行 → 路径包含 replanner 且最终成功
def test_agent_full_run_with_replan() -> None:
    _register_stub_retrieve([
        [{"chunk_id": "bad", "content": "Helios Energy capex plan for 2025.", "metadata": {"doc_id": "other"}}],
        [{"chunk_id": "good", "content": "NovaTech revenue was $10,800 million.", "metadata": {"doc_id": "nova"}}],
    ])
    runner = AgentRunner(build_agent_graph())
    final = runner.run(QUERY)
    path = _node_path(final)
    assert "replanner" in path, f"路径应包含 replanner: {path}"
    assert final["next_step"] == "end"
    assert final["retry_count"] == 1
    answers = [m for m in final["messages"] if m.get("role") == "assistant"]
    assert answers and "NovaTech" in answers[-1]["content"]


# 7.8 同一问题两次运行（第二次带 passed 上下文）→ 路径长度不同（动态性证明）
def test_agent_dynamic_path_length() -> None:
    _register_stub_retrieve([
        [{"chunk_id": "bad", "content": "Helios Energy capex plan.", "metadata": {"doc_id": "other"}}],
        [{"chunk_id": "good", "content": "NovaTech revenue grew 5%.", "metadata": {"doc_id": "nova"}}],
    ])
    runner = AgentRunner(build_agent_graph())
    # 第一次：无上下文 → 需重规划（子任务超限即降级路径除外）
    run1 = runner.run(QUERY)
    path1 = _node_path(run1)
    assert "replanner" in path1
    # 第二次：同一问题，携带上次成功的 verify_flags 上下文 → 跳过检索直接通过
    run2 = runner.run(
        QUERY,
        initial_state=_stub_state(verify_flags={QUERY: {"passed": True, "entity": "NovaTech"}}),
    )
    path2 = _node_path(run2)
    assert len(path2) < len(path1), f"预期动态路径 {path2} 短于 {path1}"
    assert "replanner" not in path2


# 7.9 AgentState 序列化往返一致
def test_agent_state_serializable() -> None:
    state = AgentState(
        messages=[{"role": "user", "content": QUERY}],
        retrieved_docs=[{"chunk_id": "d1", "content": "NovaTech data.", "metadata": {}}],
        verify_flags={QUERY: {"passed": True}},
        tool_call_history=[ToolCall(tool_name="retrieve", args={"query": QUERY}, call_id="call-1")],
        hooks=[{"node": "planner", "ts": "2026-08-29T00:00:00+00:00"}],
        retry_count=1,
        sub_tasks=[QUERY],
        current_sub_task_index=0,
        next_step="synthesizer",
    )
    dumped = state.model_dump_json()
    restored = AgentState.model_validate_json(dumped)
    assert restored.messages == state.messages
    assert restored.verify_flags == state.verify_flags
    assert restored.tool_call_history[0].tool_name == "retrieve"
    assert restored.retry_count == 1 and restored.sub_tasks == [QUERY]
    assert restored.next_step == "synthesizer"


# retriever_node 通过 stub 工具写入检索结果（辅助覆盖：节点经 Registry 调工具的路径）
def test_retriever_node_calls_tool() -> None:
    _register_stub_retrieve([
        [{"chunk_id": "d1", "content": "NovaTech revenue data.", "metadata": {"doc_id": "nova"}}],
    ])
    state = _stub_state(sub_tasks=[QUERY])
    out = retriever_node(state)
    assert out["next_step"] == "verifier"
    assert out["retrieved_docs"] and out["retrieved_docs"][0]["chunk_id"] == "d1"
    assert out["tool_call_history"] and out["tool_call_history"][0].tool_name == "retrieve"


# synthesizer_node 生成带引用的最终答案
def test_synthesizer_node_binds_citations() -> None:
    state = _stub_state(
        sub_tasks=[QUERY],
        retrieved_docs=[{"chunk_id": "d1", "content": "NovaTech revenue grew 5%.", "metadata": {"doc_id": "nova"}}],
        verify_flags={QUERY: {"passed": True, "entity": "NovaTech"}},
    )
    out = synthesizer_node(state)
    assert out["next_step"] == "end"
    assert out["messages"][0]["role"] == "assistant"
    assert "nova" in out["messages"][0]["content"]
