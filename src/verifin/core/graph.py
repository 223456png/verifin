"""LangGraph 图编译：ReAct 状态机 + 条件路由。

路由设计（Phase 3 设计 Decisions #3）：
- verifier 三向条件路由：synthesizer / replanner / retriever（多子任务回环）；
- replanner 双向条件路由：retriever / END（重试耗尽降级，防死循环）。
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from verifin.core.nodes import (
    make_calculator_node,
    make_planner_node,
    replanner_node,
    retriever_node,
    synthesizer_node,
    verifier_node,
)
from verifin.core.state import AgentState


def _load_saver():
    """按实测版本回退导入内存 checkpoint（1.x 为 InMemorySaver）。"""
    try:
        from langgraph.checkpoint.memory import MemorySaver

        return MemorySaver
    except Exception:  # pragma: no cover - 版本回退
        from langgraph.checkpoint.memory import InMemorySaver

        return InMemorySaver


def route_after_verifier(state: Any) -> str:
    """verifier 之后：synthesizer / calculator / replanner / retriever（回环），未知信号走 END。"""
    next_step = state["next_step"] if isinstance(state, dict) else state.next_step
    if next_step == "synthesizer":
        return "synthesizer"
    if next_step == "calculator":
        return "calculator"
    if next_step == "replanner":
        return "replanner"
    if next_step == "retriever":
        return "retriever"
    return "end"


def route_after_replanner(state: Any) -> str:
    """replanner 之后：retriever（继续重试）或 END（上限耗尽降级）。"""
    next_step = state["next_step"] if isinstance(state, dict) else state.next_step
    return "retriever" if next_step == "retriever" else "end"


def build_agent_graph(llm_planner=None, llm_programmer=None):
    """构建并编译 ReAct Agent 图（带内存 checkpoint）。

    Args:
        llm_planner: 可插拔 LLM 规划器（见 ``verifin.core.llm_planner.LLMPlanner``）；
            None 时 planner 走确定性规则（默认，零外部依赖）。
        llm_programmer: 可插拔 LLM 程序生成器（Phase 10，见
            ``verifin.tools.llm_programmer.LLMProgramGenerator``）；None 或不可用时
            calculator 仅走确定性模板（与 Phase 9 行为一致）。
    """
    builder = StateGraph(AgentState)

    builder.add_node("planner", make_planner_node(llm_planner, llm_programmer))
    builder.add_node("retriever", retriever_node)
    builder.add_node("verifier", verifier_node)
    builder.add_node("replanner", replanner_node)
    builder.add_node("calculator", make_calculator_node(llm_programmer))
    builder.add_node("synthesizer", synthesizer_node)

    builder.set_entry_point("planner")
    builder.add_edge("planner", "retriever")
    builder.add_edge("retriever", "verifier")
    builder.add_edge("calculator", "synthesizer")
    builder.add_edge("synthesizer", END)
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "synthesizer": "synthesizer",
            "calculator": "calculator",
            "replanner": "replanner",
            "retriever": "retriever",
            "end": END,
        },
    )
    builder.add_conditional_edges(
        "replanner",
        route_after_replanner,
        {"retriever": "retriever", "end": END},
    )

    return builder.compile(checkpointer=_load_saver()())
