"""Phase 6 — 对话状态端到端集成测试（多轮继承 / 偏好确认 / 显式覆盖 / 联合场景）。"""

from __future__ import annotations

import pytest

from verifin.core.graph import build_agent_graph
from verifin.core.runner import AgentRunner
from verifin.core.state import AgentState
from verifin.tools.registry import ToolRegistry, register_builtin_tools


@pytest.fixture(autouse=True)
def _clean_registry():
    ToolRegistry().reset()
    yield
    ToolRegistry().reset()


def _runner(responses: list[list[dict]], thread_id: str | None = None) -> AgentRunner:
    def fake(query: str, top_k: int = 10) -> list[dict]:
        index = min(calls[0], len(responses) - 1)
        calls[0] += 1
        return responses[index]

    calls = [0]
    register_builtin_tools()
    ToolRegistry().register("retrieve", fake, "stub retrieve", {"type": "object"})
    return AgentRunner(build_agent_graph(), thread_id=thread_id)


def _chunk(content: str, doc_id: str = "nova") -> dict:
    return {"chunk_id": doc_id, "content": content, "metadata": {"doc_id": doc_id}}


# 端到端 1：连续 3 轮对话，实体自动继承（后续轮次无需重复公司名）
def test_three_turn_entity_inheritance() -> None:
    runner = _runner([
        [_chunk("NovaTech revenue in 2024 was $12,400 million.", "d1")],
        [_chunk("NovaTech operating margin in 2024 was 15%.", "d2")],
        [_chunk("NovaTech gross margin in 2024 was 40%.", "d3")],
    ], thread_id="multi-turn-1")

    runner.run("What was NovaTech revenue?")
    runner.run("What about its operating margin?")
    final = runner.run("And its gross margin?")

    dialog = final["dialog_state"]
    assert dialog["entity_preference"] == "NovaTech"
    assert dialog["turn_count"] == 3
    assert final["next_step"] == "end"
    answers = [m for m in final["messages"] if m.get("role") == "assistant"]
    assert answers and "gross margin" in answers[-1]["content"]


# 端到端 2：Synthesizer 输出附带偏好确认（年报口径）
def test_preference_confirmation_in_answer() -> None:
    runner = _runner([
        [_chunk("NovaTech revenue in 2024 was $12,400 million.", "d1")],
    ])
    final = runner.run(
        "What was NovaTech revenue in 2024?",
        initial_state=AgentState(
            messages=[{"role": "user", "content": "What was NovaTech revenue in 2024?"}],
            dialog_state={
                "source_preference": "annual_report",
                "preference_source": {"source": "explicit"},
            },
        ),
    )
    answer = [m for m in final["messages"] if m.get("role") == "assistant"][-1]["content"]
    assert "年报" in answer, answer
    assert "NovaTech" in answer


# 端到端 3：轮内显式指令覆盖旧偏好（新闻稿 → 只看年报）
def test_explicit_preference_override_in_turn() -> None:
    runner = _runner([
        [_chunk("NovaTech revenue in 2024 was $12,400 million.", "d1")],
    ])
    final = runner.run(
        "只看年报，What was NovaTech revenue in 2024?",
        initial_state=AgentState(
            messages=[{"role": "user", "content": "只看年报，What was NovaTech revenue in 2024?"}],
            dialog_state={
                "source_preference": "news",
                "preference_source": {"source": "explicit"},
            },
        ),
    )
    dialog = final["dialog_state"]
    assert dialog["source_preference"] == "annual_report"
    assert dialog["preference_source"]["source"] == "explicit"
    hook = final["hooks"][0]
    assert "preference_applied" in hook


# 端到端 4：偏好 + 表格证据 + 冲突仲裁联合（表格取数 → major 冲突 → 年报偏好加权说明）
def test_preference_table_arbitration_combined() -> None:
    table_chunk = {
        "chunk_id": "tbl",
        "content": "| Metric | 2023 | 2024 |\n|---|---|---|\n| Revenue | 11,800 | 12,400 |\n",
        "metadata": {"doc_id": "10k", "source_type": "filing"},
    }
    news_chunk = {
        "chunk_id": "news",
        "content": "Consolidated revenue in 2024 was $11,000 million.",
        "metadata": {"doc_id": "press", "source_type": "news"},
    }
    runner = _runner([[table_chunk, news_chunk]])
    final = runner.run(
        "What was revenue in 2024?",
        initial_state=AgentState(
            messages=[{"role": "user", "content": "What was revenue in 2024?"}],
            dialog_state={
                "source_preference": "annual_report",
                "preference_source": {"source": "explicit"},
            },
        ),
    )
    flags = [flag for flag in final["verify_flags"].values()
             if isinstance(flag, dict) and (flag.get("conflict") or {}).get("level") == "major"]
    assert flags, "应触发 major 冲突"
    conflict = flags[0]["conflict"]
    assert "年报" in conflict.get("preference_note", "")
    # 表格来源的证据数值应正确取出 12,400
    values = {result["chunk_id"]: result["value"] for result in flags[0]["results"]}
    assert values.get("tbl") == 12400.0
    assert "replanner" in [hook["node"] for hook in final["hooks"]]