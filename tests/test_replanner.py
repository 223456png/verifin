"""Phase 5 — 带记忆的 Replanner 测试（tools/replanner.py + tools/failure_memory.py）。

设计要点（修正版设计 §1.1 带记忆 vs 无记忆）：
- FailureMemoryStore：增删查/容量淘汰/去重/摘要统计；
- Replanner：verify_flags → 失败诊断 → 确定性策略选择 → 携带完整四要素的子任务生成；
- 退出条件：重试上限 / 与上一条失败同类（无新信息）提前退出。
"""

from __future__ import annotations

from verifin.core.nodes import replanner_node
from verifin.core.state import AgentState
from verifin.prompts.replanner_prompts import STRATEGY_DESCRIPTIONS
from verifin.tools.failure_memory import FailureMemory, FailureMemoryStore
from verifin.tools.replanner import Replanner


def _flags(
    mismatches: list[str] | None = None,
    missing: list[str] | None = None,
    results: list[dict] | None = None,
    conflict: dict | None = None,
) -> dict:
    """构造 verify_flags[sub_task] 形态（Phase 4 契约）。"""
    if results is None:
        results = [
            {
                "chunk_id": "c1",
                "passed": False,
                "mismatches": mismatches or [],
                "missing": missing or [],
            }
        ]
    flags: dict = {
        "claim": {"entity": "NovaTech", "period": "2024", "metric": "revenue"},
        "results": results,
        "decision": "REJECT",
        "passed": False,
    }
    if conflict:
        flags["conflict"] = conflict
    return flags


def _memory(
    sub_task: str = "q",
    mismatches: list[str] | None = None,
    missing: list[str] | None = None,
    verify_result: dict | None = None,
    turn: int = 1,
) -> FailureMemory:
    return FailureMemory(
        turn=turn,
        sub_task=sub_task,
        claim={"entity": "NovaTech", "period": "2024", "metric": "revenue"},
        verify_result=verify_result or _flags(mismatches, missing),
        mismatches=mismatches or [],
        missing=missing or [],
    )


# 1. FailureMemoryStore：容量淘汰 / 最近条目 / 子任务去重 / 失败 chunk / 摘要
def test_failure_memory_store() -> None:
    store = FailureMemoryStore(max_entries=3)
    store.add(_memory(
        sub_task="q1",
        mismatches=["metric: expected 'revenue', got 'cost'"],
        verify_result=_flags(results=[
            {"chunk_id": "c1", "passed": False, "mismatches": ["metric: expected 'revenue', got 'cost'"], "missing": []},
            {"chunk_id": "c2", "passed": True, "mismatches": [], "missing": []},
        ]),
    ))
    store.add(_memory(sub_task="q2", missing=["period"]))
    store.add(_memory(sub_task="q3", mismatches=["metric: expected 'revenue', got 'sales'"]))
    assert len(store.entries) == 3
    assert store.get_recent(1)[0].sub_task == "q3"
    assert store.get_failed_sub_tasks() == ["q1", "q2", "q3"]
    # 失败 chunk：只含 passed=False（c2 通过不排除）
    assert store.get_failed_chunk_ids() == ["c1"]
    # 摘要：按要素类别计数（最近 5 条内 metric 2 次 / period 缺失 1 次）
    summary = store.get_diagnosis_summary()
    assert summary["mismatches"].get("metric") == 2
    assert summary["missing"].get("period") == 1
    # 子任务去重 + 容量淘汰最旧（q1 出队）
    store.add(_memory(sub_task="q2", missing=["period"]))
    assert store.get_failed_sub_tasks() == ["q2", "q3"]
    assert [e.sub_task for e in store.entries] == ["q2", "q3", "q2"]
    # 序列化往返
    restored = FailureMemoryStore.from_history(store.to_history())
    assert [e.sub_task for e in restored.entries] == [e.sub_task for e in store.entries]


# 2. 失败诊断：verify_flags.results → mismatches/missing 聚合去重
def test_failure_diagnosis() -> None:
    flags = _flags(results=[
        {"chunk_id": "a", "passed": False,
         "mismatches": ["metric: expected 'revenue', got 'cost'"], "missing": []},
        {"chunk_id": "b", "passed": False,
         "mismatches": ["metric: expected 'revenue', got 'cost'", "period: expected '2024', got '2023'"],
         "missing": ["period"]},
    ])
    replanner = Replanner()
    failure = replanner._diagnose_failure("task", flags, {"entity": "NovaTech"}, retry_count=0)
    assert failure.mismatches == [
        "metric: expected 'revenue', got 'cost'",
        "period: expected '2024', got '2023'",
    ]
    assert failure.missing == ["period"]
    assert failure.turn == 1
    assert failure.sub_task == "task"
    assert failure.mismatch_pillars() == {"metric", "period"}


# 3. 指标高频失败（记忆命中 ≥2 次）→ metric_refinement
def test_strategy_selection_metric() -> None:
    store = FailureMemoryStore()
    store.add(_memory(sub_task="q1", mismatches=["metric: expected 'revenue', got 'cost'"]))
    replanner = Replanner(failure_store=store)
    plan = replanner.replan(
        "q2", _flags(mismatches=["metric: expected 'revenue', got 'sales'"]), retry_count=1
    )
    assert plan["strategy"] == "metric_refinement"
    assert plan["continue"] is True


# 4. 期间失败 → period_refinement
def test_strategy_selection_period() -> None:
    replanner = Replanner()
    plan = replanner.replan(
        "q1", _flags(mismatches=["period: expected '2024', got '2023'"]), retry_count=0
    )
    assert plan["strategy"] == "period_refinement"


# 5. 已尝试 ≥3 个方向仍失败 → alternative_phrasing
def test_strategy_selection_alternative() -> None:
    store = FailureMemoryStore()
    store.add(_memory(sub_task="q1", mismatches=["metric: expected 'revenue', got 'cost'"]))
    store.add(_memory(sub_task="q2", mismatches=["period: expected '2024', got '2023'"]))
    store.add(_memory(sub_task="q3", mismatches=["entity: expected 'NovaTech', got 'Helios']"]))
    replanner = Replanner(failure_store=store)
    # 当前失败无要素 mismatch（如纯检索空结果）→ 与上一条不同 → 继续，方向已 3 个 → 换表述
    plan = replanner.replan("q4", _flags(results=[]), retry_count=1)
    assert plan["strategy"] == "alternative_phrasing"


# 6. 生成子任务携带完整四要素（Verifier 可直接校验）
def test_sub_task_carries_full_claim() -> None:
    replanner = Replanner()
    plan = replanner.replan(
        "What was NovaTech revenue in 2024?",
        _flags(mismatches=["period: expected '2024', got '2023'"]),
        retry_count=0,
    )
    task = plan["new_sub_task"]
    assert task is not None
    assert "NovaTech" in task
    assert "revenue" in task
    assert "2024" in task
    # 策略描述可追溯（轨迹分析用）
    assert plan["strategy"] in STRATEGY_DESCRIPTIONS


# 7. 退出条件：重试上限 → continue=False
def test_exit_condition_retry_limit() -> None:
    replanner = Replanner()
    plan = replanner.replan("q", _flags(mismatches=["metric: expected 'revenue', got 'cost'"]),
                            retry_count=3, max_retries=3)
    assert plan["continue"] is False
    assert plan["new_sub_task"] is None
    assert "retry limit" in plan["reason"]


# 8. 退出条件：与上一条失败同类（无新信息）→ 提前退出不重复尝试
def test_exit_condition_duplicate() -> None:
    mismatches = ["metric: expected 'revenue', got 'cost'"]
    store = FailureMemoryStore()
    store.add(_memory(sub_task="q", mismatches=mismatches))
    replanner = Replanner(failure_store=store)
    plan = replanner.replan("q", _flags(mismatches=mismatches), retry_count=1)
    assert plan["continue"] is False
    assert "duplicate" in plan["reason"]


# 9. 节点级集成：replanner_node 持久化 failure_history + exclude_chunk_ids
def test_replanner_node_persists_memory() -> None:
    state = AgentState(
        messages=[{"role": "user", "content": "What was NovaTech revenue?"}],
        sub_tasks=["What was NovaTech revenue?"],
        verify_flags={
            "What was NovaTech revenue?": _flags(results=[
                {"chunk_id": "bad", "passed": False,
                 "mismatches": ["entity: expected 'NovaTech', got 'Helios'"], "missing": ["metric"]},
            ]),
        },
    )
    out = replanner_node(state)
    assert out["next_step"] == "retriever"
    assert out["retry_count"] == 1
    # 失败记忆写入 state（含诊断与时间戳）
    history = out["failure_history"]
    assert len(history) == 1
    assert history[0]["sub_task"] == "What was NovaTech revenue?"
    assert history[0]["missing"] == ["metric"]
    # 失败 chunk 进入排除列表
    assert out["exclude_chunk_ids"] == ["bad"]
    # 新子任务携带完整要素（缺 metric → supplement_retrieval 补计算口径）
    assert out["sub_tasks"][-1] == "retrieve NovaTech revenue 2024 calculation method"
    # 策略记录进 hook（可观测）
    assert out["hooks"][-1]["replan_strategy"] == "supplement_retrieval"


# 10. 大差异冲突的 consolidation 重规划：低可信来源冲突项进入排除列表
#    （记忆化收敛到权威口径，Phase 7 消融中 Replanner 挽救冲突样本依赖此行为）
def test_consolidation_excludes_low_trust_conflict_sources() -> None:
    flags = _flags(
        results=[
            {"chunk_id": "filing", "passed": True, "mismatches": [], "missing": []},
            {"chunk_id": "news", "passed": True, "mismatches": [], "missing": []},
        ],
        conflict={"level": "major", "values": [
            {"chunk_id": "filing", "value": 12000.0, "source_type": "filing"},
            {"chunk_id": "news", "value": 11000.0, "source_type": "news"},
        ]},
    )
    replanner = Replanner()
    plan = replanner.replan("q", flags, retry_count=0)
    assert plan["strategy"] == "consolidation_refinement"
    assert plan["continue"] is True
    # 仅排除低可信来源（新闻稿）；年报口径的冲突项保留供重检索
    assert plan["exclude_chunk_ids"] == ["news"]
    assert "consolidated" in (plan["new_sub_task"] or "")
