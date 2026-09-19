"""Phase 9.5 — 可插拔 LLM 规划器（core/llm_planner.py + make_planner_node）测试。

设计要点（LLM 规划层，可降级）：
- ``LLMPlanner`` 通过 ``complete(prompt) -> str`` callable 注入，不绑定具体 SDK；
- 未注入 / 调用异常 / 输出非法时 ``plan`` 返回 ``None``，触发规则 planner 降级；
- ``make_planner_node(None)`` 与规则 ``planner_node`` 行为完全一致（零外部依赖兜底）；
- LLM 输出合法时：sub_tasks 采用 LLM 分解，claims 规则优先 + LLM 补充，
  calculation_spec 规则优先、LLM 仅补充规则未检测到的计算。
"""

from __future__ import annotations

from verifin.core.llm_planner import LLMPlanner
from verifin.core.nodes import make_planner_node, planner_node
from verifin.core.state import AgentState


def _state(query: str) -> AgentState:
    return AgentState(messages=[{"role": "user", "content": query}])


def _fake_complete(raw: str):
    def _complete(_prompt: str) -> str:
        return raw

    return _complete


class _StubLLMPlanner:
    """最小可插拔 planner（仅需要 available + plan）。"""

    def __init__(self, plan_result, *, available: bool = True):
        self.available = available
        self._result = plan_result
        self.calls = 0

    def plan(self, query: str, history):
        self.calls += 1
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


# 1. available：未注入 complete → False；注入 callable → True
def test_llm_planner_available() -> None:
    assert LLMPlanner().available is False
    assert LLMPlanner(lambda _p: "{}").available is True


# 2. 未注入 provider → plan 返回 None（触发降级）
def test_llm_planner_unavailable_returns_none() -> None:
    assert LLMPlanner().plan("q", []) is None


# 3. provider 抛异常 → plan 返回 None（不向上抛）
def test_llm_planner_provider_exception_returns_none() -> None:
    def boom(_p: str) -> str:
        raise RuntimeError("provider down")

    assert LLMPlanner(boom).plan("q", []) is None


# 4. 输出解析：容忍 markdown json 围栏 / 纯文本夹 JSON / 非法
def test_llm_planner_parse() -> None:
    fenced = '```json\n{"sub_tasks": ["a", "b"]}\n```'
    assert LLMPlanner._parse(fenced) == {"sub_tasks": ["a", "b"]}

    wrapped = 'Here is the plan:\n{"sub_tasks": ["a"]}\nthanks'
    assert LLMPlanner._parse(wrapped) == {"sub_tasks": ["a"]}

    assert LLMPlanner._parse("no json here") is None


# 5. 校验归一：空子任务被剔除、非 dict 的 calc_spec 归 None、claims 兜底 dict
def test_llm_planner_validate() -> None:
    out = LLMPlanner._validate(
        {
            "sub_tasks": ["  a ", "", "b"],
            "claims": {"a": {"entity": "NovaTech"}},
            "calculation_spec": "oops",
        }
    )
    assert out["sub_tasks"] == ["a", "b"]
    assert out["claims"] == {"a": {"entity": "NovaTech"}}
    assert out["calculation_spec"] is None

    # 非 dict / 空 sub_tasks → None
    assert LLMPlanner._validate("not a dict") is None
    assert LLMPlanner._validate({"sub_tasks": []}) is None
    assert LLMPlanner._validate({"sub_tasks": ["  "]}) is None


# 6. plan 端到端：合法 JSON → 结构化计划
def test_llm_planner_plan_success() -> None:
    planner = LLMPlanner(_fake_complete(
        '{"sub_tasks": ["retrieve NovaTech 2024 revenue"],'
        ' "claims": {"retrieve NovaTech 2024 revenue": {"entity": "NovaTech"}},'
        ' "calculation_spec": null}'
    ))
    plan = planner.plan("What was NovaTech revenue in 2024?", [])
    assert plan is not None
    assert plan["sub_tasks"] == ["retrieve NovaTech 2024 revenue"]
    assert plan["calculation_spec"] is None


# 7. make_planner_node(None)：与规则 planner_node 结果一致
def test_make_planner_node_fallback_without_llm() -> None:
    node = make_planner_node(None)
    query = "What was NovaTech revenue in 2024?"
    assert node(_state(query))["sub_tasks"] == planner_node(_state(query))["sub_tasks"]


# 8. 不可用 LLM（available=False）→ 走规则 planner
def test_make_planner_node_unavailable_llm_falls_back() -> None:
    node = make_planner_node(_StubLLMPlanner({"sub_tasks": ["x"]}, available=False))
    out = node(_state("What was NovaTech revenue in 2024?"))
    assert out["sub_tasks"] == planner_node(_state("What was NovaTech revenue in 2024?"))["sub_tasks"]


# 9. LLM 异常 → 降级规则 planner
def test_make_planner_node_llm_exception_falls_back() -> None:
    node = make_planner_node(_StubLLMPlanner(RuntimeError("boom")))
    out = node(_state("What was NovaTech revenue in 2024?"))
    assert out["sub_tasks"] == planner_node(_state("What was NovaTech revenue in 2024?"))["sub_tasks"]


# 10. LLM 合法输出：sub_tasks 采用 LLM 分解，claims 规则优先 + LLM 补充
def test_make_planner_node_llm_merges_subtasks_and_claims() -> None:
    stub = _StubLLMPlanner(
        {
            "sub_tasks": ["retrieve NovaTech 2024 revenue figures"],
            "claims": {
                "retrieve NovaTech 2024 revenue figures": {
                    "entity": "NovaTech", "period": "2024", "metric": "revenue", "definition": "",
                }
            },
            "calculation_spec": None,
        }
    )
    node = make_planner_node(stub)
    out = node(_state("What was NovaTech revenue in 2024?"))

    assert stub.calls == 1
    assert out["sub_tasks"] == ["retrieve NovaTech 2024 revenue figures"]
    # 规则 query claim 保留，LLM 新子任务 claim 补充进来
    original = "What was NovaTech revenue in 2024?"
    assert original in out["claims"]
    assert out["claims"]["retrieve NovaTech 2024 revenue figures"]["metric"] == "revenue"


# 11. LLM 补充规则未检测到的计算：仅在规则无 calc 时采用 LLM calc_spec
def test_make_planner_node_llm_supplements_calculation() -> None:
    llm_spec = {
        "kind": "growth_pct",
        "base_period": "2023",
        "target_period": "2024",
        "metric": "revenue",
        "entity": "NovaTech",
        "reversed": False,
    }
    stub = _StubLLMPlanner(
        {
            "sub_tasks": ["retrieve NovaTech revenue"],
            "claims": {},
            "calculation_spec": llm_spec,
        }
    )
    node = make_planner_node(stub)
    # 无计算关键词的问答 → 规则不检测 calc，LLM 补充
    out = node(_state("How has NovaTech performed over the past two years?"))
    assert out["calculation_requested"] is True
    assert out["calculation_spec"]["kind"] == "growth_pct"
