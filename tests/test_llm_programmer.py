"""Phase 10 — LLM 程序生成器测试（mock provider，锁定降级语义）。

诚实边界：无 API key 不跑真实 provider；LLM 路径以 mock complete 锁定
「合法输出可执行 / 非法输出降级 None」的契约，确定性默认路径零依赖。
"""

from __future__ import annotations

from verifin.core.graph import build_agent_graph
from verifin.tools.llm_programmer import LLMProgramGenerator, validate_dsl

_CANDIDATES = [
    {"id": "v1", "value": 8.1, "context": "row: leased facilities | col: total",
     "chunk_id": "c1"},
    {"id": "v2", "value": 56.1, "context": "row: total facilities | col: total",
     "chunk_id": "c1"},
    {"id": "t1", "value": [10.0, 20.0, 30.0], "context": "row: operating expenses",
     "chunk_id": "c2"},
]


class _MockComplete:
    """记录 prompt 的 mock provider。"""

    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


class _Boom:
    def __call__(self, prompt: str) -> str:
        raise RuntimeError("provider timeout")


# 1. validate_dsl：合法程序规范化返回；非法（幻觉引用/前向引用/未知 token）拒绝
def test_validate_dsl() -> None:
    ids = {"v1", "v2", "t1"}
    assert validate_dsl("divide(v1, v2)", ids) == "divide(v1, v2)"
    assert validate_dsl(
        "subtract(v2, v1), divide(#0, v2), multiply(#1, const_100)", ids
    )
    # 聚合算子引用值组绑定
    assert validate_dsl("table_sum(t1)", ids)
    assert validate_dsl("table_average(t1), divide(#0, v2)", ids)
    # 幻觉候选引用
    assert validate_dsl("divide(v9, v2)", ids) is None
    # 前向引用 / 越界
    assert validate_dsl("divide(#0, v1)", ids) is None
    assert validate_dsl("divide(#1, v1), add(#0, v2)", ids) is None
    # 未知 token（非数字 / const / #N / vN / tN）
    assert validate_dsl("divide(foo, v1)", ids) is None
    # 非法算子 / 非法结构
    assert validate_dsl("sin(v1, v2)", ids) is None
    assert validate_dsl("not a program", ids) is None
    # 数字与 const 字面量合法
    assert validate_dsl("subtract(5829, 5735), divide(#0, 5735)", ids)


# 2. 生成成功：JSON 输出 → program + 全量候选绑定
def test_generate_success() -> None:
    mock = _MockComplete('{"program": "divide(v1, v2)"}')
    gen = LLMProgramGenerator(complete=mock)
    assert gen.available
    out = gen.generate("what percentage of total facilities are leased?", _CANDIDATES)
    assert out is not None
    assert out["program"] == "divide(v1, v2)"
    assert out["bindings"] == {"v1": 8.1, "v2": 56.1, "t1": [10.0, 20.0, 30.0]}
    # prompt 含问题与候选上下文（行标签供 LLM 消歧）
    assert "what percentage of total facilities are leased?" in mock.prompts[0]
    assert "leased facilities" in mock.prompts[0]
    assert "v1 = 8.1" in mock.prompts[0]


# 3. 容错解析：markdown 代码块 / 前后杂文 / 多步程序
def test_generate_tolerant_parsing() -> None:
    fenced = _MockComplete(
        'Here is the program:\n```json\n{"program": "subtract(v2, v1), '
        'divide(#0, v2)"}\n```\nDone.'
    )
    out = LLMProgramGenerator(complete=fenced).generate("ratio?", _CANDIDATES)
    assert out is not None
    assert out["program"] == "subtract(v2, v1), divide(#0, v2)"
    # 值组候选渲染（LLM 可用 table_sum(tN)）
    agg = _MockComplete('{"program": "table_sum(t1)"}')
    out2 = LLMProgramGenerator(complete=agg).generate("sum?", _CANDIDATES)
    assert out2 is not None
    assert "t1 = [10, 20, 30]" in agg.prompts[0]


# 4. 降级：未注入 provider / 异常 / 非 JSON / 非法程序 / 空候选 → None
def test_generate_fallbacks() -> None:
    # 未注入 → 不可用
    assert LLMProgramGenerator().available is False
    assert LLMProgramGenerator().generate("q", _CANDIDATES) is None
    # provider 异常 → None（不抛出）
    assert LLMProgramGenerator(complete=_Boom()).generate("q", _CANDIDATES) is None
    # 非 JSON 输出
    assert LLMProgramGenerator(
        complete=_MockComplete("I cannot answer that.")
    ).generate("q", _CANDIDATES) is None
    # JSON 但无 program 字段
    assert LLMProgramGenerator(
        complete=_MockComplete('{"answer": 42}')
    ).generate("q", _CANDIDATES) is None
    # 程序引用幻觉候选 v9 → 校验拒绝
    assert LLMProgramGenerator(
        complete=_MockComplete('{"program": "divide(v9, v1)"}')
    ).generate("q", _CANDIDATES) is None
    # 前向引用
    assert LLMProgramGenerator(
        complete=_MockComplete('{"program": "divide(#1, v1)"}')
    ).generate("q", _CANDIDATES) is None
    # 空候选
    mock = _MockComplete('{"program": "divide(v1, v2)"}')
    assert LLMProgramGenerator(complete=mock).generate("q", []) is None
    assert mock.prompts == []  # 空候选不发起调用


# 5. 图注入冒烟：llm_programmer 注入后图可编译（默认路径不受影响）
def test_graph_compiles_with_llm_programmer() -> None:
    gen = LLMProgramGenerator(complete=_MockComplete('{"program": "add(1, 2)"}'))
    graph = build_agent_graph(llm_programmer=gen)
    assert graph is not None
    # 默认（不注入）路径保持可用
    assert build_agent_graph() is not None
