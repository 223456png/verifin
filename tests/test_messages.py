"""Phase 5 — 人性化诊断消息测试（ui/messages.py）。"""

from __future__ import annotations

from verifin.core.nodes import synthesizer_node
from verifin.core.state import AgentState
from verifin.ui.messages import friendly_message, metric_label


def _assert_no_technical_jargon(text: str) -> None:
    lowered = text.lower()
    for token in ("expected", "got '", "mismatch", "traceback", "exception", "error:"):
        assert token not in lowered, f"诊断信息不应含技术黑话: {token!r} in {text!r}"


# 指标错配 → 纯中文业务解释（含指标中文名）
def test_friendly_metric_mismatch() -> None:
    message = friendly_message("MISMATCH_METRIC", expected="revenue", got="gross_margin")
    assert "营收" in message and "毛利" in message
    assert message.endswith("重新为您检索更精确的数据。")
    _assert_no_technical_jargon(message)


# NO_EVIDENCE 与 PoT 拦截原因中文化
def test_no_evidence_and_calc_error() -> None:
    none_message = friendly_message("NO_EVIDENCE")
    assert "未检索到相关数据" in none_message
    _assert_no_technical_jargon(none_message)

    calc_message = friendly_message(
        "EVAL_REJECT", reason="expression too long (max 400 chars)"
    )
    assert "表达式" in calc_message and "过长" in calc_message
    _assert_no_technical_jargon(calc_message)

    assert metric_label("operating_margin") == "营业利润率"
    assert metric_label("EBITDA") == "EBITDA"


# synthesizer 失败路径：终态消息全中文、无 Traceback
def test_synthesizer_friendly_failure() -> None:
    state = AgentState(
        messages=[{"role": "user", "content": "What was NovaTech revenue?"}],
        sub_tasks=["What was NovaTech revenue?"],
        verify_flags={
            "What was NovaTech revenue?": {
                "passed": False,
                "decision": "REJECT",
                "results": [
                    {"chunk_id": "d1", "passed": False,
                     "mismatches": ["metric: expected 'revenue', got 'gross_margin'"]}
                ],
            }
        },
    )
    out = synthesizer_node(state)
    content = out["messages"][0]["content"]
    assert "营收" in content
    assert "Based on verified evidence" not in content
    _assert_no_technical_jargon(content)