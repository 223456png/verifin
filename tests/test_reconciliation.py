"""Phase 5 — 多文档冲突检测与仲裁测试（resolve_conflicts）。"""

from __future__ import annotations

import pytest

from verifin.core.graph import build_agent_graph
from verifin.core.runner import AgentRunner
from verifin.tools.registry import ToolRegistry, register_builtin_tools
from verifin.tools.verifier import resolve_conflicts, verify_claim_batch

QUERY = "What was NovaTech revenue in 2024?"
CLAIM = {"entity": "NovaTech", "period": "2024", "metric": "revenue"}


def _item(chunk_id: str, value: float, unit: str, period: str = "2024",
          source_type: str = "filing") -> dict:
    return {"chunk_id": chunk_id, "value": value, "unit": unit,
            "period": period, "source_type": source_type}


def _chunk(content: str, doc_id: str, source_type: str) -> dict:
    return {"chunk_id": doc_id, "content": content,
            "metadata": {"doc_id": doc_id, "source_type": source_type}}


# 冲突仲裁：同指标不同值，差异 ≤2% → 四舍五入误差，取均值
def test_conflict_rounding_diff_mean() -> None:
    result = resolve_conflicts([
        _item("a", 12400.0, "million"),
        _item("b", 12410.0, "million"),
    ])
    assert result["level"] == "none"
    assert result["conflict"] is False
    assert result["reconciled_value"] == pytest.approx(12405.0)
    assert result["reconciled_unit"] == "million"


# 差异 >5% → major 冲突，不伪造数值（reconciled None）
def test_conflict_major_no_fabrication() -> None:
    result = resolve_conflicts([
        _item("a", 12400.0, "million"),
        _item("b", 11000.0, "million"),
    ])
    assert result["level"] == "major"
    assert result["conflict"] is True
    assert result["reconciled_value"] is None
    assert "consolidated" in result["note"].lower()


# 时间 + 来源可信度加权：最新财报权重更高
def test_conflict_weighted_by_time_and_trust() -> None:
    result = resolve_conflicts([
        _item("newer", 12400.0, "million", period="FY2025", source_type="filing"),
        _item("older", 12380.0, "million", period="FY2023", source_type="news"),
    ])
    assert result["level"] == "none"
    # 加权均值应偏向更新的财报数值（> 算术均值 12390 且 < 12400）
    assert result["reconciled_value"] is not None
    assert 12390.0 < result["reconciled_value"] < 12400.0


# 单位换算：$12,400 million ≡ $12.4 billion → 归一后一致
def test_conflict_unit_normalization() -> None:
    result = resolve_conflicts([
        _item("a", 12400.0, "million"),
        _item("b", 12.4, "billion"),
    ])
    assert result["level"] == "none"
    assert result["reconciled_value"] == pytest.approx(12400.0)
    assert result["reconciled_unit"] == "million"


# 批量校验集成：两条都通过四要素、数值冲突 major → decision REJECT（触发 Replan）
def test_verify_batch_conflict_major_reject() -> None:
    batch = verify_claim_batch(CLAIM, [
        _chunk("NovaTech revenue in 2024 was $12,400 million.", "nova-filing", "filing"),
        _chunk("NovaTech revenue in 2024 was $11,000 million.", "nova-news", "news"),
    ])
    assert all(result["passed"] for result in batch["results"])
    assert batch["conflict"]["level"] == "major"
    assert batch["decision"] == "REJECT"


@pytest.fixture(autouse=True)
def _clean_registry():
    ToolRegistry().reset()
    yield
    ToolRegistry().reset()


# Agent 集成：双文档大差异冲突 → replan 且重规划子任务要求合并报表口径
def test_agent_conflict_replan_integration() -> None:
    register_builtin_tools()
    ToolRegistry().register(
        "retrieve",
        lambda query, top_k=10: [
            _chunk("NovaTech revenue in 2024 was $12,400 million.", "nova-filing", "filing"),
            _chunk("NovaTech revenue in 2024 was $11,000 million.", "nova-news", "news"),
        ],
        "stub retrieve",
        {"type": "object"},
    )
    runner = AgentRunner(build_agent_graph())
    final = runner.run(QUERY)
    path = [hook["node"] for hook in final["hooks"]]
    assert "replanner" in path, f"大差异冲突应触发重规划: {path}"
    assert any("consolidated" in task for task in final["sub_tasks"])
    assert any(
        (flag.get("conflict") or {}).get("level") == "major"
        for flag in final["verify_flags"].values()
        if isinstance(flag, dict)
    )