"""Phase 9 — FinQA program DSL 模板执行器测试（detect_program / ProgramExecutor）。"""

from __future__ import annotations

import pytest

from verifin.tools.program_executor import (
    ProgramExecutor,
    ProgramSpec,
    ValueCandidate,
    detect_program,
)
from verifin.tools.evidence import EvidenceExtractor


# 1. growth 模板：双年份 + 计算关键词（含 compare-to 反序）
def test_detect_growth_program() -> None:
    spec = detect_program("What was the percentage growth in revenue from 2023 to 2024?")
    assert spec is not None
    assert spec.kind == "growth_pct"
    assert spec.base_period == "2023"
    assert spec.target_period == "2024"
    # 反序表达："X in 2024 compare to 2023" → 基期 2023 在后
    reverse = detect_program("What is the percentage change in cash flow hedges in 2011 compare to the 2010?")
    assert reverse is not None
    assert reverse.base_period == "2010"
    assert reverse.target_period == "2011"


# 2. difference 模板：单年份 + 净变动关键词 → 基期推断为上一年
def test_detect_difference_program() -> None:
    spec = detect_program("what is the net change in net revenue during 2015 for entergy corporation?")
    assert spec is not None
    assert spec.kind == "difference"
    assert spec.base_period == "2014"
    assert spec.target_period == "2015"


# 3. 防误伤：无关键词 / 无年份 / growth 无双年份 → None
def test_detect_program_negative() -> None:
    assert detect_program("What was NovaTech revenue in 2024?") is None
    assert detect_program("What was NovaTech growth in 2024?") is None
    assert detect_program("Tell me about the company.") is None


# 4. 单位归一：billion vs million 组合可执行（12B - 10,000M → +20%）
def test_execute_unit_normalization() -> None:
    spec = ProgramSpec(kind="growth_pct", base_period="2023", target_period="2024")
    result = ProgramExecutor().execute(spec, {
        "2023": [ValueCandidate(value=10000.0, unit="million", period="2023")],
        "2024": [ValueCandidate(value=12.0, unit="billion", period="2024")],
    })
    assert result.value == pytest.approx(20.0)
    assert result.fraction == pytest.approx(0.2)
    assert result.difference == pytest.approx(2000.0)


# 5. 合理性剪枝：量纲冲突（% vs million）与荒谬增速组合被拒
def test_execute_pruning() -> None:
    spec = ProgramSpec(kind="growth_pct", base_period="2023", target_period="2024")
    # % 与 million 跨量纲 → 组合拒绝 → 无可用组合
    mixed = ProgramExecutor().execute(spec, {
        "2023": [ValueCandidate(value=10000.0, unit="million", period="2023")],
        "2024": [ValueCandidate(value=42.0, unit="%", period="2024")],
    })
    assert mixed.value is None
    assert "plausible" in (mixed.error or "")
    # 增长率超 2000% 的锚定错误组合被剪枝
    absurd = ProgramExecutor().execute(spec, {
        "2023": [ValueCandidate(value=1.0, unit=None, period="2023")],
        "2024": [ValueCandidate(value=90000.0, unit=None, period="2024")],
    })
    assert absurd.value is None


# 6. 多候选枚举：primary 取锚定分最优组合，alternates ≤2
def test_execute_alternates() -> None:
    spec = ProgramSpec(kind="growth_pct", base_period="2023", target_period="2024")
    result = ProgramExecutor().execute(spec, {
        "2023": [
            ValueCandidate(value=10000.0, unit="million", period="2023", anchor_score=0.0),
            ValueCandidate(value=9000.0, unit="million", period="2023", anchor_score=30.0),
        ],
        "2024": [
            ValueCandidate(value=12000.0, unit="million", period="2024", anchor_score=0.0),
        ],
    })
    assert result.value == pytest.approx(20.0)  # primary：锚定分最优 (10000, 12000)
    assert len(result.alternates) == 1
    assert result.alternates[0]["fraction"] == pytest.approx(0.333333, rel=1e-4)


# 7. 缺失任一期候选 → error 降级（不抛出）
def test_execute_missing_candidates() -> None:
    spec = ProgramSpec(kind="difference", base_period="2014", target_period="2015")
    result = ProgramExecutor().execute(spec, {
        "2015": [ValueCandidate(value=5829.0, unit="million", period="2015")],
    })
    assert result.value is None
    assert "missing" in (result.error or "")


# 8. 年份锚定多候选收集：按到年份的字符距离升序，窗口外丢弃
def test_collect_year_values() -> None:
    content = (
        "the 2014 net revenue of amount ( in millions ) is $ 5735 ; "
        "the 2015 net revenue of amount ( in millions ) is $ 5829 ; "
        "an unrelated figure 999 appears far from any year mention and "
        "another 777 value also floats without a nearby year."
    )
    values_2015 = EvidenceExtractor.collect_year_values(content, "2015")
    assert values_2015, "2015 窗口内应有候选"
    assert values_2015[0]["value"] == pytest.approx(5829.0)
    assert values_2015[0]["anchor_score"] <= values_2015[-1]["anchor_score"]
    assert EvidenceExtractor.collect_year_values(content, "1999") == []
