"""Phase 9/10 — FinQA program DSL 模板执行器测试（detect_program / ProgramExecutor / execute_dsl）。"""

from __future__ import annotations

import pytest

from verifin.tools.program_executor import (
    ProgramExecutor,
    ProgramSpec,
    ValueCandidate,
    detect_program,
    execute_dsl,
    parse_dsl,
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


# ---------------------------------------------------------------------------
# Phase 9.7 — 实体键控模板（cross_entity_diff / argmax_relay）

# 9. cross_entity_diff 检测：比较级 + 双实体 + 单年份
def test_detect_cross_entity_diff() -> None:
    spec = detect_program(
        "How much higher was Vertex revenue than NovaTech revenue in 2024?"
    )
    assert spec is not None
    assert spec.kind == "cross_entity_diff"
    assert spec.entity_a == "Vertex"
    assert spec.entity_b == "NovaTech"
    assert spec.base_period == "2024"
    # difference between 形态
    between = detect_program(
        "What was the difference between NovaTech revenue and Orion revenue in 2024?"
    )
    assert between is not None
    assert between.kind == "cross_entity_diff"
    assert between.entity_a == "NovaTech"
    assert between.entity_b == "Orion"


# 10. argmax_relay 检测：三实体 + highest + 胜者第二指标
def test_detect_argmax_relay() -> None:
    spec = detect_program(
        "Among NovaTech, Orion and Vertex, which company had the highest "
        "revenue in 2024, and what was its gross margin?"
    )
    assert spec is not None
    assert spec.kind == "argmax_relay"
    assert spec.entities == ["NovaTech", "Orion", "Vertex"]
    assert spec.metric == "revenue"
    assert spec.second_metric == "gross_margin"
    assert spec.base_period == "2024"
    assert spec.reversed is False
    # lowest → 取极小值方向
    lowest = detect_program(
        "Among NovaTech, Orion and Vertex, which company had the lowest "
        "revenue in 2024, and what was its operating margin?"
    )
    assert lowest is not None and lowest.reversed is True


# 11. 防误伤：真实 FinQA 全小写问题（无大写实体）不触发新模板
def test_detect_entity_templates_not_triggered_by_lowercase() -> None:
    # 比较级 + than 但无大写实体 → None（不劫持既有路径）
    assert detect_program(
        "what percent lower is the carrying value than the fair value?"
    ) is None
    # growth 反序/双年份问题仍走 growth_pct
    growth = detect_program(
        "What was the percentage growth in NovaTech revenue from 2023 to 2024?"
    )
    assert growth is not None and growth.kind == "growth_pct"


# 12. 跨实体差值执行：单位归一 + 差值刻度 + 量纲冲突剪枝
def test_execute_cross_entity_diff() -> None:
    spec = detect_program(
        "How much higher was Vertex revenue than NovaTech revenue in 2024?"
    )
    result = ProgramExecutor().execute(spec, {
        "Vertex": [ValueCandidate(value=20000.0, unit="million", period="2024")],
        "NovaTech": [ValueCandidate(value=12000.0, unit="million", period="2024")],
    })
    assert result.difference == pytest.approx(8000.0)
    assert result.difference_unit == "million"
    assert result.expression == "(20000.0 - 12000.0)"
    assert result.evidence == [None, None]  # 直接传候选无 chunk_id
    # 量纲冲突（% vs million）→ 无可用组合
    mixed = ProgramExecutor().execute(spec, {
        "Vertex": [ValueCandidate(value=44.0, unit="%", period="2024")],
        "NovaTech": [ValueCandidate(value=12000.0, unit="million", period="2024")],
    })
    assert mixed.difference is None
    # 缺一侧实体候选 → error 降级
    missing = ProgramExecutor().execute(spec, {
        "Vertex": [ValueCandidate(value=20000.0, unit="million", period="2024")],
    })
    assert missing.difference is None
    assert "missing" in (missing.error or "")


# 13. argmax 接力执行：比较取胜者 → 第二指标查找
def test_execute_argmax_relay() -> None:
    spec = detect_program(
        "Among NovaTech, Orion and Vertex, which company had the highest "
        "revenue in 2024, and what was its gross margin?"
    )
    candidates = {
        "NovaTech": [ValueCandidate(value=12000.0, unit="million", period="2024")],
        "Orion": [ValueCandidate(value=8500.0, unit="million", period="2024")],
        "Vertex": [ValueCandidate(value=20000.0, unit="million", period="2024")],
        "second::Vertex": [ValueCandidate(value=44.0, unit="%", period="2024")],
    }
    result = ProgramExecutor().execute(spec, candidates)
    # Vertex 收入最高（20,000）→ 其 gross margin 44% 为答案
    assert result.value == pytest.approx(44.0)
    assert result.unit == "%"
    assert "argmax" in (result.expression or "")
    assert "Vertex" in (result.expression or "")
    # 胜者第二指标缺失 → error 降级
    broken = ProgramExecutor().execute(spec, {
        k: v for k, v in candidates.items() if k != "second::Vertex"
    })
    assert broken.value is None
    assert "second-metric" in (broken.error or "")


# 14. 实体锚定扫描：目标实体/指标命中、防跨指标串值
def test_collect_entity_values() -> None:
    margin_chunk = (
        "Vertex gross margin in 2024 was 44% according to the consolidated "
        "financial statements."
    )
    revenue_chunk = (
        "Vertex revenue in 2024 was $20,000 million according to the "
        "consolidated financial statements."
    )
    # 命中：实体 + 指标 + 年份三信号
    hit = EvidenceExtractor.collect_entity_values(margin_chunk, "Vertex", "2024", "gross_margin")
    assert hit and hit[0]["value"] == pytest.approx(44.0)
    assert hit[0]["unit"] == "%"
    # 实体不在 chunk → 空
    assert EvidenceExtractor.collect_entity_values(margin_chunk, "Orion", "2024", "gross_margin") == []
    # chunk 未提及目标指标 → 空（防 gross margin 问题误收 revenue 值）
    assert EvidenceExtractor.collect_entity_values(revenue_chunk, "Vertex", "2024", "gross_margin") == []
    # 年份不符 → 空
    assert EvidenceExtractor.collect_entity_values(margin_chunk, "Vertex", "2023", "gross_margin") == []


# ---------------------------------------------------------------------------
# Phase 10 — ratio 模板（"what percentage of X are Y" → divide(Y值, X值)）

# 15. ratio 检测：分子/分母短语抽取（FinQA test 占 17% 的比率题型）
def test_detect_ratio_program() -> None:
    spec = detect_program(
        "what percentage of the total number of leased and owned properties "
        "were leased?"
    )
    assert spec is not None
    assert spec.kind == "ratio"
    assert spec.denominator == "the total number of leased and owned properties"
    assert spec.numerator == "leased"
    # 带年份单年份形态：year 记录在 base/target period
    with_year = detect_program(
        "what portion of total facilities are leased facilities in 2017?"
    )
    assert with_year is not None and with_year.kind == "ratio"
    assert with_year.base_period == "2017"
    # 双年份比率表达归 growth_pct（不劫持既有路径）
    two_years = detect_program(
        "what percentage of 2018 revenue is 2017 revenue?"
    )
    assert two_years is not None and two_years.kind == "growth_pct"


# 16. ratio 执行：num/den 候选 → divide(部分, 整体)，双刻度输出
def test_execute_ratio() -> None:
    spec = ProgramSpec(
        kind="ratio", base_period="2017", target_period="2017",
        numerator="leased facilities", denominator="total facilities",
    )
    result = ProgramExecutor().execute(spec, {
        "num": [ValueCandidate(value=8.1, unit=None, period="2017")],
        "den": [ValueCandidate(value=56.1, unit=None, period="2017")],
    })
    assert result.fraction == pytest.approx(8.1 / 56.1)
    assert result.value == pytest.approx(8.1 / 56.1 * 100)
    assert result.unit == "%"
    assert result.expression == "(8.1 / 56.1)"
    assert result.base["value"] == pytest.approx(56.1)   # 整体（分母）
    assert result.target["value"] == pytest.approx(8.1)  # 部分（分子）
    # 缺一侧候选 → error 降级
    missing = ProgramExecutor().execute(spec, {
        "num": [ValueCandidate(value=8.1, unit=None, period="2017")],
    })
    assert missing.fraction is None
    assert "missing" in (missing.error or "")
    # 量纲冲突（% vs million）与零分母组合被剪枝
    mixed = ProgramExecutor().execute(spec, {
        "num": [ValueCandidate(value=8.1, unit="%", period="2017")],
        "den": [ValueCandidate(value=56.1, unit="million", period="2017")],
    })
    assert mixed.fraction is None
    assert "plausible" in (mixed.error or "")
    # 同一数值自除（chunk_id 相同且值相同）非有效组合
    same = ProgramExecutor().execute(spec, {
        "num": [ValueCandidate(value=56.1, unit=None, period="2017", chunk_id="c1")],
        "den": [ValueCandidate(value=56.1, unit=None, period="2017", chunk_id="c1")],
    })
    assert same.fraction is None


# 17. ratio 排序偏好：|fraction| ≤ 2（部分 ≤ 整体）优先于反向配对
def test_execute_ratio_ordering() -> None:
    spec = ProgramSpec(
        kind="ratio", base_period="", target_period="",
        numerator="leased", denominator="total",
    )
    result = ProgramExecutor().execute(spec, {
        "num": [ValueCandidate(value=56.1, unit=None, period="", chunk_id="c1")],
        "den": [
            # 反向：56.1/8.1 ≈ 6.92；正向：56.1/56.1 = 1（chunk_id 不同，
            # 防「同 chunk 同值自除」剪枝误伤）
            ValueCandidate(value=8.1, unit=None, period="", chunk_id="c2"),
            ValueCandidate(value=56.1, unit=None, period="", chunk_id="c3"),
        ],
    })
    # 同锚分下 |fraction|≤2 的组合胜出（1.0 < 6.92）
    assert result.fraction == pytest.approx(1.0)


# 18. 短语锚定表格扫描：行标签 × 短语实词重叠（无年份比率题的候选来源）
def test_collect_phrase_values() -> None:
    content = (
        "| ( in millions ) | amount |\n"
        "| leased facilities | 56 |\n"
        "| total facilities | 100 |\n"
        "| owned properties | 44 |\n"
    )
    leased = EvidenceExtractor.collect_phrase_values(content, "leased facilities")
    assert leased and leased[0]["value"] == pytest.approx(56.0)
    assert leased[0]["row_label"] == "leased facilities"
    assert leased[0]["anchor_score"] == 0.0  # 双向全覆盖命中
    total = EvidenceExtractor.collect_phrase_values(content, "total facilities")
    assert total and total[0]["value"] == pytest.approx(100.0)
    # 无关短语：无行命中 → 全部候选均为低置信（anchor ≥ 1，无强锚定）
    unrelated = EvidenceExtractor.collect_phrase_values(content, "goodwill impairment")
    assert unrelated and all(item["anchor_score"] >= 1.0 for item in unrelated)
    # 空短语 → 空
    assert EvidenceExtractor.collect_phrase_values(content, "") == []


# ---------------------------------------------------------------------------
# Phase 10 — DSL 多步执行器（LLM program 生成的求值底座）

# 19. parse_dsl：顶层逗号切分 + 算子白名单 + 元数校验
def test_parse_dsl() -> None:
    steps = parse_dsl("subtract(5829, 5735), divide(#0, 5735)")
    assert steps == [("subtract", ["5829", "5735"]), ("divide", ["#0", "5735"])]
    # 嵌套调用不是合法步骤（顶层切分后 op 为 add 但参数被切碎 → 元数不符）
    with pytest.raises(ValueError):
        parse_dsl("add(divide(1, 2), 3)")
    with pytest.raises(ValueError):
        parse_dsl("unknown_op(1, 2)")
    with pytest.raises(ValueError):
        parse_dsl("add(1)")  # 元数不符
    with pytest.raises(ValueError):
        parse_dsl("")


# 20. execute_dsl：#N 步骤引用 + 数字/const/绑定参数
def test_execute_dsl_steps() -> None:
    out = execute_dsl("subtract(5829, 5735), divide(#0, 5735)")
    assert out["value"] == pytest.approx(94.0 / 5735.0)
    assert out["expression"] == "(94 / 5735)"  # 末步表达式（#0 已解析为 94）
    assert len(out["steps"]) == 2
    assert out["steps"][0]["value"] == pytest.approx(94.0)
    # 候选绑定 vN + 常量 const_K
    out2 = execute_dsl(
        "divide(v1, v2), multiply(#0, const_100)",
        {"v1": 8.1, "v2": 56.1},
    )
    assert out2["value"] == pytest.approx(8.1 / 56.1 * 100)
    # greater 原生比较（bool → 1.0/0.0）
    assert execute_dsl("greater(5, 3)")["value"] == 1.0
    assert execute_dsl("greater(3, 5)")["value"] == 0.0


# 21. execute_dsl：聚合算子（值组 tN 展开）
def test_execute_dsl_aggregations() -> None:
    assert execute_dsl("table_sum(t1)", {"t1": [1.5, 2.5, 3.0]})["value"] == pytest.approx(7.0)
    assert execute_dsl("table_average(t1)", {"t1": [1.0, 2.0, 4.0]})["value"] == pytest.approx(7.0 / 3.0)
    assert execute_dsl("table_max(t1)", {"t1": [1.0, 9.0, 4.0]})["value"] == pytest.approx(9.0)
    assert execute_dsl("table_min(t1)", {"t1": [1.0, 9.0, 4.0]})["value"] == pytest.approx(1.0)
    # 聚合后可被后续步骤引用（#0）
    out = execute_dsl(
        "table_sum(t1), subtract(#0, v1)",
        {"t1": [10.0, 20.0], "v1": 5.0},
    )
    assert out["value"] == pytest.approx(25.0)


# 22. execute_dsl：非法输入 → error 字典（不抛出）
def test_execute_dsl_errors() -> None:
    # 前向引用
    assert "error" in execute_dsl("divide(#1, v1)", {"v1": 1.0})
    # 引用越界
    assert "error" in execute_dsl("add(#5, 1), divide(#0, 2)")
    # 绑定缺失（幻觉引用）
    assert "error" in execute_dsl("divide(v3, v1)", {"v1": 1.0})
    # 除零
    assert "error" in execute_dsl("divide(1, 0)")
    # 非法算子
    assert "error" in execute_dsl("sin(1, 2)")
    # 聚合参数不是值组
    assert "error" in execute_dsl("table_sum(v1)", {"v1": 1.0})
    # 空值组
    assert "error" in execute_dsl("table_sum(t1)", {"t1": []})
