"""Phase 9/10 — FinQA program DSL 模板执行器测试（detect_program / ProgramExecutor / execute_dsl）。"""

from __future__ import annotations

import pytest

from verifin.tools.evidence import EvidenceExtractor
from verifin.tools.program_executor import (
    ProgramExecutor,
    ProgramSpec,
    ValueCandidate,
    detect_program,
    execute_dsl,
    parse_dsl,
)


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


# 3. 防误伤：无关键词 / 无年份 / growth 无年份 → None
def test_detect_program_negative() -> None:
    assert detect_program("What was NovaTech revenue in 2024?") is None
    assert detect_program("Tell me about the company.") is None
    # Phase 12 行为变更（D5）："growth" 单年份现为 difference 模板
    # （"growth in 2024" = 2024 vs 2023，FinQA gold 同为
    # subtract(cur, prev), divide(#0, prev)）；无年份 growth 仍 None
    assert detect_program("What was NovaTech growth?") is None
    growth_1yr = detect_program("What was NovaTech growth in 2024?")
    assert growth_1yr is not None
    assert growth_1yr.kind == "difference"
    assert growth_1yr.base_period == "2023"
    assert growth_1yr.target_period == "2024"


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


# 17a. Phase 10.1：负值操作数剪枝（部分/整体非负）
def test_execute_ratio_negative_prune() -> None:
    spec = ProgramSpec(
        kind="ratio", base_period="", target_period="",
        numerator="deferred tax", denominator="purchase price",
    )
    result = ProgramExecutor().execute(spec, {
        "num": [
            ValueCandidate(value=-9500.0, unit=None, period="", chunk_id="c1"),
            ValueCandidate(value=2400.0, unit=None, period="", chunk_id="c1"),
        ],
        "den": [ValueCandidate(value=73200.0, unit=None, period="", chunk_id="c1")],
    })
    # -9500/73200 被剪枝，唯一存活组合 2400/73200（fraction 保留 6 位小数）
    assert result.fraction == pytest.approx(2400.0 / 73200.0, abs=1e-5)


# 17b. Phase 10.1：分母 total 行偏好 + 分子 total 行惩罚
def test_execute_ratio_total_row_preference() -> None:
    spec = ProgramSpec(
        kind="ratio", base_period="", target_period="",
        numerator="due in 2018", denominator="future minimum rental payments",
    )
    # 分母短语与行标签零词重叠：total 行回退候选（anchor 0.75）
    # 对普通行候选（anchor 0.25 但语义不符）——本例中只有 total 行
    result = ProgramExecutor().execute(spec, {
        "num": [
            ValueCandidate(value=301.0, unit=None, period="", chunk_id="c1",
                           row_label="2018"),
        ],
        "den": [
            ValueCandidate(value=1160.0, unit=None, period="", chunk_id="c1",
                           row_label="2021 - thereafter", anchor_score=0.5),
            ValueCandidate(value=2575.0, unit=None, period="", chunk_id="c1",
                           row_label="total", anchor_score=0.75),
        ],
    })
    # total 行偏好（-0.35 → 0.40 < 0.5）：分母选 2575 而非 1160
    assert result.fraction == pytest.approx(301.0 / 2575.0, abs=1e-5)
    # 分子 total 行惩罚：分子若含 total 行候选则靠后
    spec2 = ProgramSpec(
        kind="ratio", base_period="", target_period="",
        numerator="part", denominator="whole",
    )
    result2 = ProgramExecutor().execute(spec2, {
        "num": [
            ValueCandidate(value=8.0, unit=None, period="", chunk_id="c1",
                           row_label="part", anchor_score=0.3),
            ValueCandidate(value=100.0, unit=None, period="", chunk_id="c1",
                           row_label="total part", anchor_score=0.3),
        ],
        "den": [ValueCandidate(value=100.0, unit=None, period="", chunk_id="c2",
                               row_label="whole")],
    })
    # total 行分子 +0.2 惩罚 → part 行胜出（8/100 而非 100/100 自除）
    assert result2.fraction == pytest.approx(0.08)


# 17c. Phase 10.1：列一致性（分母短语含 "total" → total 列的 num/den 配对）
def test_execute_ratio_column_preference() -> None:
    spec = ProgramSpec(
        kind="ratio", base_period="", target_period="",
        numerator="leased",
        denominator="total facilities as measured in square feet",
    )
    # 三列同锚分并列（unitedstates/othercountries/total 列）：
    # 列头与分母短语词重叠的 total 列配对胜出
    result = ProgramExecutor().execute(spec, {
        "num": [
            ValueCandidate(value=2.1, unit=None, period="", chunk_id="c1",
                           row_label="leased facilities", column="unitedstates"),
            ValueCandidate(value=6.0, unit=None, period="", chunk_id="c1",
                           row_label="leased facilities", column="othercountries"),
            ValueCandidate(value=8.1, unit=None, period="", chunk_id="c1",
                           row_label="leased facilities", column="total"),
        ],
        "den": [
            ValueCandidate(value=32.8, unit=None, period="", chunk_id="c1",
                           row_label="total facilities", column="unitedstates"),
            ValueCandidate(value=23.2, unit=None, period="", chunk_id="c1",
                           row_label="total facilities", column="othercountries"),
            ValueCandidate(value=56.0, unit=None, period="", chunk_id="c1",
                           row_label="total facilities", column="total"),
        ],
    })
    assert result.fraction == pytest.approx(8.1 / 56.0)


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
    # 无关短语：零词重叠行不再成候选（Phase 10.1 严格匹配，
    # 旧版零重叠行以 anchor 1.25 兜底会产生自除假阳性）
    unrelated = EvidenceExtractor.collect_phrase_values(content, "goodwill impairment")
    assert unrelated == []
    # 空短语 → 空
    assert EvidenceExtractor.collect_phrase_values(content, "") == []


# 18a. Phase 10.1：年份 token 匹配（"due in 2018" → 纯年份行标签）
def test_collect_phrase_values_year_token() -> None:
    content = (
        "| $ in millions | as of december 2015 |\n"
        "| 2016 | 317 |\n"
        "| 2017 | 313 |\n"
        "| 2018 | 301 |\n"
        "| total | 2575 |\n"
    )
    due_2018 = EvidenceExtractor.collect_phrase_values(content, "due in 2018")
    assert due_2018 and due_2018[0]["value"] == pytest.approx(301.0)
    assert due_2018[0]["row_label"] == "2018"
    # "after 2020" → 年份上界语义：命中 thereafter 行，不命中 ≤2020 的年份行
    maturity = (
        "| $ in millions | maturities |\n"
        "| 2019 | 258 |\n"
        "| 2020 | 226 |\n"
        "| 2021 - thereafter | 1160 |\n"
    )
    after_2020 = EvidenceExtractor.collect_phrase_values(maturity, "due after 2020")
    assert after_2020 and after_2020[0]["value"] == pytest.approx(1160.0)
    assert after_2020[0]["row_label"] == "2021 - thereafter"
    values = [item["value"] for item in after_2020]
    assert 226.0 not in values  # 2020 行本身不命中
    # 普通年份短语不触发上界语义
    due_2019 = EvidenceExtractor.collect_phrase_values(maturity, "due in 2019")
    assert due_2019 and due_2019[0]["value"] == pytest.approx(258.0)


# 18b. Phase 10.1：正文句级扫描（分子在脚注正文而非表格）
def test_collect_phrase_values_text_scan() -> None:
    content = (
        "( 1 ) 42749 shares were repurchased in open-market transactions "
        "under the plan .\n"
        "| period | shares |\n"
        "| total: | 45686 |\n"
    )
    num = EvidenceExtractor.collect_phrase_values(
        content, "repurchased in open-market transactions"
    )
    values = [item["value"] for item in num]
    assert 42749.0 in values
    assert 1.0 not in values  # 脚注标记 "( 1 )" 剥离，不污染候选
    text_item = next(item for item in num if item["value"] == 42749.0)
    assert text_item["source"] == "text"
    # 表格行仍优先于正文（正文来源 +0.1 轻惩罚）
    den = EvidenceExtractor.collect_phrase_values(content, "total number of shares")
    assert den and den[0]["value"] == pytest.approx(45686.0)
    assert den[0]["row_label"] == "total:"


# 18c. Phase 10.1：total 行回退（"percentage of X" 的 X 与行标签零重叠时）
def test_collect_total_values() -> None:
    content = (
        "| $ in millions | commitments |\n"
        "| 2016 | 317 |\n"
        "| 2018 | 301 |\n"
        "| total | 2575 |\n"
        "| other table header | x |\n"
        "| grand total | 999 |\n"
    )
    totals = EvidenceExtractor.collect_total_values(content)
    values = [item["value"] for item in totals]
    assert 2575.0 in values
    assert 999.0 in values  # "grand total" 尾缀形态也命中
    assert all(item["anchor_score"] == 0.75 for item in totals)
    assert all("total" in item["row_label"].lower() for item in totals)
    # 非合计行不入回退
    assert 317.0 not in values and 301.0 not in values


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


# ---------------------------------------------------------------------------
# Phase 12 — ratio 句式扩展 + average 模板 + growth/difference 关键词扩展

# 23. ratio_of_to："ratio of NUM to DEN"（48 题未检出的最大缺口）
def test_detect_ratio_of_to() -> None:
    spec = detect_program(
        "what is the ratio of the total flight attendants to total "
        "maintenance personnel ?"
    )
    assert spec is not None
    assert spec.kind == "ratio"
    assert spec.numerator == "the total flight attendants"
    assert spec.denominator == "total maintenance personnel"
    # 双年份放行（D3）：年份在分子/分母短语内，base_period 为空
    two_years = detect_program(
        "what was the ratio of the purchase in december 2012 to the "
        "purchase in january 2013 ?"
    )
    assert two_years is not None and two_years.kind == "ratio"
    assert two_years.base_period == ""
    assert "2012" in (two_years.numerator or "")
    assert "2013" in (two_years.denominator or "")


# 24. hyphen_ratio："debt-to-asset ratio" → divide(debt, asset)
def test_detect_hyphen_ratio() -> None:
    spec = detect_program("what is the debt-to-asset ratio ?")
    assert spec is not None
    assert spec.kind == "ratio"
    assert spec.numerator == "debt"
    assert spec.denominator == "asset"


# 25. pct_of_to："percent of NUM to DEN" + 假连接词排除（D2）
def test_detect_pct_of_to() -> None:
    spec = detect_program(
        "in 2010 what was the percent of the income tax benefit to the "
        "stock based compensation cost ?"
    )
    assert spec is not None
    assert spec.kind == "ratio"
    assert spec.numerator == "the income tax benefit"
    assert spec.denominator == "the stock based compensation cost"
    # "not leased" 否定形态：percent of NUM（not leased）to DEN
    negated = detect_program(
        "as of december 2012 what is the percent of the square footage "
        "not leased to the total square footage ?"
    )
    assert negated is not None and negated.kind == "ratio"
    assert "not leased" in (negated.numerator or "")
    assert negated.denominator == "the total square footage"


# 25a. pct_of_to 假连接词：due to / compared to 等不触发
def test_detect_pct_of_to_false_connectives() -> None:
    # "due to" 是短语内连接词，非分母引导词——既有 _RATIO_RE 先匹配
    # （"are" 连接），语义正确且分母不是 "expire"
    spec = detect_program(
        "what percentage of obligations are due to expire in 2018 ?"
    )
    assert spec is not None
    assert spec.kind == "ratio"
    assert spec.denominator == "obligations"
    assert spec.numerator == "due to expire in 2018"
    # 明确的假连接词形态：compared to（无年份，不会走 growth）
    assert detect_program(
        "what was the percent of revenue compared to prior year ?"
    ) is None


# 26. as_pct_of / represented："NUM as a percentage of DEN" 与倒装
def test_detect_as_pct_of_and_represented() -> None:
    spec = detect_program(
        "what is the borrowing under the term loan facility as a "
        "percentage of the total contractual maturities of debt ?"
    )
    assert spec is not None
    assert spec.kind == "ratio"
    assert "borrowing under the term loan facility" in (spec.numerator or "")
    assert spec.denominator == "the total contractual maturities of debt"
    # 倒装："NUM represented what percentage of DEN"
    rep = detect_program(
        "brazilian paper sales represented what percentage of printing "
        "papers in 2006 ?"
    )
    assert rep is not None and rep.kind == "ratio"
    assert "brazilian paper sales" in (rep.numerator or "")
    assert "printing papers" in (rep.denominator or "")
    assert rep.base_period == "2006"


# 27. average：between/from Y1 and/to Y2 → 年份区间
def test_detect_average() -> None:
    spec = detect_program(
        "what was the average net revenue between 2016 and 2017 in millions ?"
    )
    assert spec is not None
    assert spec.kind == "average"
    assert spec.periods == ["2016", "2017"]
    assert spec.base_period == "2016"
    assert spec.target_period == "2017"
    # from ... to ... 三年区间
    span = detect_program("what was the average cash flow from 2004 to 2006 ?")
    assert span is not None and span.kind == "average"
    assert span.periods == ["2004", "2005", "2006"]
    # average 不劫持双年份 growth 问句
    growth = detect_program(
        "what was the percentage change in revenue between 2016 and 2017 ?"
    )
    assert growth is not None and growth.kind == "growth_pct"
    # 无年份区间的 average 不触发（走提取/LLM 路径）
    assert detect_program("what was the average share price in 2015 ?") is None


# 28. average 执行：逐年取值求均值 + 单位一致剪枝 + 缺期降级
def test_execute_average() -> None:
    spec = ProgramSpec(
        kind="average", base_period="2016", target_period="2017",
        periods=["2016", "2017"],
    )
    result = ProgramExecutor().execute(spec, {
        "2016": [ValueCandidate(value=703.1, unit="million", period="2016")],
        "2017": [ValueCandidate(value=705.4, unit="million", period="2017")],
    })
    assert result.value == pytest.approx(704.25)
    assert result.unit == "million"
    assert result.expression == "(703.1 + 705.4) / 2"
    # 三年区间均值
    span = ProgramSpec(
        kind="average", base_period="2004", target_period="2006",
        periods=["2004", "2005", "2006"],
    )
    span_result = ProgramExecutor().execute(span, {
        "2004": [ValueCandidate(value=900.0, unit="million", period="2004")],
        "2005": [ValueCandidate(value=957.4, unit="million", period="2005")],
        "2006": [ValueCandidate(value=819.5, unit="million", period="2006")],
    })
    assert span_result.value == pytest.approx((900.0 + 957.4 + 819.5) / 3)
    # 量纲冲突（% vs million）→ 无可用组合
    mixed = ProgramExecutor().execute(spec, {
        "2016": [ValueCandidate(value=703.1, unit="million", period="2016")],
        "2017": [ValueCandidate(value=42.0, unit="%", period="2017")],
    })
    assert mixed.value is None
    assert "plausible" in (mixed.error or "")
    # 缺一年候选 → error 降级
    missing = ProgramExecutor().execute(spec, {
        "2016": [ValueCandidate(value=703.1, unit="million", period="2016")],
    })
    assert missing.value is None
    assert "missing" in (missing.error or "")


# 29. ROI / return on investment / rate of return → growth_pct（D6）
def test_detect_roi_as_growth() -> None:
    spec = detect_program(
        "what is the roi of an investment in ups from 2008 to 2009 ?"
    )
    assert spec is not None
    assert spec.kind == "growth_pct"
    assert spec.base_period == "2008"
    assert spec.target_period == "2009"
    roi = detect_program(
        "what is the return on investment for s&p500 from 2004 to 2006 ?"
    )
    assert roi is not None and roi.kind == "growth_pct"
    ror = detect_program(
        "what is the rate of return in cadence design systems inc . of "
        "an investment from 2010 to 2011 ?"
    )
    assert ror is not None and ror.kind == "growth_pct"


# 30. 新句式不抢既有模板流量（D1 顺序保障）
def test_new_patterns_do_not_hijack_existing() -> None:
    # 既有 _RATIO_RE 形态仍先匹配（"are" 连接，无 " to "）
    spec = detect_program(
        "what percentage of total facilities as measured in square feet "
        "are leased ?"
    )
    assert spec is not None and spec.kind == "ratio"
    assert spec.denominator == "total facilities as measured in square feet"
    assert spec.numerator == "leased"
    # growth 双年份不被 ratio 句式劫持（"percentage ... from 2023 to 2024"
    # 含 growth 关键词 → growth_pct 优先）
    growth = detect_program(
        "what was the percentage of revenue from 2023 to 2024 ?"
    )
    assert growth is not None and growth.kind == "growth_pct"
    # cross_entity_diff 形态不受 ratio 变体影响
    cross = detect_program(
        "How much higher was Vertex revenue than NovaTech revenue in 2024?"
    )
    assert cross is not None and cross.kind == "cross_entity_diff"
