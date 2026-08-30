"""Phase 9.6 — RAGAs 风格生成质量指标（规则近似）测试。

口径锁定：
- faithfulness：答案数值声明被「证据文本 ∪ PoT 计算产物」支持的比例；
  无数值声明 → 1.0；空答案 → 0.0；单位归一 + 1% 容差；
- answer_relevancy：问题内容词（去停用词 + 数值 token）覆盖率；
- summarize_results 聚合两指标（逐样本均值，跳过 None）。
"""

from __future__ import annotations

from verifin.benchmark.metrics import (
    answer_relevancy,
    faithfulness,
    summarize_results,
)


# 1. faithfulness：全部数值有证据支持 → 1.0
def test_faithfulness_fully_supported() -> None:
    evidence = ["NovaTech revenue in 2024 was $12,000 million according to the filing."]
    answer = "NovaTech revenue in 2024 was $12,000 million."
    assert faithfulness(answer, evidence) == 1.0


# 2. 部分支持：1/2 数值无据 → 0.5
def test_faithfulness_partially_supported() -> None:
    evidence = ["NovaTech revenue in 2024 was $12,000 million."]
    answer = "NovaTech revenue was $12,000 million, up from $11,000 million."
    assert faithfulness(answer, evidence) == 0.5


# 3. 计算产物支持：增长率不在证据文本，但等于 PoT 结果 → 有据
def test_faithfulness_calc_supported() -> None:
    evidence = [
        "NovaTech revenue in 2023 was $10,000 million.",
        "NovaTech revenue in 2024 was $12,000 million.",
    ]
    answer = "NovaTech revenue grew 20.0% from 2023 to 2024."
    calc = [(20.0, "%")]
    assert faithfulness(answer, evidence, calc_candidates=calc) == 1.0
    # 无计算产物 → 增长率无据（年份 2023/2024 有据）→ 2/3
    assert abs(faithfulness(answer, evidence) - 2 / 3) < 1e-9


# 4. 单位归一：12.0 billion 与 $12,000 million 等价 → 支持
def test_faithfulness_unit_normalization() -> None:
    evidence = ["NovaTech revenue in 2024 was $12,000 million."]
    answer = "NovaTech revenue was $12 billion."
    assert faithfulness(answer, evidence) == 1.0


# 5. 边界：无数值声明 → 1.0；空答案 → 0.0
def test_faithfulness_boundaries() -> None:
    assert faithfulness("No evidence found.", []) == 1.0
    assert faithfulness("", ["some evidence"]) == 0.0


# 6. answer_relevancy：内容词全覆盖 → 1.0；部分覆盖按比例
def test_answer_relevancy() -> None:
    query = "What was NovaTech revenue in 2024?"
    full = "NovaTech revenue in 2024 was $12,000 million."
    assert answer_relevancy(query, full) == 1.0
    partial = answer_relevancy(query, "The revenue was $12,000 million.")
    assert 0.0 < partial < 1.0
    assert answer_relevancy(query, "Helios capex was $900 million.") == 0.0


# 7. 停用词不计入分母（纯停用词问题 → 有答案即 1.0）
def test_answer_relevancy_stopwords_only() -> None:
    assert answer_relevancy("What about it?", "Revenue was $12,000 million.") == 1.0
    assert answer_relevancy("What about it?", "") == 0.0


# 8. 数值 token 计入覆盖（年份 2024 必须在答案中出现才满分）
def test_answer_relevancy_numeric_tokens() -> None:
    query = "What was NovaTech revenue in 2024?"
    with_year = "NovaTech revenue in 2024 was $12,000 million."
    without_year = "NovaTech revenue was $12,000 million."
    assert answer_relevancy(query, with_year) > answer_relevancy(query, without_year)


# 9. summarize_results 聚合：逐样本均值（跳过 None）
def test_summarize_includes_generation_quality() -> None:
    results = [
        {"is_correct": True, "faithfulness": 1.0, "answer_relevancy": 1.0},
        {"is_correct": False, "faithfulness": 0.5, "answer_relevancy": 0.5},
        {"is_correct": True},  # 无该字段（旧结果兼容）→ 跳过
    ]
    summary = summarize_results(results)
    assert summary["faithfulness"] == 0.75
    assert summary["answer_relevancy"] == 0.75
    # 全无字段 → None（报告渲染 N/A）
    assert summarize_results([{"is_correct": True}])["faithfulness"] is None
