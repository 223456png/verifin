# -*- coding: utf-8 -*-
"""查询归一化（会话入口层）规格测试。

中文指标词 / 中文年份 / 虚词剥离 / 实体大小写修正——只覆盖 API 入口的
normalize_query 行为；benchmark 直跑 graph 不经此层（口径隔离）。
"""

from __future__ import annotations

from verifin.query_norm import normalize_query
from verifin.tools.verifier import extract_claim


def test_chinese_metric_mapped_to_canonical() -> None:
    out = normalize_query("NovaTech 2023年营收是多少？")
    assert "revenue" in out
    assert "营收" not in out
    claim = extract_claim(out)
    assert claim["metric"] == "revenue"
    assert claim["period"] == "2023"
    assert claim["entity"] == "NovaTech"


def test_chinese_year_word_boundary_restored() -> None:
    out = normalize_query("NovaTech 2024年净利润")
    claim = extract_claim(out)
    assert claim["period"] == "2024"
    assert claim["metric"] == "profit"


def test_growth_term_triggers_calculation_detection() -> None:
    from verifin.core.nodes import _detect_calculation

    out = normalize_query("NovaTech 2023到2024年营收增长率是多少？")
    spec = _detect_calculation(out)
    assert spec is not None
    assert spec.get("base_period") == "2023"
    assert spec.get("target_period") == "2024"


def test_known_entity_case_correction() -> None:
    out = normalize_query(
        "what was novatech revenue in 2023",
        known_entities=["NovaTech"],
    )
    claim = extract_claim(out)
    assert claim["entity"] == "NovaTech"


def test_zh_fillers_stripped() -> None:
    out = normalize_query("请问NovaTech的毛利率是多少")
    assert "的" not in out
    assert "请问" not in out
    claim = extract_claim(out)
    assert claim["metric"] == "gross_margin"


def test_english_query_passthrough() -> None:
    q = "What was NovaTech revenue in 2023?"
    out = normalize_query(q)
    assert out == q


def test_empty_and_whitespace_fall_back() -> None:
    assert normalize_query("") == ""
    assert normalize_query("   ") == "   "


def test_fullwidth_digits_normalized() -> None:
    out = normalize_query("NovaTech ２０２３年营收")
    assert "2023" in out


def test_mixed_source_preference_phrase_survives() -> None:
    out = normalize_query("只看年报口径，重新告诉我")
    assert "年报" in out or "consolidated" in out
