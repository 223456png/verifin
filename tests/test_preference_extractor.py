"""Phase 6 — 偏好抽取测试（core/preference_extractor.py + config/preference_rules.json）。"""

from __future__ import annotations

from verifin.core.preference_extractor import extract_preferences, load_preference_rules


# 显式：只看年报 → source_preference = annual_report
def test_extract_annual_report_preference() -> None:
    assert extract_preferences("只看年报，其它来源不要") == {"source": "annual_report"}
    assert extract_preferences("请基于 10-K 数据回答") == {"source": "annual_report"}


# 显式：合并口径 → consolidation_preference True
def test_extract_consolidation_preference() -> None:
    assert extract_preferences("我要合并口径的数据") == {"consolidation": True}
    assert extract_preferences("只看单体口径") == {"consolidation": False}


# 显式：期间偏好（只看2024 / FY2024）
def test_extract_period_preference() -> None:
    assert extract_preferences("只看2024年的数据") == {"period": "2024"}
    assert extract_preferences("只看 FY2024 的数据") == {"period": "FY2024"}


# 覆盖：新指令覆盖旧偏好（不看年报 → 新闻稿）
def test_extract_override_to_news() -> None:
    prefs = extract_preferences("不看年报了，看新闻稿")
    assert prefs.get("source") == "news"


# 研报来源 + 规则文件加载
def test_extract_research_and_rules_loaded() -> None:
    assert extract_preferences("给我第三方研报的数据") == {"source": "research"}
    rules = load_preference_rules()
    assert isinstance(rules, dict) and "explicit" in rules and "implicit" in rules