"""Phase 5 — 数字单位智能归一化测试（tools/unit_parser.py）。"""

from __future__ import annotations

import pytest

from verifin.tools.unit_parser import convert_unit, normalize_to_base


# billion → million
def test_convert_billion_to_million() -> None:
    assert convert_unit(12.4, "billion", "million") == pytest.approx(12400.0)
    assert convert_unit(1.0, "million", "billion") == pytest.approx(0.001)


# 中文单位：亿 = 100 million
def test_convert_chinese_unit() -> None:
    assert convert_unit(1, "亿", "million") == pytest.approx(100.0)
    assert convert_unit(1, "万", "million") == pytest.approx(0.01)


# 归一化到基准（million）：跨单位口径一致；百分比原样保留
def test_normalize_values_consistent() -> None:
    assert normalize_to_base(12400, "million") == pytest.approx((12400.0, "million"))
    assert normalize_to_base(12.4, "billion") == pytest.approx((12400.0, "million"))
    assert normalize_to_base(42.5, "%") == pytest.approx((42.5, "%"))
    # 未知单位 → None（不可比较）
    assert normalize_to_base(1.0, "widget") is None