"""Phase 4 — PoT 计算器安全求值测试（tools/calculator.py）。"""

from __future__ import annotations

import pytest

from verifin.schemas import ExpressionResult
from verifin.tools.calculator import ExpressionCalculator


# 9.3.1 简单算术：(3910-2470)/2470*100 ≈ 58.3，unit="%"
def test_calc_simple_arithmetic() -> None:
    calc = ExpressionCalculator()
    result = calc.evaluate("(3910-2470)/2470*100")
    assert isinstance(result, ExpressionResult)
    assert result.is_valid is True
    assert result.value == pytest.approx(58.3, abs=0.01)
    assert result.unit == "%"
    assert result.error is None


# 9.3.2 数学函数：sqrt(16) → 4.0；round(3.14159, 2) → 3.14
def test_calc_with_math_func() -> None:
    calc = ExpressionCalculator()
    assert calc.evaluate("sqrt(16)").value == pytest.approx(4.0)
    assert calc.evaluate("round(3.14159, 2)").value == pytest.approx(3.14)
    assert calc.evaluate("abs(-5) + pow(2, 3)").value == pytest.approx(13.0)


# 9.3.3 不安全表达式 → 拒绝（不执行）
def test_calc_unsafe_expression() -> None:
    calc = ExpressionCalculator()
    for expression in ("__import__('os')", "open('/etc/passwd')", "().__class__"):
        result = calc.evaluate(expression)
        assert result.is_valid is False, f"应拒绝: {expression}"
        assert result.error, f"拒绝必须带原因: {expression}"
        assert result.value is None
    # 语句形态（非表达式）→ 语法拒绝
    assert calc.evaluate("import os").is_valid is False


# 9.3.4 超长/超大表达式 → 快速确定性拒绝（guard 优先于真实超时）
def test_calc_timeout_and_bounds() -> None:
    calc = ExpressionCalculator()
    # 超长表达式：长度 guard 直接拒绝
    long_expr = "9" * 500
    assert calc.evaluate(long_expr).is_valid is False
    # 超大幂运算：指数 guard 直接拒绝（防内存爆炸，测试秒回）
    huge = calc.evaluate("2**4000")
    assert huge.is_valid is False
    assert huge.error
    # 指数链炸弹亦被安全拒绝（MemoryError 或 guard，均须被捕获）
    chain = calc.evaluate("2**2**2**2**2**2**2**2**9")
    assert chain.is_valid is False
    assert chain.error


# 9.3.5 unit 启发式：* 100 → %
def test_calc_unit_detection() -> None:
    calc = ExpressionCalculator()
    assert calc.evaluate("1440 / 2470 * 100").unit == "%"
    assert calc.evaluate("sqrt(16)").unit is None
    assert calc.evaluate("1 + 2").unit is None


# 除零等运行异常 → invalid + error，不抛出
def test_calc_runtime_error() -> None:
    calc = ExpressionCalculator()
    result = calc.evaluate("1/0")
    assert result.is_valid is False
    assert result.error
