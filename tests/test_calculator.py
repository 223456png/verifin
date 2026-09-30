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


# 2026-09-30 安全审计修复：非常量/嵌套幂指数必须在 AST 层快速拒绝——
# 部署线程与 Windows 没有 SIGALRM 兜底，9**9**9 这类表达式曾可占用
# 执行器直到算完/内存耗尽。反向测试锁定：毫秒级返回，绝不长跑。
def test_calc_rejects_nested_power_bomb_fast() -> None:
    import time as _time

    calc = ExpressionCalculator()
    for expr in ("9**9**9", "2**(2**20)", "2**2**20"):
        t0 = _time.monotonic()
        result = calc.evaluate(expr)
        elapsed = _time.monotonic() - t0
        assert result.is_valid is False
        assert result.error == "power exponent must be an integer constant"
        # 任何环境下都必须快速确定拒绝（此前该用例可长跑数十秒）
        assert elapsed < 1.0, f"{expr} took {elapsed:.2f}s"


def test_calc_rejects_non_int_power_exponent() -> None:
    calc = ExpressionCalculator()
    # float 指数同样无法静态估计（拒）；合法幂走 math.pow 函数路径不受影响
    assert calc.evaluate("2.0**10.0").is_valid is False
    assert calc.evaluate("2.0**10.0").error == "power exponent must be an integer constant"


def test_calc_legitimate_powers_still_pass() -> None:
    calc = ExpressionCalculator()
    # 常量指数 ≤ 上限正常放行
    assert calc.evaluate("2**999").is_valid is True
    assert calc.evaluate("2**10").value == 1024
    # 合法幂但结果超出 float 表示范围 → 确定性错误（修复前是未捕获 OverflowError 炸穿调用方）
    big = calc.evaluate("(9**9)**999")
    assert big.is_valid is False
    assert big.error == "result too large to represent (float overflow)"


# 2026-09-30 复验补充：inf / nan 曾以 is_valid=True 返回——
# 对"可校验计算"而言，把 nan 当作已验证数值送进 Claim-Evidence 链路
# 是静默错误值（比抛异常更危险：下游无从察觉）。
# 反向测试锁定：非有限结果必须是确定性错误。
def test_calc_rejects_non_finite_results() -> None:
    calc = ExpressionCalculator()
    non_finite = (
        "1e400",               # 字面量直接溢出 → inf
        "-1e400",              # → -inf
        "1e308*10",            # 乘法溢出 → inf
        "1e400 - 1e400",       # inf - inf → nan
        "1e400*0",             # inf * 0 → nan
        "1e400/1e400",         # inf / inf → nan
        "sqrt(1e308*1e308)",   # math 函数路径 → inf
    )
    for expr in non_finite:
        result = calc.evaluate(expr)
        assert result.is_valid is False, f"{expr} 被放行，value={result.value!r}"
        assert result.error == "non-finite result (inf/nan)", f"{expr} -> {result.error}"


def test_calc_finite_values_not_affected_by_non_finite_guard() -> None:
    calc = ExpressionCalculator()
    # 接近边界但仍是有限值 → 正常放行，新守卫不能误伤
    assert calc.evaluate("1e308").is_valid is True
    assert calc.evaluate("1e308").value == 1e308
    assert calc.evaluate("2**999").is_valid is True
    assert calc.evaluate("100").value == 100.0
    assert calc.evaluate("0").value == 0.0
    assert calc.evaluate("-0.5").value == -0.5
