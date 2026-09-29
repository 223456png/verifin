"""PoT 计算器：受限表达式安全求值（不依赖 LangGraph）。

三层防御：
- Layer 1: ``ast.parse(mode="eval")`` + 节点白名单（拒绝 Import/Attribute/未注册调用等）；
- Layer 2: ``eval`` 命名空间禁用 ``__builtins__``，仅暴露 math 函数白名单；
- Layer 3: SIGALRM 超时（仅主线程生效）+ 决定性 guard（长度/指数上限）保证快速确定拒绝。
"""

from __future__ import annotations

import ast
import math
import re
import signal
import threading
import time
from dataclasses import asdict
from typing import Dict, Optional

from verifin.schemas import ExpressionResult

_MAX_EXPRESSION_LENGTH = 400
_MAX_POW_EXPONENT = 1000

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Name, ast.Constant, ast.Call,
    ast.Compare, ast.BoolOp, ast.Tuple, ast.Load,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd, ast.Not,
    ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
)
_ALLOWED_CONSTANT_TYPES = (int, float, bool, type(None))

_UNIT_PERCENT_RE = re.compile(r"(?<!\*)\*100(?![0-9.])")


class _TimeoutError(TimeoutError):
    """eval 超时专用（signal handler 触发）。"""


def _timeout_handler(signum, frame) -> None:  # noqa: ARG001 - signal 回调签名
    raise _TimeoutError("expression evaluation timed out")


class ExpressionCalculator:
    """受限表达式求值器。"""

    _math_functions: Dict[str, object] = {
        "sqrt": math.sqrt,
        "pow": math.pow,
        "sin": math.sin,
        "cos": math.cos,
        "log": math.log,
        "exp": math.exp,
        "abs": abs,
        "round": round,
    }

    def __init__(self, timeout_seconds: float = 1.0) -> None:
        self.timeout = max(timeout_seconds, 0.01)

    # ------------------------------------------------------------------ 接口

    def evaluate(
        self, expression: str, context: Optional[dict] = None
    ) -> ExpressionResult:
        """安全求值；任何异常均转为 :attr:`ExpressionResult.error` 而不抛出。"""
        start = time.perf_counter()

        def fail(error: str, elapsed: float) -> ExpressionResult:
            return ExpressionResult(
                expression=expression,
                error=error,
                is_valid=False,
                execution_time_ms=round(elapsed * 1000.0, 2),
            )

        elapsed = time.perf_counter() - start
        if not expression or not expression.strip():
            return fail("empty expression", elapsed)
        if len(expression) > _MAX_EXPRESSION_LENGTH:
            return fail("expression too long (max 400 chars)", elapsed)

        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return fail(f"SyntaxError: {exc.msg}", time.perf_counter() - start)

        guard_error = self._validate(
            tree, allowed_names=set(context or {})
        )
        if guard_error:
            return fail(guard_error, time.perf_counter() - start)

        safe_globals: dict = {"__builtins__": {}, **self._math_functions}
        safe_locals: dict = dict(context) if context else {}
        code = compile(tree, "<verifin-calc>", "eval")

        can_signal = (
            hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        )
        if can_signal:
            old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
            signal.setitimer(signal.ITIMER_REAL, self.timeout)
        try:
            raw = eval(code, safe_globals, safe_locals)  # noqa: S307 - 白名单 AST + 空 builtins
        except _TimeoutError:
            return fail("evaluation timed out", time.perf_counter() - start)
        except Exception as exc:  # ZeroDivisionError / MemoryError / ValueError ...
            return fail(f"{type(exc).__name__}: {exc}", time.perf_counter() - start)
        finally:
            if can_signal:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        unit = "%" if _UNIT_PERCENT_RE.search(re.sub(r"\s+", "", expression)) else None
        return ExpressionResult(
            expression=expression,
            value=value,
            unit=unit,
            is_valid=True,
            execution_time_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )

    # ------------------------------------------------------------------ 校验

    def _validate(self, tree: ast.AST, allowed_names: Optional[set] = None) -> Optional[str]:
        """白名单节点检查 + 超大幂 guard；返回错误信息，None 表示通过。"""
        allowed_names = allowed_names or set()
        for node in ast.walk(tree):
            if not isinstance(node, _ALLOWED_NODES):
                return f"unsafe node: {type(node).__name__}"
            if isinstance(node, ast.Name):
                if node.id not in self._math_functions and node.id not in allowed_names:
                    return f"unsafe name: {node.id}"
            if isinstance(node, ast.Constant):
                if not isinstance(node.value, _ALLOWED_CONSTANT_TYPES):
                    return f"unsafe constant: {type(node.value).__name__}"
            if isinstance(node, ast.Call):
                if not (
                    isinstance(node.func, ast.Name)
                    and node.func.id in self._math_functions
                ):
                    return "unsafe function call"
            if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
                # 决定性 guard：常量指数过大直接拒绝（防内存炸弹，先于超时兜底）
                right = node.right
                if (
                    isinstance(right, ast.Constant)
                    and isinstance(right.value, int)
                    and abs(right.value) > _MAX_POW_EXPONENT
                ):
                    return f"exponent too large (max {_MAX_POW_EXPONENT})"
        return None


def calc_expression(expr: str) -> dict:
    """Tool 包装：字符串表达式 → ExpressionResult dict（JSON 可序列化）。"""
    return asdict(ExpressionCalculator().evaluate(expr))
