"""可插拔 LLM 程序生成器：问题 + 编号候选值 → FinQA DSL 程序（Phase 10）。

设计对齐 :class:`verifin.core.llm_planner.LLMPlanner`（Phase 9.5）：
- provider 通过 ``complete(prompt) -> str`` callable 注入，不绑定具体 SDK；
- 未注入 / 调用失败 / 输出非法（非 JSON、未知算子、引用越界、绑定缺失）
  时 ``generate`` 返回 ``None``，calculator 降级回确定性模板路径；
- 程序执行复用 :func:`verifin.tools.program_executor.execute_dsl`
  （算子白名单 + ``#N`` 引用校验 + PoT 沙箱求值），LLM 输出**永不直接
  执行**——先过 ``validate_dsl`` 结构校验，再逐步求值。

诚实边界：默认路径（无 LLM）行为与 Phase 9 完全一致，评测数字零外部
依赖可复现；LLM 路径为可选增强，未接入真实 provider 跑全量评测
（无 API key 不虚报数字），以单测 mock provider 锁定语义。
"""

from __future__ import annotations

import json
import re
from string import Template
from typing import Callable, Dict, List, Optional

from loguru import logger

from verifin.tools.program_executor import parse_dsl

# string.Template（$query / $candidates）：JSON 示例花括号与 format 冲突
_PROMPT = Template("""\
You are a financial numerical reasoning programmer. Given a financial question \
and candidate values extracted from financial documents (numbered v1..vN, with \
row labels; t1..tM are groups of values from the same row), write a program in \
the FinQA DSL that computes the answer.

Operations: add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b), \
greater(a, b), table_sum(t), table_average(t), table_max(t), table_min(t)
Arguments: numbers, vN (a candidate value), tN (a value group, only for \
table_* operations), const_100 / const_1000 (constants), #N (result of step N, \
0-based).
Scale: percentage questions usually answer the bare fraction (divide without \
multiplying by 100) unless the question asks for a percent explicitly; plain \
sums/differences answer the raw value.

Respond with ONLY a JSON object, no markdown fences, no commentary:
{"program": "divide(v1, v2)"}

Question: $query
Candidates:
$candidates
""")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def validate_dsl(program: str, candidate_ids: set) -> Optional[str]:
    """校验 LLM 生成的 DSL 程序；非法返回 None，合法返回规范化程序串。

    校验项（结构层，语义由 :func:`execute_dsl` 求值时兜底）：
    - ``parse_dsl`` 通过（算子白名单 + 元数）；
    - 标量参数为 数字 / ``const_K`` / ``#N``（不前向引用、不越界）/
      ``vN``（候选值绑定）；聚合参数为 ``tN``（值组绑定）。
    """
    try:
        steps = parse_dsl(program)
    except ValueError:
        return None
    for step_index, (_op, args) in enumerate(steps):
        for arg in args:
            if arg in candidate_ids:
                continue  # vN / tN 绑定（tN 仅聚合算子合法，求值时再验）
            if arg.startswith("#"):
                if not arg[1:].isdigit() or int(arg[1:]) >= step_index:
                    return None  # 前向引用 / 越界
                continue
            if re.match(r"^const_\d+$", arg) or re.match(r"^-?\d+(\.\d+)?$", arg):
                continue
            return None  # 未知 token（幻觉引用）
    return program.strip()


class LLMProgramGenerator:
    """LLM 驱动的 FinQA DSL 程序生成器（可插拔 provider，失败降级）。

    Args:
        complete: ``(prompt) -> str`` 的 LLM 调用函数；None 表示不可用。
    """

    def __init__(self, complete: Optional[Callable[[str], str]] = None) -> None:
        self._complete = complete

    @property
    def available(self) -> bool:
        """是否可用（已注入 complete）。"""
        return self._complete is not None

    def generate(
        self, query: str, candidates: List[dict]
    ) -> Optional[Dict[str, object]]:
        """生成可执行的 DSL 程序；任何失败返回 ``None``（触发模板降级）。

        Args:
            query: 用户问题。
            candidates: ``[{"id": "v1", "value": 8.1, "context": "row: leased
                facilities | col: total"}, ...]``——calculator 收集的编号候选
                （单值 ``vN``；值组 ``tN`` 的 value 为数值列表）。

        Returns:
            ``{"program": str, "bindings": {id: value}}`` 或 ``None``。
        """
        if not self.available or not candidates:
            return None
        prompt = _PROMPT.substitute(
            query=query, candidates=self._render_candidates(candidates)
        )
        try:
            raw = self._complete(prompt)
            data = self._parse(raw)
            if data is None:
                return None
            program = str(data.get("program") or "").strip()
            bindings = {
                str(c["id"]): c.get("value")
                for c in candidates
                if isinstance(c, dict) and c.get("id") is not None
            }
            # 隐式过滤：LLM 未用到的绑定不影响执行；校验引用完整性
            program = validate_dsl(program, set(bindings))
            if program is None:
                return None
            return {"program": program, "bindings": bindings}
        except Exception as exc:  # noqa: BLE001 - 任何 provider 异常都降级
            logger.warning("LLM 程序生成失败，降级确定性模板: {}", exc)
            return None

    # ------------------------------------------------------------------

    @staticmethod
    def _render_candidates(candidates: List[dict]) -> str:
        """候选列表渲染为 prompt 文本（值 + 行标签上下文）。"""
        lines = []
        for cand in candidates:
            cid = cand.get("id")
            value = cand.get("value")
            context = str(cand.get("context") or "").strip()
            if isinstance(value, list):
                value_repr = "[" + ", ".join(f"{v:g}" for v in value) + "]"
            elif isinstance(value, (int, float)):
                value_repr = f"{value:g}"
            else:
                continue
            line = f"{cid} = {value_repr}"
            if context:
                line += f"  ({context})"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> Optional[dict]:
        """从 LLM 输出提取 JSON（容忍 markdown 代码块与前后杂文）。"""
        raw = (raw or "").strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
        else:
            m = _JSON_RE.search(raw)
            if not m:
                return None
            raw = m.group(0)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None
