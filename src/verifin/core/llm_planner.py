"""可插拔 LLM 规划器：用 LLM 把金融问题分解为检索子任务。

设计目标（Phase 9.5）：
- provider 通过 ``complete(prompt) -> str`` callable 注入，不绑定具体 SDK
  （OpenAI / Anthropic / 本地模型均可，调用方自行封装）；
- 未注入 ``complete`` 或调用失败/输出非法时，``plan`` 返回 ``None``，
  由 ``make_planner_node`` 降级到确定性规则 planner；
- 默认路径（无 LLM）行为不变，保证零外部依赖即可运行。
"""

from __future__ import annotations

import json
import re
from string import Template
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

# 用 string.Template（$query / $history）而非 str.format：JSON 示例含大量花括号，
# format 会误判为占位符；Template 只认 $ 占位符，JSON 花括号原样保留。
_PROMPT = Template("""\
You are a financial question decomposition planner for a retrieval-augmented \
evidence-verification agent.

Given a financial question and optional conversation history, decompose it into \
one or more retrieval sub-queries and extract the four verification elements \
(entity, period, metric, definition) for each sub-query. If the question requires \
arithmetic (a growth percentage, or a difference between two periods), also emit \
a calculation_spec.

Respond with ONLY a JSON object, no markdown fences, no commentary:

{
  "sub_tasks": ["sub query 1", "sub query 2"],
  "claims": {
    "sub query 1": {
      "entity": "", "period": "", "metric": "", "definition": ""
    }
  },
  "calculation_spec": {
    "kind": "growth_pct",
    "base_period": "2023",
    "target_period": "2024",
    "metric": "revenue",
    "entity": "NovaTech",
    "reversed": false
  }
}

Rules:
- sub_tasks: at least one entry; prefer concise retrieval-friendly phrases that
  include entity + metric + period when present.
- calculation_spec is null when no arithmetic is needed; otherwise kind must be
  one of "growth_pct" or "difference".
- Leave a field as empty string "" when the element is absent.

Question: $query
History: $history
""")


class LLMPlanner:
    """LLM 驱动的问题分解器（可插拔 provider，失败降级）。

    Args:
        complete: ``(prompt) -> str`` 的 LLM 调用函数；None 表示不可用。
    """

    def __init__(self, complete: Optional[Callable[[str], str]] = None) -> None:
        self._complete = complete

    @property
    def available(self) -> bool:
        """是否可用（已注入 complete 且未显式关闭）。"""
        return self._complete is not None

    def plan(self, query: str, history: List[Any]) -> Optional[dict]:
        """生成结构化计划；任何失败返回 None（触发规则降级）。

        Returns:
            ``{"sub_tasks": [...], "claims": {...}, "calculation_spec": {...|None}}``
            或 ``None``（provider 不可用/异常/输出非法）。
        """
        if not self.available:
            return None
        prompt = _PROMPT.substitute(query=query, history=self._render_history(history))
        try:
            raw = self._complete(prompt)
            data = self._parse(raw)
            return self._validate(data) if data is not None else None
        except Exception as exc:  # noqa: BLE001 - 任何 provider 异常都降级
            logger.warning("LLM 规划失败，降级规则 planner: {}", exc)
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _render_history(history: List[Any]) -> str:
        """把历史消息渲染为可读文本（空历史返回占位符）。"""
        if not history:
            return "(none)"
        lines: List[str] = []
        for msg in history[-6:]:  # 最多 6 条历史，控制 prompt 长度
            if isinstance(msg, dict):
                role = msg.get("role") or msg.get("type") or "?"
                content = msg.get("content")
            else:
                role = getattr(msg, "role", "?")
                content = getattr(msg, "content", str(msg))
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> Optional[dict]:
        """从 LLM 输出提取 JSON（容忍 markdown 代码块与前后杂文）。"""
        raw = raw.strip()
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            raw = m.group(1)
        else:
            start, end = raw.find("{"), raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                return None
            raw = raw[start:end + 1]
        return json.loads(raw)

    @staticmethod
    def _validate(data: Any) -> Optional[dict]:
        """校验并归一输出结构；非法返回 None。"""
        if not isinstance(data, dict):
            return None
        sub_tasks = data.get("sub_tasks")
        if not isinstance(sub_tasks, list):
            return None
        sub_tasks = [str(t).strip() for t in sub_tasks if str(t).strip()]
        if not sub_tasks:
            return None
        claims = data.get("claims") or {}
        if not isinstance(claims, dict):
            claims = {}
        calc_spec = data.get("calculation_spec")
        return {
            "sub_tasks": sub_tasks,
            "claims": claims,
            "calculation_spec": calc_spec if isinstance(calc_spec, dict) else None,
        }