"""Minimal tool-calling harness（Phase 13）：LLM 自主决定工具调用的通用循环。

与 :mod:`verifin.core.graph` 的固定状态机互补而非替代：

- graph = 确定性路由（规则 planner + 模板执行），可评测、可复现，是 EM 基线；
- harness = 模型驱动循环（OpenAI function calling 协议），模型自己决定
  「调什么工具、什么时候停」，适用于规则流程覆盖不了的开放任务。

设计约束（约 150 行，零框架依赖）：

- 协议最小化：LLM 侧只需实现 ``chat(messages, tools) -> dict | None``——
  返回 ``{"content": str, "tool_calls": [{"id", "name", "args"}]}``，失败返回
  ``None``（与 :meth:`verifin.llm_provider.OpenAICompatLLM.chat` 对齐，
  永不抛异常契约同源）；
- 工具层复用 :class:`verifin.tools.registry.ToolRegistry`，与 graph 节点
  共用同一套工具注册 / 执行 / 历史记录，无第二套抽象；
- 三停机条件：模型不再请求工具（finished）/ 轮数上限（max_turns，防
  死循环）/ LLM 失败（llm_error，answer 为空串由上层决定降级）；
- 工具错误回喂：未知工具 / 执行异常不中断循环，错误以 tool 消息回喂
  给模型自主纠正（ReAct 的核心收益）。

注意 harness 不进 benchmark 基线：graph 的 29.0% EM 是系统成绩，harness 是能力层，
混跑会污染口径。行为契约由 mock provider 单测锁定。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, List, Optional

from verifin.schemas import ToolCall, ToolResult
from verifin.tools.registry import ToolRegistry

#: 单条工具结果回喂模型的最大字符数（SearchResultSet 等大对象截断，防上下文爆炸）。
MAX_TOOL_CONTENT_CHARS = 4000

_STOPPED_FINISHED = "finished"
_STOPPED_MAX_TURNS = "max_turns"
_STOPPED_LLM_ERROR = "llm_error"


@dataclass
class HarnessResult:
    """一次 harness 运行的完整产出（可回放）。

    Attributes:
        answer: 模型最终文本答案（llm_error / max_turns 时可能为空串）。
        stopped_reason: finished | max_turns | llm_error。
        turns: 模型调用轮数（含最终无工具调用的一轮）。
        messages: 完整消息历史（回放 / 调试 / 断言用）。
        tool_calls: 按序发生的全部工具调用（与 registry.history 对齐）。
    """

    answer: str = ""
    stopped_reason: str = _STOPPED_FINISHED
    turns: int = 0
    messages: List[dict] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)


class ToolCallingHarness:
    """模型驱动的工具调用循环。

    用法::

        harness = ToolCallingHarness(llm=OpenAICompatLLM(api_key=...))
        result = harness.run("What was Apple's 2024 revenue?")
        result.answer, result.tool_calls, result.stopped_reason

    Args:
        llm: 实现了 ``chat(messages, tools) -> dict | None`` 的任意对象
            （:class:`OpenAICompatLLM` 或测试 mock）。
        registry: 工具注册表；默认全局单例（与 graph 共用）。
        system_prompt: 系统提示词；None 时用内置默认。
        max_turns: 模型调用轮数上限（防死循环），默认 10。
    """

    def __init__(
        self,
        llm: Any,
        registry: Optional[ToolRegistry] = None,
        system_prompt: Optional[str] = None,
        max_turns: int = 10,
    ) -> None:
        self.llm = llm
        self.registry = registry if registry is not None else ToolRegistry()
        self.system_prompt = (
            system_prompt
            if system_prompt is not None
            else (
                "You are a financial QA assistant. Use the provided tools to "
                "retrieve and verify evidence before answering; cite the "
                "evidence you used."
            )
        )
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------

    def run(self, query: str, tools: Optional[List[dict]] = None) -> HarnessResult:
        """执行一次「模型 ↔ 工具」循环，直到模型给出答案或触发停机条件。

        Args:
            query: 用户问题。
            tools: 工具 schema 列表（``registry.list_tools()`` 形态）；
                None 时用注册表全部工具。
        """
        tools = self.registry.list_tools() if tools is None else tools
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": query})

        result = HarnessResult(messages=messages)

        for turn in range(1, self.max_turns + 1):
            result.turns = turn
            resp = self._chat(messages, tools)
            if resp is None:
                result.stopped_reason = _STOPPED_LLM_ERROR
                return result

            tool_calls = resp.get("tool_calls") or []
            if not tool_calls:
                messages.append(
                    {"role": "assistant", "content": str(resp.get("content") or "")}
                )
                result.answer = str(resp.get("content") or "")
                result.stopped_reason = _STOPPED_FINISHED
                return result

            # assistant 轮：原样记录模型请求的工具调用（OpenAI 消息形态）
            messages.append(
                {
                    "role": "assistant",
                    "content": str(resp.get("content") or "") or None,
                    "tool_calls": [
                        {
                            "id": c.get("id") or uuid.uuid4().hex,
                            "type": "function",
                            "function": {
                                "name": c["name"],
                                "arguments": json.dumps(
                                    c.get("args") or {}, ensure_ascii=False
                                ),
                            },
                        }
                        for c in tool_calls
                    ],
                }
            )

            # 工具轮：逐个执行，结果（或错误）回喂
            for call in tool_calls:
                tc = ToolCall(
                    tool_name=str(call["name"]),
                    args=dict(call.get("args") or {}),
                    call_id=str(call.get("id") or uuid.uuid4().hex),
                )
                tool_result = self.registry.execute(tc)
                result.tool_calls.append(tc)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.call_id,
                        "content": _serialize_tool_result(tool_result),
                    }
                )

        result.stopped_reason = _STOPPED_MAX_TURNS
        return result

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    def _chat(self, messages: List[dict], tools: List[dict]) -> Optional[dict]:
        """调 LLM，防御性兜底：协议外异常按 llm_error 处理，不冒泡。"""
        try:
            resp = self.llm.chat(messages, tools)
        except Exception:  # noqa: BLE001 - harness 层兜底契约
            return None
        if not isinstance(resp, dict):
            return None
        return resp


def _serialize_tool_result(tr: ToolResult) -> str:
    """ToolResult → 回喂模型的 JSON 文本（失败给 error，超长截断）。"""
    payload: dict[str, Any] = (
        {"output": tr.output} if tr.success else {"error": tr.error or "unknown error"}
    )
    text = json.dumps(payload, ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_CONTENT_CHARS:
        text = text[:MAX_TOOL_CONTENT_CHARS] + '..."}'
    return text
