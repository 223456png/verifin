"""AgentRunner：图执行的同步/异步封装。

每次 ``run`` 默认分配新的 thread_id（内存 checkpoint 相互隔离），
需要多轮会话时显式传入相同 thread_id 以复用状态。
"""

from __future__ import annotations

import uuid
from typing import Optional

from verifin.core.state import AgentState


class AgentRunner:
    """执行 ReAct Agent 图的轻量封装。"""

    def __init__(self, graph, thread_id: Optional[str] = None) -> None:
        self.graph = graph
        self.default_thread_id = thread_id

    def _config(self, thread_id: Optional[str]) -> dict:
        return {
            "configurable": {
                "thread_id": thread_id or self.default_thread_id or f"run-{uuid.uuid4().hex}"
            }
        }

    def run(
        self,
        query: str,
        initial_state: Optional[AgentState] = None,
        thread_id: Optional[str] = None,
    ) -> dict:
        """同步运行 Agent，返回最终状态 dict（messages 已归一为 dict 形态）。

        Args:
            query: 用户问题。
            initial_state: 可选初始状态（如携带 verify_flags 上下文）。
            thread_id: checkpoint 线程 id（缺省每次运行自动生成新 id）。
        """
        state = initial_state or AgentState()
        if not state.messages:
            state = AgentState(messages=[{"role": "user", "content": query}])
        result = self.graph.invoke(state, config=self._config(thread_id))
        return _normalize_messages(result)

    async def astream(self, query: str, thread_id: Optional[str] = None):
        """异步流式运行（事件流）。"""
        state = AgentState(messages=[{"role": "user", "content": query}])
        async for event in self.graph.astream(state, config=self._config(thread_id)):
            yield event


def _normalize_messages(result: dict) -> dict:
    """把 langchain Message 对象还原为可序列化的 dict 形态（便于消费与测试）。"""
    normalized = []
    for message in result.get("messages") or []:
        if isinstance(message, dict):
            normalized.append(message)
            continue
        role = {
            "human": "user",
            "ai": "assistant",
            "system": "system",
            "tool": "tool",
        }.get(getattr(message, "type", ""), "assistant")
        normalized.append({"role": role, "content": getattr(message, "content", "")})
    result["messages"] = normalized
    return result
