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
            },
            # 防失控兜底：planner→retriever→verifier→replanner 循环异常时
            # 由递归上限强制终止（LangGraph 抛 GraphRecursionError），run() 捕获
            # 后转为失败答案，保证服务线程不被死循环饿死（GIL 饥饿会拖垮 /health）
            "recursion_limit": 60,
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
        try:
            result = self.graph.invoke(state, config=self._config(thread_id))
        except Exception as exc:
            # GraphRecursionError（步数兜底）等图执行异常 → 转失败答案，
            # 不让异常穿透到 API 层；开发者细节看日志
            from loguru import logger

            logger.warning("Agent 图执行异常（{}），已转为失败答案", type(exc).__name__)
            name = type(exc).__name__
            result = {
                "messages": [
                    {"role": "user", "content": query},
                    {
                        "role": "assistant",
                        "content": (
                            "本次执行超出系统安全步数上限，已自动终止。"
                            "请尝试更明确的问题表述（公司名 + 年份 + 指标）后重试。"
                            f"（{name}）"
                        ),
                    },
                ],
            }
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
