"""ToolRegistry：Agent 工具的线程安全单例注册表。

领域工具（retriever/verify_claim/calc_expression）均为纯函数，本模块只负责
注册、查找与执行记录，不依赖 LangGraph。
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from verifin.schemas import ToolCall, ToolResult


class ToolRegistry:
    """线程安全单例工具注册表。

    用法::

        registry = ToolRegistry()
        registry.register("calc", lambda expr: eval_ast(expr), "计算表达式", schema)
        result = registry.execute(ToolCall(tool_name="calc", args={...}, call_id=...))
    """

    _instance: Optional["ToolRegistry"] = None
    _lock = threading.Lock()
    _schema: dict = {"type": "object"}

    def __new__(cls) -> "ToolRegistry":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._tools: Dict[str, dict] = {}
                    instance._history: List[ToolCall] = []
                    cls._instance = instance
        return cls._instance

    def register(
        self, name: str, func: Callable, description: str, schema: dict
    ) -> None:
        """注册工具（同名覆盖，便于测试注入 stub）。"""
        self._tools[name] = {
            "func": func,
            "description": description,
            "schema": schema or self._schema,
        }

    def get(self, name: str) -> Optional[Callable]:
        """按名取工具函数；未注册返回 None。"""
        tool = self._tools.get(name)
        return tool["func"] if tool else None

    def list_tools(self) -> List[dict]:
        """返回工具描述列表（供 LLM function calling 使用）。"""
        return [
            {"name": name, "description": tool["description"], "parameters": tool["schema"]}
            for name, tool in self._tools.items()
        ]

    def execute(self, call: ToolCall) -> ToolResult:
        """执行工具调用，记录历史与耗时；失败不抛出。"""
        tool = self._tools.get(call.tool_name)
        if tool is None:
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=f"Tool {call.tool_name} not found",
            )
        start = time.perf_counter()
        try:
            output = tool["func"](**call.args)
            duration_ms = (time.perf_counter() - start) * 1000.0
            self._history.append(call)
            return ToolResult(
                call_id=call.call_id, success=True, output=output, duration_ms=duration_ms
            )
        except Exception as exc:
            logger.warning("工具 {} 执行失败: {}", call.tool_name, exc)
            return ToolResult(
                call_id=call.call_id,
                success=False,
                error=str(exc),
                duration_ms=(time.perf_counter() - start) * 1000.0,
            )

    def history(self) -> List[ToolCall]:
        """已执行工具调用的有序快照（不修改内部状态；轨迹分析用）。"""
        return list(self._history)

    def reset(self) -> None:
        """清空全部工具与历史（测试隔离用）。"""
        self._tools.clear()
        self._history.clear()


def get_preferences(dialog: dict) -> dict:
    """会话偏好读取工具（Phase 6）：返回当前已设置的非空偏好键值对。"""
    return {
        key: value
        for key, value in (dialog or {}).items()
        if key.endswith("_preference") and value is not None
    }


def set_preference(dialog: dict, key: str, value: Any, source: str = "explicit") -> dict:
    """会话偏好写入工具（Phase 6）：设置一项偏好，返回更新后的 dialog_state dict。

    key ∈ {entity, metric, period, source, consolidation}；来源默认 explicit（可追溯）。
    """
    updated = dict(dialog or {})
    updated[f"{key}_preference"] = value
    sources = dict(updated.get("preference_source") or {})
    sources[key] = source
    updated["preference_source"] = sources
    return updated


def register_builtin_tools() -> None:
    """注册内置工具（Agent 初始化时调用；同名覆盖便于测试注入 stub）。"""
    from verifin.tools.retriever import retrieve

    registry = ToolRegistry()
    registry.register(
        name="retrieve",
        func=retrieve,
        description="Retrieve relevant document chunks for a query (BM25 + Dense + RRF + rerank)",
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
                "use_reranker": {"type": "boolean"},
                "filter_metadata": {"type": "object"},
            },
            "required": ["query"],
        },
    )
    register_verifier_tools()
    registry.register(
        name="get_preferences",
        func=get_preferences,
        description="Read current session preferences from dialog state (Phase 6)",
        schema={"type": "object", "properties": {"dialog": {"type": "object"}}},
    )
    registry.register(
        name="set_preference",
        func=set_preference,
        description="Set a session preference (key: entity/metric/period/source/consolidation)",
        schema={
            "type": "object",
            "properties": {
                "dialog": {"type": "object"},
                "key": {"type": "string"},
                "value": {},
                "source": {"type": "string"},
            },
            "required": ["dialog", "key", "value"],
        },
    )


def register_verifier_tools() -> None:
    """注册 Phase 4/5 校验/计算/证据抽取/单位换算工具（由 register_builtin_tools 调用）。"""
    from verifin.tools.calculator import calc_expression
    from verifin.tools.evidence import extract_evidence
    from verifin.tools.unit_parser import convert_unit
    from verifin.tools.verifier import verify_claim, verify_claim_batch

    registry = ToolRegistry()
    registry.register(
        name="verify_claim",
        func=verify_claim,
        description=(
            "Verify a claim against a single evidence chunk using "
            "entity/period/metric/definition matching; returns passed + mismatches"
        ),
        schema={
            "type": "object",
            "properties": {
                "claim": {"type": "object"},
                "chunk": {"type": "object"},
            },
            "required": ["claim", "chunk"],
        },
    )
    registry.register(
        name="verify_claim_batch",
        func=verify_claim_batch,
        description=(
            "Verify a claim against evidence chunks in batch; returns per-chunk results "
            "plus ACCEPT/PARTIAL/REJECT decision for the sub-task"
        ),
        schema={
            "type": "object",
            "properties": {
                "claim": {"type": "object"},
                "chunks": {"type": "array"},
            },
            "required": ["claim", "chunks"],
        },
    )
    registry.register(
        name="calc_expression",
        func=calc_expression,
        description="Safely evaluate a numeric expression in a sandbox (PoT calculator)",
        schema={"type": "object", "properties": {"expr": {"type": "string"}}},
    )
    registry.register(
        name="extract_evidence",
        func=extract_evidence,
        description="Extract structured evidence (entity/period/metric/value) from a document chunk",
        schema={"type": "object", "properties": {"chunk": {"type": "object"}}},
    )
    registry.register(
        name="convert_unit",
        func=convert_unit,
        description="Convert a numeric value between units (million/billion/thousand/万/亿; % excluded)",
        schema={
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "from_unit": {"type": "string"},
                "to_unit": {"type": "string"},
            },
            "required": ["value", "from_unit", "to_unit"],
        },
    )