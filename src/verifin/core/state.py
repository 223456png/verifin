"""AgentState：ReAct 循环的序列化状态（Pydantic）。

设计要点（Phase 3 设计 Decisions #4/#10）：
- ``next_step`` 是显式路由信号，节点返回局部更新 dict，由 LangGraph 合并；
- ``messages`` 使用 ``add_messages`` reducer 支持多轮对话积累；
- 所有可变容器字段用 default_factory，避免跨实例共享。
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Annotated, Any, Dict, List, Literal

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field, field_validator

from verifin.schemas import ToolCall


def _merge_dialog_state(existing: dict, update: dict) -> dict:
    """dialog_state 合并 reducer：新值覆盖旧值；空更新不覆盖（多轮对话持久化）。"""
    merged = dict(existing or {})
    merged.update(update or {})
    return merged


class AgentState(BaseModel):
    """LangGraph 图的完整状态。

    Attributes:
        messages: 对话历史（LangGraph add_messages reducer 自动合并）。
        retrieved_docs: 当前子任务检索到的文档（dict 形态，含 chunk_id/content/metadata）。
        verify_flags: 四要素校验结果与历史跳过标记（key 为 sub_task 文本）。
            Phase 4 结构（schemas.VerifyFlags 的 dict 形态）：
            ``{"passed": bool, "claim": {...}, "results": [{chunk_id, passed, mismatches, ...}],
            "passed_count", "total_count", "decision": "ACCEPT"|"PARTIAL"|"REJECT"}``；
            与 Phase 3 兼容：值含 ``passed=True`` 即视为已通过（动态路径跳过）。
        tool_call_history: 工具调用记录（Phase 7 轨迹分析数据源）。
        hooks: 可观测钩子（每节点访问追加 {"node", "ts", "retry_count", "sub_task"}）。
        claims: 各子任务的最终 claim（Phase 5 上下文继承）：
            planner 写入 ``claims[sub_task] = {entity, period, metric, definition}``，
            verifier 优先读取（fallback 对子任务文本重抽）。
        dialog_state: 会话级偏好记忆（Phase 6，core.DialogState 的 dict 形态）：
            实体/指标/期间/来源/口径偏好 + 轮次 + preference_source 追溯；
            随 checkpoint 在多轮对话间延续（新线程即新对话，天然重置）。
        failure_history: 重规划失败记忆（Phase 5 修正版，FailureMemory 的 dict 列表）：
            每轮 REJECT 追加一条 {turn, sub_task, claim, verify_result, mismatches,
            missing, timestamp}；随 checkpoint 在同一问题的重试链内累积，
            新用户提问由 planner 重置。
        exclude_chunk_ids: 已四要素校验失败的 chunk id（Phase 5 修正版）：
            重规划后检索过滤这些 chunk（带记忆：不重复检索已失败方向）。
        calculation_requested: 当前提问是否需要数值计算（Phase 8）：
            Planner 检测增长/百分比类多跳问题（关键词 + 双年份）置 True；
        calculation_spec: 计算规格 {"kind": "growth_pct", "base_period": "2023",
            "target_period": "2024"}（Planner 写入，Calculator 消费）。
        calculation_result: Calculator 节点产出 {"expression", "value", "unit",
            "base", "target", "evidence", "error"}；Synthesizer 输出计算答案、
            benchmark 数值提取优先取该值（Phase 8）。
        retry_count: 当前重试次数。
        max_retries: 重试上限，耗尽后降级路由 END。
        sub_tasks: 子任务队列（Planner 生成初始队列，Replanner 可动态追加）。
        current_sub_task_index: 当前执行的子任务下标。
        next_step: 路由信号（planner/retriever/verifier/replanner/synthesizer/end）。
    """

    messages: Annotated[List[Any], add_messages] = Field(default_factory=list)
    retrieved_docs: List[dict] = Field(default_factory=list)
    verify_flags: Dict[str, Any] = Field(default_factory=dict)
    tool_call_history: List[ToolCall] = Field(default_factory=list)
    hooks: List[dict] = Field(default_factory=list)
    claims: Dict[str, dict] = Field(default_factory=dict)
    dialog_state: Annotated[Dict[str, Any], _merge_dialog_state] = Field(default_factory=dict)
    failure_history: List[dict] = Field(default_factory=list)
    exclude_chunk_ids: List[str] = Field(default_factory=list)
    calculation_requested: bool = False
    calculation_spec: Dict[str, Any] = Field(default_factory=dict)
    calculation_result: Dict[str, Any] = Field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    sub_tasks: List[str] = Field(default_factory=list)
    current_sub_task_index: int = 0
    next_step: Literal[
        "planner", "retriever", "verifier", "replanner", "synthesizer", "calculator", "end"
    ] = "planner"

    @field_validator("verify_flags", mode="before")
    @classmethod
    def _normalize_flags(cls, value: Any) -> Any:
        """类型强化：VerifyFlags（dataclass/BaseModel）实例统一归一为 dict 形态。"""
        if not isinstance(value, dict):
            return value

        def to_dict(item: Any) -> Any:
            if isinstance(item, BaseModel):
                return item.model_dump()
            if is_dataclass(item) and not isinstance(item, type):
                return asdict(item)
            return item

        return {str(key): to_dict(item) for key, item in value.items()}

    @field_validator("dialog_state", mode="before")
    @classmethod
    def _normalize_dialog(cls, value: Any) -> Any:
        """类型强化：DialogState（dataclass）实例统一归一为 dict 形态。"""
        if value is None:
            return {}
        if isinstance(value, BaseModel):
            return value.model_dump()
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        return value if isinstance(value, dict) else {}
