"""失败记忆存储（不依赖 LangGraph）：跨重规划轮次积累失败诊断。

带记忆 vs 无记忆 Replanner 的核心差异（Phase 5 设计 §1.1）：

=====================  =====================
无记忆 Replanner       带记忆 Replanner
=====================  =====================
重复生成相同细化查询    根据失败历史选择差异化策略
重复检索同一批错误文档  排除已失败 chunk（exclude_chunk_ids）
3 次重试重复劳动        重复失败提前退出，逐步收敛
=====================  =====================

``FailureMemory`` 从 Phase 4 的 ``verify_flags`` 结构诊断而来
（results[].mismatches / missing），``FailureMemoryStore`` 提供去重、
淘汰、摘要统计与序列化（``state.failure_history`` 持久化）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import List


def _pillar(mismatch: str) -> str:
    """从 mismatch 文本提取要素名（"metric: expected 'revenue', got 'cost'" → "metric"）。"""
    return mismatch.split(":", 1)[0].strip().lower()


def _dedup(items: List[str]) -> List[str]:
    """去重保序（mismatch 文本可能跨 chunk 重复）。"""
    seen: set = set()
    ordered: List[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


@dataclass
class FailureMemory:
    """一次重规划失败的结构化记忆。

    Attributes:
        turn: 第几次重试（retry_count + 1）。
        sub_task: 失败的子任务文本。
        claim: 该子任务的 claim 四要素（与 Planner 共享抽取）。
        verify_result: 完整的 ``verify_flags[sub_task]`` 结构（Phase 4 契约）。
        mismatches: 不匹配项列表（人类可读，如 "metric: expected 'revenue', got 'cost'"）。
        missing: 证据缺失的要素列表（如 ["metric", "period"]）。
        timestamp: UTC ISO 时间戳。
    """

    turn: int = 0
    sub_task: str = ""
    claim: dict = field(default_factory=dict)
    verify_result: dict = field(default_factory=dict)
    mismatches: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "FailureMemory":
        data = data or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{key: value for key, value in data.items() if key in known})

    def mismatch_pillars(self) -> set:
        """mismatch 文本的要素类别集合（{"metric", "period", ...}）。"""
        return {_pillar(mismatch) for mismatch in self.mismatches}

    def is_different_from(self, other: "FailureMemory") -> bool:
        """判断两次失败是否本质不同。

        规则（Phase 5 设计 §4.1）：**精确 mismatch 文本**集合重叠度 > 80% 视为
        同一类失败（无新诊断信息，应提前退出而非重复尝试）；双方均无 mismatch
        （如纯冲突失败）也视为同类。

        注意按要素类别（metric/period/...）判定会过严：同一要素在不同 chunk 上
        的错配值不同（got 'cost' vs got 'sales'）携带新信息，不应判为重复；
        只有完全相同的失败签名才退出。
        """
        mine = set(self.mismatches)
        theirs = set(other.mismatches)
        if not mine and not theirs:
            return False
        larger = max(len(mine), len(theirs))
        overlap = len(mine & theirs)
        return overlap / larger <= 0.8


class FailureMemoryStore:
    """失败记忆的进程内存储（容量受限的滑动窗口）。

    序列化契约：``to_history`` / ``from_history`` 与
    ``AgentState.failure_history``（List[dict]）互转，供 LangGraph
    checkpoint 跨轮次持久化。
    """

    def __init__(self, max_entries: int = 10) -> None:
        self.entries: List[FailureMemory] = []
        self.max_entries = max_entries

    def add(self, failure: FailureMemory) -> None:
        """追加一条失败记忆；超容量时淘汰最旧条目。"""
        self.entries.append(failure)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)

    def get_recent(self, n: int = 3) -> List[FailureMemory]:
        """最近 n 条失败（按时间升序）。"""
        return self.entries[-n:] if n > 0 else []

    def get_failed_sub_tasks(self) -> List[str]:
        """所有已失败的子任务文本（去重保序，供策略切换判断）。"""
        return _dedup([entry.sub_task for entry in self.entries])

    def get_failed_chunk_ids(self) -> List[str]:
        """四要素校验失败的 chunk id 集合（供检索排除，去重保序）。

        只含 ``passed=False`` 的 chunk：大差异冲突中四要素通过的 chunk
        不在此列（它们是"通过但互相矛盾"，排除应由冲突仲裁处理）。
        """
        chunk_ids: List[str] = []
        for entry in self.entries:
            results = (entry.verify_result or {}).get("results") or []
            for result in results:
                if (
                    isinstance(result, dict)
                    and result.get("passed") is False
                    and result.get("chunk_id")
                ):
                    chunk_ids.append(str(result["chunk_id"]))
        return _dedup(chunk_ids)

    def get_diagnosis_summary(self) -> dict:
        """汇总最近 5 条失败模式，供策略选择。

        Returns:
            ``{"mismatches": {要素: 次数}, "missing": {要素: 次数}}``
            （mismatch 按要素类别计数，如 "metric" 出现 2 次）。
        """
        summary: dict = {"mismatches": {}, "missing": {}}
        for entry in self.entries[-5:]:
            for mismatch in entry.mismatches:
                key = _pillar(mismatch)
                summary["mismatches"][key] = summary["mismatches"].get(key, 0) + 1
            for item in entry.missing:
                summary["missing"][item] = summary["missing"].get(item, 0) + 1
        return summary

    def to_history(self) -> List[dict]:
        """导出为可 JSON 序列化的 List[dict]（写入 state.failure_history）。"""
        return [entry.to_dict() for entry in self.entries]

    @classmethod
    def from_history(cls, history: List[dict]) -> "FailureMemoryStore":
        """从 state.failure_history（List[dict]）重建存储。"""
        store = cls()
        for item in history or []:
            if isinstance(item, dict):
                store.entries.append(FailureMemory.from_dict(item))
        return store
