"""带记忆的 Replanner（不依赖 LangGraph）：失败诊断 → 策略选择 → 模板化子任务生成。

与 Planner 共享同一套 claim 抽取（``tools.verifier.extract_claim``），
保证：
1. Planner 生成的子任务 → Verifier 可直接校验；
2. Replanner 生成的子任务 → 携带完整四要素，Verifier 可直接校验；
3. 同一问题多轮生成的一致性。

确定性规则全链路（无 LLM）：策略选择基于失败记忆的摘要统计
（如 metric 不匹配连续出现 ≥2 次 → metric_refinement），
模板渲染见 :mod:`verifin.prompts.replanner_prompts`。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from loguru import logger

from verifin.prompts.replanner_prompts import STRATEGY_DESCRIPTIONS, render_sub_task
from verifin.tools.failure_memory import FailureMemory, FailureMemoryStore
from verifin.tools.verifier import _source_trust, extract_claim

DEFAULT_MAX_RETRIES = 3


class Replanner:
    """基于失败记忆的重规划器。

    Usage::

        store = FailureMemoryStore.from_history(state["failure_history"])
        plan = Replanner(failure_store=store).replan(
            sub_task=failed_task,
            verify_flags=state["verify_flags"][failed_task],
            retry_count=state["retry_count"],
            max_retries=state["max_retries"],
        )
    """

    def __init__(self, failure_store: Optional[FailureMemoryStore] = None) -> None:
        self.failure_store = failure_store or FailureMemoryStore()

    # ---- 对外入口 -------------------------------------------------------

    def replan(
        self,
        sub_task: str,
        verify_flags: dict,
        retry_count: int,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> dict:
        """基于失败记忆生成新的子任务。

        Args:
            sub_task: 失败的子任务文本。
            verify_flags: ``verify_flags[sub_task]`` 结构（Phase 4 契约，
                含 claim/results/decision/conflict）。
            retry_count: 当前重试次数。
            max_retries: 重试上限。

        Returns:
            ``{"new_sub_task": Optional[str], "strategy": str,
            "exclude_chunk_ids": List[str], "continue": bool,
            "reason": str, "failure": FailureMemory}``
        """
        flags = verify_flags if isinstance(verify_flags, dict) else {}
        claim = self._claim_for(sub_task, flags)
        failure = self._diagnose_failure(sub_task, flags, claim, retry_count)

        should_continue, reason = self._should_continue(failure, retry_count, max_retries)
        if not should_continue:
            self.failure_store.add(failure)  # 终止也记录，供事后轨迹分析
            return {
                "new_sub_task": None,
                "strategy": "give_up",
                "exclude_chunk_ids": [],
                "continue": False,
                "reason": reason,
                "failure": failure,
            }

        self.failure_store.add(failure)
        strategy = self._determine_strategy(failure, self.failure_store.get_diagnosis_summary())
        new_sub_task = self._generate_sub_task(sub_task, claim, strategy)
        exclude_chunk_ids = self._exclude_chunk_ids(failure, strategy)
        logger.info(
            "重规划：turn={} strategy={} task={!r} exclude={}",
            failure.turn, strategy, new_sub_task, exclude_chunk_ids,
        )
        return {
            "new_sub_task": new_sub_task,
            "strategy": strategy,
            "exclude_chunk_ids": exclude_chunk_ids,
            "continue": True,
            "reason": STRATEGY_DESCRIPTIONS.get(strategy, ""),
            "failure": failure,
        }

    # ---- 内部步骤 -------------------------------------------------------

    @staticmethod
    def _claim_for(sub_task: str, verify_flags: dict) -> dict:
        """优先复用 verify_flags 内嵌 claim；否则对子任务文本重新抽取（与 Planner 共享）。"""
        claim = verify_flags.get("claim")
        if isinstance(claim, dict) and claim:
            return dict(claim)
        return dict(extract_claim(sub_task))

    def _diagnose_failure(
        self,
        sub_task: str,
        verify_flags: dict,
        claim: dict,
        retry_count: int,
    ) -> FailureMemory:
        """从 verify_flags 的 results 汇总 mismatches/missing，生成 FailureMemory。"""
        mismatches: list = []
        missing: list = []
        for result in verify_flags.get("results") or []:
            if not isinstance(result, dict):
                continue
            mismatches.extend(str(item) for item in result.get("mismatches") or [])
            missing.extend(str(item) for item in result.get("missing") or [])
        seen_m, seen_x = set(), set()
        unique_mismatches = [m for m in mismatches if not (m in seen_m or seen_m.add(m))]
        unique_missing = [m for m in missing if not (m in seen_x or seen_x.add(m))]
        return FailureMemory(
            turn=retry_count + 1,
            sub_task=sub_task,
            claim=dict(claim or {}),
            verify_result=dict(verify_flags),
            mismatches=unique_mismatches,
            missing=unique_missing,
        )

    def _determine_strategy(self, failure: FailureMemory, summary: dict) -> str:
        """确定性策略选择（优先级从高到低，Phase 5 设计 §5.3）。

        1. 证据缺失要素 → supplement_retrieval（补定义/计算口径）；
        2. 多文档大差异冲突 → consolidation_refinement（合并报表口径定向，
           Phase 5 冲突仲裁联动）；
        3. 指标不匹配且记忆中反复出现（≥2 次）→ metric_refinement；
        4. 期间不匹配 → period_refinement；
        5. 实体不匹配 → entity_clarification；
        6. 已尝试 ≥3 个不同子任务仍失败 → alternative_phrasing（换表述）；
        7. 默认 → general_broadening（广度扩展年报）。
        """
        if failure.missing:
            return "supplement_retrieval"

        conflict = (failure.verify_result or {}).get("conflict")
        if isinstance(conflict, dict) and conflict.get("level") == "major":
            return "consolidation_refinement"

        pillars = failure.mismatch_pillars()
        if "metric" in pillars and summary.get("mismatches", {}).get("metric", 0) >= 2:
            return "metric_refinement"
        if "period" in pillars:
            return "period_refinement"
        if "entity" in pillars:
            return "entity_clarification"

        if len(self.failure_store.get_failed_sub_tasks()) >= 3:
            return "alternative_phrasing"
        return "general_broadening"

    def _generate_sub_task(
        self,
        original_sub_task: str,
        claim: dict,
        strategy: str,
    ) -> str:
        """按策略模板生成携带完整四要素的子任务；claim 全空回退 broader terms。"""
        rendered = render_sub_task(strategy, claim or {})
        if rendered:
            return rendered
        return f"retrieve {original_sub_task} with broader terms"

    def _exclude_chunk_ids(self, failure: FailureMemory, strategy: str) -> List[str]:
        """计算本轮应排除的 chunk id（带记忆：不重复失败方向）。

        - 常规：四要素校验失败（passed=False）的 chunk；
        - ``consolidation_refinement``（大差异冲突）：额外排除**低可信来源**的
          冲突参与项（如新闻稿），让重检索收敛到年报/10-K 等权威口径——
          冲突源排除是「合并报表口径定向」的记忆化实现。
        """
        excludes = [str(item) for item in self.failure_store.get_failed_chunk_ids()]
        if strategy == "consolidation_refinement":
            conflict = (failure.verify_result or {}).get("conflict")
            if isinstance(conflict, dict):
                for item in conflict.get("values") or []:
                    if not isinstance(item, dict) or not item.get("chunk_id"):
                        continue
                    source_type = str(item.get("source_type") or "unknown")
                    if _source_trust(source_type) < 1.0:
                        chunk_id = str(item["chunk_id"])
                        if chunk_id not in excludes:
                            excludes.append(chunk_id)
        return excludes

    def _should_continue(
        self,
        failure: FailureMemory,
        retry_count: int,
        max_retries: int,
    ) -> Tuple[bool, str]:
        """确定性退出条件：重试上限 / 与上一条失败同类（无新信息）。"""
        if retry_count >= max_retries:
            return False, f"retry limit reached ({retry_count}/{max_retries})"
        previous = self.failure_store.get_recent(1)
        if previous and not failure.is_different_from(previous[0]):
            return False, "duplicate failure: no new diagnostic signal"
        return True, ""
