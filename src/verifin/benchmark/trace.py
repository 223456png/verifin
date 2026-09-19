"""轨迹分析器：Agent 最终 state（hooks / verify_flags / tool_call_history）
→ 行为指标与错误归因（不依赖 LangGraph）。

行为指标口径（Phase 7 §二）：
- ``steps``：state.hooks 的节点访问次数（一次节点访问记 1 步）；
- ``replan_triggered``：hooks 中出现 replanner 节点；
- ``verifier_decision``：最后一条子任务的校验裁决（无校验 → "N/A"）；
- 错误归因优先级：多跳计算缺口 > 检索失败 > 校验拒绝 > 其他。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from verifin.tools.verifier import _DEFAULT_VERIFIER, extract_claim

AnswerValue = Optional[Tuple[float, Optional[str]]]

_GROWTH_KEYWORDS = (
    "growth", "percentage", "change", "increase", "decrease",
    "grew", "grow", "rose", "yoy",
)


def extract_answer_value(state: dict, query: str) -> list:
    """从最终 state 提取 Agent 的答案数值候选列表（不解析回答文本，直接取证据值）。

    优先级：
    0. 计算答案（Phase 8）：difference / fraction / percentage 三种刻度候选
       （FinQA 裸数值 GT 刻度不一，逐一覆盖）——优先于 REJECT 门：
       计算问题的末个子任务常被四要素校验 REJECT，但计算产物仍可作为答案；
    1. 最终裁决 REJECT → 无答案（空列表）；
    2. 冲突仲裁的 reconciled_value（时间/来源加权均值，最权威）；
    3. 裁决非 REJECT 的校验通过的证据值；
    4. 无校验场景（no_verifier 消融）：对检索文档做证据抽取，取与 claim
       指标一致的第一条。
    """
    calc = state.get("calculation_result") or {}
    if isinstance(calc, dict):
        candidates = []
        if calc.get("value") is not None:
            candidates.append((float(calc["value"]), calc.get("unit")))
        if calc.get("fraction") is not None:
            candidates.append((float(calc["fraction"]), None))
        if calc.get("difference") is not None:
            candidates.append((float(calc["difference"]), calc.get("difference_unit")))
        # Phase 9：备选程序组合（候选程序枚举口径，≤2 组合的比率/差值刻度）
        # Phase 12：average 备选的 value 刻度（均值无 fraction/difference）
        for alternate in calc.get("alternates") or []:
            if not isinstance(alternate, dict):
                continue
            if alternate.get("value") is not None:
                candidates.append((float(alternate["value"]), alternate.get("unit")))
            if alternate.get("fraction") is not None:
                candidates.append((float(alternate["fraction"]), None))
            if alternate.get("difference") is not None:
                candidates.append(
                    (float(alternate["difference"]), alternate.get("difference_unit"))
                )
        if candidates:
            return candidates
    flags = state.get("verify_flags") or {}
    flag_values = [flag for flag in flags.values() if isinstance(flag, dict)]
    if flag_values and str(flag_values[-1].get("decision", "")) == "REJECT":
        return []
    for flag in flag_values:
        conflict = flag.get("conflict") if isinstance(flag.get("conflict"), dict) else {}
        if conflict.get("reconciled_value") is not None:
            return [(float(conflict["reconciled_value"]), conflict.get("reconciled_unit"))]
    for flag in flag_values:
        if str(flag.get("decision", "")) == "REJECT":
            continue
        for result in flag.get("results") or []:
            if (
                isinstance(result, dict)
                and result.get("passed")
                and result.get("value") is not None
            ):
                return [(float(result["value"]), result.get("unit"))]
    docs = [doc for doc in (state.get("retrieved_docs") or []) if isinstance(doc, dict)]
    if docs:
        claim_metric = extract_claim(query).get("metric")
        for doc in docs:
            evidence = _DEFAULT_VERIFIER.extractor.extract(doc)
            if evidence.value is None:
                continue
            if claim_metric is None or evidence.metric == claim_metric:
                return [(evidence.value, evidence.unit)]
        for doc in docs:
            evidence = _DEFAULT_VERIFIER.extractor.extract(doc)
            if evidence.value is not None:
                return [(evidence.value, evidence.unit)]
    return []


def extract_trajectory(
    state: dict,
    tools_used: List[str],
    tool_families_used: List[str],
    retrieved_count: int,
    gt_in_retrieved: bool,
    verifier_enabled: bool,
) -> dict:
    """最终 state + 工具轨迹 → 行为指标 dict（写入样本结果的 trajectory 字段）。"""
    hooks = state.get("hooks") or []
    nodes = [str(hook.get("node")) for hook in hooks if isinstance(hook, dict)]
    replan_count = nodes.count("replanner")
    replan_strategies = [
        str(hook.get("replan_strategy"))
        for hook in hooks
        if isinstance(hook, dict) and hook.get("node") == "replanner"
        and hook.get("replan_strategy")
    ]
    decisions = [
        str(flag.get("decision"))
        for flag in (state.get("verify_flags") or {}).values()
        if isinstance(flag, dict) and flag.get("decision")
    ]
    verifier_decision = decisions[-1] if decisions else ("N/A" if verifier_enabled else "disabled")
    return {
        "steps": len(hooks),
        "node_path": nodes,
        "tools_used": tools_used,
        "tool_families": tool_families_used,
        "replan_triggered": replan_count > 0,
        "replan_count": replan_count,
        "replan_strategies": replan_strategies,
        "verifier_decision": verifier_decision,
        "retrieved_count": retrieved_count,
        "gt_in_retrieved": gt_in_retrieved,
    }


def classify_error(result: dict) -> Optional[str]:
    """错误归因（仅对答错的样本调用）。

    优先级：多跳计算缺口 → 检索失败 → 校验拒绝 → 其他。
    """
    if result.get("is_correct"):
        return None
    query = (result.get("query") or "").lower()
    trajectory = result.get("trajectory", {})
    if any(keyword in query for keyword in _GROWTH_KEYWORDS):
        return "calculation_error"
    if not trajectory.get("gt_in_retrieved"):
        return "retrieval_failure"
    decision = str(trajectory.get("verifier_decision") or "")
    if decision == "REJECT" or decision == "N/A":
        return "verifier_reject"
    return "other"
