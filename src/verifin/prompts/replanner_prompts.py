"""重规划子任务生成模板（确定性规则，无 LLM 依赖；Phase 7 可升级为 LLM 生成）。

设计要点（Phase 5 修正版设计 §5.4）：
- 每个策略一个模板，生成的子任务**携带完整 claim 四要素**（entity/period/metric），
  Verifier 可直接校验，无需重新抽取；
- 模板与策略选择逻辑分离（数据与控制分离），便于独立测试与 Phase 7 LLM 化替换。
"""

from __future__ import annotations

from typing import Optional

# 策略清单与人类可读描述（供日志 / 轨迹分析 / Phase 7 报告）
STRATEGY_DESCRIPTIONS = {
    "supplement_retrieval": "证据缺失要素 → 补充检索定义/计算口径",
    "consolidation_refinement": "多文档数值大差异 → 定向合并报表口径检索",
    "metric_refinement": "指标反复不匹配（记忆命中）→ 精确指标名细化",
    "period_refinement": "期间不匹配 → 定向期间细化",
    "entity_clarification": "实体不匹配 → 实体财报定向检索",
    "alternative_phrasing": "已尝试方向过多 → 更换表述角度",
    "general_broadening": "默认策略 → 广度扩展到年报全文",
}


def _metric_label(metric: Optional[str]) -> str:
    """canonical 指标名 → 人读形态（operating_margin → operating margin）。"""
    return (metric or "").replace("_", " ").strip()


def render_sub_task(strategy: str, claim: dict) -> Optional[str]:
    """按策略渲染携带完整四要素的重规划子任务。

    Args:
        strategy: 策略名（STRATEGY_DESCRIPTIONS 的键）。
        claim: 四要素字典（entity/period/metric/definition，值可为 None）。

    Returns:
        携带要素的子任务文本；claim 要素全空时返回 None（调用方回退
        "broader terms" 策略，与 Phase 3 行为兼容）。
    """
    entity = str(claim.get("entity") or "").strip()
    period = str(claim.get("period") or "").strip()
    metric = _metric_label(claim.get("metric"))
    definition = str(claim.get("definition") or "").strip()

    # 要素主体：entity + metric + period（Phase 5 既有顺序，检索 token 全覆盖）
    parts = " ".join(part for part in (entity, metric, period) if part)
    if not parts:
        return None

    if strategy == "supplement_retrieval":
        # 缺失要素 → 定向补定义/计算口径（definition 已知则直接带上）
        suffix = f"definition: {definition}" if definition else "calculation method"
        return f"retrieve {parts} {suffix}"
    if strategy == "consolidation_refinement":
        # 大差异冲突 → 要求合并报表口径下的明确数值（Phase 5 冲突联动）
        return f"retrieve {parts} consolidated financial statements specifically"
    if strategy == "metric_refinement":
        return f"retrieve {parts} with exact metric name"
    if strategy == "period_refinement":
        if period:
            return f"retrieve {parts} specifically for {period}"
        return f"retrieve {parts} with exact period"
    if strategy == "entity_clarification":
        head = entity or "the company"
        tail = " ".join(part for part in (period, metric) if part)
        return f"retrieve {head} financial statements for {tail}".rstrip()
    if strategy == "alternative_phrasing":
        base = " ".join(part for part in (entity, period) if part)
        return f"retrieve {base} financial performance including {metric}".strip()
    # general_broadening：放宽到年报全文，但保留全部 claim 要素（§1.2 不变量）
    return f"retrieve {parts} annual report"
