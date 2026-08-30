"""Phase 7 指标计算：准确率 / Tool F1（pseudo-gold）/ 重规划有效率 / 轨迹步数。

**诚实声明（报告必引）**：FinQA 等数据集不附带工具轨迹标签，Tool F1 采用
规则生成的 pseudo-gold；pseudo-gold 按「工具族」归一（``verify_claim`` 与
``verify_claim_batch`` 同属 verify 族），衡量的是**该不该调某类工具**，
而非逐调用参数的精确轨迹。
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from verifin.tools.unit_parser import normalize_to_base, parse_number_with_unit

AnswerValue = Optional[Tuple[float, Optional[str]]]

# 数值比对的相对容差（1%）
DEFAULT_TOLERANCE = 0.01

# 文本数值扫描（answer_in_text 用）：$/%/词单位数值 token
_NUM_TOKEN_RE = re.compile(
    r"\$?\s*(\d[\d,]*\.?\d*)\s*(%|(?:million|billion|[MB])\b)?", re.IGNORECASE
)

# 触发「需要计算」伪标签的多跳关键词
_GROWTH_KEYWORDS = (
    "growth", "percentage", "change", "increase", "decrease",
    "grew", "grow", "rose", "yoy",
)

# 工具名 → 工具族（pseudo-gold 与预测轨迹都先归一到族再比较）
TOOL_FAMILIES = {
    "retrieve": "retrieve",
    "expand_document": "retrieve",  # 父文档补全同属检索族（small-to-big）
    "verify_claim": "verify",
    "verify_claim_batch": "verify",
    "extract_evidence": "extract",
    "calc_expression": "calc",
    "convert_unit": "convert",
    "get_preferences": "preference",
    "set_preference": "preference",
}

# 错误类型 → 人类可读标签（报告/轨迹分析共用）
ERROR_LABELS = {
    "retrieval_failure": "检索失败（gold 证据未召回）",
    "verifier_reject": "校验拒绝（证据在但被拒/冲突未解）",
    "calculation_error": "计算缺口（多跳需计算，未接入计算工具）",
    "other": "其他",
}


def exact_match(
    predicted,
    ground_truth: str,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """数值归一化匹配（多候选 + 单位折算 + 1% 容差）。

    Args:
        predicted: 单个 ``(value, unit)``、候选列表（Phase 8 计算答案多刻度），
            或 None（无答案）。
        ground_truth: 标准答案文本（如 "$12,000 million" / "15%" / 裸数值 "0.09864"）。

    规则（任一候选命中即 True）：
    - 两侧单位都能归一 → 同单位量纲下的相对容差比较（million vs % 不跨类）；
    - 任一侧无单位（FinQA 的裸数值 GT / 无单位证据值）→ 原始数值相对容差比较；
    - 相对差异 ≤ tolerance 视为相等。
    """
    if predicted is None:
        return False
    candidates = predicted if isinstance(predicted, list) else [predicted]
    gt = parse_number_with_unit(ground_truth)
    if gt is None:
        return False
    gt_value, gt_unit = gt
    for candidate in candidates:
        if not candidate:
            continue
        pred_value, pred_unit = candidate
        if pred_value is None:
            continue
        # 优先：单位归一化后同量纲比较（v billion ≡ v×1000 million）
        pred_base = normalize_to_base(float(pred_value), pred_unit)
        gt_base = normalize_to_base(gt_value, gt_unit)
        if pred_base is not None and gt_base is not None and pred_base[1] == gt_base[1]:
            scale = max(abs(pred_base[0]), abs(gt_base[0]), 1e-9)
            if abs(pred_base[0] - gt_base[0]) / scale <= tolerance:
                return True
            continue  # 同量纲但不相等 → 换下一候选
        # 回退：任一侧无单位（FinQA 裸数值 GT / 无单位证据值）→ 原始数值容差比较
        if pred_unit in (None, "") or gt_unit in (None, ""):
            scale = max(abs(float(pred_value)), abs(gt_value), 1e-9)
            if abs(float(pred_value) - gt_value) / scale <= tolerance:
                return True
    return False


def ground_truth_in_text(ground_truth: str, text: str) -> bool:
    """标准答案数值是否以常见写法出现在证据文本中（供错误归因）。

    例：GT "$12,000 million" → 检查 "12,000" / "12000" / "12.0" 等形态。
    """
    parsed = parse_number_with_unit(ground_truth)
    if parsed is None:
        return ground_truth in text
    value, unit = parsed
    forms = {f"{value:g}", f"{value:,.0f}" if value == int(value) else f"{value:g}"}
    if unit == "%":
        forms |= {f"{value:g}%", f"{value:,.0f}%"}
    return any(form in text for form in forms)


def answer_in_text(
    ground_truth: str, text: str, tolerance: float = DEFAULT_TOLERANCE
) -> bool:
    """GT 数值是否以常见写法出现在 gold 文档全文（FinQA 提取/推导题口径）。

    对文本内全部数值 token 做单位归一 + 相对容差比较（任一命中即 True）；
    GT 为 yes/no 等非数值形式返回 False（布尔题不属于数值可答子集）。
    """
    parsed = parse_number_with_unit(ground_truth)
    if parsed is None:
        return False
    gt_value, gt_unit = parsed
    for match in _NUM_TOKEN_RE.finditer(text or ""):
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        unit_raw = (match.group(2) or "").lower()
        unit = "%" if unit_raw == "%" else unit_raw or None
        pred_base = normalize_to_base(value, unit)
        gt_base = normalize_to_base(gt_value, gt_unit)
        if pred_base is not None and gt_base is not None and pred_base[1] == gt_base[1]:
            scale = max(abs(pred_base[0]), abs(gt_base[0]), 1e-9)
            if abs(pred_base[0] - gt_base[0]) / scale <= tolerance:
                return True
            continue
        if unit in (None, "") or gt_unit in (None, ""):
            scale = max(abs(value), abs(gt_value), 1e-9)
            if abs(value - gt_value) / scale <= tolerance:
                return True
    return False


def accuracy(results: List[dict]) -> float:
    """整体准确率（空结果返回 0.0）。"""
    if not results:
        return 0.0
    return sum(1 for result in results if result.get("is_correct")) / len(results)


def generate_pseudo_gold(query: str) -> List[str]:
    """规则生成 pseudo-gold 工具族标签。

    - 所有样本：retrieve + verify（证据必须过校验）；
    - 增长/百分比/变化类多跳问题：额外要求 calc（计算结果而非直接引用）。
    """
    required = ["retrieve", "verify"]
    lowered = (query or "").lower()
    if any(keyword in lowered for keyword in _GROWTH_KEYWORDS):
        required.append("calc")
    return required


def tool_families(tools: List[str]) -> set:
    """工具名列表 → 工具族集合（未知名保留原名）。"""
    return {TOOL_FAMILIES.get(tool, tool) for tool in tools}


def tool_f1(predicted_tools: List[str], gold_tools: List[str]) -> float:
    """工具调用 F1（工具族级；空标要求记为 1.0，见 docstring 的 honest 声明）。"""
    pred_set = set(predicted_tools)
    gold_set = set(gold_tools)
    if not gold_set:
        return 1.0
    precision = len(pred_set & gold_set) / len(pred_set) if pred_set else 0.0
    recall = len(pred_set & gold_set) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def tool_f1_avg(results: List[dict]) -> float:
    """逐样本 Tool F1 的均值（pseudo-gold 由 query 规则生成）。"""
    if not results:
        return 0.0
    scores = [
        tool_f1(
            result.get("trajectory", {}).get("tool_families") or [],
            generate_pseudo_gold(result.get("query", "")),
        )
        for result in results
    ]
    return sum(scores) / len(scores)


_ACCEPTED_DECISIONS = {"ACCEPT", "PARTIAL"}


def replan_hit_rate(results: List[dict]) -> float:
    """重规划有效率：触发 Replan 的样本中，最终校验裁决非 REJECT 的比例。

    口径说明：PARTIAL（至少一条证据通过）计入「挽救成功」。
    """
    replanned = [
        result for result in results
        if result.get("trajectory", {}).get("replan_triggered")
    ]
    if not replanned:
        return 0.0
    hits = [
        result for result in replanned
        if (result.get("trajectory", {}).get("verifier_decision") or "") in _ACCEPTED_DECISIONS
    ]
    return len(hits) / len(replanned)


def avg_trajectory_steps(results: List[dict]) -> float:
    """平均轨迹步数（state.hooks 的节点访问次数；空结果返回 0.0）。"""
    if not results:
        return 0.0
    return sum(
        result.get("trajectory", {}).get("steps", 0) for result in results
    ) / len(results)


def accuracy_multiturn(results: List[dict]) -> float:
    """多轮子集（conversation_id 非空）的准确率；无多轮样本返回 None。"""
    subset = [result for result in results if result.get("conversation_id")]
    if not subset:
        return None
    return accuracy(subset)


def retrieval_recall(results: List[dict]) -> float:
    """文档级召回 recall@k（gold doc_id 出现在检索结果中的样本占比；空结果返回 0.0）。"""
    if not results:
        return 0.0
    return sum(
        1 for result in results
        if result.get("trajectory", {}).get("gt_in_retrieved")
    ) / len(results)


def summarize_results(results: List[dict]) -> dict:
    """单配置结果 → 摘要指标（含错误归因计数 + Phase 8 答案类型分解）。"""
    errors: dict = {}
    replanned = 0
    replan_hits = 0
    decisions = _ACCEPTED_DECISIONS
    for result in results:
        trajectory = result.get("trajectory", {})
        if trajectory.get("replan_triggered"):
            replanned += 1
            if (trajectory.get("verifier_decision") or "") in decisions:
                replan_hits += 1
        if not result.get("is_correct"):
            label = (result.get("error_type") or "other")
            errors[label] = errors.get(label, 0) + 1
    type_accuracy: dict = {}
    for type_name in ("extractive", "derived", "boolean", "unknown"):
        subset = [result for result in results if result.get("gold_answer_type") == type_name]
        if subset:
            type_accuracy[type_name] = {
                "count": len(subset),
                "accuracy": accuracy(subset),
            }
    calc_rows = [
        result for result in results
        if "calculator" in (result.get("trajectory", {}).get("node_path") or [])
    ]
    summary = {
        "accuracy": accuracy(results),
        "avg_steps": avg_trajectory_steps(results),
        "replan_hit_rate": replan_hit_rate(results),
        "tool_f1": tool_f1_avg(results),
        "retrieval_recall": retrieval_recall(results),
        "total": len(results),
        "correct": sum(1 for result in results if result.get("is_correct")),
        "replanned": replanned,
        "replan_hits": replan_hits,
        "error_counts": errors,
        "multi_turn_accuracy": accuracy_multiturn(results),
        "type_accuracy": type_accuracy,
        "calc_subset": {
            "count": len(calc_rows),
            "accuracy": accuracy(calc_rows),
        },
    }
    return summary