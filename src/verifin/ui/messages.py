"""人性化诊断消息（业务语义中文模板，零技术黑话/零 Traceback）。

面向终端用户；开发者细节仍保留在 ``verify_flags`` 与日志中（双受众分离）。
"""

from __future__ import annotations

from typing import Optional

_METRIC_LABELS_ZH = {
    "revenue": "营收",
    "sales": "销售收入",
    "operating_margin": "营业利润率",
    "operating income": "营业利润",
    "gross_margin": "毛利率",
    "gross profit": "毛利",
    "net_margin": "净利率",
    "net_income": "净利润",
    "EBITDA": "EBITDA",
    "cost": "成本费用",
    "profit": "净利润",
    "growth_rate": "增速",
}

_SOURCE_LABELS_ZH = {
    "annual_report": "年报",
    "annual report": "年报",
    "filing": "年报",
    "10-k": "年报",
    "news": "新闻稿",
    "press release": "新闻稿",
    "research": "第三方研报",
    "analyst": "第三方研报",
}


def source_label(name: Optional[str]) -> str:
    """来源偏好值 → 中文名（未收录原样返回）。"""
    if not name:
        return ""
    return _SOURCE_LABELS_ZH.get(str(name), str(name))


def metric_label(name: Optional[str]) -> str:
    """指标 canonical key → 中文名；未收录原样返回。"""
    if not name:
        return ""
    return _METRIC_LABELS_ZH.get(str(name), str(name))


def preference_confirmation(dialog: Optional[dict]) -> Optional[str]:
    """生成偏好确认消息（Synthesizer 成功路径前置一行）；无偏好返回 None。

    - 来源偏好："已按您偏好的「年报」口径呈现结果。"
    - 合并口径："已采用合并报表口径。"
    """
    if not dialog:
        return None
    parts: list = []
    if dialog.get("source_preference"):
        label = source_label(str(dialog["source_preference"]))
        parts.append(f"已按您偏好的「{label}」口径呈现结果。")
    if dialog.get("consolidation_preference"):
        parts.append("已采用合并报表口径。")
    return " ".join(parts) or None


def friendly_message(code: str, **context: str) -> str:
    """按错误码渲染中文业务消息。

    支持错误码：
    MISMATCH_METRIC / MISMATCH_ENTITY / MISMATCH_PERIOD / MISMATCH_DEFINITION
    / NO_EVIDENCE / CONFLICT_MAJOR / RETRY_EXHAUSTED / EVAL_REJECT
    """
    if code == "MISMATCH_METRIC":
        expected = metric_label(context.get("expected"))
        got = metric_label(context.get("got"))
        got_text = f"「{got}」" if got else "其他口径"
        return (
            f"未找到与您查询的「{expected}」完全匹配的字段，"
            f"系统已自动对比了{got_text}，为确保准确性将重新为您检索更精确的数据。"
        )
    if code == "MISMATCH_ENTITY":
        expected = context.get("expected") or ""
        return f"未找到关于「{expected}」的直接证据，请确认公司名称或换用全称后重试。"
    if code == "MISMATCH_PERIOD":
        expected = context.get("expected") or ""
        return f"未检索到 {expected} 期间的数据，已尝试为您查找相邻期间。"
    if code == "MISMATCH_DEFINITION":
        return "证据中的指标口径与您查询的定义不一致，已按最接近的口径重新检索。"
    if code == "NO_EVIDENCE":
        return "未检索到相关数据，建议您缩小时间范围或提供更精确的公司名称。"
    if code == "CONFLICT_MAJOR":
        return "检测到多份文档对同一指标给出了差异较大的数值，为保障准确性，系统已请求合并报表口径下的明确数值。"
    if code == "RETRY_EXHAUSTED":
        return "多次尝试后仍未获得一致证据，建议您补充更详细的问题描述后重试。"
    if code == "EVAL_REJECT":
        reason = str(context.get("reason") or "")
        lowered = reason.lower()
        if "too long" in lowered or "过长" in lowered:
            return "表达式过长，已被安全拦截，请精简后重试。"
        if "too large" in lowered or "过大" in lowered or "exponent" in lowered:
            return "表达式中常量的指数过大，已被安全拦截，请调小数值后重试。"
        if "unsafe" in lowered or "timed out" in lowered or "timeout" in lowered:
            return "表达式包含不安全内容或执行超时，已被安全拦截。"
        return "计算未通过，请检查表达式后重试。"
    return "操作未完成，请稍后重试。"