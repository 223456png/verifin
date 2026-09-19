"""Markdown 报告生成器：消融对比表 + 关键结论 + Agent 行为指标 + 错误分析 + 诚实声明。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from verifin.benchmark.ablation import CONFIG_LABELS
from verifin.benchmark.metrics import ERROR_LABELS

_TARGETS = {
    "replan_hit_rate": "≥ 60%",
    "tool_f1": "≥ 85%",
    "avg_steps": "≤ 4 轮",
}


def _pct(value: Optional[float]) -> str:
    """比例 → 百分数字符串（None → "N/A"）。"""
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def _num(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _pp(delta: Optional[float]) -> str:
    """准确率差值 → 百分点字符串。"""
    if delta is None:
        return "N/A"
    return f"{delta * 100:+.1f}"


def _abs_pp(delta: Optional[float]) -> str:
    """准确率差值 → 绝对值百分点字符串（用于「下降 X pp」表述）。"""
    if delta is None:
        return "N/A"
    return f"{abs(delta) * 100:.1f}"


class ReportGenerator:
    """从摘要 dict 生成报告并写盘（results_dir/report.md）。"""

    def __init__(self, results_dir: Path) -> None:
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, summary: dict) -> str:
        """生成完整 Markdown 报告文本并保存为 report.md。"""
        text = self._render(summary)
        (self.results_dir / "report.md").write_text(text, encoding="utf-8")
        return text

    # ------------------------------------------------------------------

    def _render(self, summary: dict) -> str:
        configs = summary.get("configs", {})
        full = configs.get("full", {})
        no_verifier = configs.get("no_verifier", {})
        no_replanner = configs.get("no_replanner", {})
        no_preferences = configs.get("no_preferences", {})
        no_dense = configs.get("no_dense", {})

        lines: list = []
        lines.append("# VeriFin Phase 8: Benchmark Report")
        lines.append("")
        lines.extend(self._section_config(summary))
        lines.extend(self._section_ablation(configs))
        if summary.get("dataset_source") != "synthetic":
            lines.extend(self._section_answer_types(full, summary))
        lines.extend(self._section_conclusions(
            full, no_verifier, no_replanner, no_preferences, no_dense, summary
        ))
        lines.extend(self._section_behavior(full))
        lines.extend(self._section_generation_quality(full))
        lines.extend(self._section_errors(full))
        lines.extend(self._section_honesty(summary))
        return "\n".join(lines)

    def _section_config(self, summary: dict) -> list:
        run_date = summary.get(
            "run_date", datetime.now(timezone.utc).strftime("%Y-%m-%d")
        )
        return [
            "## 实验配置",
            "",
            f"- 数据集: {summary.get('dataset', 'unknown')}"
            f"（{'内置合成语料' if summary.get('dataset_source') == 'synthetic' else '真实数据'}）",
            f"- 样本数: {summary.get('total_samples', 0)}（每个消融配置）",
            f"- 多轮评估: {'开' if summary.get('multi_turn') else '关'}",
            f"- 运行日期: {run_date}",
            "",
        ]

    def _section_ablation(self, configs: dict) -> list:
        lines = [
            "## 消融实验结果",
            "",
            "| 配置 | 准确率 | 平均步数 | 重规划有效率 | Tool F1 | 多轮准确率 |",
            "|------|--------|----------|-------------|---------|-----------|",
        ]
        order = ["full", "no_verifier", "no_replanner", "no_preferences", "no_dense"]
        for name in order:
            stats = configs.get(name)
            if not stats:
                continue
            lines.append(
                f"| {CONFIG_LABELS.get(name, name)} "
                f"| {_pct(stats.get('accuracy'))} "
                f"| {_num(stats.get('avg_steps'))} "
                f"| {self._replan_cell(name, stats)} "
                f"| {_pct(stats.get('tool_f1'))} "
                f"| {_pct(stats.get('multi_turn_accuracy'))} |"
            )
        lines.append("")
        return lines

    @staticmethod
    def _replan_cell(name: str, stats: dict) -> str:
        """no_verifier 无验证 → 无重规划 → N/A；no_replanner 恒为 0%。"""
        if name == "no_verifier":
            return "N/A"
        if name == "no_replanner":
            return "0%（无 Replanner）"
        return _pct(stats.get("replan_hit_rate"))

    _TYPE_LABELS = {
        "extractive": "提取型（答案在原文）",
        "derived": "推导型（需 program 算术）",
        "boolean": "布尔型（yes/no）",
        "unknown": "未标注",
    }

    def _section_answer_types(self, full: dict, summary: dict) -> list:
        """Phase 8：答案类型分解（诚实口径：规则引擎的能力边界按题型分开陈述）。"""
        type_accuracy = full.get("type_accuracy") or {}
        if not type_accuracy:
            return []
        total = full.get("total", 0)
        lines = [
            "## 答案类型分解（真实数据诚实口径）",
            "",
            "| 类型 | 样本数 | 占评估集 | 准确率 |",
            "|------|--------|---------|--------|",
        ]
        for kind in ("extractive", "derived", "boolean", "unknown"):
            entry = type_accuracy.get(kind)
            if not entry:
                continue
            lines.append(
                f"| {self._TYPE_LABELS.get(kind, kind)} "
                f"| {entry['count']} "
                f"| {entry['count'] / max(total, 1) * 100:.0f}% "
                f"| {_pct(entry['accuracy'])} |"
            )
        calc = full.get("calc_subset") or {}
        if calc.get("count"):
            lines.append(
                f"| 其中：PoT 计算子集（轨迹经 calculator） | {calc['count']} | "
                f"{calc['count'] / max(total, 1) * 100:.0f}% | {_pct(calc['accuracy'])} |"
            )
        lines.append("")
        lines.append(
            "说明：推导型/布尔型需要执行 program 算术（subtract/divide/table_sum/…）"
            "与语义比较，超出当前确定性规则引擎的能力边界；规则引擎的评测重点是"
            "提取型上的检索-校验-抽取链路与 PoT 增长计算。"
        )
        lines.append("")
        return lines

    def _section_conclusions(
        self, full, no_verifier, no_replanner, no_preferences, no_dense, summary: dict
    ) -> list:
        verifier_delta = (
            (full.get("accuracy") or 0.0) - (no_verifier.get("accuracy") or 0.0)
            if full and no_verifier else None
        )
        replanner_delta = (
            (full.get("accuracy") or 0.0) - (no_replanner.get("accuracy") or 0.0)
            if full and no_replanner else None
        )
        dense_delta = (
            (full.get("accuracy") or 0.0) - (no_dense.get("accuracy") or 0.0)
            if full and no_dense else None
        )
        dense_recall_delta = (
            (full.get("retrieval_recall") or 0.0) - (no_dense.get("retrieval_recall") or 0.0)
            if full and no_dense else None
        )
        replanned = full.get("replanned", 0)
        replan_hits = full.get("replan_hits", 0)

        prefs_lines: list = []
        if summary.get("multi_turn"):
            full_mt = full.get("multi_turn_accuracy")
            nopref_mt = no_preferences.get("multi_turn_accuracy")
            if full_mt is not None and nopref_mt is not None:
                prefs_lines.append(
                    f"3. 偏好记忆对多轮体验的提升: 多轮子集准确率 "
                    f"{_pct(full_mt)} → {_pct(nopref_mt)}（下降 "
                    f"{_abs_pp(full_mt - nopref_mt)} pp）"
                )
            else:
                prefs_lines.append("3. 偏好记忆对多轮体验的提升: 本子集无多轮样本，N/A")
        else:
            prefs_lines.append(
                "3. 偏好记忆对多轮体验的提升: 单轮模式下偏好为中性"
                "（符合预期，多轮增益见 --multi-turn 运行）"
            )

        return [
            "## 关键结论",
            "",
            f"1. Verifier 对准确率的贡献: {_pp(verifier_delta)} pp"
            "（完整系统 vs 去掉 Verifier）",
            f"2. Replanner 挽救率: 触发重规划 {replanned} 次，挽救 {replan_hits} 次"
            f"（有效率 {_pct(full.get('replan_hit_rate'))}）；"
            f"去掉 Replanner 后准确率下降 "
            f"{_abs_pp(replanner_delta)} pp",
            *prefs_lines,
            f"4. 稠密检索（Dense）贡献: 完整系统 vs 仅 BM25 = {_pp(dense_delta)} pp"
            f"（文档级召回 recall@20: {_pct(full.get('retrieval_recall'))} vs "
            f"{_pct(no_dense.get('retrieval_recall') if no_dense else None)}，"
            f"增益 {_pp(dense_recall_delta)} pp）",
            "",
        ]

    def _section_behavior(self, full: dict) -> list:
        replan_hit = full.get("replan_hit_rate")
        tool_f1 = full.get("tool_f1")
        steps = full.get("avg_steps")
        replanned = full.get("replanned", 0)

        return [
            "## Agent 行为指标",
            "",
            "| 指标 | 实测 | 目标 | 评价 |",
            "|------|------|------|------|",
            f"| 重规划有效率 | {_pct(replan_hit)} | {_TARGETS['replan_hit_rate']}"
            f" | {self._rate_verdict(replan_hit, replanned)} |",
            f"| 工具调用精准率 (Tool F1) | {_pct(tool_f1)} | {_TARGETS['tool_f1']}"
            f" | {self._f1_verdict(tool_f1)} |",
            f"| 平均轨迹步数 | {_num(steps)} | {_TARGETS['avg_steps']}"
            f" | {self._steps_verdict(steps)} |",
            "",
        ]

    @staticmethod
    def _rate_verdict(value, replanned: int) -> str:
        if value is None or replanned == 0:
            return "无重规划样本（N/A）"
        return "达标 ✓" if value >= 0.6 else "未达标 ✗"

    @staticmethod
    def _f1_verdict(value) -> str:
        if value is None:
            return "N/A"
        return "达标 ✓" if value >= 0.85 else "未达标 ✗"

    @staticmethod
    def _steps_verdict(value) -> str:
        if value is None:
            return "N/A"
        return "达标 ✓" if value <= 4 else "未达标 ✗"

    def _section_generation_quality(self, full: dict) -> list:
        """RAGAs 风格生成质量指标（规则近似）：让数字说行业标准语言。"""
        return [
            "## 生成质量指标（RAGAs 风格・规则近似）",
            "",
            "| 指标 | 实测 | 口径 |",
            "|------|------|------|",
            f"| Faithfulness | {_pct(full.get('faithfulness'))} "
            "| 答案数值声明被「通过四要素校验的证据 ∪ PoT 计算产物」支持的比例 |",
            f"| Answer Relevancy | {_pct(full.get('answer_relevancy'))} "
            "| 问题内容词（去停用词 + 数值 token）在答案中的覆盖率 |",
            "",
            "说明：两指标为 RAGAs 同名指标的**确定性规则近似**（非 LLM-as-judge），",
            "优点是零外部依赖、可复现、可进 CI；与 LLM 评测的相关性未标定，",
            "适用于跨配置相对比较（消融表中各配置同口径计算），不宜与 LLM 评测",
            "绝对值直接对标（诚实边界）。",
            "",
        ]

    def _section_errors(self, full: dict) -> list:
        counts = full.get("error_counts") or {}
        wrong = full.get("total", 0) - full.get("correct", 0)
        lines = ["## 错误分析", ""]
        if not counts or wrong <= 0:
            lines.append("本子集无错误样本。")
            lines.append("")
            return lines
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        top = ranked[:3]
        for rank, (kind, count) in enumerate(top, start=1):
            share = count / wrong * 100 if wrong else 0.0
            lines.append(
                f"{rank}. {ERROR_LABELS.get(kind, kind)}: {count} 例"
                f"（占错误答案 {share:.0f}%）"
            )
        lines.append("")
        return lines

    def _section_honesty(self, summary: dict) -> list:
        real = summary.get("dataset_source") != "synthetic"
        return [
            "## 诚实声明",
            "",
            "1. Tool F1 使用 pseudo-gold 标签（规则生成，非人工标注）："
            "增长类多跳问题要求 calc 工具族，其余样本要求 retrieve + verify；"
            "工具名按「工具族」归一（verify_claim 与 verify_claim_batch 同族）",
            f"2. 本报告样本数 N={summary.get('total_samples', 0)}"
            f"（实际总样本数 {summary.get('dataset_total', 'N/A')}），"
            "如受计算资源限制未全量运行",
            "3. 准确率口径 = 证据数值 vs ground_truth 数值（单位归一 + 1% 容差），"
            "不解析回答文本；REJECT 子任务的证据值不作为答案",
            (
                f"4. 真实数据评测（{summary.get('dataset', 'real')}，共 {summary.get('dataset_total', 'N/A')} 题）：索引经 "
                "scripts/build_index.py 由语料构建；gold 文档级召回以 doc_id 命中计；"
                "答案类型按 gold program/原文数值命中标注（提取/推导/布尔），"
                "推导型与布尔型超出规则引擎能力边界，如实报告"
                if real else
                "4. 语料为内置合成语料（无外网环境下 Kaggle FinQA 不可用时的离线回退），"
                "真实 FinQA（2,789 文档 / 1,147 问答）评测结果见 results/finqa_full"
            ),
            "5. Agent 为确定性规则引擎（无 LLM 调用）；Phase 8 已将 PoT 计算器接入主循环"
            "（增长/百分比多跳 → 年份锚定取证 → calc_expression 求增长率），"
            "计算轨迹可见 calculator 节点与 calc 工具族调用",
            "6. 重规划有效率口径：触发 replanner 的样本中最终裁决非 REJECT 的比例"
            "（PARTIAL 计入挽救成功）",
            "",
        ]
