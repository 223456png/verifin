"""FinQA program DSL 模板执行器（Phase 9）。

思路对齐 FinQA SOTA（EMNLP 2021 官方与后续 PoT 工作）：数值推理题的答案由
program（``subtract(5829, 5735), divide(#0, 5735)`` 等）在表格/文本数值上
执行得出。本项目为确定性规则引擎，用「问题模板 → ProgramSpec → 候选数值
组合枚举 + 合理性剪枝」近似 program 生成；表达式执行复用 Phase 4 的
:class:`ExpressionCalculator`（PoT 受限安全求值，AST 白名单三层防御）。

诚实边界（报告必引）：
- 模板覆盖 ``growth_pct``（双年份增长率/占比变化）与 ``difference``
  （单年份净变动，基期推断为上一年，FinQA 高频模式）两类；
- ``table_sum`` / ``exp_avg`` / ``greater`` 等聚合与比较算子未覆盖，
  属 LLM program 生成范畴（Phase 10 规划）；
- 每年候选数值 ≤3（按到年份提及的字符距离排序），组合 ≤9，任一刻度
  命中 GT 记正确——候选程序枚举是 PoT 方法的确定性近似口径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from verifin.tools.calculator import calc_expression
from verifin.tools.unit_parser import normalize_to_base

# growth 模板：计算关键词 + 双年份（from X to Y / compare to / between X and Y）
_GROWTH_KEYWORDS = (
    "growth", "percentage", "increase", "decrease", "change",
    "grew", "grown", "rose", "yoy", "compare", "difference",
)
# difference 模板：净变动关键词 + 单年份（基期 = 上一年，FinQA 高频模式）
_DIFFERENCE_KEYWORDS = (
    "net change", "change", "difference", "decrease", "increase",
)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")
_REVERSE_RE = re.compile(r"compar(?:e|ed)?\s+to|relative\s+to|versus|\bvs\b")

# 合理性剪枝：|增长率| 上限（2000%——财报典型变动远小于此，超限视为锚定错误）
_MAX_ABS_GROWTH = 20.0
# 每年/每组合的候选上限（表格感知 + 距离锚定混合候选，4×4=16 组合内可枚举）
_MAX_CANDIDATES_PER_YEAR = 4


@dataclass
class ProgramSpec:
    """从问题中检测出的计算程序规格。"""

    kind: str                      # "growth_pct" | "difference"
    base_period: str               # 基期年份（"2014"）
    target_period: str             # 比较期年份（"2015"）
    metric: Optional[str] = None   # planner 填充（claim 四要素）
    entity: Optional[str] = None
    # FinQA 惯例（Phase 9）："percentage DECREASE from 2009 to 2010" 的
    # program 为 subtract(2009值, 2010值), divide(#0, 2009值)——降幅取正，
    # 与标准方向 (target-base)/base 相反
    reversed: bool = False


@dataclass
class ValueCandidate:
    """单个数值候选（来源：校验证据或年份锚定扫描）。"""

    value: float
    unit: Optional[str]
    period: str
    chunk_id: Optional[str] = None
    # 锚定分（越小越可信：0 = 四要素校验通过的证据；>0 = 到年份提及的字符距离）
    anchor_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "chunk_id": self.chunk_id,
        }


@dataclass
class ProgramResult:
    """程序执行结果（primary 组合 + 备选组合的多刻度答案）。"""

    kind: str
    expression: Optional[str] = None
    value: Optional[float] = None          # 百分比刻度（fraction × 100）
    unit: Optional[str] = None             # "%"
    fraction: Optional[float] = None       # 比率刻度（FinQA 裸数值 GT 常见）
    difference: Optional[float] = None     # 差值刻度
    difference_unit: Optional[str] = None
    base: Optional[dict] = None
    target: Optional[dict] = None
    alternates: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    evidence: List[Optional[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "expression": self.expression,
            "value": self.value,
            "unit": self.unit,
            "fraction": self.fraction,
            "difference": self.difference,
            "difference_unit": self.difference_unit,
            "base": self.base,
            "target": self.target,
            "alternates": self.alternates,
            "error": self.error,
            "evidence": self.evidence,
        }


def detect_program(query: str) -> Optional[ProgramSpec]:
    """问题 → ProgramSpec（检测不到计算需求返回 None）。

    检测顺序（先具体后一般）：
    1. **growth_pct**：计算关键词 + 两个不同年份（"from 2023 to 2024"）；
       反序表达（"X in 2024 compare to 2023"）交换基期与比较期；
    2. **difference**：净变动关键词 + 单一年份（"net change in X during 2015"）
       → 基期推断为上一年（FinQA 高频模式 ``subtract(cur, prev)``）；
    3. 无关键词或无年份 → None（普通提取路径，防误伤）。
    """
    lowered = (query or "").lower()
    if not lowered.strip():
        return None
    years = list(dict.fromkeys(_YEAR_RE.findall(query or "")))

    # 1. growth_pct：关键词 + 双年份
    if any(keyword in lowered for keyword in _GROWTH_KEYWORDS) and len(years) >= 2:
        if _REVERSE_RE.search(lowered):
            base_period, target_period = years[1], years[0]
        else:
            base_period, target_period = years[0], years[1]
        return ProgramSpec(kind="growth_pct", base_period=base_period,
                           target_period=target_period,
                           reversed="decrease" in lowered)

    # 2. difference：净变动关键词 + 单年份（基期 = 上一年）
    if (
        any(keyword in lowered for keyword in _DIFFERENCE_KEYWORDS)
        and len(years) == 1
    ):
        year = int(years[0])
        if 2000 < year < 2100:
            return ProgramSpec(kind="difference", base_period=str(year - 1),
                               target_period=str(year))
    return None


class ProgramExecutor:
    """候选数值组合枚举 + 单位归一 + 合理性剪枝 + PoT 安全求值。

    Args:
        evaluator: 表达式求值函数（默认直接调 :func:`calc_expression`；
            Agent 节点注入 ToolRegistry 版本以保留 calc 工具族调用轨迹）。
    """

    def __init__(self, evaluator=None) -> None:
        self._evaluator = evaluator or calc_expression

    def execute(
        self,
        spec: ProgramSpec,
        candidates: Optional[Dict[str, List[ValueCandidate]]] = None,
    ) -> ProgramResult:
        """执行程序规格。

        Args:
            spec: 问题检测出的程序规格。
            candidates: ``{period: [ValueCandidate]}``（按可信度升序）；
                缺失任一期的候选 → error 降级。

        Returns:
            :class:`ProgramResult`（primary 组合最优 + ≤2 个 alternates）。
        """
        candidates = candidates or {}

        def _sorted_bucket(period: str) -> List[ValueCandidate]:
            """按 anchor_score 升序截断（稳定排序，防低置信候选挤占名额）。"""
            bucket = list(candidates.get(period) or [])
            bucket.sort(key=lambda c: c.anchor_score)
            return bucket[:_MAX_CANDIDATES_PER_YEAR]

        base_list = _sorted_bucket(spec.base_period)
        target_list = _sorted_bucket(spec.target_period)
        if not base_list or not target_list:
            return ProgramResult(
                kind=spec.kind,
                error="missing base/target evidence values for calculation",
                evidence=[c.chunk_id for lst in candidates.values() for c in lst],
            )

        combos: List[dict] = []
        for base in base_list:
            for target in target_list:
                # 同一 chunk 的同一数值不是有效的双期组合
                if (base.chunk_id == target.chunk_id
                        and base.value == target.value):
                    continue
                evaluated = self._eval_pair(spec, base, target)
                if evaluated is not None:
                    # 跨 chunk 组合惩罚：双期数值通常位于同一表格/文档片段
                    # （FinQA program 的操作数几乎总是同表），跨片段组合多为
                    # 巧合配对——锚分相同时间隔片段的组合应劣后
                    if base.chunk_id and target.chunk_id and base.chunk_id != target.chunk_id:
                        evaluated["anchor_score"] += 1.0
                    combos.append(evaluated)

        if not combos:
            return ProgramResult(
                kind=spec.kind,
                error="no plausible value combination passed sanity pruning",
                evidence=[c.chunk_id for c in base_list + target_list],
            )

        # primary 排序：锚定分（证据 > 近距离扫描）→ 典型增速区间（|pct|≤200%）
        # → 原始顺序（稳定）
        combos.sort(key=lambda c: (
            c["anchor_score"],
            0 if c["fraction"] is not None and abs(c["fraction"]) <= 2.0 else 1,
            c["index"],
        ))
        primary = combos[0]
        result = ProgramResult(
            kind=spec.kind,
            expression=primary["expression"],
            value=primary["value"],
            unit="%",
            fraction=primary["fraction"],
            difference=primary["difference"],
            difference_unit=primary["difference_unit"],
            base=primary["base"].to_dict(),
            target=primary["target"].to_dict(),
            alternates=[
                {
                    "fraction": c["fraction"],
                    "difference": c["difference"],
                    "difference_unit": c["difference_unit"],
                    "base": c["base"].to_dict(),
                    "target": c["target"].to_dict(),
                }
                for c in combos[1:3]
            ],
            evidence=[primary["base"].chunk_id, primary["target"].chunk_id],
        )
        return result

    # ------------------------------------------------------------------

    def _eval_pair(
        self, spec: ProgramSpec, base: ValueCandidate, target: ValueCandidate
    ) -> Optional[dict]:
        """单组合求值：单位归一 → PoT 求值 → 合理性剪枝；不可行返回 None。"""
        base_norm = normalize_to_base(base.value, base.unit)
        target_norm = normalize_to_base(target.value, target.unit)
        if base_norm is None or target_norm is None:
            return None  # 未知单位不可比较
        if base_norm[1] != target_norm[1]:
            return None  # 量纲冲突（% vs million）
        bv, tv = base_norm[0], target_norm[0]
        if abs(bv) < 1e-9:
            return None  # 零基期无法求增长率

        # 方向（Phase 9）：标准 (target-base)/|base|；"decrease" 措辞按 FinQA
        # 惯例取 (base-target)/|base|（降幅为正）
        left, right = (bv, tv) if spec.reversed else (tv, bv)
        expression = f"({left} - {right}) / abs({bv})"
        out = self._evaluator(expression)
        out = out if isinstance(out, dict) else {}
        fraction = out.get("value")
        if out.get("is_valid") is False or fraction is None:
            return None
        if abs(float(fraction)) > _MAX_ABS_GROWTH:
            return None  # 合理性剪枝：增长率超 2000% 视为锚定错误

        difference = left - right
        # 差值单位：双方原始单位一致时保留原单位口径，否则用归一口径
        if base.unit == target.unit:
            difference_unit = base.unit
        else:
            difference_unit = base_norm[1] if base_norm[1] == "%" else None
        return {
            "expression": expression,
            "value": round(float(fraction) * 100, 4),
            "fraction": round(float(fraction), 6),
            "difference": round(difference, 4),
            "difference_unit": difference_unit,
            "base": base,
            "target": target,
            "anchor_score": base.anchor_score + target.anchor_score,
            "index": 0,
        }
