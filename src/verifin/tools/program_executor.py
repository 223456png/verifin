"""FinQA program DSL 模板执行器（Phase 9）。

思路对齐 FinQA SOTA（EMNLP 2021 官方与后续 PoT 工作）：数值推理题的答案由
program（``subtract(5829, 5735), divide(#0, 5735)`` 等）在表格/文本数值上
执行得出。本项目为确定性规则引擎，用「问题模板 → ProgramSpec → 候选数值
组合枚举 + 合理性剪枝」近似 program 生成；表达式执行复用 Phase 4 的
:class:`ExpressionCalculator`（PoT 受限安全求值，AST 白名单三层防御）。

诚实边界（报告必引）：
- 确定性模板覆盖五类：``growth_pct``（双年份增长率/占比变化）、``difference``
  （单年份净变动，基期推断为上一年，FinQA 高频模式）、``cross_entity_diff``
  （同期跨实体差值，"How much higher was A than B in 2024"）、
  ``argmax_relay``（多实体比较取极值 + 胜者第二指标接力查找）、``ratio``
  （"what percentage of X is Y" 比率题，占 FinQA test 17%，Phase 10）；
- 模板外的推导型问题由 Phase 10 :class:`LLMProgramGenerator` 生成 FinQA DSL
  程序、:func:`execute_dsl` 多步执行（``#N`` 步骤引用）——LLM 不可用时
  自动降级回确定性模板，行为与 Phase 9 一致；
- 每年/每实体/每短语候选数值 ≤4（按锚定分排序），组合 ≤16，任一刻度
  命中 GT 记正确——候选程序枚举是 PoT 方法的确定性近似口径。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
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

# 跨实体比较模板（Phase 9.7）：比较级 + than（"How much higher was A than B"）
_COMPARATIVE_RE = re.compile(
    r"\b(?:higher|lower|more|less|larger|smaller|greater|bigger)\b[^?.!]*\bthan\b",
)
_DIFF_BETWEEN_RE = re.compile(r"\bdifference\s+between\b")
# argmax 接力模板（Phase 9.7）："which company had the highest <m1> in <year>,
# and what was its <m2>?" —— 比较取极值 + 胜者第二指标查找（3-hop 链）
_ARGMAX_RELAY_RE = re.compile(
    r"\b(?:which|what)\b.{0,60}?\b(?:company|companies|firm|entity)\b"
    r".*?\b(?P<order>highest|largest|greatest|lowest|smallest)\s+"
    r"(?P<metric>[a-z][a-z ]*?)\s+in\s+(?P<year>20\d{2})\b"
    r".*?\bits\s+(?P<metric2>[a-z][a-z ]*?)\s*\?",
    re.IGNORECASE | re.DOTALL,
)
# ratio 模板（Phase 10）："what percentage/percent/portion/fraction of DEN
# is/are NUM" → divide(NUM值, DEN值)——FinQA test 占 17% 的比率题型
_RATIO_RE = re.compile(
    r"\bwhat\s+(?:percentage|percent|portion|fraction)\s+of\s+"
    r"(?P<den>.+?)\s+(?:is|are|was|were|do|does|did)\s+(?P<num>.+?)\s*[?.!]*\s*$",
    re.IGNORECASE | re.DOTALL,
)
# 实体 token 过滤（问句虚词/比较级不作为实体）
_ENTITY_STOPWORDS = {
    "what", "who", "which", "when", "where", "how", "among", "was", "is",
    "were", "are", "did", "do", "does", "had", "has", "have", "the", "a",
    "an", "in", "on", "at", "to", "from", "by", "of", "and", "or", "than",
    "its", "their", "much", "more", "less", "company", "companies", "firm",
    "entity", "highest", "lowest", "largest", "smallest", "greatest",
    "greater", "bigger", "higher", "lower", "percentage", "growth",
}

# 合理性剪枝：|增长率| 上限（2000%——财报典型变动远小于此，超限视为锚定错误）
_MAX_ABS_GROWTH = 20.0
# 每年/每组合的候选上限（表格感知 + 距离锚定混合候选，4×4=16 组合内可枚举）
_MAX_CANDIDATES_PER_YEAR = 4


@dataclass
class ProgramSpec:
    """从问题中检测出的计算程序规格。"""

    kind: str                      # "growth_pct" | "difference" | "cross_entity_diff" | "argmax_relay"
    base_period: str               # 基期年份（"2014"）；cross_entity_diff 为比较年份
    target_period: str             # 比较期年份（"2015"）
    metric: Optional[str] = None   # planner 填充（claim 四要素）
    entity: Optional[str] = None
    # FinQA 惯例（Phase 9）："percentage DECREASE from 2009 to 2010" 的
    # program 为 subtract(2009值, 2010值), divide(#0, 2009值)——降幅取正，
    # 与标准方向 (target-base)/base 相反
    reversed: bool = False
    # cross_entity_diff（Phase 9.7）：比较级两侧的实体（A higher than B → A-B）
    entity_a: Optional[str] = None
    entity_b: Optional[str] = None
    # argmax_relay（Phase 9.7）：参与比较的实体列表 + 胜者第二指标
    entities: List[str] = field(default_factory=list)
    second_metric: Optional[str] = None
    # ratio（Phase 10）：分子短语（部分值，"are leased" 的 leased 侧）与
    # 分母短语（整体值，"of total facilities" 侧）→ divide(NUM, DEN)
    numerator: Optional[str] = None
    denominator: Optional[str] = None


@dataclass
class ValueCandidate:
    """单个数值候选（来源：校验证据或年份锚定扫描）。"""

    value: float
    unit: Optional[str]
    period: str
    chunk_id: Optional[str] = None
    # 锚定分（越小越可信：0 = 四要素校验通过的证据；>0 = 到年份提及的字符距离）
    anchor_score: float = 0.0
    # Phase 10.1：短语锚定来源的行标签/列头（比率流列一致性与 total 行偏好）
    row_label: str = ""
    column: str = ""

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


def _entity_tokens(text: str) -> List[str]:
    """文本中的实体 token（首字母大写、过滤问句虚词/比较级）。"""
    return [
        token for token in re.findall(r"\b[A-Z][A-Za-z0-9]*\b", text or "")
        if token.lower() not in _ENTITY_STOPWORDS
    ]


def _canonical_metric(phrase: Optional[str]) -> Optional[str]:
    """指标短语 → 词典 canonical key（"gross margin" → "gross_margin"）。"""
    if not phrase:
        return None
    from verifin.tools.evidence import _match_metric

    return _match_metric(phrase) or phrase.strip().lower().replace(" ", "_")


def detect_program(query: str) -> Optional[ProgramSpec]:
    """问题 → ProgramSpec（检测不到计算需求返回 None）。

    检测顺序（先具体后一般）：
    1. **argmax_relay**："which company had the highest <m1> in <year>,
       and what was its <m2>?" —— 多实体比较取极值 + 胜者第二指标接力；
    2. **growth_pct**：计算关键词 + 两个不同年份（"from 2023 to 2024"）；
       反序表达（"X in 2024 compare to 2023"）交换基期与比较期；
    3. **ratio**（Phase 10）："what percentage/percent/portion/fraction
       of <整体> is/are <部分>？" → ``divide(部分值, 整体值)``；年份 ≤1
       （双年份归 growth_pct），FinQA test 占 17% 的比率题型；
    4. **cross_entity_diff**：比较级 + than / difference between + 双实体
       （"How much higher was A revenue than B revenue in 2024" → A-B）；
       年份 ≤1 个（双年份归 growth_pct）；
    5. **difference**：净变动关键词 + 单一年份（"net change in X during 2015"）
       → 基期推断为上一年（FinQA 高频模式 ``subtract(cur, prev)``）；
    6. 无关键词或无年份 → None（普通提取路径，防误伤）。
    """
    lowered = (query or "").lower()
    if not lowered.strip():
        return None
    years = list(dict.fromkeys(_YEAR_RE.findall(query or "")))

    # 1. argmax_relay：比较 + 接力查找（最特异，优先检测）
    argmax = _ARGMAX_RELAY_RE.search(query or "")
    if argmax:
        entities = _entity_tokens(query)
        if len(entities) >= 2:
            order = (argmax.group("order") or "").lower()
            return ProgramSpec(
                kind="argmax_relay",
                base_period=argmax.group("year"),
                target_period=argmax.group("year"),
                metric=_canonical_metric(argmax.group("metric")),
                second_metric=_canonical_metric(argmax.group("metric2")),
                entities=entities,
                # lowest/smallest → 取极小值（reversed 复用为方向标记）
                reversed=order in ("lowest", "smallest"),
            )

    # 2. growth_pct：关键词 + 双年份
    if any(keyword in lowered for keyword in _GROWTH_KEYWORDS) and len(years) >= 2:
        if _REVERSE_RE.search(lowered):
            base_period, target_period = years[1], years[0]
        else:
            base_period, target_period = years[0], years[1]
        return ProgramSpec(kind="growth_pct", base_period=base_period,
                           target_period=target_period,
                           reversed="decrease" in lowered)

    # 3. ratio（Phase 10）：比率题型（年份 ≤1，双年份归 growth_pct）
    if len(years) <= 1:
        ratio = _RATIO_RE.search(query or "")
        if ratio:
            year = years[0] if years else ""
            return ProgramSpec(
                kind="ratio",
                base_period=year,
                target_period=year,
                numerator=(ratio.group("num") or "").strip(),
                denominator=(ratio.group("den") or "").strip(),
            )

    # 4. cross_entity_diff：比较级 + 双实体（年份 ≤1，双年份归 growth_pct）
    if len(years) <= 1 and (
        _COMPARATIVE_RE.search(lowered) or _DIFF_BETWEEN_RE.search(lowered)
    ):
        # 按原始大小写切分（实体识别依赖首字母大写）
        if re.search(r"\s+than\s+", lowered):
            head, tail = re.split(r"\s+than\s+", query, maxsplit=1)
        else:
            head, tail = re.split(r"\s+and\s+", query, maxsplit=1)
        entities_a = _entity_tokens(head)
        entities_b = _entity_tokens(tail)
        if entities_a and entities_b:
            year = years[0] if years else ""
            return ProgramSpec(
                kind="cross_entity_diff",
                base_period=year,
                target_period=year,
                entity_a=entities_a[0],
                entity_b=entities_b[0],
            )

    # 5. difference：净变动关键词 + 单年份（基期 = 上一年）
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

        # Phase 9.7：实体键控模板分派（candidates 键为实体名而非年份）
        if spec.kind == "cross_entity_diff":
            return self._execute_cross_entity_diff(spec, candidates)
        if spec.kind == "argmax_relay":
            return self._execute_argmax_relay(spec, candidates)
        # Phase 10：比率模板（candidates 键为 "num" / "den"）
        if spec.kind == "ratio":
            return self._execute_ratio(spec, candidates)

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

    # ------------------------------------------------------------------
    # Phase 10：比率模板（"what percentage of X are Y" → divide(部分, 整体)）

    def _execute_ratio(
        self, spec: ProgramSpec, candidates: Dict[str, List[ValueCandidate]]
    ) -> ProgramResult:
        """比率："what percentage of <整体> are <部分>?" → ``divide(部分, 整体)``。

        candidates 键为 ``"num"`` / ``"den"``（calculator 按分子/分母短语
        锚定扫描收集）。GT 刻度：FinQA 比率题 exe_ans 多为**裸小数**
        （0.14464），部分带 ``multiply(#0, const_100)``（16.94）→ 输出
        fraction（裸）+ value（×100）双刻度，EM 池逐一尝试。

        剪枝：|fraction| > 20 视为锚定错误（财报比率极少超 2000%）；
        负值操作数跳过（部分/整体非负）；
        排序偏好（Phase 10.1）：
        1. 锚定分（分母 total 行加权 −0.35："percentage of X" 的 X 几乎
           总是合计行；分子 total 行 +0.2 反之）；
        2. 列一致性：num/den 同列且列头词与分母短语重叠
           （"total facilities as measured in square feet" → total 列，
           防止取到 US/其他国别列的子值）；
        3. |fraction| ≤ 2（部分 ≤ 整体是常态，反向多为方向配错）。
        """
        _TOTAL_ROW_RE = re.compile(r"(?:^|[^a-z])total\b", re.IGNORECASE)

        def _is_total_row(cand: ValueCandidate) -> bool:
            return bool(_TOTAL_ROW_RE.search(cand.row_label or ""))

        den_words = set(re.findall(r"[a-z]+", (spec.denominator or "").lower()))
        has_total_in_den_phrase = "total" in den_words

        def _adjusted(cand: ValueCandidate, key: str) -> float:
            """角色调整后的有效锚分（total 行偏好/惩罚计入组合排序）。"""
            if _is_total_row(cand) and not has_total_in_den_phrase:
                return cand.anchor_score + (-0.35 if key == "den" else 0.2)
            return cand.anchor_score

        def _bucket(key: str) -> List[ValueCandidate]:
            bucket = list(candidates.get(key) or [])
            # 有效锚分替换 anchor_score：组合层排序直接继承角色偏好
            bucket = [
                replace(c, anchor_score=_adjusted(c, key)) for c in bucket
            ]
            bucket.sort(key=lambda c: c.anchor_score)
            return bucket[:_MAX_CANDIDATES_PER_YEAR]

        num_list, den_list = _bucket("num"), _bucket("den")
        if not num_list or not den_list:
            return ProgramResult(
                kind=spec.kind,
                error="missing numerator/denominator evidence values for ratio",
                evidence=[c.chunk_id for lst in candidates.values() for c in lst],
            )

        combos: List[dict] = []
        for num_c in num_list:
            for den_c in den_list:
                num_norm = normalize_to_base(num_c.value, num_c.unit)
                den_norm = normalize_to_base(den_c.value, den_c.unit)
                if num_norm is None or den_norm is None:
                    continue  # 未知单位
                if num_norm[1] != den_norm[1]:
                    continue  # 量纲冲突（% vs million）
                nv, dv = num_norm[0], den_norm[0]
                if abs(dv) < 1e-9:
                    continue  # 零分母
                if nv < 0 or dv < 0:
                    continue  # 部分/整体非负（Phase 10.1）
                if num_c.chunk_id == den_c.chunk_id and num_c.value == den_c.value:
                    continue  # 同一数值自除恒为 1，非有效组合
                expression = f"({nv} / {dv})"
                out = self._evaluator(expression)
                out = out if isinstance(out, dict) else {}
                fraction = out.get("value")
                if out.get("is_valid") is False or fraction is None:
                    continue
                if abs(float(fraction)) > _MAX_ABS_GROWTH:
                    continue  # 合理性剪枝：比率超 2000% 视为锚定错误
                # 列一致性（Phase 10.1）：num/den 同列且列头词命中分母短语
                col_lower = (den_c.column or "").lower()
                same_column = bool(
                    num_c.column and den_c.column
                    and num_c.column.lower() == col_lower
                )
                column_pref = same_column and bool(
                    set(re.findall(r"[a-z]+", col_lower)) & den_words
                )
                combos.append({
                    "expression": expression,
                    "fraction": round(float(fraction), 6),
                    "num": num_c,
                    "den": den_c,
                    "anchor_score": num_c.anchor_score + den_c.anchor_score,
                    "column_pref": 0 if column_pref else 1,
                    "same_column": 0 if same_column else 1,
                    "index": 0,
                })

        if not combos:
            return ProgramResult(
                kind=spec.kind,
                error="no plausible value combination passed sanity pruning",
                evidence=[c.chunk_id for c in num_list + den_list],
            )

        combos.sort(key=lambda c: (
            c["anchor_score"],
            c["column_pref"],
            c["same_column"],
            0 if abs(c["fraction"]) <= 2.0 else 1,
            c["index"],
        ))
        primary = combos[0]
        return ProgramResult(
            kind=spec.kind,
            expression=primary["expression"],
            value=round(primary["fraction"] * 100, 4),
            unit="%",
            fraction=primary["fraction"],
            base=primary["den"].to_dict(),   # 整体（分母）
            target=primary["num"].to_dict(),  # 部分（分子）
            alternates=[
                {"fraction": c["fraction"]}
                for c in combos[1:3]
            ],
            evidence=[primary["num"].chunk_id, primary["den"].chunk_id],
        )

    # ------------------------------------------------------------------
    # Phase 9.7：实体键控模板（跨实体差值 / argmax 接力）

    def _execute_cross_entity_diff(
        self, spec: ProgramSpec, candidates: Dict[str, List[ValueCandidate]]
    ) -> ProgramResult:
        """跨实体差值："How much higher was A <metric> than B <metric> in <year>"
        → ``subtract(A值, B值)``。candidates 键为实体名。

        双实体数值天然位于不同 chunk（跨文档证据链），故**不施加**
        同 chunk 偏好与跨 chunk 惩罚——这与 growth_pct（操作数几乎总在
        同表）相反，是多跳语义的差异所在。
        """
        def _bucket(entity: Optional[str]) -> List[ValueCandidate]:
            bucket = list(candidates.get(entity or "") or [])
            bucket.sort(key=lambda c: c.anchor_score)
            return bucket[:_MAX_CANDIDATES_PER_YEAR]

        a_list, b_list = _bucket(spec.entity_a), _bucket(spec.entity_b)
        if not a_list or not b_list:
            return ProgramResult(
                kind=spec.kind,
                error="missing entity evidence values for cross-entity diff",
                evidence=[c.chunk_id for lst in candidates.values() for c in lst],
            )

        combos: List[dict] = []
        for cand_a in a_list:
            for cand_b in b_list:
                a_norm = normalize_to_base(cand_a.value, cand_a.unit)
                b_norm = normalize_to_base(cand_b.value, cand_b.unit)
                if a_norm is None or b_norm is None or a_norm[1] != b_norm[1]:
                    continue  # 未知单位 / 量纲冲突（% vs million）
                if cand_a.chunk_id == cand_b.chunk_id and cand_a.value == cand_b.value:
                    continue  # 同一 chunk 的同一数值不是有效的双实体组合
                av, bv = a_norm[0], b_norm[0]
                expression = f"({av} - {bv})"
                out = self._evaluator(expression)
                out = out if isinstance(out, dict) else {}
                diff = out.get("value")
                if out.get("is_valid") is False or diff is None:
                    continue
                combos.append({
                    "expression": expression,
                    "difference": round(float(diff), 4),
                    "difference_unit": (
                        cand_a.unit if cand_a.unit == cand_b.unit
                        else (a_norm[1] if a_norm[1] == "%" else None)
                    ),
                    "base": cand_a,
                    "target": cand_b,
                    "anchor_score": cand_a.anchor_score + cand_b.anchor_score,
                })

        if not combos:
            return ProgramResult(
                kind=spec.kind,
                error="no plausible value combination passed sanity pruning",
                evidence=[c.chunk_id for c in a_list + b_list],
            )

        combos.sort(key=lambda c: c["anchor_score"])
        primary = combos[0]
        return ProgramResult(
            kind=spec.kind,
            expression=primary["expression"],
            difference=primary["difference"],
            difference_unit=primary["difference_unit"],
            base=primary["base"].to_dict(),
            target=primary["target"].to_dict(),
            alternates=[
                {
                    "difference": c["difference"],
                    "difference_unit": c["difference_unit"],
                    "base": c["base"].to_dict(),
                    "target": c["target"].to_dict(),
                }
                for c in combos[1:3]
            ],
            evidence=[primary["base"].chunk_id, primary["target"].chunk_id],
        )

    def _execute_argmax_relay(
        self, spec: ProgramSpec, candidates: Dict[str, List[ValueCandidate]]
    ) -> ProgramResult:
        """argmax 接力："Among A, B and C, which had the highest <m1> in
        <year>, and what was its <m2>?" → ``greater`` 比较链 + 胜者
        ``lookup``。candidates 键为实体名（主指标）与 ``second::<entity>``
        （第二指标）。
        """
        best_per_entity: Dict[str, ValueCandidate] = {}
        for entity in spec.entities:
            bucket = sorted(
                candidates.get(entity) or [], key=lambda c: c.anchor_score
            )
            if bucket:
                best_per_entity[entity] = bucket[0]
        if len(best_per_entity) < 2:
            return ProgramResult(
                kind=spec.kind,
                error="missing entity evidence values for comparison",
                evidence=[c.chunk_id for lst in candidates.values() for c in lst],
            )

        # 单位归一后比较取极值（量纲不一致的实体值不参与比较）
        normalized: Dict[str, tuple] = {}
        for entity, cand in best_per_entity.items():
            norm = normalize_to_base(cand.value, cand.unit)
            if norm is not None:
                normalized[entity] = norm
        if len(normalized) < 2:
            return ProgramResult(
                kind=spec.kind,
                error="dimension mismatch across entity values",
                evidence=[c.chunk_id for c in best_per_entity.values()],
            )
        dimension = next(iter(normalized.values()))[1]
        comparable = {
            e: n[0] for e, n in normalized.items() if n[1] == dimension
        }
        # "lowest/smallest" 取极小值；默认（highest/largest/greatest）取极大值
        winner = (
            min(comparable, key=comparable.get) if spec.reversed
            else max(comparable, key=comparable.get)
        )
        winner_cand = best_per_entity[winner]

        # 胜者第二指标接力查找
        second_bucket = sorted(
            candidates.get(f"second::{winner}") or [], key=lambda c: c.anchor_score
        )
        if not second_bucket:
            return ProgramResult(
                kind=spec.kind,
                error=f"missing winner second-metric evidence for {winner}",
                evidence=[c.chunk_id for c in best_per_entity.values()],
            )
        second = second_bucket[0]
        values_repr = ", ".join(
            f"{entity}={normalize_to_base(best_per_entity[entity].value, best_per_entity[entity].unit)[0]:g}"
            for entity in spec.entities if entity in comparable
        )
        op = "argmin" if spec.reversed else "argmax"
        return ProgramResult(
            kind=spec.kind,
            expression=(
                f"{op}({values_repr}) -> {winner}; "
                f"lookup({spec.second_metric}, {winner})"
            ),
            value=second.value,
            unit=second.unit,
            base=winner_cand.to_dict(),
            target=second.to_dict(),
            evidence=[winner_cand.chunk_id, second.chunk_id],
        )


# ----------------------------------------------------------------------
# Phase 10：FinQA DSL 多步执行器（LLM program 生成的求值底座）
# ----------------------------------------------------------------------

# 算子白名单 → 元数（参数个数）
_DSL_OPS: Dict[str, int] = {
    "add": 2, "subtract": 2, "multiply": 2, "divide": 2, "exp": 2,
    "greater": 2,
    "table_sum": 1, "table_average": 1, "table_max": 1, "table_min": 1,
}
# 聚合算子（参数为值组绑定 tN，展开为算术表达式后过 evaluator）
_DSL_AGG = {"table_sum", "table_average", "table_max", "table_min"}
_DSL_CONST_RE = re.compile(r"^const_(\d+)$")
_DSL_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_DSL_STEP_RE = re.compile(r"^([a-z_]+)\s*\((.*)\)$")


def _split_dsl_steps(program: str) -> List[str]:
    """顶层逗号切分（括号内的逗号不切）。"""
    steps: List[str] = []
    depth, buf = 0, []
    for ch in program:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            steps.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        steps.append(tail)
    return [s for s in steps if s]


def parse_dsl(program: str) -> List[tuple]:
    """解析 FinQA DSL 程序 → ``[(op, [arg, ...]), ...]``。

    形态：``subtract(5829, 5735), divide(#0, 5735)``（步骤间 ``#N`` 引用，
    N 为 0 起的步骤序号）。参数为扁平 token（数字 / ``#N`` / ``const_K`` /
    ``vN`` 候选值 / ``tN`` 值组），不支持嵌套调用（嵌套形态 ``program_re``
    需先展开）。非法结构抛 :class:`ValueError`。
    """
    steps: List[tuple] = []
    for raw in _split_dsl_steps(program or ""):
        match = _DSL_STEP_RE.match(raw)
        if not match:
            raise ValueError(f"malformed DSL step: {raw!r}")
        op = match.group(1)
        args_raw = match.group(2).strip()
        args = [a.strip() for a in args_raw.split(",")] if args_raw else []
        if op not in _DSL_OPS:
            raise ValueError(f"unknown DSL op: {op}")
        if len(args) != _DSL_OPS[op]:
            raise ValueError(f"op {op} expects {_DSL_OPS[op]} args, got {len(args)}")
        steps.append((op, args))
    if not steps:
        raise ValueError("empty DSL program")
    return steps


def execute_dsl(
    program: str,
    bindings: Optional[Dict[str, object]] = None,
    evaluator=None,
) -> dict:
    """执行 FinQA DSL 多步程序（Phase 10 LLM program 生成的求值底座）。

    Args:
        program: ``"subtract(5829, 5735), divide(#0, 5735)"`` 形态的步骤串。
        bindings: ``{"v1": 8.1, ..., "t1": [1.0, 2.0]}``——``vN`` 单值候选、
            ``tN`` 值组（聚合算子参数）。
        evaluator: 表达式求值函数（默认 :func:`calc_expression`；Agent 节点
            注入 ToolRegistry 版本保留 calc 工具族调用轨迹）。

    Returns:
        ``{"value", "expression", "steps": [{"op", "args", "value"}]}``；
        任何非法（解析失败 / 未知算子 / 引用越界 / 除零）→ ``{"error": ...}``
        不抛出（LLM 输出不可信，失败即降级）。

    求值策略：算术算子逐步经 ``evaluator``（PoT 沙箱，保留轨迹）；
    聚合算子展开为算术表达式（``table_sum(t) → (a + b + c)``）后同样过
    evaluator；``greater`` 原生比较（bool 不经 eval）。
    """
    evaluator = evaluator or calc_expression
    bindings = bindings or {}
    try:
        parsed = parse_dsl(program)
    except ValueError as exc:
        return {"error": str(exc)}

    def _resolve_values(arg: str, step_index: int) -> Optional[float]:
        """标量参数 → float；非法返回 None。"""
        if _DSL_NUMBER_RE.match(arg):
            return float(arg)
        if _DSL_CONST_RE.match(arg):
            return float(arg[len("const_"):])
        if arg.startswith("#"):
            if not arg[1:].isdigit() or int(arg[1:]) >= step_index:
                return None  # 引用越界或前向引用
            return results[int(arg[1:])]
        value = bindings.get(arg)
        return float(value) if isinstance(value, (int, float)) else None

    results: List[float] = []
    records: List[dict] = []
    final_expression = ""
    for step_index, (op, args) in enumerate(parsed):
        if op in _DSL_AGG:
            # 聚合算子：参数须为值组绑定 tN
            group = bindings.get(args[0])
            if not isinstance(group, list) or not group:
                return {"error": f"{op} expects a non-empty value group: {args[0]}"}
            try:
                values = [float(v) for v in group]
            except (TypeError, ValueError):
                return {"error": f"non-numeric value group: {args[0]}"}
            if op == "table_sum":
                expr = "(" + " + ".join(f"{v:g}" for v in values) + ")"
            elif op == "table_average":
                expr = (
                    "(" + " + ".join(f"{v:g}" for v in values) + ")"
                    f" / {len(values)}"
                )
            else:  # table_max / table_min：原生求值（calc 无 max/min）
                picked = max(values) if op == "table_max" else min(values)
                results.append(float(picked))
                records.append({"op": op, "args": args, "value": float(picked)})
                final_expression = f"{op}({args[0]})"
                continue
            out = evaluator(expr)
            out = out if isinstance(out, dict) else {}
            if out.get("is_valid") is False or out.get("value") is None:
                return {"error": f"{op} evaluation failed: {out.get('error')}"}
            results.append(float(out["value"]))
            records.append({"op": op, "args": args, "value": float(out["value"])})
            final_expression = expr
            continue

        vals = []
        for arg in args:
            resolved = _resolve_values(arg, step_index)
            if resolved is None:
                return {"error": f"unresolvable arg: {arg!r} in {op}(...)"}
            vals.append(resolved)
        if op == "greater":
            picked = 1.0 if vals[0] > vals[1] else 0.0
            results.append(picked)
            records.append({"op": op, "args": args, "value": picked})
            final_expression = f"greater({vals[0]:g}, {vals[1]:g})"
            continue
        a, b = vals
        symbol = {"add": "+", "subtract": "-", "multiply": "*",
                  "divide": "/", "exp": "**"}[op]
        if symbol == "/" and abs(b) < 1e-12:
            return {"error": "division by zero"}
        expr = f"({a:g} {symbol} {b:g})"
        out = evaluator(expr)
        out = out if isinstance(out, dict) else {}
        if out.get("is_valid") is False or out.get("value") is None:
            return {"error": f"{op} evaluation failed: {out.get('error')}"}
        results.append(float(out["value"]))
        records.append({"op": op, "args": args, "value": float(out["value"])})
        final_expression = expr

    return {
        "value": results[-1],
        "expression": final_expression,
        "steps": records,
    }
