"""四要素校验器：Entity / Period / Metric / Definition 逐项比对（不依赖 LangGraph）。

匹配规则：
- Entity：大小写不敏感精确匹配 → 词边界包含匹配（"Nova" in "NovaTech" 因边界拒绝）；
- Period：双方 4 位年份集合取交集（FY2024 ⊇ 2024）；
- Metric：指标词典 canonical key 相等（同义词归一）；
- Definition：token 交集 / min(双方 token 数) ≥ 0.5。
"""

from __future__ import annotations

import re
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from verifin.schemas import Evidence, VerificationResult
from verifin.tools.evidence import (
    _DEFINITION_RE,
    _PERIOD_RE,
    _SORTED_METRIC_TERMS,
    METRIC_KEYWORDS,
    EvidenceExtractor,
    extract_entity_candidates,
)
from verifin.tools.unit_parser import normalize_to_base

_YEAR_IN_PERIOD_RE = re.compile(r"20\d{2}")
_DEFINITION_OVERLAP_THRESHOLD = 0.5


def _is_table_content(content: str) -> bool:
    """表格 chunk 判定（Phase 11）：markdown 管道行 ≥2 视为表格。

    表格行标签（"revenue | 5829"）无散文四要素文本，四要素校验失败是
    结构性的而非 chunk 无关——据此标记供排除豁免。
    """
    return sum(1 for line in content.splitlines() if line.strip().startswith("|")) >= 2

# ---- Phase 5: 多文档冲突仲裁 ----
_CONFLICT_ROUNDING = 0.02   # ≤2%：四舍五入误差 → 融合均值
_CONFLICT_MAJOR = 0.05      # >5%：强制 Replan（合并报表口径）
_SOURCE_TRUST = {
    "filing": 1.0, "annual report": 1.0, "10-k": 1.0, "10k": 1.0, "sec": 1.0,
    "news": 0.7, "press release": 0.7,
    "research": 0.5, "analyst": 0.5,
    "unknown": 0.6,
}
_YEAR_WEIGHTS = (1.0, 0.5, 0.25)  # 按年份新旧排序：最新/次新/更早

# 偏好来源值 → 证据 source_type 别名组（Phase 6 偏好加权命中判定）
_PREFERENCE_SOURCE_ALIASES = {
    "annual_report": {"filing", "annual report", "10-k", "10k", "sec", "annual_report"},
    "news": {"news", "press release", "news_release"},
    "research": {"research", "analyst", "analyst report"},
}


def _normalize_source_alias(value: Any) -> str:
    """归一来源名（下划线/空格/大小写），落在别名组则返回组键（annual_report/news/research）。"""
    normalized = re.sub(r"[\s_]+", "_", str(value or "").strip().lower())
    for group, members in _PREFERENCE_SOURCE_ALIASES.items():
        if normalized in members:
            return group
    return normalized


def _source_trust(source_type: Any) -> float:
    return _SOURCE_TRUST.get(str(source_type or "unknown").lower(), 0.6)


def _relative_diff(a: float, b: float) -> float:
    """相对差异（除以较大绝对值；对 0/负值也稳定）。"""
    scale = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / scale


def resolve_conflicts(results: List[dict], preferences: Optional[dict] = None) -> dict:
    """多文档同指标异值仲裁（时间 × 来源可信度加权；偏好来源可信度 +0.2）。

    输入：passed 结果列表，每项至少含 ``value``（可选 ``unit``/``period``/``source_type``）。
    ``preferences``：会话偏好 dict（如 ``{"source_preference": "annual_report"}``），
    命中来源别名组时该来源 +0.2 权重并记录 ``preference_note``。
    输出：::

        {"level": "none"|"minor"|"major", "conflict": bool,
         "reconciled_value": Optional[float], "reconciled_unit": Optional[str],
         "note": str, "preference_note": Optional[str], "values": [...]}

    规则：
    - 单位统一折算到基准（million；% 独立口径），不可折算 → major；
    - 相对差异 ≤2% → 视为四舍五入误差，按时间/来源加权融合均值；
    - 2%~5% → minor（保留双值，不伪造数值）；
    - >5% → major（note 要求合并报表口径，reconciled 为 None）。
    """
    default: dict = {
        "level": "none", "conflict": False,
        "reconciled_value": None, "reconciled_unit": None,
        "note": "", "preference_note": None, "values": [],
    }
    preferences = preferences or {}
    preferred_source = _normalize_source_alias(preferences.get("source_preference"))

    numeric: List[dict] = []
    for item in results:
        value = item.get("value") if isinstance(item, dict) else None
        if value is None:
            continue
        try:
            base = normalize_to_base(float(value), item.get("unit"))
        except (TypeError, ValueError):
            continue
        if base is None:
            continue
        match = _YEAR_IN_PERIOD_RE.search(str(item.get("period") or ""))
        numeric.append({
            "chunk_id": item.get("chunk_id"),
            "value": float(value),
            "unit": item.get("unit"),
            "base_value": base[0],
            "base_unit": base[1],
            "year": int(match.group(0)) if match else None,
            "source_type": item.get("source_type", "unknown"),
        })
    default["values"] = numeric
    if len(numeric) < 2:
        return default

    base_units = {item["base_unit"] for item in numeric}
    if len(base_units) > 1:
        default["level"] = "major"
        default["conflict"] = True
        default["note"] = "values use incompatible units; requires consolidated financial statement figure"
        return default

    # 偏好来源加权（来源别名组命中 → 可信度 +0.2）
    if preferred_source:
        matched = False
        for item in numeric:
            if _normalize_source_alias(item.get("source_type")) == preferred_source:
                item["trust_bonus"] = 0.2
                matched = True
        if matched:
            from verifin.ui.messages import source_label  # 延迟导入避免层间耦合扩散

            default["preference_note"] = (
                f"用户偏好「{source_label(preferences['source_preference'])}」，"
                "已上调该来源可信度"
            )

    values = [item["base_value"] for item in numeric]
    max_diff = max(
        _relative_diff(a, b) for i, a in enumerate(values) for b in values[i + 1:]
    )

    if max_diff <= _CONFLICT_ROUNDING:
        # 权重 = 年份权重（新旧排序）×（来源信任度 + 偏好加成）
        years = sorted({item["year"] for item in numeric if item["year"]}, reverse=True)
        for item in numeric:
            year_weight = 0.5 if item["year"] is None else (
                _YEAR_WEIGHTS[years.index(item["year"])] if item["year"] in years else 0.25
            )
            item["weight"] = year_weight * (
                _source_trust(item["source_type"]) + item.get("trust_bonus", 0.0)
            )
        total_weight = sum(item["weight"] for item in numeric) or 1.0
        reconciled = sum(
            item["base_value"] * item["weight"] for item in numeric
        ) / total_weight
        default["reconciled_value"] = round(reconciled, 4)
        default["reconciled_unit"] = numeric[0]["base_unit"]
        default["note"] = "rounding difference; reconciled by time/source weighted mean"
        return default

    if max_diff <= _CONFLICT_MAJOR:
        default["level"] = "minor"
        default["conflict"] = True
        default["note"] = "minor discrepancy kept both values"
        return default
    default["level"] = "major"
    default["conflict"] = True
    default["note"] = "material conflict; requires consolidated financial statement figure"
    return default


def _normalize_metric(text: str) -> Optional[str]:
    """把任意指标表述归一为 canonical key（"operating margin" / "operating_margin" 均可）。"""
    if not text:
        return None
    cleaned = " ".join(text.strip().lower().replace("_", " ").split())
    for canonical in METRIC_KEYWORDS:
        if cleaned == canonical.lower():
            return canonical
    for canonical, synonym in _SORTED_METRIC_TERMS:
        if re.search(r"\b" + re.escape(synonym) + r"\b", cleaned):
            return canonical
    return None


def extract_claim(text: str) -> Dict[str, Optional[str]]:
    """从用户问题/子任务文本中抽取 claim 四要素字典。

    - entity: 首字母大写短语候选的第一个（过滤问句首词）；
    - period: 年份/FY/季度模式的第一个命中；
    - metric: 指标词典 canonical key；
    - definition: "defined as ..." 模式（问题中罕见，None 为主）。
    """
    candidates = extract_entity_candidates(text)
    period_match = _PERIOD_RE.search(text)
    definition_match = _DEFINITION_RE.search(text)
    return {
        "entity": candidates[0] if candidates else None,
        "period": period_match.group(0) if period_match else None,
        "metric": _normalize_metric(text),
        "definition": (
            " ".join(definition_match.group(1).strip().lower().split())
            if definition_match
            else None
        ),
    }


def _entity_match(claim_entity: str, evidence_entity: str) -> bool:
    """大小写不敏感精确匹配，回退词边界包含匹配（双向）。"""
    a = " ".join(claim_entity.strip().lower().split())
    b = " ".join(evidence_entity.strip().lower().split())
    if not a or not b:
        return False
    if a == b:
        return True
    def contained(needle, haystack):
        return re.search(
            r"(?<![a-z0-9])" + re.escape(needle) + r"(?![a-z0-9])", haystack
        ) is not None
    return contained(a, b) or contained(b, a)


def _period_match(claim_period: str, evidence_period: str) -> bool:
    """年份集合交集非空（"2024" in "FY2024" → True；跨年 → False）。"""
    claim_years = {int(year) for year in _YEAR_IN_PERIOD_RE.findall(claim_period)}
    evidence_years = {int(year) for year in _YEAR_IN_PERIOD_RE.findall(evidence_period)}
    if not claim_years or not evidence_years:
        return False
    return bool(claim_years & evidence_years)


def _definition_match(claim_definition: str, evidence_definition: str) -> bool:
    """token 关键词重叠度 ≥ 50%。"""
    claim_tokens = set(re.findall(r"[a-z0-9]+", claim_definition.lower()))
    evidence_tokens = set(re.findall(r"[a-z0-9]+", evidence_definition.lower()))
    if not claim_tokens or not evidence_tokens:
        return False
    overlap = len(claim_tokens & evidence_tokens)
    return overlap / min(len(claim_tokens), len(evidence_tokens)) >= _DEFINITION_OVERLAP_THRESHOLD


class Verifier:
    """规则模式四要素校验器（``mode="llm"`` 预留 Phase 7）。"""

    def __init__(self, mode: str = "rule") -> None:
        self.mode = mode
        self.extractor = EvidenceExtractor()

    def verify(self, claim: dict, evidence: Evidence) -> VerificationResult:
        """对单条证据校验 claim 四要素；任一提供要素不匹配 → REJECT。"""
        mismatches: List[str] = []
        missing: List[str] = []
        checked = 0
        matched = 0

        # Entity
        claim_entity = claim.get("entity")
        entity_match: Optional[bool] = None
        if claim_entity:
            checked += 1
            if not evidence.entity:
                mismatches.append(f"entity: expected '{claim_entity}', not extracted")
                missing.append("entity")
                entity_match = False
            elif _entity_match(str(claim_entity), evidence.entity):
                entity_match = True
                matched += 1
            else:
                mismatches.append(
                    f"entity: expected '{claim_entity}', got '{evidence.entity}'"
                )
                entity_match = False

        # Period
        claim_period = claim.get("period")
        period_match: Optional[bool] = None
        if claim_period:
            checked += 1
            if not evidence.period:
                mismatches.append(f"period: expected '{claim_period}', not extracted")
                missing.append("period")
                period_match = False
            elif _period_match(str(claim_period), evidence.period):
                period_match = True
                matched += 1
            else:
                mismatches.append(
                    f"period: expected '{claim_period}', got '{evidence.period}'"
                )
                period_match = False

        # Metric（canonical key 相等）
        claim_metric = claim.get("metric")
        metric_match: Optional[bool] = None
        if claim_metric:
            checked += 1
            canonical = _normalize_metric(str(claim_metric))
            if not evidence.metric:
                mismatches.append(f"metric: expected '{canonical}', not extracted")
                missing.append("metric")
                metric_match = False
            elif canonical and evidence.metric.lower() == canonical.lower():
                metric_match = True
                matched += 1
            else:
                mismatches.append(
                    f"metric: expected '{canonical or claim_metric}', got '{evidence.metric}'"
                )
                metric_match = False

        # Definition（仅 claim 提供时校验）
        claim_definition = claim.get("definition")
        definition_match: Optional[bool] = None
        if claim_definition:
            checked += 1
            if not evidence.definition:
                mismatches.append(
                    f"definition: expected '{claim_definition}', not extracted"
                )
                missing.append("definition")
                definition_match = False
            elif _definition_match(str(claim_definition), evidence.definition):
                definition_match = True
                matched += 1
            else:
                mismatches.append(
                    f"definition: expected '{claim_definition}', got '{evidence.definition}'"
                )
                definition_match = False

        confidence = round(matched / checked, 4) if checked else 0.0
        return VerificationResult(
            chunk_id=evidence.chunk_id,
            passed=not mismatches,
            entity_match=entity_match,
            period_match=period_match,
            metric_match=metric_match,
            definition_match=definition_match,
            mismatches=mismatches,
            missing=missing,
            confidence=confidence,
        )

    def verify_batch(
        self, claim: dict, evidence_list: List[Evidence]
    ) -> List[VerificationResult]:
        """批量校验多条证据。"""
        return [self.verify(claim, evidence) for evidence in evidence_list]


_DEFAULT_VERIFIER = Verifier()


def _result_to_dict(result: VerificationResult) -> dict:
    data = asdict(result)
    return {
        "chunk_id": data["chunk_id"],
        "passed": data["passed"],
        "entity_match": data["entity_match"],
        "period_match": data["period_match"],
        "metric_match": data["metric_match"],
        "definition_match": data["definition_match"],
        "mismatches": data["mismatches"],
        "missing": data["missing"],
        "confidence": data["confidence"],
    }


def verify_claim(claim: dict, chunk: dict) -> dict:
    """Tool 包装（单条）：claim dict + 原始 chunk dict → 校验结果 dict。"""
    if not any(claim.get(k) for k in ("entity", "period", "metric", "definition")):
        # 四要素全空（如 claim="qwerty 123"）：无任何可校验要素，不得真空通过——
        # 否则任意无意义 claim 配任意证据都会 passed=true，误导 MCP 调用方。
        return {
            "chunk_id": chunk.get("chunk_id", "evidence"),
            "passed": False,
            "entity_match": None,
            "period_match": None,
            "metric_match": None,
            "definition_match": None,
            "mismatches": [
                "claim has no checkable elements "
                "(entity/period/metric/definition all empty)"
            ],
            "missing": ["entity", "period", "metric", "definition"],
            "confidence": 0.0,
        }
    evidence = _DEFAULT_VERIFIER.extractor.extract(chunk)
    return _result_to_dict(_DEFAULT_VERIFIER.verify(claim, evidence))


def verify_claim_batch(
    claim: dict, chunks: List[dict], preferences: Optional[dict] = None
) -> Dict[str, Any]:
    """Tool 包装（批量）：返回写入 ``AgentState.verify_flags[sub_task]`` 的完整结构。

    返回::

        {
            "claim": {...}, "results": [{chunk_id, passed, mismatches, ...}],
            "passed_count": int, "total_count": int,
            "decision": "ACCEPT" | "PARTIAL" | "REJECT",
            "passed": bool,   # Phase 3 兼容语义：decision != "REJECT"
            "confidence": float,
        }
    """
    filtered_chunks = [chunk for chunk in chunks if isinstance(chunk, dict)]
    evidence_list = [
        _DEFAULT_VERIFIER.extractor.extract(chunk) for chunk in filtered_chunks
    ]
    results = _DEFAULT_VERIFIER.verify_batch(claim, evidence_list)
    result_dicts = [_result_to_dict(result) for result in results]

    # Phase 5：结果富化（数值/单位/期间/来源）→ 多文档冲突仲裁
    for result_dict, evidence, chunk in zip(result_dicts, evidence_list, filtered_chunks, strict=False):
        metadata = (chunk.get("metadata") or {}) if isinstance(chunk.get("metadata"), dict) else {}
        result_dict["value"] = evidence.value
        result_dict["unit"] = evidence.unit
        result_dict["period"] = evidence.period
        result_dict["source_type"] = metadata.get("source_type", "unknown")
        # Phase 11：表格 chunk 标记（markdown 管道行 ≥2）——四要素校验对表格
        # 天然过严（行标签无散文四要素文本），下游据此豁免排除（见
        # FailureMemoryStore.get_failed_chunk_ids），防 gold 表格被永久拉黑
        result_dict["is_table"] = _is_table_content(str(chunk.get("content") or ""))

    passed_items = [item for item in result_dicts if item.get("passed")]
    conflict = resolve_conflicts(passed_items, preferences)

    passed_count = sum(1 for result in result_dicts if result["passed"])
    total_count = len(result_dicts)
    if total_count and passed_count == total_count:
        decision = "ACCEPT"
    elif passed_count:
        decision = "PARTIAL"
    else:
        decision = "REJECT"
    # 大差异冲突强制 Replan（即使单条四要素均通过）
    if conflict["level"] == "major":
        claim_has_constraints = any(
            claim.get(key) for key in ("entity", "period", "metric", "definition")
        )
        if claim_has_constraints:
            decision = "REJECT"
        else:
            # Phase 8（真实数据适配）：claim 四要素全空（question 中抽取不到
            # 实体/期间/指标）时，多文档数值必然互异——此时把「融合序首位
            # 证据值」作为建议答案写入 reconciled，不再硬 REJECT（否则真实
            # 语料下无约束问题会被 major 冲突全量否决）。
            if passed_items:
                top = passed_items[0]
                if top.get("value") is not None:
                    conflict["reconciled_value"] = float(top["value"])
                    conflict["reconciled_unit"] = top.get("unit")
                    conflict["note"] = (
                        "unconstrained claim: top-ranked evidence value selected"
                    )
    confidence = max((result["confidence"] for result in result_dicts), default=0.0)
    return {
        "claim": claim,
        "results": result_dicts,
        "passed_count": passed_count,
        "total_count": total_count,
        "decision": decision,
        "passed": decision != "REJECT",
        "confidence": round(confidence, 4),
        "conflict": conflict,
    }
