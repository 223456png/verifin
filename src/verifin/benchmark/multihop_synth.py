"""多跳 QA 合成管线（Phase 9.6）：从原子种子事实合成跨文档多跳评测基准。

流程（对标业界 multi-hop benchmark 合成思路）：
1. **种子抽取**：从带来源标识（chunk_id）的原子 chunk 抽取
   ``(company, metric, year, value)`` 事实；指标不在词典内的种子被拒
   （语义校验的一部分，如 capex）；
2. **跨文档候选配对**：同实体跨年份（增长链）/ 同指标跨实体（比较链）
   两种配对方向；
3. **按推理深度合并**：
   - 2-hop ``growth_chain``：实体内两期数值 → 增长率（program 算术）；
   - 2-hop ``cross_company_diff``：两实体同期数值 → 差值（program 算术）；
   - 3-hop ``argmax_relay``：三实体同期比较取最大 → 再查胜者的第二指标
     （比较 + 二次检索链）；
4. **四重校验**（任一不过即丢弃，计数进 stats）：
   a. 语义校验：模板槽位完整（实体/指标/年份齐全且指标在词典内）；
   b. 推理校验：gold 由 FinQA 风格 program 在种子事实上确定性推出；
   c. 来源校验：gold_evidence 跨 ≥2（2-hop）/ ≥3（3-hop）个不同文档；
   d. 反伪多跳校验（算术推导型）：gold 数值不与任何单一 seed chunk 的
      现成数值等价（答案可从单文档现成读出的是伪多跳；数值级比较而非
      子串匹配，避免年份 "2023" 误含 "20"）。

合成全程确定性（固定遍历序，无随机数），基准可复现、可进 CI。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List, Optional, Tuple

from verifin.benchmark.dataset import synth_chunks
from verifin.tools.unit_parser import parse_number_with_unit

# 种子事实抽取："<Company> <metric> in <year> was <value> according to ..."
# value 取到 "according to" 前的完整文本（含 "$12,000 million" 的千分位逗号）
_FACT_RE = re.compile(
    r"^(?P<company>[A-Z]\w+)\s+(?P<metric>[a-z][a-z ]*?)\s+in\s+(?P<year>20\d{2})\s+"
    r"was\s+(?P<value>.+?)\s+according to",
)

# 语义校验：指标必须在词典内（capex 等词典外指标种子被拒）
_KNOWN_METRICS = {
    "revenue", "operating margin", "gross margin", "net profit",
}

# 生成上限（确定性遍历 + 截断，保证基准规模稳定）
_MAX_PER_TYPE = {"growth_chain": 6, "cross_company_diff": 8, "argmax_relay": 4}


@dataclass
class SeedFact:
    """原子种子事实（带来源标识）。"""

    company: str
    metric: str
    year: int
    value: float
    unit: Optional[str]
    chunk_id: str


@dataclass
class SynthStats:
    """合成管线统计（校验通过/拒绝计数，供报告与测试断言）。"""

    seeds_total: int = 0
    seeds_rejected: int = 0
    candidates: int = 0
    rejected: Dict[str, int] = field(default_factory=dict)
    accepted: int = 0

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1


# ---------------------------------------------------------------------------
# 1. 种子抽取（带来源标识）
# ---------------------------------------------------------------------------

def extract_seed_facts(chunks=None) -> Tuple[List[SeedFact], SynthStats]:
    """从语料 chunk 抽取种子事实（词典外指标种子被拒并计数）。"""
    stats = SynthStats()
    facts: List[SeedFact] = []
    for chunk in chunks if chunks is not None else synth_chunks():
        stats.seeds_total += 1
        match = _FACT_RE.match(chunk.content.strip())
        if not match:
            stats.seeds_rejected += 1
            continue
        metric = match.group("metric").strip()
        if metric not in _KNOWN_METRICS:  # 语义校验：指标词典
            stats.seeds_rejected += 1
            continue
        parsed = parse_number_with_unit(match.group("value"))
        if parsed is None:
            stats.seeds_rejected += 1
            continue
        facts.append(SeedFact(
            company=match.group("company"),
            metric=metric,
            year=int(match.group("year")),
            value=parsed[0],
            unit=parsed[1],
            chunk_id=chunk.chunk_id,
        ))
    return facts, stats


def _fact_index(facts: List[SeedFact]) -> Dict[Tuple[str, str, int], SeedFact]:
    """(company, metric, year) → SeedFact 索引。"""
    return {(f.company, f.metric, f.year): f for f in facts}


# ---------------------------------------------------------------------------
# 4d. 反伪多跳校验
# ---------------------------------------------------------------------------

def _values_equal(a: float, au: Optional[str], b: float, bu: Optional[str]) -> bool:
    """两数值单位归一 + 1% 容差等价（与 benchmark EM 同口径）。"""
    from verifin.tools.unit_parser import normalize_to_base

    ab, bb = normalize_to_base(a, au), normalize_to_base(b, bu)
    if ab is not None and bb is not None and ab[1] == bb[1]:
        return abs(ab[0] - bb[0]) / max(abs(ab[0]), abs(bb[0]), 1e-9) <= 0.01
    if au in (None, "") or bu in (None, ""):
        return abs(a - b) / max(abs(a), abs(b), 1e-9) <= 0.01
    return False


def _gold_in_single_chunk(gold: str, facts: List[SeedFact]) -> bool:
    """gold 数值是否与任一单一 seed chunk 中的现成数值等价（伪多跳信号）。

    数值级比较而非子串匹配：年份 "2023" 不含 "20" 的误判必须排除；
    chunk 内数值 token 仅含 (年份, 指标值)，逐一做单位归一 + 容差比较。
    """
    parsed = parse_number_with_unit(gold)
    if parsed is None:
        return False
    gold_value, gold_unit = parsed
    for fact in facts:
        if _values_equal(gold_value, gold_unit, fact.value, fact.unit):
            return True
        if gold_unit in (None, "") and abs(gold_value - fact.year) < 1e-9:
            return True  # gold 恰为年份数值 → 单文档可读
    return False


def _fmt(value: float, unit: Optional[str]) -> str:
    """数值 → 常见写法（gold 文本，供 EM 与反伪多跳校验共用）。"""
    if unit == "%":
        return f"{value:.1f}%"
    if value == int(value):
        return f"${value:,.0f} million"
    return f"{value:g}"


# ---------------------------------------------------------------------------
# 3. 按推理深度合并（模板 + 四重校验）
# ---------------------------------------------------------------------------

def _make_sample(
    query: str, gold: str, evidence_facts: List[SeedFact],
    hops: int, synth_type: str, program: str,
) -> dict:
    return {
        "query": query,
        "ground_truth": gold,
        "gold_evidence": [
            {"doc_id": f.chunk_id, "chunk_id": f.chunk_id} for f in evidence_facts
        ],
        "hops": hops,
        "synth_type": synth_type,
        "program": program,
        "gold_answer_type": "derived",
        "conversation_id": None,
    }


def _validate_sample(
    sample: dict, facts: List[SeedFact], stats: SynthStats,
    min_docs: int,
) -> bool:
    """四重校验（b 推理 / c 来源 / d 反伪多跳；a 语义在模板填充时已保证）。

    d 反伪多跳仅适用于**算术推导型**（growth_chain / cross_company_diff，
    gold 是计算产物，若与某单文档现成数值重合即伪多跳）；argmax_relay 的
    答案值必然存在于胜者的第二指标文档（多跳性由三文档比较链保证，
    不能用「答案在单文档出现」判伪）。
    """
    # b. 推理校验：gold 必须可解析为数值（program 产物）
    if parse_number_with_unit(sample["ground_truth"]) is None:
        stats.reject("reasoning_unparseable")
        return False
    # c. 来源校验：gold_evidence 跨文档数达到跳数要求
    doc_ids = {e["doc_id"] for e in sample["gold_evidence"]}
    if len(doc_ids) < min_docs:
        stats.reject("source_span_insufficient")
        return False
    # d. 反伪多跳校验（仅算术推导型）
    if sample["synth_type"] in ("growth_chain", "cross_company_diff") and \
            _gold_in_single_chunk(sample["ground_truth"], facts):
        stats.reject("pseudo_multihop")
        return False
    return True


def synthesize_multihop_samples(
    facts: Optional[List[SeedFact]] = None,
) -> Tuple[List[dict], SynthStats]:
    """合成多跳评测样本（确定性；返回样本列表与校验统计）。"""
    if facts is None:
        facts, stats = extract_seed_facts()
    else:
        stats = SynthStats(seeds_total=len(facts))
    index = _fact_index(facts)
    companies = sorted({f.company for f in facts})
    samples: List[dict] = []

    # ---- 2-hop growth_chain：同实体同指标跨年份 → 增长率 ----
    for company in companies:
        for metric in sorted({f.metric for f in facts if f.company == company}):
            years = sorted(
                f.year for f in facts if f.company == company and f.metric == metric
            )
            for y1, y2 in zip(years, years[1:]):
                if len([s for s in samples if s["synth_type"] == "growth_chain"]) >= \
                        _MAX_PER_TYPE["growth_chain"]:
                    break
                base, target = index[(company, metric, y1)], index[(company, metric, y2)]
                if base.value == 0:
                    continue
                growth = (target.value - base.value) / base.value * 100
                gold = _fmt(growth, "%")
                sample = _make_sample(
                    f"What was the percentage growth in {company} {metric} "
                    f"from {y1} to {y2}?",
                    gold, [base, target], hops=2, synth_type="growth_chain",
                    program=f"subtract({target.value:g}, {base.value:g}), "
                            f"divide(#0, {base.value:g}), multiply(#1, 100)",
                )
                stats.candidates += 1
                if _validate_sample(sample, facts, stats, min_docs=2):
                    samples.append(sample)
                    stats.accepted += 1

    # ---- 2-hop cross_company_diff：同指标同年份跨实体 → 差值 ----
    for metric in sorted({f.metric for f in facts}):
        years = sorted({f.year for f in facts if f.metric == metric})
        for year in years:
            rows = sorted(
                (f for f in facts if f.metric == metric and f.year == year),
                key=lambda f: (-f.value, f.company),
            )
            for high, low in combinations(rows, 2):
                if len([s for s in samples if s["synth_type"] == "cross_company_diff"]) >= \
                        _MAX_PER_TYPE["cross_company_diff"]:
                    break
                if high.unit != low.unit:
                    continue  # 单位不一致不可比（语义校验）
                diff = high.value - low.value
                gold = _fmt(diff, high.unit)
                sample = _make_sample(
                    f"How much higher was {high.company} {metric} than "
                    f"{low.company} {metric} in {year}?",
                    gold, [high, low], hops=2, synth_type="cross_company_diff",
                    program=f"subtract({high.value:g}, {low.value:g})",
                )
                stats.candidates += 1
                if _validate_sample(sample, facts, stats, min_docs=2):
                    samples.append(sample)
                    stats.accepted += 1

    # ---- 3-hop argmax_relay：三实体比较取最大 → 胜者第二指标 ----
    metric_primary = "revenue"
    for year in sorted({f.year for f in facts if f.metric == metric_primary}):
        rows = sorted(
            (f for f in facts if f.metric == metric_primary and f.year == year),
            key=lambda f: (-f.value, f.company),
        )
        if len(rows) < 3:
            continue
        top3 = rows[:3]
        winner = top3[0]
        for metric_second in ("gross margin", "operating margin"):
            if len([s for s in samples if s["synth_type"] == "argmax_relay"]) >= \
                    _MAX_PER_TYPE["argmax_relay"]:
                break
            second = index.get((winner.company, metric_second, year))
            if second is None:
                continue
            gold = _fmt(second.value, second.unit)
            query = (
                f"Among {top3[1].company}, {top3[2].company} and {top3[0].company}, "
                f"which company had the highest {metric_primary} in {year}, "
                f"and what was its {metric_second}?"
            )
            sample = _make_sample(
                query, gold, list(top3) + [second], hops=3,
                synth_type="argmax_relay",
                program=(
                    f"greater({top3[0].value:g}, {top3[1].value:g}), "
                    f"greater(#0, {top3[2].value:g}), "
                    f"lookup({metric_second}, winner)"
                ),
            )
            stats.candidates += 1
            if _validate_sample(sample, facts, stats, min_docs=3):
                samples.append(sample)
                stats.accepted += 1

    return samples, stats


class MultihopSynthDataset:
    """合成多跳基准数据集（协议同 Dataset；确定性可复现）。"""

    name = "synthetic_multihop"
    is_synthetic = True

    def __init__(self) -> None:
        samples, self.stats = synthesize_multihop_samples()
        self.queries = samples

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> dict:
        return self.queries[idx]

    def get_batch(self, indices: List[int]) -> List[dict]:
        return [self.queries[i] for i in indices]

    def build_chunks(self):
        return synth_chunks()
