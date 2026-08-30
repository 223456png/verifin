"""Phase 9.6 — 多跳 QA 合成管线（benchmark/multihop_synth.py）测试。

四重校验的确定性锁定：
- 种子抽取：带来源标识、词典外指标（capex）被拒；
- 反伪多跳：gold 与单文档现成数值重合 → 拒（数值级比较，年份不误判）；
- 来源校验：2-hop ≥2 文档 / 3-hop ≥3 文档；
- 推理校验：gold 可解析（growth 数值正确性逐一断言）；
- 合成确定性：两次调用产出完全一致（可复现、可进 CI）。
"""

from __future__ import annotations

from verifin.benchmark.dataset import load_dataset, synth_chunks
from verifin.benchmark.multihop_synth import (
    SeedFact,
    SynthStats,
    extract_seed_facts,
    synthesize_multihop_samples,
)


# 1. 种子抽取：21 条原子事实（21 个指标行 chunk），新闻/合并/干扰 chunk 被拒
def test_extract_seed_facts() -> None:
    facts, stats = extract_seed_facts()
    assert stats.seeds_total == len(synth_chunks()) == 24
    assert len(facts) == 21
    assert stats.seeds_rejected == 3  # 新闻稿 / 合并口径 / helios 干扰
    # 带来源标识：chunk_id 可回溯
    by_key = {(f.company, f.metric, f.year): f for f in facts}
    nt = by_key[("NovaTech", "revenue", 2024)]
    assert nt.value == 12000.0 and nt.unit == "million"
    assert nt.chunk_id == "nt-fy2024-revenue"


# 2. 词典外指标种子被拒（capex 不在指标词典 → 语义校验拒绝）
def test_seed_metric_dictionary_rejects_capex() -> None:
    facts, stats = extract_seed_facts()
    assert all(f.metric != "Energy capex" for f in facts)
    assert all("capex" not in f.metric for f in facts)


# 3. 合成产出：16 条（2-hop 14 / 3-hop 2），类型分布稳定
def test_synthesize_multihop_samples() -> None:
    samples, stats = synthesize_multihop_samples()
    assert stats.accepted == len(samples) == 16
    assert stats.candidates == 19
    assert stats.rejected["pseudo_multihop"] == 3  # 反伪多跳过滤真实生效
    assert {s["hops"] for s in samples} == {2, 3}
    assert sum(1 for s in samples if s["hops"] == 2) == 14
    assert sum(1 for s in samples if s["hops"] == 3) == 2
    types = {s["synth_type"] for s in samples}
    assert types == {"growth_chain", "cross_company_diff", "argmax_relay"}


# 4. 来源校验：gold_evidence 跨文档数 ≥ 跳数
def test_source_span_validation() -> None:
    samples, _ = synthesize_multihop_samples()
    for sample in samples:
        doc_ids = {e["doc_id"] for e in sample["gold_evidence"]}
        assert len(doc_ids) >= sample["hops"], sample["query"]
        # 证据 chunk_id 可回溯到种子
        assert all(e["chunk_id"] == e["doc_id"] for e in sample["gold_evidence"])


# 5. 推理校验：growth_chain 的 gold 数值正确（program 算术可复算）
def test_growth_chain_gold_correctness() -> None:
    samples, _ = synthesize_multihop_samples()
    growth = [s for s in samples if s["synth_type"] == "growth_chain"]
    assert len(growth) == 6
    by_query = {s["query"]: s for s in growth}
    # NovaTech revenue: (12000 - 10000) / 10000 = 20.0%
    nt = by_query[
        "What was the percentage growth in NovaTech revenue from 2023 to 2024?"
    ]
    assert nt["ground_truth"] == "20.0%"
    # program 记录了 FinQA 风格推导链
    assert "subtract(12000, 10000)" in nt["program"].replace(", ", ", ")


# 6. 反伪多跳：gold 与单文档现成数值重合 → 拒
#    （gross margin 差值 44-35=9 与 Quantum operating margin 9% 重合）
def test_pseudo_multihop_rejection() -> None:
    samples, stats = synthesize_multihop_samples()
    assert stats.rejected.get("pseudo_multihop", 0) >= 1
    # 被拒样本不出现在基准中
    queries = {s["query"] for s in samples}
    assert "How much higher was Vertex gross margin than Orion gross margin in 2024?" \
        not in queries


# 7. 数值级比较：年份 "2023" 不含 "20"（子串误判必须排除）
def test_gold_in_single_chunk_numeric_not_substring() -> None:
    from verifin.benchmark.multihop_synth import _gold_in_single_chunk

    facts = [SeedFact("NovaTech", "revenue", 2023, 10000.0, "million", "c1")]
    # 20.0% 是增长率（不在任何 chunk），年份 2023 的子串 "20" 不得误判
    assert _gold_in_single_chunk("20.0%", facts) is False
    # 2023 恰为年份数值 → 单文档可读 → 伪多跳
    assert _gold_in_single_chunk("2023", facts) is True
    # 10000 million 与 chunk 现成值重合 → 伪多跳
    assert _gold_in_single_chunk("$10,000 million", facts) is True


# 8. 3-hop argmax_relay：三实体比较 + 胜者第二指标
def test_argmax_relay_structure() -> None:
    samples, _ = synthesize_multihop_samples()
    relays = [s for s in samples if s["synth_type"] == "argmax_relay"]
    assert len(relays) == 2
    for relay in relays:
        assert relay["hops"] == 3
        assert len({e["doc_id"] for e in relay["gold_evidence"]}) >= 3
        assert "highest revenue" in relay["query"]
        assert "greater(" in relay["program"]


# 9. 合成确定性：两次调用产出一致（基准可复现）
def test_synthesis_deterministic() -> None:
    first, _ = synthesize_multihop_samples()
    second, _ = synthesize_multihop_samples()
    assert first == second


# 10. 数据集工厂接入：synthetic_multihop 可加载且带 stats
def test_load_dataset_multihop() -> None:
    dataset = load_dataset("synthetic_multihop", "data")
    assert dataset.name == "synthetic_multihop"
    assert len(dataset) == 16
    assert dataset[0]["gold_answer_type"] == "derived"
    assert len(dataset.build_chunks()) == 24
    assert dataset.stats.accepted == 16
