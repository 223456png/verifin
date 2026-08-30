"""Phase 2 — Hybrid Retrieval + Rerank 规格测试。

离线环境事实：sentence-transformers 未安装 → Reranker 走降级透传路径；
hash-dense 为词法级信号，召回对比断言采用「∀query hybrid ≥ bm25 且 ∃query 严格 >」
（设计 Decisions #7 的可辩护形态），严格整体提升将在 bge 模式下（Phase 7）验证。

召回语料构造原理（设计 Decisions #8）：
- GT 文档「短而精确」：hash-dense 的 L2 归一使其余弦相似度占优；
- 干扰文档「长且高频重复查询词」：BM25 的 tf 使其稀疏分占优；
- 语料含 6 个查询、3 组可证明的严格提升案例（Titan 组 / Lunar 组 / 公共词
  revenue 组——BM25 的 tf 淹没 GT、dense 的 L2 归一以短精确文档找回），
  其余案例验证「≥」保持语义。
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from verifin.embedding import HashEmbedding
from verifin.indexing.bm25_index import BM25Index
from verifin.indexing.metadata_store import MetadataStore
from verifin.indexing.vector_index import VectorIndex
from verifin.retrieval.fusion import RRFusion, weighted_fuse
from verifin.retrieval.hybrid_retriever import HybridRetriever
from verifin.retrieval.reranker import Reranker
from verifin.schemas import DocumentChunk, SearchResultSet
from verifin.tools.retriever import get_retriever, retrieve, set_retriever


def _chunk(doc_id: str, content: str, year: int) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=uuid.uuid4().hex,
        doc_id=doc_id,
        doc_name=f"{doc_id}.md",
        content=content,
        page_num=1,
        metadata={"year": year},
    )


# 召回语料：g1/g4 为严格提升案例的 GT（短精确）；d* 为对应干扰（长、高频重复查询词）
def _build_corpus() -> tuple[list[DocumentChunk], dict]:
    g1 = _chunk("doc_g1", "Titan Q4 revenue.", 2024)
    g2 = _chunk("doc_g2", "Orion operating margin for fiscal 2023 was 19.8 percent.", 2023)
    g3 = _chunk("doc_g3", "Meridian gross margin for fiscal 2024 was 47.8 percent.", 2024)
    g4 = _chunk("doc_g4", "Lunar Q2 capex.", 2024)
    titan_distractors = [
        _chunk(
            f"doc_d{i}",
            "Titan " * 6 + "revenue revenue revenue revenue q4 in the latest filing.",
            2024,
        )
        for i in range(1, 11)
    ]
    lunar_distractors = [
        _chunk(
            f"doc_l{i}",
            "Lunar Lunar Lunar capex capex capex q2 in the filing.",
            2024,
        )
        for i in range(1, 11)
    ]
    chunks = [g1, *titan_distractors, g2, g3, g4, *lunar_distractors]
    gt = {
        "g1": g1.chunk_id,
        "g2": g2.chunk_id,
        "g3": g3.chunk_id,
        "g4": g4.chunk_id,
    }
    return chunks, gt


@pytest.fixture()
def hybrid_fixture(tmp_path: Path):
    """在临时目录构建 BM25 + VectorIndex + MetadataStore 的最小混合检索环境。"""
    chunks, gt_ids = _build_corpus()
    bm25 = BM25Index()
    bm25.build(chunks)
    vector = VectorIndex(persist_dir=str(tmp_path / "vec"), embedding=HashEmbedding(dim=384))
    vector.add_chunks(chunks)
    store = MetadataStore(path=str(tmp_path / "meta.db"))
    store.upsert_chunks(chunks)
    hybrid = HybridRetriever(bm25_index=bm25, vector_index=vector, metadata_store=store)
    return {"chunks": chunks, "bm25": bm25, "vector": vector, "store": store,
            "hybrid": hybrid, "gt": gt_ids}


def _recall_at_k(result_ids: list[str], gt_ids: list[str], k: int = 10) -> float:
    top = result_ids[:k]
    return sum(1 for gt in gt_ids if gt in top) / len(gt_ids)


def _ids_of(result_set: SearchResultSet) -> list[str]:
    return [result.chunk_id for result in result_set.results]


# 8.1 两个排序列表 → RRF 融合 → 断言融合后排序正确
def test_rrf_fusion() -> None:
    fusion = RRFusion(k=60)
    lists = [[("a", 9.0), ("b", 8.0)], [("b", 7.0), ("c", 6.0)]]
    fused = fusion.fuse(lists, top_k=3)
    # b 同时出现在两个列表首位 → 融合分最高；a（rank1）高于 c（rank2）
    assert fused[0][0] == "b"
    assert fused[0][1] > fused[1][1] >= fused[2][1]
    assert [cid for cid, _ in fused] == ["b", "a", "c"]


# 8.2 检查 Reranker 可用状态 → 降级透传行为正确
def test_reranker_available() -> None:
    reranker = Reranker(model_name="__unavailable_for_test__")
    assert reranker.available is False  # 懒加载：未触发加载前必然不可用
    out = reranker.rerank("q", [("a", "text a"), ("b", "text b")], top_k=2)
    # 降级契约：保持候选原序、零分透传（有/无 st 环境行为一致）
    assert [cid for cid, _ in out] == ["a", "b"]
    assert all(score == 0.0 for _, score in out)


# 8.3 HybridRetriever.search → 返回 top_k 条且字段完整
def test_hybrid_retriever_search(hybrid_fixture) -> None:
    hybrid: HybridRetriever = hybrid_fixture["hybrid"]
    result = hybrid.search("revenue", top_k=3, use_reranker=False)
    assert isinstance(result, SearchResultSet)
    assert result.query == "revenue"
    assert len(result.results) == 3
    assert result.total_count == 3
    assert result.fusion_method == "rrf"
    assert result.reranker_used is False
    assert result.retrieval_time_ms >= 0
    for item in result.results:
        assert 0.0 <= item.score <= 1.0
        assert item.content
        assert isinstance(item.metadata, dict)


# 8.4 召回对比（关键）：BM25-only vs Hybrid vs Hybrid+Rerank
def test_retrieval_recall_improvement(hybrid_fixture) -> None:
    bm25: BM25Index = hybrid_fixture["bm25"]
    hybrid: HybridRetriever = hybrid_fixture["hybrid"]
    gt = hybrid_fixture["gt"]
    # 6 查询：3 个严格提升（BM25 的 tf 淹没 GT，dense 以短精确文档找回）、
    # 3 个等召回（两路同序，验证「≥」保持语义）
    cases = [
        ("Titan Q4 revenue", [gt["g1"]], True),   # bm25=0.0 → hybrid=1.0（实测）
        ("Lunar Q2 capex", [gt["g4"]], True),     # bm25=0.0 → hybrid=1.0（实测）
        ("revenue", [gt["g1"]], True),            # bm25=0.0 → hybrid=1.0（实测）
        ("Orion operating margin 2023", [gt["g2"]], False),
        ("Meridian gross margin", [gt["g3"]], False),
        ("fiscal margin", [gt["g2"]], False),
    ]
    strict_seen = 0
    for query, gt_ids, expect_strict in cases:
        bm25_ids = [cid for cid, _ in bm25.search(query, top_k=10)]
        hybrid_ids = _ids_of(hybrid.search(query, top_k=10, use_reranker=False))
        rerank_ids = _ids_of(hybrid.search(query, top_k=10, use_reranker=True))
        b = _recall_at_k(bm25_ids, gt_ids)
        h = _recall_at_k(hybrid_ids, gt_ids)
        r = _recall_at_k(rerank_ids, gt_ids)
        assert h >= b, f"{query}: hybrid {h} < bm25 {b}"
        assert r >= h, f"{query}: rerank {r} < hybrid {h}"
        if expect_strict:
            assert h > b, f"{query}: 预期 hybrid({h}) 严格优于 bm25({b})"
            strict_seen += 1
    assert strict_seen == 3, "应存在 3 个严格提升案例"


# 8.5 元数据过滤：year=2024 → 结果全部来自 2024 文档
def test_metadata_filter(hybrid_fixture) -> None:
    hybrid: HybridRetriever = hybrid_fixture["hybrid"]
    result = hybrid.search("revenue", top_k=5, use_reranker=False,
                           filter_metadata={"year": 2024})
    assert result.results, "过滤后结果集不应为空"
    assert all(item.metadata.get("year") == 2024 for item in result.results)
    assert all(item.metadata.get("doc_id") != "doc_g2" for item in result.results)


# 8.6 工具函数 retrieve() → 返回 SearchResultSet
def test_retriever_tool(hybrid_fixture) -> None:
    set_retriever(hybrid_fixture["hybrid"])
    result = retrieve("revenue", top_k=2)
    assert isinstance(result, SearchResultSet)
    assert len(result.results) == 2
    assert get_retriever() is hybrid_fixture["hybrid"]


# 加权融合单元测试（覆盖空列表跳过 / 退化值 0.5 / 权重长度校验）
def test_weighted_fusion() -> None:
    out = weighted_fuse(
        [[("a", 9.0), ("b", 8.0)], [("b", 7.0), ("c", 6.0)]], [0.5, 0.5], top_k=2
    )
    assert set(cid for cid, _ in out) == {"a", "b"}
    # 单列表 + 单元素：归一退化 0.5 × 权重 0.5 = 0.25
    single = weighted_fuse([[("a", 9.0)], []], [0.5, 0.5], top_k=2)
    assert single == [("a", 0.25)]
    with pytest.raises(ValueError):
        weighted_fuse([[("a", 9.0)], [("b", 8.0)]], [0.5], top_k=2)


# 加权融合集成：fusion_method="weighted" 走通完整流水线
def test_hybrid_weighted_mode(hybrid_fixture) -> None:
    weighted = HybridRetriever(
        bm25_index=hybrid_fixture["bm25"],
        vector_index=hybrid_fixture["vector"],
        metadata_store=hybrid_fixture["store"],
        fusion_method="weighted",
        dense_weight=0.5,
    )
    result = weighted.search("revenue", top_k=3, use_reranker=False)
    assert result.fusion_method == "weighted"
    assert len(result.results) == 3