"""混合检索器：BM25 + Dense 双路召回 → RRF/加权融合 → Cross-Encoder 精排（可降级）→ 元数据过滤。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from verifin.indexing.bm25_index import BM25Index
from verifin.indexing.metadata_store import MetadataStore
from verifin.indexing.vector_index import VectorIndex
from verifin.retrieval.fusion import RRFusion, minmax_normalize, weighted_fuse
from verifin.retrieval.reranker import Reranker
from verifin.schemas import DocumentChunk, SearchResult, SearchResultSet

PathLike = Union[str, Path]


class HybridRetriever:
    """混合检索流水线编排器。

    流程：BM25 召回 top_k*2 → Dense 召回 top_k*2（距离转相似度 1/(1+d)）
    → rrf/weighted 融合取 top_k*2 → [可选] Reranker 精排到 top_k
    → filter_metadata 过滤 → 组装 SearchResultSet（最终分 min-max 归一 0-1）。
    """

    def __init__(
        self,
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        metadata_store: MetadataStore,
        reranker: Optional[Reranker] = None,
        fusion: Optional[RRFusion] = None,
        dense_weight: float = 0.5,
        fusion_method: str = "rrf",
    ) -> None:
        if fusion_method not in {"rrf", "weighted"}:
            raise ValueError(f"未知融合方式: {fusion_method!r}")
        self.bm25 = bm25_index
        self.vector = vector_index
        self.metadata = metadata_store
        self.reranker = reranker or Reranker()
        self.fusion = fusion or RRFusion()
        self.dense_weight = dense_weight
        self.fusion_method = fusion_method

    def _match_filter(self, chunk_id: str, filter_metadata: dict) -> bool:
        """通用元数据过滤：扩展字段精确匹配 + doc_id/section_title/page_num 特判。"""
        chunk = self.metadata.get(chunk_id)
        if chunk is None:
            return False
        for key, expected in filter_metadata.items():
            if key == "doc_id":
                actual = chunk.doc_id
            elif key == "section_title":
                actual = chunk.section_title
            elif key == "page_num":
                actual = chunk.page_num
            else:
                actual = chunk.metadata.get(key)
            if actual != expected:
                return False
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        use_reranker: bool = True,
        filter_metadata: Optional[dict] = None,
    ) -> SearchResultSet:
        """执行完整混合检索流水线。

        Args:
            query: 查询文本。
            top_k: 最终返回条数。
            use_reranker: 是否启用 Cross-Encoder 精排（不可用时自动降级）。
            filter_metadata: 元数据过滤条件 dict（如 {"year": 2024}），
                键支持扩展字段精确匹配与 doc_id/section_title/page_num 特判。
                注意：过滤在精排截断之后执行，命中过滤条件的候选不足 top_k 时
                结果数可能少于 top_k（不会从融合池回填）。

        Returns:
            :class:`SearchResultSet`。
        """
        start = time.perf_counter()
        pool = top_k * 2

        # 1. 稀疏路
        bm25_hits: List[Tuple[str, float]] = self.bm25.search(query, top_k=pool)
        bm25_scores: Dict[str, float] = dict(bm25_hits)

        # 2. 稠密路（Chroma distance → similarity）
        dense_hits: List[Tuple[str, float]] = [
            (chunk_id, 1.0 / (1.0 + distance))
            for chunk_id, distance in self.vector.search(query, top_k=pool)
        ]
        dense_scores: Dict[str, float] = dict(dense_hits)

        # 3. 融合
        if self.fusion_method == "weighted":
            fused = weighted_fuse(
                [bm25_hits, dense_hits],
                [1.0 - self.dense_weight, self.dense_weight],
                top_k=pool,
            )
        else:
            fused = self.fusion.fuse([bm25_hits, dense_hits], top_k=pool)
        fused_scores: Dict[str, float] = dict(fused)

        # 4. [可选] Reranker 精排（不可用时保持融合顺序）
        reranker_used = False
        rerank_scores: Dict[str, float] = {}
        if use_reranker:
            candidates: List[Tuple[str, str]] = []
            for chunk_id, _ in fused:
                chunk = self.metadata.get(chunk_id)
                if chunk is not None:
                    candidates.append((chunk_id, chunk.content))
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            if self.reranker.available:
                reranker_used = True
                rerank_scores = dict(reranked)
                ordered: List[Tuple[str, float]] = reranked
            else:
                ordered = [(chunk_id, fused_scores[chunk_id]) for chunk_id, _ in reranked]
        else:
            ordered = [(chunk_id, fused_scores[chunk_id]) for chunk_id, _ in fused[:top_k]]

        # 5. 元数据过滤
        if filter_metadata:
            ordered = [
                (chunk_id, score)
                for chunk_id, score in ordered
                if self._match_filter(chunk_id, filter_metadata)
            ]

        # 6. 组装：最终分 min-max 归一化到 0-1（全同分退化值为 0.0）
        raw_scores = [score for _, score in ordered]
        normalized = minmax_normalize(raw_scores, degenerate=0.0)
        results: List[SearchResult] = []
        for (chunk_id, _), norm_score in zip(ordered, normalized):
            chunk = self.metadata.get(chunk_id)
            if chunk is None:
                continue
            result_meta: dict = {
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_num": chunk.page_num,
                "section_title": chunk.section_title,
            }
            result_meta.update(chunk.metadata)
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    content=chunk.content,
                    score=norm_score,
                    bm25_score=bm25_scores.get(chunk_id),
                    dense_score=dense_scores.get(chunk_id),
                    rerank_score=rerank_scores.get(chunk_id),
                    metadata=result_meta,
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return SearchResultSet(
            query=query,
            results=results,
            total_count=len(results),
            retrieval_time_ms=elapsed_ms,
            fusion_method=self.fusion_method,
            reranker_used=reranker_used,
        )


def build_hybrid_retriever(
    persist_dir: PathLike = ".",
    use_reranker: bool = True,
    **kwargs,
) -> HybridRetriever:
    """从 Phase 1 索引产物组装 HybridRetriever。

    Args:
        persist_dir: 索引产物根目录（含 indexes/bm25.pkl、chroma_db/、verifin_metadata.db）。
        use_reranker: 是否构造 Reranker（False 时完全跳过精排组件）。
        **kwargs: 透传给 HybridRetriever（如 fusion_method/dense_weight）。
    """
    root = Path(persist_dir)
    bm25 = BM25Index.load(root / "indexes" / "bm25.pkl")
    vector = VectorIndex(persist_dir=str(root / "chroma_db"))
    store = MetadataStore(path=str(root / "verifin_metadata.db"))
    reranker = Reranker() if use_reranker else None
    return HybridRetriever(
        bm25_index=bm25, vector_index=vector, metadata_store=store, reranker=reranker, **kwargs
    )