"""Agent 可调用的检索工具（Phase 3 将包装为 LangGraph Tool 的纯函数前置）。

当前为纯函数单例形态，不 import LangGraph；Phase 3 的 core/ 层负责 Tool 包装。
"""

from __future__ import annotations

from typing import Optional

from verifin.retrieval.hybrid_retriever import HybridRetriever, build_hybrid_retriever
from verifin.schemas import SearchResultSet

_retriever: Optional[HybridRetriever] = None


def get_retriever(
    persist_dir: str = ".",
    force_rebuild: bool = False,
    **kwargs,
) -> HybridRetriever:
    """惰性单例：首次调用时从 persist_dir 索引产物构建 HybridRetriever。"""
    global _retriever
    if _retriever is None or force_rebuild:
        _retriever = build_hybrid_retriever(persist_dir=persist_dir, **kwargs)
    return _retriever


def set_retriever(retriever: HybridRetriever) -> None:
    """注入替换实例（测试与多环境切换场景）。"""
    global _retriever
    _retriever = retriever


def retrieve(
    query: str,
    top_k: int = 10,
    use_reranker: bool = True,
    filter_metadata: Optional[dict] = None,
) -> SearchResultSet:
    """Agent 可调用的检索工具：执行 BM25 + Dense + RRF + [Rerank] 流水线。"""
    return get_retriever().search(
        query, top_k=top_k, use_reranker=use_reranker, filter_metadata=filter_metadata
    )
