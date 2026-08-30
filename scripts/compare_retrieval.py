#!/usr/bin/env python3
"""检索升级对比：hash vs bge embedding + reranker 的 doc recall@K。

用法（需先构建两种索引）::

    # hash 基线（默认路径）
    python scripts/build_index.py --dataset finqa --data-dir data/finqa --embedding hash

    # bge 语义 embedding（独立目录，避免覆盖 hash 索引）
    HF_ENDPOINT=https://hf-mirror.com \
      python scripts/build_index.py --dataset finqa --data-dir data/finqa \
        --embedding bge --persist-dir ./bge_env

    # 对比（模型已缓存时用 offline 加速加载）
    HF_HUB_OFFLINE=1 python scripts/compare_retrieval.py --max-samples 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.retrieval.hybrid_retriever import build_hybrid_retriever  # noqa: E402


def _doc_recall_at_k(retriever, query: str, gold: set[str], k: int,
                     use_reranker: bool) -> bool:
    """检索 top_k，判断是否命中任一 gold doc_id。"""
    res = retriever.search(query, top_k=k, use_reranker=use_reranker)
    got = {item.metadata.get("doc_id") for item in res.results}
    return bool(gold & got)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/finqa")
    parser.add_argument("--hash-dir", default=".", help="hash 基线索引根目录")
    parser.add_argument("--bge-dir", default="./bge_env", help="bge 索引根目录")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path(args.data_dir), split="test")
    n = min(args.max_samples, len(dataset))
    k = args.top_k

    hash_r = build_hybrid_retriever(persist_dir=Path(args.hash_dir), use_reranker=False)
    bge_r = build_hybrid_retriever(persist_dir=Path(args.bge_dir), use_reranker=False)
    bge_rerank_r = build_hybrid_retriever(
        persist_dir=Path(args.bge_dir), use_reranker=True
    )

    hits = {"hash+hybrid": 0, "bge+hybrid": 0, "bge+hybrid+rerank": 0}
    for i in range(n):
        sample = dataset[i]
        gold = {str(e.get("doc_id")) for e in (sample.get("gold_evidence") or [])}
        query = str(sample["query"])
        if _doc_recall_at_k(hash_r, query, gold, k, use_reranker=False):
            hits["hash+hybrid"] += 1
        if _doc_recall_at_k(bge_r, query, gold, k, use_reranker=False):
            hits["bge+hybrid"] += 1
        if _doc_recall_at_k(bge_rerank_r, query, gold, k, use_reranker=True):
            hits["bge+hybrid+rerank"] += 1

    print(f"samples={n}  top_k={k}")
    for name, cnt in hits.items():
        print(f"  {name:18s} doc recall@{k}: {cnt / n * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())