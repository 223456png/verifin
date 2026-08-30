#!/usr/bin/env python3
"""离线检索参数扫描：doc recall@K × (BM25/hybrid) × 每文档 chunk 上限。"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.retrieval.hybrid_retriever import build_hybrid_retriever  # noqa: E402


def search_with_doc_cap(bm25, metadata, query: str, top_k: int, cap: int):
    """BM25 top_k*3 候选 → 每文档最多 cap 个 chunk → 截断 top_k。"""
    hits = bm25.search(query, top_k=top_k * 3)
    out, doc_count = [], {}
    for cid, score in hits:
        chunk = metadata.get(cid)
        if chunk is None:
            continue
        if doc_count.get(chunk.doc_id, 0) >= cap:
            continue
        doc_count[chunk.doc_id] = doc_count.get(chunk.doc_id, 0) + 1
        out.append((cid, chunk.doc_id))
        if len(out) >= top_k:
            break
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/finqa")
    parser.add_argument("--index-dir", default=".")
    parser.add_argument("--max-samples", type=int, default=100)
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path(args.data_dir), split="test")
    r = build_hybrid_retriever(persist_dir=Path(args.index_dir), use_reranker=False)

    configs = OrderedDict()
    for k in (20, 30, 40):
        configs[f"bm25@{k}"] = ("bm25", k, 99)
        for cap in (1, 2, 3):
            configs[f"bm25@{k} cap{cap}"] = ("bm25", k, cap)
    configs["hybrid@20"] = ("hybrid", 20, 99)

    hits = {name: 0 for name in configs}
    n = min(args.max_samples, len(dataset))
    for i in range(n):
        sample = dataset[i]
        gold = {str(e.get("doc_id")) for e in (sample.get("gold_evidence") or [])}
        query = str(sample["query"])
        for name, (mode, k, cap) in configs.items():
            if mode == "bm25":
                if cap >= 99:
                    got = {metadata_doc for _, metadata_doc
                           in [(cid, r.metadata.get(cid).doc_id)
                               for cid, _ in r.bm25.search(query, top_k=k)
                               if r.metadata.get(cid)]}
                else:
                    got = {doc for _, doc in search_with_doc_cap(
                        r.bm25, r.metadata, query, k, cap)}
            else:
                res = r.search(query, top_k=k, use_reranker=False)
                got = {x.metadata.get("doc_id") for x in res.results}
            if gold & got:
                hits[name] += 1

    print(f"samples: {n}")
    for name in configs:
        print(f"  {name:18s} doc recall: {hits[name]/n*100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
