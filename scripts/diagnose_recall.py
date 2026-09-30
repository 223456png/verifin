#!/usr/bin/env python3
"""Phase 11 前置诊断：端到端 doc recall 损耗归因。

对照口径（FinQA test 前 100 样本，gold doc_id 命中）：
  A. 单查询直接检索（原始问题，top_k=25，无 rerank）× hash / bge 索引
  B. 子任务路径模拟（planner_node 生成 sub_tasks → 逐个检索 → 合并）
     × hash / bge 索引 × rerank 开/关（端到端默认 rerank=True）

目的：把「单查询 bge 86% vs 端到端 56%」的 30pp 差距分解为
索引选择 / 子任务改写 / rerank 三项损耗，为 Phase 11 设计提供依据。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.core.nodes import planner_node  # noqa: E402
from verifin.retrieval.hybrid_retriever import build_hybrid_retriever  # noqa: E402


def _gold_docs(sample: dict) -> set:
    return {str(e.get("doc_id")) for e in (sample.get("gold_evidence") or [])}


def _sub_tasks(query: str) -> list:
    """复现端到端 planner 的子任务分解（确定性规则路径，无 LLM）。"""
    state = {"messages": [{"role": "user", "content": query}]}
    updates = planner_node(state)
    return list(updates.get("sub_tasks") or [query])


def _doc_ids(retriever, queries: list, top_k: int, use_reranker: bool) -> set:
    """多查询合并（模拟 retriever_node：逐查 top_k → doc_id 并集）。"""
    got: set = set()
    for q in queries:
        res = retriever.search(q, top_k=top_k, use_reranker=use_reranker)
        got |= {item.metadata.get("doc_id") for item in res.results}
    return got


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/finqa")
    parser.add_argument("--hash-dir", default=".")
    parser.add_argument("--bge-dir", default="./bge_env")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path(args.data_dir), split="test")
    if dataset.is_synthetic:
        print(
            "错误：需要真实 FinQA 数据（test 集），当前目录下只有 sample。\n"
            f"     加载器已降级为内置合成语料。请将 FinQA test.jsonl 放入 "
            f"{args.data_dir}/ 后重试。"
        )
        return 2

    hash_r = build_hybrid_retriever(persist_dir=Path(args.hash_dir), use_reranker=False)
    bge_r = build_hybrid_retriever(persist_dir=Path(args.bge_dir), use_reranker=False)

    settings = {
        "A1 hash  单查询     norerank": (hash_r, "single", False),
        "A2 bge   单查询     norerank": (bge_r, "single", False),
        "B1 hash  子任务     norerank": (hash_r, "subtasks", False),
        "B2 bge   子任务     norerank": (bge_r, "subtasks", False),
        "B3 hash  子任务     rerank": (hash_r, "subtasks", True),
        "B4 bge   子任务     rerank": (bge_r, "subtasks", True),
    }

    n = min(args.max_samples, len(dataset))
    hits = {name: 0 for name in settings}
    sub_task_counts: list = []
    for i in range(n):
        sample = dataset[i]
        gold = _gold_docs(sample)
        if not gold:
            continue
        query = str(sample["query"])
        tasks = _sub_tasks(query)
        sub_task_counts.append(len(tasks))
        for name, (retriever, mode, rerank) in settings.items():
            queries = [query] if mode == "single" else tasks
            got = _doc_ids(retriever, queries, args.top_k, rerank)
            if gold & got:
                hits[name] += 1

    print(f"samples={n}  top_k={args.top_k}  "
          f"avg_sub_tasks={sum(sub_task_counts)/max(len(sub_task_counts),1):.2f}")
    for name, cnt in hits.items():
        print(f"  {name}: {cnt / n * 100:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
