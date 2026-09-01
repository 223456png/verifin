#!/usr/bin/env python3
"""Phase 11 前置诊断 2：gold doc 在图运行中的丢失点追踪。

对 FinQA test 样本跑完整图（与 benchmark 相同路径），检查：
  1. 终态 retrieved_docs 是否含 gold doc（gt_in_retrieved 口径复现）；
  2. 终态 exclude_chunk_ids 是否包含 gold doc 的 chunk（拉黑实锤）；
  3. 用终态子任务重放检索（不过滤 excluded）能否命中 gold doc
     ——命中则说明损耗来自 exclude 过滤而非检索层。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.benchmark.runner import (  # noqa: E402
    FULL_CONFIG, BenchmarkRunner, _register_retrieve_tool,
)
from verifin.core.runner import AgentRunner  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, nargs="+", required=True)
    parser.add_argument("--data-dir", default="data/finqa")
    parser.add_argument("--index-dir", default=".")
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path(args.data_dir), split="test")
    runner = BenchmarkRunner(
        config_name="full", dataset=dataset, config=dict(FULL_CONFIG),
        output_dir=Path("results/.debug_trace"), index_dir=Path(args.index_dir),
    )

    for idx in args.idx:
        sample = dataset[idx]
        gold_docs = {str(e.get("doc_id")) for e in (sample.get("gold_evidence") or [])}
        query = str(sample["query"])
        print(f"\n=== sample {idx} ===")
        print("query:", query[:90])
        print("gold docs:", gold_docs)

        final = AgentRunner(runner.graph).run(query)
        docs = [d for d in (final.get("retrieved_docs") or []) if isinstance(d, dict)]
        got_docs = {
            str((d.get("metadata") or {}).get("doc_id")) for d in docs
        }
        excluded = set(final.get("exclude_chunk_ids") or [])
        sub_tasks = list(final.get("sub_tasks") or [])
        print(f"sub_tasks ({len(sub_tasks)}):", [t[:50] for t in sub_tasks])
        print(f"final retrieved: {len(docs)} chunks, gold hit: {bool(gold_docs & got_docs)}")
        print(f"excluded chunks: {len(excluded)}")

        # 拉黑实锤：excluded 里是否有 gold doc 的 chunk
        gold_chunk_ids = set()
        if gold_docs:
            for chunk in runner.env.metadata.filter(doc_id=sorted(gold_docs)[0]):
                gold_chunk_ids.add(chunk.chunk_id)
        blacklisted = gold_chunk_ids & excluded
        print(f"gold doc chunks: {len(gold_chunk_ids)}, blacklisted: {len(blacklisted)}")

        # 无过滤重放：终态子任务逐个检索（不过滤 excluded）
        replay_hit = False
        for task in sub_tasks:
            res = runner.env.hybrid.search(task, top_k=25, use_reranker=True)
            hit = any(
                str((r.metadata or {}).get("doc_id")) in gold_docs
                for r in res.results
            )
            if hit:
                replay_hit = True
                break
        print(f"replay (no exclude filter) gold hit: {replay_hit}")

        verdict = []
        if gold_docs & got_docs:
            verdict.append("GOLD-RETRIEVED（终态未丢失）")
        if blacklisted:
            verdict.append(f"BLACKLISTED({len(blacklisted)})")
        if replay_hit and not (gold_docs & got_docs):
            verdict.append("LOSS-BY-EXCLUDE（检索可召回但被过滤）")
        if not replay_hit and not (gold_docs & got_docs):
            verdict.append("LOSS-BY-RETRIEVAL（子任务重放也无法召回）")
        print("verdict:", " + ".join(verdict) or "UNCLEAR")
    return 0


if __name__ == "__main__":
    sys.exit(main())
