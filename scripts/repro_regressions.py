#!/usr/bin/env python3
"""复现 Phase 12.1 引入的 3 个 regression 样本，dump 完整中间状态。

用法：
    python scripts/repro_regressions.py --index-dir ./bge_env --indices 24,43,72
    python scripts/repro_regressions.py --index-dir ./bge_env --indices 24 --dump-raw

输出（results/repro_regressions/<idx>.json）：
- answer / ground_truth / is_correct
- calculation_result（模板路由 + 锚定选中的操作数 + 表达式）
- verify_flags（四要素校验逐条裁决，含 chunk_id）
- retrieved_docs（chunk_id + doc_id + 前 200 字符）
- node 轨迹与工具调用序列
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# 允许仓库根直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from verifin.benchmark.dataset import load_dataset
from verifin.benchmark.runner import (  # noqa: E402
    FULL_CONFIG,
    RetrievalEnvironment,
    _register_retrieve_tool,
    build_benchmark_graph,
)
from verifin.core.runner import AgentRunner
from verifin.tools.registry import ToolRegistry, register_builtin_tools
from verifin.benchmark.trace import extract_answer_value
from verifin.benchmark.metrics import exact_match


def _brief_doc(doc: dict, n: int = 200) -> dict:
    content = str(doc.get("content") or "")
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    return {
        "chunk_id": doc.get("chunk_id"),
        "doc_id": doc.get("doc_id") or meta.get("doc_id"),
        "score": doc.get("score"),
        "content_head": content[:n],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", default="bge_env", help="预构建索引目录")
    parser.add_argument("--data-dir", default="data/finqa", help="FinQA 数据目录")
    parser.add_argument("--indices", default="24,43,72", help="样本索引（逗号分隔）")
    parser.add_argument("--output", default="results/repro_regressions", help="输出目录")
    parser.add_argument("--dump-raw", action="store_true", help="附带完整 final state 原始 dump")
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path(args.data_dir), split="test")
    env = RetrievalEnvironment.from_persist_dir(Path(args.index_dir))
    graph = build_benchmark_graph(dict(FULL_CONFIG))
    register_builtin_tools()
    _register_retrieve_tool(env, dense=True)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in [int(x) for x in args.indices.split(",")]:
        sample = dataset[idx]
        query = str(sample["query"])
        gold = str(sample["ground_truth"])
        registry = ToolRegistry()
        before = len(registry.history())

        final = AgentRunner(graph).run(query)

        tools_used = [c.tool_name for c in registry.history()[before:]]
        candidates = extract_answer_value(final, query)
        calc = final.get("calculation_result") or {}

        report = {
            "index": idx,
            "query": query,
            "ground_truth": gold,
            "gold_program": sample.get("gold_program"),
            "gold_evidence": sample.get("gold_evidence"),
            "answer_value": [[v, u] for v, u in candidates],
            "is_correct": exact_match(candidates, gold),
            "calculation_result": calc,
            "tools_used": tools_used,
            "nodes": [h.get("node") for h in final.get("hooks") or []],
            "verifier_decisions": [
                f.get("decision") for f in (final.get("verify_flags") or {}).values()
                if isinstance(f, dict)
            ],
            "retrieved_docs": [
                _brief_doc(d) for d in final.get("retrieved_docs") or []
                if isinstance(d, dict)
            ],
            "verify_flags": final.get("verify_flags") or {},
        }
        if args.dump_raw:
            report["_raw_final_state"] = final

        path = out_dir / f"{idx:04d}.json"
        path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        status = "OK " if report["is_correct"] else "FAIL"
        print(
            f"[{status}] idx={idx} gold={gold} got={report['answer_value']} "
            f"-> {path.name}"
        )

    stamp = datetime.now(timezone.utc).isoformat()
    (out_dir / "_meta.txt").write_text(
        f"rerun at {stamp}; index_dir={args.index_dir}; branch=see git\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
