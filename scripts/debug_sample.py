#!/usr/bin/env python3
"""单个 FinQA 样本回放诊断：候选收集 + 程序执行细节。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.benchmark.runner import (  # noqa: E402
    FULL_CONFIG,
    BenchmarkRunner,
    _register_retrieve_tool,
)
from verifin.core.runner import AgentRunner  # noqa: E402
from verifin.core.graph import build_agent_graph  # noqa: E402
from verifin.tools.registry import ToolRegistry, register_builtin_tools  # noqa: E402
from verifin.tools.program_executor import detect_program  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    parser.add_argument("--data-dir", default="data/finqa")
    parser.add_argument("--index-dir", default=".")
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path(args.data_dir), split="test")
    sample = dataset[args.idx]
    print("query:", sample["query"])
    print("ground_truth:", sample["ground_truth"])
    print("gold_evidence:", sample.get("gold_evidence"))

    spec = detect_program(sample["query"])
    print("detected spec:", spec)

    runner = BenchmarkRunner(
        config_name="full", dataset=dataset, config=dict(FULL_CONFIG),
        output_dir=Path("results/.debug"), index_dir=Path(args.index_dir),
    )
    final = AgentRunner(runner.graph).run(str(sample["query"]))

    print("\n=== node_path ===")
    print(final.get("node_path") or [h.get("node") for h in final.get("hooks") or []])
    print("\n=== calc spec ===")
    print(json.dumps(final.get("calculation_spec"), ensure_ascii=False, default=str))
    print("\n=== calculation_result ===")
    print(json.dumps(final.get("calculation_result"), ensure_ascii=False, indent=1, default=str))

    docs = [d for d in (final.get("retrieved_docs") or []) if isinstance(d, dict)]
    print("\n=== retrieved docs (top 8) ===")
    for d in docs[:8]:
        meta = d.get("metadata") or {}
        print(" ", meta.get("doc_id"), "|", meta.get("doc_name", "")[:40],
              "|", str(d.get("content"))[:60].replace("\n", " "))

    gold_docs = {str(e.get("doc_id")) for e in (sample.get("gold_evidence") or [])}
    print("gold doc retrieved:", any(
        str((d.get("metadata") or {}).get("doc_id")) in gold_docs for d in docs))

    # 候选重演：对检索文档跑 collect_table_year_values
    if spec:
        from verifin.tools.evidence import EvidenceExtractor
        for year in (spec.base_period, spec.target_period):
            print(f"\n=== table candidates for {year} (top docs) ===")
            for d in docs[:5]:
                items = EvidenceExtractor.collect_table_year_values(
                    str(d.get("content")), year,
                    spec_metric=(final.get("calculation_spec") or {}).get("metric"),
                    query=str(sample["query"]),
                )
                if items:
                    print(" doc", (d.get("metadata") or {}).get("doc_id"))
                    for it in items[:4]:
                        print("   ", it)
    return 0


if __name__ == "__main__":
    sys.exit(main())
