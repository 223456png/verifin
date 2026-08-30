#!/usr/bin/env python3
"""回放单个样本并 dump calculator 的候选桶（monkeypatch ProgramExecutor.execute）。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.benchmark.runner import (  # noqa: E402
    FULL_CONFIG,
    BenchmarkRunner,
)
from verifin.core.runner import AgentRunner  # noqa: E402
from verifin.tools.program_executor import ProgramExecutor  # noqa: E402

_orig = ProgramExecutor.execute


def patched(self, spec, candidates=None):
    print(f"\n=== candidates for {spec.kind} {spec.base_period}->{spec.target_period} ===")
    for period, bucket in (candidates or {}).items():
        print(f"  [{period}]")
        for c in sorted(bucket, key=lambda x: x.anchor_score)[:8]:
            print(f"    value={c.value} unit={c.unit} anchor={c.anchor_score} chunk={str(c.chunk_id)[:10]}")
    result = _orig(self, spec, candidates)
    print("  => primary:", result.expression, "| value:", result.value,
          "| fraction:", result.fraction, "| difference:", result.difference)
    return result


ProgramExecutor.execute = patched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--idx", type=int, required=True)
    args = parser.parse_args()

    dataset = load_dataset("finqa", Path("data/finqa"), split="test")
    sample = dataset[args.idx]
    print("query:", sample["query"])
    print("ground_truth:", sample["ground_truth"])
    runner = BenchmarkRunner(
        config_name="full", dataset=dataset, config=dict(FULL_CONFIG),
        output_dir=Path("results/.debug"), index_dir=Path("."),
    )
    AgentRunner(runner.graph).run(str(sample["query"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
