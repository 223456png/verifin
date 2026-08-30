#!/usr/bin/env python3
"""多跳 QA 合成管线入口（Phase 9.6）。

用法：
    python scripts/synthesize_multihop.py --output ./results/multihop_synth

产出：
    benchmark.json —— 合成的多跳评测基准（统一样本格式 + hops/program 标注）
    stdout —— 四重校验统计（种子拒绝 / 候选 / 各校验拒绝计数 / 通过）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from verifin.benchmark.multihop_synth import synthesize_multihop_samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/multihop_synth", help="输出目录")
    args = parser.parse_args()

    samples, stats = synthesize_multihop_samples()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = out_dir / "benchmark.json"
    benchmark_path.write_text(
        json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    hops2 = [s for s in samples if s["hops"] == 2]
    hops3 = [s for s in samples if s["hops"] == 3]
    by_type: dict[str, int] = {}
    for sample in samples:
        by_type[sample["synth_type"]] = by_type.get(sample["synth_type"], 0) + 1

    print(f"种子 chunk: {stats.seeds_total}（词典外/不可解析拒绝 {stats.seeds_rejected}）")
    print(f"候选样本: {stats.candidates} → 四重校验通过 {stats.accepted}")
    if stats.rejected:
        for reason, count in sorted(stats.rejected.items(), key=lambda kv: -kv[1]):
            print(f"  拒绝[{reason}]: {count}")
    print(f"产出基准: {len(samples)} 条（2-hop {len(hops2)} / 3-hop {len(hops3)}）")
    print(f"类型分布: {by_type}")
    print(f"基准已保存到 {benchmark_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
