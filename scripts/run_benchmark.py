#!/usr/bin/env python3
"""VeriFin Benchmark 入口（Phase 7 收官）。

用法：
    python scripts/run_benchmark.py --dataset finqa --max-samples 100 --output ./results
    python scripts/run_benchmark.py --dataset finqa --ablation --max-samples 50
    python scripts/run_benchmark.py --dataset convfinqa --multi-turn
    python scripts/run_benchmark.py --dataset synthetic --ablation --multi-turn --max-samples 20
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from verifin.benchmark.ablation import AblationRunner, CONFIG_LABELS
from verifin.benchmark.dataset import load_dataset
from verifin.benchmark.metrics import summarize_results
from verifin.benchmark.report import ReportGenerator
from verifin.benchmark.runner import FULL_CONFIG, BenchmarkRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", default="finqa",
        choices=["finqa", "convfinqa", "tatqa", "synthetic", "synthetic_multihop"],
        help="评测数据集（synthetic_multihop 为多跳合成基准；真实数据缺失时自动降级内置合成语料）",
    )
    parser.add_argument("--data-dir", default="data", help="真实数据集目录")
    parser.add_argument("--split", default="test", help="数据集拆分（train/val/test）")
    parser.add_argument("--max-samples", type=int, default=None, help="最多评估样本数")
    parser.add_argument("--ablation", action="store_true", help="运行 5 组消融实验")
    parser.add_argument("--multi-turn", action="store_true", help="启用多轮对话评估")
    parser.add_argument("--output", default="results", help="输出目录")
    parser.add_argument(
        "--index-dir", default=None,
        help="真实数据集的预构建索引目录（build_index.py 产物：indexes/ chroma_db/ *.db）",
    )
    parser.add_argument("--resume", action="store_true", help="从断点续跑")
    return parser.parse_args()


def _build_summary(
    args: argparse.Namespace,
    results_map: dict,
    dataset,
) -> dict:
    configs = {
        name: summarize_results(rows) for name, rows in results_map.items()
    }
    first = next(iter(results_map.values()))
    return {
        "dataset": dataset.name,
        "dataset_source": "synthetic" if getattr(dataset, "is_synthetic", False) else "real",
        "dataset_total": len(dataset),
        "total_samples": len(first) if first else 0,
        "multi_turn": args.multi_turn,
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "configs": configs,
    }


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output)
    dataset = load_dataset(args.dataset, Path(args.data_dir), split=args.split)
    if getattr(dataset, "is_synthetic", False) and \
            args.dataset not in ("synthetic", "synthetic_multihop"):
        print(f"提示: {args.dataset} 真实数据不可用（目录 {args.data_dir}），已降级内置合成语料（{len(dataset)} 个样本）")
    print(f"数据集 {dataset.name}: {len(dataset)} 个样本 → 评估 {args.max_samples or len(dataset)} 个")

    if args.ablation:
        runner = AblationRunner(
            dataset, output_dir, multi_turn=args.multi_turn,
            index_dir=Path(args.index_dir) if args.index_dir else None,
        )
        results = runner.run_all(max_samples=args.max_samples, resume=args.resume)
    else:
        runner = BenchmarkRunner(
            config_name="full",
            dataset=dataset,
            config=dict(FULL_CONFIG),
            output_dir=output_dir,
            multi_turn=args.multi_turn,
            index_dir=Path(args.index_dir) if args.index_dir else None,
        )
        results = {"full": runner.run_all(max_samples=args.max_samples, resume=args.resume)}

    summary = _build_summary(args, results, dataset)
    report_text = ReportGenerator(output_dir).generate(summary)
    for name, stats in summary["configs"].items():
        print(
            f"[{CONFIG_LABELS.get(name, name)}] "
            f"准确率 {stats['accuracy'] * 100:.1f}% | "
            f"平均步数 {stats['avg_steps']:.2f} | "
            f"重规划有效率 {stats['replan_hit_rate'] * 100:.1f}% | "
            f"Tool F1 {stats['tool_f1'] * 100:.1f}%"
        )
    report_path = output_dir / "report.md"
    _ = report_text  # 报告已由 ReportGenerator 落盘
    print(f"Benchmark 完成。报告已保存到 {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())