#!/usr/bin/env python3
"""LLM 消融结果自动归档：两组 run 完成后生成对比数字（results_llm_summary.md）。

由后台 watcher 调用（/tmp/watch_llm.sh），也可手动重跑。
"""
import json
import glob
from collections import Counter
from pathlib import Path

ROOT = Path("/home/z/my-project/repos/verifin")
RUNS = {
    "rule（基线）": ROOT / "results_v122_norank",
    "L1 llm=both": ROOT / "results_llm_both",
    "L2 llm=programmer": ROOT / "results_llm_prog",
}

def load_run(d: Path):
    f = d / "full" / "results.json"
    if not f.exists():
        return None
    return json.load(open(f))

def load_llm_stats(d: Path):
    out = {}
    for kind in ("planner", "programmer"):
        f = d / f"llm_calls_{kind}.json"
        if f.exists():
            out[kind] = json.load(open(f))["summary"]
    return out

rows, details = [], []
for name, d in RUNS.items():
    r = load_run(d)
    if r is None:
        rows.append((name, None, None))
        continue
    ok = sum(1 for x in r if x.get("is_correct"))
    errs = Counter(x.get("error_type") for x in r if not x.get("is_correct"))
    rows.append((name, f"{ok}/{len(r)} = {100*ok/len(r):.1f}%", dict(errs)))
    details.append((name, r))

lines = ["# LLM 消融对比（100 样本，降级 reranker 口径，GLM via z-ai 桥接）", ""]
for name, acc, errs in rows:
    lines.append(f"- **{name}**: {acc or '未完成'} | 错误分布: {errs or '-'}")

# 翻转分析（L1 vs rule）
if len(details) >= 2:
    base = details[0][1]
    for name, r in details[1:]:
        flips = [(i, a["is_correct"], b["is_correct"]) for i, (a, b) in enumerate(zip(base, r)) if a["is_correct"] != b["is_correct"]]
        up = [i for i, a, b in flips if b]
        down = [i for i, a, b in flips if a]
        lines.append(f"- **{name} vs 基线翻转**: +{len(up)} / -{len(down)}（新对 {up}，转错 {down}）")

# LLM 调用统计
for name, d in RUNS[1:]:
    stats = load_llm_stats(d)
    if stats:
        lines.append(f"- **{name} 调用统计**:")
        for kind, s in stats.items():
            lines.append(
                f"  - [{kind}] {s['ok']}/{s['calls']} 成功，平均 {s['avg_latency_ms']:.0f}ms，"
                f"p95 {s['p95_latency_ms']:.0f}ms"
            )

out = ROOT / "results_llm_summary.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
