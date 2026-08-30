#!/usr/bin/env python3
"""理想检索口径：ratio 模板锚定精度验证（Phase 10.1）。

对 FinQA test 前 100 样本中检出 ratio 模板的题，把 gold 记录全文
（pre_text + table + post_text，与 loader 同口径序列化）直接作为唯一
"检索结果" 喂给候选收集 + 比率执行器，统计 EM——剥离检索层，单独度量
短语锚定的精度（端到端 EM 的上界分解依据）。

用法：.venv/bin/python scripts/verify_ratio_ideal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from verifin.benchmark.dataset import load_dataset  # noqa: E402
from verifin.ingestion.loader import _extract_text  # noqa: E402
from verifin.tools.evidence import EvidenceExtractor  # noqa: E402
from verifin.tools.program_executor import (  # noqa: E402
    ProgramExecutor, ValueCandidate, detect_program,
)


def gold_content(record: dict) -> str:
    text = _extract_text(record)
    return text or ""


def collect(record_text: str, spec) -> dict:
    """镜像 nodes.py 的 ratio 候选收集（单 gold chunk，无父文档补全）。"""
    year = spec.base_period or None
    candidates = {"num": [], "den": []}
    total_fallback = []

    def _add(bucket: str, item: dict) -> None:
        lst = candidates[bucket]
        if any(c.value == item["value"] and c.unit == item["unit"] for c in lst):
            return
        lst.append(ValueCandidate(
            value=float(item["value"]), unit=item.get("unit"),
            period=str(item.get("period") or year or ""), chunk_id="gold",
            anchor_score=float(item.get("anchor_score", 0.0)),
            row_label=str(item.get("row_label") or ""),
            column=str(item.get("column") or ""),
        ))

    for bucket, phrase in (("num", spec.numerator), ("den", spec.denominator)):
        if not phrase:
            continue
        items = EvidenceExtractor.collect_phrase_values(record_text, phrase, year)
        if not items and year:
            items = EvidenceExtractor.collect_year_values(record_text, year)
        for item in items:
            _add(bucket, item)
    for item in EvidenceExtractor.collect_total_values(record_text):
        total_fallback.append(item)
    if not candidates["den"]:
        for item in total_fallback:
            _add("den", item)
    return candidates


def match(gt: str, result) -> bool:
    try:
        gt_v = float(str(gt).replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return False
    pool = []
    if result.fraction is not None:
        pool.append(float(result.fraction))
    if result.value is not None:
        pool.append(float(result.value))
    for v in pool:
        if abs(v) <= 1e-12 and abs(gt_v) > 1e-12:
            continue
        if abs(gt_v) > 1e-12 and abs(v - gt_v) / max(abs(gt_v), 1e-9) <= 0.01:
            return True
        if abs(gt_v) <= 1e-12 and abs(v) <= 1e-9:
            return True
    return False


def main() -> int:
    records = json.loads(Path("data/finqa/test.json").read_text())
    dataset = load_dataset("finqa", Path("data/finqa"), split="test")
    assert not dataset.is_synthetic

    exec_ = ProgramExecutor()
    n_total = n_ok = 0
    for idx in range(min(100, len(dataset))):
        sample = dataset[idx]
        spec = detect_program(sample["query"])
        if not spec or spec.kind != "ratio":
            continue
        record = records[idx]
        text = gold_content(record)
        candidates = collect(text, spec)
        result = exec_.execute(spec, candidates)
        ok = result.error is None and match(sample["ground_truth"], result)
        n_total += 1
        n_ok += ok
        print(
            f"{idx:3d} {'PASS' if ok else 'FAIL'} gt={sample['ground_truth']:<12} "
            f"frac={result.fraction} err={result.error} :: {sample['query'][:70]}"
        )
    print(f"\nideal-retrieval ratio EM: {n_ok}/{n_total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
