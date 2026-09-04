#!/usr/bin/env python3
"""LLM-as-judge 评测（Phase 12.4）：faithfulness + relevancy 逐样本判定。

对指定 benchmark run 的 results.json 逐样本取证：
- 证据 = verify_flags 中四要素校验**通过**的 chunk 文本（与验证闭环绑定，
  而非任意检索结果——与 RAGAs 风格规则指标的口径一致）；
- judge 模型对 (query, answer, evidence) 输出
  ``{"faithful": bool, "relevancy": 1-5, "reason": str}``；
- 聚合 faithfulness 率 / 平均 relevancy，落盘 llm_judge.json + 报告。

用法：
  python scripts/llm_judge.py --run-dir results_llm_prog --index-dir ./bge_env \
      [--api-key sk-xxx | --bridge http://127.0.0.1:8642] [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_PROMPT = """你是金融问答系统的忠实性评审。给定【问题】【系统答案】【证据】，判定：
1. faithful：答案中的每个数值/结论是否都能被证据支持（无证据支撑的编造 = false）；
2. relevancy：答案是否回答了问题本身（1=完全没回答 5=精准回答）。

只输出 JSON：{{"faithful": true|false, "relevancy": 1-5, "reason": "一句话"}}

【问题】{query}
【系统答案】{answer}
【证据】
{evidence}"""


def _load_chunk_texts(db_path: Path, chunk_ids: list) -> dict:
    conn = sqlite3.connect(str(db_path))
    out = {}
    for cid in chunk_ids:
        row = conn.execute(
            "SELECT doc_id, content FROM chunks WHERE chunk_id=?", (cid,)
        ).fetchone()
        if row:
            out[cid] = (row[0], row[1])
    conn.close()
    return out


def _evidence_texts(result: dict, db_path: Path, max_chunks: int = 3) -> list:
    """收集该样本四要素校验通过的 chunk 文本（去重，截断）。"""
    ids: list = []
    for flags in (result.get("verify_flags") or {}).values():
        if not isinstance(flags, dict):
            continue
        for r in flags.get("results") or []:
            if isinstance(r, dict) and r.get("passed") and r.get("chunk_id"):
                cid = str(r["chunk_id"])
                if cid not in ids:
                    ids.append(cid)
    texts = _load_chunk_texts(db_path, ids[:max_chunks])
    out = []
    for cid in ids[:max_chunks]:
        if cid in texts:
            doc_id, content = texts[cid]
            out.append(f"[{doc_id}] {content[:400]}")
    return out


def _parse(raw: str) -> dict | None:
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    else:
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e <= s:
            return None
        raw = raw[s:e + 1]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "faithful" not in data:
        return None
    rel = data.get("relevancy")
    data["relevancy"] = int(rel) if isinstance(rel, (int, float)) and 1 <= rel <= 5 else None
    return data


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="benchmark 输出目录（含 full/results.json）")
    ap.add_argument("--index-dir", default="./bge_env")
    ap.add_argument("--bridge", default="http://127.0.0.1:8642")
    ap.add_argument("--api-key", default=None, help="OpenAI 兼容直连（优先于 bridge）")
    ap.add_argument("--api-base", default="https://api.deepseek.com")
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--limit", type=int, default=None, help="最多评判 N 个样本")
    args = ap.parse_args()

    from verifin.llm_provider import BridgedLLM, OpenAICompatLLM

    if args.api_key:
        prov = OpenAICompatLLM(args.api_key, base_url=args.api_base, model=args.model, kind="judge")
    else:
        prov = BridgedLLM(args.bridge, kind="judge")

    run_dir = Path(args.run_dir)
    results = json.load(open(run_dir / "full" / "results.json"))
    db_path = Path(args.index_dir) / "verifin_metadata.db"
    samples = results[: args.limit] if args.limit else results

    judgments, parsed_fail = [], 0
    for i, x in enumerate(samples):
        evidence = _evidence_texts(x, db_path)
        if not evidence:
            judgments.append({"idx": i, "faithful": None, "relevancy": None,
                              "reason": "no verified evidence"})
            continue
        prompt = _PROMPT.format(
            query=x.get("query", ""), answer=str(x.get("answer", ""))[:600],
            evidence="\n".join(evidence)[:1500],
        )
        raw = prov.complete(prompt)
        data = _parse(raw) if raw else None
        if data is None:
            parsed_fail += 1
            judgments.append({"idx": i, "faithful": None, "relevancy": None,
                              "reason": f"judge unavailable/parse fail: {raw[:60]}"})
        else:
            judgments.append({"idx": i, **data})

    valid = [j for j in judgments if j.get("faithful") is not None]
    rels = [j["relevancy"] for j in valid if j.get("relevancy") is not None]
    summary = {
        "samples": len(samples),
        "judged": len(valid),
        "no_evidence": sum(1 for j in judgments if j["reason"] == "no verified evidence"),
        "judge_fail": parsed_fail,
        "faithfulness": round(sum(1 for j in valid if j["faithful"]) / len(valid), 3) if valid else None,
        "avg_relevancy": round(sum(rels) / len(rels), 2) if rels else None,
        "call_stats": prov.summary(),
    }
    out_json = run_dir / "llm_judge.json"
    out_json.write_text(json.dumps({"summary": summary, "judgments": judgments},
                                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"明细已保存: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
