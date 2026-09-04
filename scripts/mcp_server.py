#!/usr/bin/env python3
"""VeriFin MCP Server（Phase 12.4）：把金融证据校验能力暴露为 MCP 工具。

任何 MCP 客户端（Claude Desktop / 自建 Agent）即可复用 VeriFin 的
检索 + 校验 + 计算能力，无需理解其内部实现。

工具（只读 + 安全计算，无副作用）：
- retrieve(query, top_k, use_reranker)  混合检索（BM25+Dense+RRF±rerank）
- expand_document(doc_id)               父文档补全（表格跨 chunk 补救）
- verify_claim(claim, evidence_text)    四要素规则校验（实体/期间/指标/定义）
- calculate(expression)                 PoT 安全表达式计算（AST 白名单，无 eval）

启动（stdio 传输，IDE / Claude Desktop 配置指向本脚本即可）：
    VERIFIN_INDEX_DIR=./bge_env python scripts/mcp_server.py

安全边界：只读检索与规则校验，计算器为 AST 白名单实现（verifin/tools/calculator.py），
不暴露任何写路径与 shell。所有工具返回 JSON 字符串（FastMCP 对裸 list
会做单元素截断，显式序列化保证结构可预期）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp.server.fastmcp import FastMCP

from verifin.benchmark.runner import RetrievalEnvironment
from verifin.tools.calculator import calc_expression
from verifin.tools.verifier import verify_claim as _verify_claim_tool
from verifin.tools.verifier import extract_claim

INDEX_DIR = Path(os.environ.get("VERIFIN_INDEX_DIR", "./bge_env"))

mcp = FastMCP("verifin")

print(f"[mcp-verifin] 加载索引: {INDEX_DIR}", file=sys.stderr)
_env = RetrievalEnvironment.from_persist_dir(INDEX_DIR)
print("[mcp-verifin] 索引就绪", file=sys.stderr)


@mcp.tool()
def retrieve(query: str, top_k: int = 10, use_reranker: bool = True) -> str:
    """混合检索金融文档证据（BM25 + Dense 向量 + RRF 融合，可选 Cross-Encoder 精排）。

    返回 JSON 数组：[{chunk_id, doc_id, score, content(截断 1000 字), metadata}, ...]。
    """
    result = _env.hybrid.search(query, top_k=min(max(top_k, 1), 20), use_reranker=use_reranker)
    out = []
    for item in result.results:
        meta = item.metadata or {}
        out.append({
            "chunk_id": item.chunk_id,
            "doc_id": meta.get("doc_id"),
            "score": round(item.score, 4),
            "content": (item.content or "")[:1000],
            "metadata": meta,
        })
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def expand_document(doc_id: str) -> str:
    """返回文档全部 chunk（父文档补全：表格被分块切断时的标准补救）。"""
    chunks = _env.metadata.filter(doc_id=doc_id)
    out = [
        {"chunk_id": c.chunk_id, "content": (c.content or "")[:2000],
         "metadata": c.metadata}
        for c in chunks
    ]
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def verify_claim(claim: str, evidence_text: str) -> str:
    """四要素规则校验：从 claim 文本抽取（实体/期间/指标/定义）并对证据逐项核对。

    claim 形如 "revenue in 2015"；返回逐要素匹配明细与总体裁决 PASS/REJECT。
    """
    extracted = extract_claim(claim)
    return json.dumps(
        _verify_claim_tool(extracted, {"chunk_id": "evidence", "content": evidence_text}),
        ensure_ascii=False,
    )


@mcp.tool()
def calculate(expression: str) -> str:
    """PoT 安全计算器：白名单 AST 求值（支持 + - * / ** % 与括号），无 eval。"""
    try:
        return json.dumps({"expression": expression, "value": calc_expression(expression)})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"expression": expression, "error": str(exc)})


if __name__ == "__main__":
    mcp.run()  # stdio 传输
