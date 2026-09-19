#!/usr/bin/env python3
"""VeriFin MCP Server（Phase 12.4）：把金融证据校验能力暴露为 MCP 工具。

任何 MCP 客户端（Claude Desktop / Cursor / 自建 Agent）即可复用 VeriFin 的
检索 + 校验 + 计算能力，无需理解其内部实现。

工具（只读 + 安全计算，无副作用）：
- retrieve(query, top_k, use_reranker)  混合检索（BM25+Dense+RRF±rerank）
- expand_document(doc_id)               父文档补全（表格跨 chunk 补救）
- verify_claim(claim, evidence_text)    四要素规则校验（实体/期间/指标/定义）
- calculate(expression)                 PoT 安全表达式计算（AST 白名单受限求值）

启动（stdio 传输，IDE / Claude Desktop 配置指向本模块即可）：
    VERIFIN_INDEX_DIR=./bge_env python -m verifin.mcp.server

安全边界：只读检索与规则校验，不暴露任何写路径与 shell。计算器采用三层防御
（AST 节点白名单 + ``__builtins__`` 置空 + SIGALRM 超时 + 幂次决定性 guard），
详见 ``verifin/tools/calculator.py``。

版本兼容：MCP Python SDK 2.x 将 ``FastMCP`` 更名为 ``MCPServer``（模块路径
``mcp.server.mcpserver``）。本模块在导入期自动适配 1.x / 2.x 两套命名，
``pyproject.toml`` 声明 ``mcp>=1.0.0`` 无需上限约束。

索引在**首次工具调用**时惰性加载——导入本模块不会触发磁盘 IO，
因此在没有索引的环境中也可以安全 import（CI 友好）。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:  # MCP SDK 2.x：FastMCP 更名为 MCPServer
    from mcp.server.mcpserver import MCPServer as _MCPServer
except ImportError:  # MCP SDK 1.x：保留 FastMCP 名称
    from mcp.server.fastmcp import FastMCP as _MCPServer

from verifin.tools.calculator import calc_expression
from verifin.tools.verifier import extract_claim
from verifin.tools.verifier import verify_claim as _verify_claim_tool

# 索引目录：环境变量优先，其次项目根下的 bge_env（bge 语义索引）
_DEFAULT_INDEX_DIR = Path(__file__).resolve().parents[3] / "bge_env"
INDEX_DIR = Path(os.environ.get("VERIFIN_INDEX_DIR") or _DEFAULT_INDEX_DIR)

mcp = _MCPServer("verifin")

_env: Optional[Any] = None


def _get_env():
    """惰性加载检索环境（进程内单例）。

    首次调用时从 ``INDEX_DIR`` 读取持久化索引并打印进度到 stderr，
    避免污染 stdio 上的 JSON-RPC 流。
    """
    global _env
    if _env is None:
        from verifin.benchmark.runner import RetrievalEnvironment

        print(f"[mcp-verifin] 加载索引: {INDEX_DIR}", file=sys.stderr)
        _env = RetrievalEnvironment.from_persist_dir(INDEX_DIR)
        print("[mcp-verifin] 索引就绪", file=sys.stderr)
    return _env


@mcp.tool()
def retrieve(query: str, top_k: int = 10, use_reranker: bool = True) -> str:
    """混合检索金融文档证据（BM25 + Dense 向量 + RRF 融合，可选 Cross-Encoder 精排）。

    返回 JSON 数组：[{chunk_id, doc_id, score, content(截断 1000 字), metadata}, ...]。
    """
    result = _get_env().hybrid.search(
        query, top_k=min(max(top_k, 1), 20), use_reranker=use_reranker
    )
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
    chunks = _get_env().metadata.filter(doc_id=doc_id)
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
    """PoT 安全计算器：AST 白名单受限求值（支持 + - * / ** % 与括号、math 白名单函数）。

    实现为「白名单 AST + 空 builtins + 超时」的受限 eval，不接受任意代码与 IO。

    返回 JSON 对象：``{expression, value, unit, error, is_valid, execution_time_ms}``。
    求值失败不抛异常，改为 ``error`` 非空、``value`` 为 null。
    """
    try:
        return json.dumps(calc_expression(expression), ensure_ascii=False)
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"expression": expression, "value": None, "error": str(exc), "is_valid": False},
            ensure_ascii=False,
        )


def main() -> None:
    """stdio 传输入口。"""
    mcp.run()


if __name__ == "__main__":
    main()
