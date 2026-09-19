#!/usr/bin/env python3
"""MCP server 协议级冒烟：stdio 子进程握手 → tools/list → 四工具逐个调用。

用法：python scripts/test_mcp_server.py
（需已构建索引；索引目录由 VERIFIN_INDEX_DIR 指定，默认项目根下 ./bge_env）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = [sys.executable, "-m", "verifin.mcp.server"]

_id = 0


def rpc(method: str, params: dict | None = None, notify: bool = False) -> dict | None:
    global _id
    msg: dict = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    if not notify:
        _id += 1
        msg["id"] = _id
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()
    if notify:
        return None
    while True:
        line = proc.stdout.readline()
        if not line:
            raise RuntimeError("server closed stream")
        resp = json.loads(line)
        if resp.get("id") == _id:
            return resp


_ENV = {
    **os.environ,
    "HF_HUB_OFFLINE": "1",
    # 未以 -e 安装时也能以源码模式启动子进程
    "PYTHONPATH": os.pathsep.join(
        p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p
    ),
}

proc = subprocess.Popen(
    SERVER, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL, env=_ENV,
)

# 1. initialize 握手
init = rpc("initialize", {
    "protocolVersion": "2024-11-05",
    "capabilities": {},
    "clientInfo": {"name": "verifin-smoke", "version": "0.1"},
})
name = init["result"]["serverInfo"]["name"]
print(f"[1] initialize OK: server={name}, protocol={init['result']['protocolVersion']}")
rpc("notifications/initialized", {}, notify=True)

# 2. tools/list
tools = rpc("tools/list")["result"]["tools"]
print(f"[2] tools/list OK: {[t['name'] for t in tools]}")
assert {t["name"] for t in tools} >= {"retrieve", "expand_document", "verify_claim", "calculate"}

# 3. 四工具调用
calc = rpc("tools/call", {"name": "calculate",
                          "arguments": {"expression": "(11806 - 9216) / 9216 * 100"}})
val = calc["result"]["content"][0]["text"]
print(f"[3] calculate: {val}")
assert "28.1" in val  # (11806-9216)/9216*100 ≈ 28.12

ver = rpc("tools/call", {"name": "verify_claim", "arguments": {
    "claim": "revenue in 2015",
    "evidence_text": "total net revenue in 2015 was $ 11806 million"}})
verdict = json.loads(ver["result"]["content"][0]["text"])
print(f"[4] verify_claim: {verdict.get('verdict', verdict)}")

ret = rpc("tools/call", {"name": "retrieve",
                         "arguments": {"query": "total net revenue 2015", "top_k": 3,
                                       "use_reranker": False}})
hits = json.loads(ret["result"]["content"][0]["text"])
print(f"[5] retrieve: {len(hits)} hits, top doc={hits[0]['doc_id']}, score={hits[0]['score']}")
assert hits

doc_id = hits[0]["doc_id"]
exp = rpc("tools/call", {"name": "expand_document", "arguments": {"doc_id": doc_id}})
chunks = json.loads(exp["result"]["content"][0]["text"])
print(f"[6] expand_document: {len(chunks)} chunks of {doc_id}")
assert chunks

proc.stdin.close()
proc.wait(timeout=10)
print("MCP 冒烟全部通过 ✔")
