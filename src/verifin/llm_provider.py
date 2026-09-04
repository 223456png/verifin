"""LLM provider（Phase 12.3）：``complete(prompt) -> str`` 的两种实现。

1. :class:`BridgedLLM`——本地桥接服务（scripts/llm_bridge.mjs，z-ai SDK）；
2. :class:`OpenAICompatLLM`——任意 OpenAI 兼容端点直连（DeepSeek 等，
   ``POST {base}/chat/completions``，Bearer 鉴权）。

设计契约（与 :class:`verifin.core.llm_planner.LLMPlanner` /
:class:`verifin.tools.llm_programmer.LLMProgramGenerator` 对齐）：
- ``complete`` **永不抛异常**——失败返回空串，调用方解析层把空串判非法
  并走确定性降级（plan 返回 None / generate 返回 None）；
- 每次调用记录 (kind, latency_ms, 字符数, ok, error)，``dump_stats()``
  落盘供成本/延迟归因；
- 重试 3 次（指数退避 3s/6s/12s），覆盖限流（429）与瞬时 5xx。
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class _BaseProvider:
    """共用：调用统计 + 重试循环 + 永不抛异常契约。"""

    kind = "llm"

    def __init__(self, timeout: float = 120.0, retries: int = 3) -> None:
        self.timeout = timeout
        self.retries = retries
        self.stats: list = []

    def _post(self, payload: dict) -> str:
        """单次 HTTP 调用，返回原始文本；失败抛异常（由 complete 兜底）。"""
        raise NotImplementedError

    def complete(self, prompt: str) -> str:
        t0 = time.perf_counter()
        last_err = ""
        for attempt in range(self.retries + 1):
            try:
                text = self._post({"prompt": prompt})
                self.stats.append({
                    "kind": self.kind,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "prompt_chars": len(prompt),
                    "resp_chars": len(text),
                    "ok": True,
                    "attempt": attempt + 1,
                })
                return text
            except Exception as exc:  # noqa: BLE001 - 契约：永不抛出
                last_err = f"{type(exc).__name__}: {exc}"[:200]
                self.stats.append({
                    "kind": self.kind,
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                    "prompt_chars": len(prompt),
                    "resp_chars": 0,
                    "ok": False,
                    "attempt": attempt + 1,
                    "error": last_err,
                })
                if attempt < self.retries:
                    time.sleep(3.0 * (2 ** attempt))  # 3s/6s/12s
        return ""

    def summary(self) -> Dict[str, Any]:
        oks = [s for s in self.stats if s["ok"]]
        lats = sorted(s["latency_ms"] for s in oks) or [0]
        p95 = lats[min(len(lats) - 1, int(len(lats) * 0.95))]
        return {
            "kind": self.kind,
            "calls": len(self.stats),
            "ok": len(oks),
            "fail": len(self.stats) - len(oks),
            "success_rate": round(len(oks) / len(self.stats), 3) if self.stats else 0.0,
            "avg_latency_ms": round(sum(lats) / len(lats), 1),
            "p95_latency_ms": round(p95, 1),
            "prompt_chars_total": sum(s["prompt_chars"] for s in self.stats),
            "resp_chars_total": sum(s["resp_chars"] for s in self.stats),
        }

    def dump_stats(self, path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"summary": self.summary(), "calls": self.stats},
                f, ensure_ascii=False, indent=2,
            )


class BridgedLLM(_BaseProvider):
    """本地桥接端点：``POST {base_url}/complete {"prompt"} → {"text"}``。"""

    def __init__(self, base_url: str = "http://127.0.0.1:8642", **kw) -> None:
        super().__init__(**kw)
        self.base_url = base_url.rstrip("/")
        self.kind = "llm_bridge"

    def _post(self, payload: dict) -> str:
        body = json.dumps({"prompt": payload["prompt"]}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/complete", data=body, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return str(json.loads(resp.read()).get("text") or "")


class OpenAICompatLLM(_BaseProvider):
    """OpenAI 兼容直连（DeepSeek 等）：temperature=0，JSON 友好。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        kind: str = "deepseek",
        **kw,
    ) -> None:
        super().__init__(**kw)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.kind = kind

    def _post(self, payload: dict) -> str:
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": payload["prompt"]}],
            "temperature": 0,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())
        return str(
            (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        )


def make_llm_stack(
    mode: str = "both",
    bridge_url: Optional[str] = None,
    api_key: Optional[str] = None,
    api_base: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
) -> tuple:
    """按 mode 与通道构造 (llm_planner, llm_programmer, providers)。

    mode: ``both`` | ``planner`` | ``programmer``——单变量消融用。
    通道二选一：``bridge_url``（本地桥接）或 ``api_key``（OpenAI 兼容直连）。
    """
    from verifin.core.llm_planner import LLMPlanner
    from verifin.tools.llm_programmer import LLMProgramGenerator

    def _make(kind: str):
        if api_key:
            return OpenAICompatLLM(api_key, base_url=api_base, model=model, kind=kind)
        if bridge_url:
            return BridgedLLM(bridge_url, kind=kind)
        return None

    planner_prov = _make("planner") if mode in ("both", "planner") else None
    prog_prov = _make("programmer") if mode in ("both", "programmer") else None
    llm_planner = LLMPlanner(planner_prov.complete) if planner_prov else None
    llm_programmer = LLMProgramGenerator(prog_prov.complete) if prog_prov else None
    providers = [p for p in (planner_prov, prog_prov) if p is not None]
    return llm_planner, llm_programmer, providers
