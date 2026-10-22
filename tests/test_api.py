"""FastAPI 服务层测试：/health、/ask 单轮与多轮、错误处理。"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from verifin.api.app import _build_service, create_app


@pytest.fixture(scope="module")
def client() -> TestClient:
    service = _build_service(index_dir=None, dense=True)
    app = create_app(service=service)
    return TestClient(app)


def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["corpus"]["source"] == "synthetic"
    assert body["sessions"] >= 0


def test_ask_single_turn_extraction(client: TestClient) -> None:
    resp = client.post("/ask", json={"query": "What was NovaTech revenue in 2023?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["conversation_id"]
    assert "10,000" in body["answer"] or "10000" in body["answer"]
    # 提取型问题的证据链
    assert body["evidence"], "应返回通过校验的证据"
    assert any(e["chunk_id"] for e in body["evidence"])
    assert body["trajectory"]["node_path"][0] == "planner"
    assert body["trajectory"]["steps"] >= 3


def test_ask_calculation_growth(client: TestClient) -> None:
    resp = client.post(
        "/ask",
        json={"query": "What was the percentage growth in NovaTech revenue from 2023 to 2024?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    calc = body["calculation"]
    assert calc.get("kind") == "growth_pct"
    assert calc.get("fraction") == pytest.approx(0.2, rel=1e-3)
    assert calc.get("value") == pytest.approx(20.0, rel=1e-3)
    assert calc.get("expression")


def test_ask_multi_turn_preference(client: TestClient) -> None:
    """多轮：同一 conversation_id 复用对话状态（指代消解 + 偏好继承）。"""
    conv = "test-conv-1"
    first = client.post(
        "/ask",
        json={"query": "What was NovaTech revenue in 2024?", "conversation_id": conv},
    ).json()
    assert first["conversation_id"] == conv
    assert first["dialog_state"], "应返回会话状态"
    # 回归：session_id 曾恒为空串（图中默认 "" 被 setdefault 短路），
    # 现在必须回填 API 层的 conversation_id
    assert first["dialog_state"].get("session_id") == conv

    second = client.post(
        "/ask",
        json={"query": "只看年报口径，重新告诉我", "conversation_id": conv},
    ).json()
    assert second["conversation_id"] == conv
    # 会话状态在轮间累积（偏好记忆生效）
    assert second["dialog_state"]

    third = client.post(
        "/ask",
        json={"query": "What about its gross margin?", "conversation_id": conv},
    ).json()
    # 指代消解：its → NovaTech（偏好继承后的指标切换应返回毛利率相关证据）
    assert third["answer"]


def test_ask_rejects_empty_query(client: TestClient) -> None:
    resp = client.post("/ask", json={"query": "   "})
    assert resp.status_code == 422


def test_ask_rejects_missing_query(client: TestClient) -> None:
    resp = client.post("/ask", json={})
    assert resp.status_code == 422


def test_root_serves_demo_or_api_info(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    # web/index.html 存在时返回 HTML，否则返回 API 描述 JSON
    if Path("web/index.html").exists():
        assert "html" in resp.headers.get("content-type", "")
    else:
        assert resp.json()["name"] == "VeriFin API"


# 2026-09-30 安全审计修复：会话表 LRU 上限 / 并发闸 503 / 状态目录绝对路径
def test_session_table_lru_eviction() -> None:
    """会话表超上限逐出最旧条目：长跑不再无限累积。"""

    from verifin.api.app import _MAX_SESSIONS

    service = _build_service(index_dir=None, dense=True)
    n = _MAX_SESSIONS + 10
    for i in range(n):
        service._register_session(f"conv-{i}")
    assert service.session_count == _MAX_SESSIONS
    # 最旧的 conv-0..9 已被逐出，最新 conv-n-1 还在
    assert "conv-0" not in service._threads
    assert f"conv-{n - 1}" in service._threads
    # 累计新建数独立于跟踪数（P2-1：语义不再混淆）
    assert service.sessions_total == n
    # 复用已有会话不增加累计数，且刷新 LRU 顺序
    total_before = service.sessions_total
    service._register_session(f"conv-{n - 1}")
    assert service.sessions_total == total_before


def test_concurrency_gate_returns_503() -> None:
    """并发闸占满 → /ask 快速 503，而不是在单线程执行器前无限堆积。"""
    import threading as _threading

    from verifin.api.app import _GATE_WAIT_SECONDS, ServiceBusyError

    service = _build_service(index_dir=None, dense=True)
    gate = _threading.BoundedSemaphore(1)
    gate.acquire()  # 人为占满闸
    service._ask_gate = gate
    app = create_app(service=service)
    with TestClient(app) as c:
        resp = c.post("/ask", json={"query": "NovaTech 2024 revenue"})
        assert resp.status_code == 503
        assert "服务忙" in resp.json()["detail"]
    gate.release()
    assert ServiceBusyError is not None
    assert _GATE_WAIT_SECONDS > 0


def test_workspace_state_dir_is_absolute() -> None:
    """合成语料模式的状态目录必须在系统临时目录下（不随启动 cwd 漂移）。"""
    import tempfile as _tempfile

    service = _build_service(index_dir=None, dense=True)
    assert service._env is not None
    root = Path(service._env.root)
    assert root.is_absolute()
    assert str(root).startswith(_tempfile.gettempdir())


# ---------------------------------------------------------------------------
# 2026-10 修复：仲裁采纳值必须进答案主句 + 证据卡标注未采纳 + 排队快速 503
# ---------------------------------------------------------------------------

def test_answer_uses_arbitrated_value_not_first_evidence(client: TestClient) -> None:
    """同指标异值（新闻稿 11000 / 年报 12000）→ 主句必须是仲裁采纳的 12000。

    回归背景：synthesizer 原来取「第一条通过校验的 evidence」，检索顺序一变
    答案就在 11000 与 12000 之间跳变，与本项目自己的"来源可信度仲裁"结论
    自相矛盾。
    """
    for query in ("NovaTech 2024 revenue", "What was NovaTech revenue in 2024?"):
        body = client.post("/ask", json={"query": query}).json()
        assert "12000" in body["answer"], body["answer"]
        assert "11000.0 million" not in body["answer"].split("\n")[0]
        # 仲裁结果对客户端可见（可核对，不用去猜）
        assert body["conflict"]["reconciled_value"] == 12000.0
        assert body["conflict"]["level"] == "none"
        # 未采纳的数值仍然展示，但标成 excluded + 原因
        losers = [
            e for e in body["evidence"]
            if e["chunk_id"] == "nt-news-revenue-2024"
        ]
        assert losers, "冲突来源应保留在证据卡里"
        assert losers[0]["status"] == "excluded"
        assert losers[0]["reason"] == "conflict_not_adopted"


def test_openapi_documents_verdict_and_conflict(client: TestClient) -> None:
    """response_model 必须覆盖 verdict/conflict，否则客户端按文档取值会 KeyError。"""
    schema = client.get("/openapi.json").json()
    props = schema["components"]["schemas"]["AskResponse"]["properties"]
    assert "verdict" in props
    assert "conflict" in props


def test_executor_busy_returns_503_fast() -> None:
    """执行器被占满时，排队请求应快速 503，而不是排队到超时才失败。"""
    import time

    from verifin.api.app import ServiceBusyError, _build_service

    service = _build_service(index_dir=None, dense=True)
    blocker = __import__("threading").Event()
    service._executor.submit(blocker.wait)  # 占住唯一的执行线程
    started = time.monotonic()
    try:
        service.ask("NovaTech 2024 revenue", None)
    except ServiceBusyError as exc:
        assert "执行器繁忙" in str(exc)
    else:  # pragma: no cover - 失败时的明确信号
        raise AssertionError("executor busy 应抛 ServiceBusyError")
    finally:
        blocker.set()
    # 排队上限 8s，必须远小于旧的 120s 整体超时
    assert time.monotonic() - started < 20
