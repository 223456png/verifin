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
