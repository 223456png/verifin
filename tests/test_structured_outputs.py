"""Phase 13.1 — Structured Outputs 测试：response_format 透传与自动回退。

诚实边界：无 API key 不打真实端点；请求构造（body 含 response_format）与
回退语义（400/422 摘除后立即重试）用 monkeypatch 替换 ``urlopen`` 锁定，
与 W2 所练手法一致。schema 常量与 make_llm_stack 接线为纯构造断言。
"""

from __future__ import annotations

import io
import json
import urllib.error

from verifin.core.llm_planner import PLANNER_RESPONSE_FORMAT
from verifin.llm_provider import OpenAICompatLLM, make_llm_stack
from verifin.tools.llm_programmer import PROGRAMMER_RESPONSE_FORMAT


class _FakeResp(io.BytesIO):
    """满足 urlopen 上下文管理器协议的假响应。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _ok_response(content: str) -> _FakeResp:
    payload = {
        "choices": [{"message": {"content": content}}]
    }
    return _FakeResp(json.dumps(payload).encode())


def _make_provider(response_format=None, **kw) -> OpenAICompatLLM:
    return OpenAICompatLLM(
        "test-key", base_url="http://fake.local", kind="test",
        response_format=response_format, **kw
    )


# ----------------------------------------------------------------------
# 请求构造：response_format 透传
# ----------------------------------------------------------------------

def test_request_body_contains_response_format(monkeypatch) -> None:
    bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        return _ok_response('{"sub_tasks": ["a"]}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _make_provider(response_format=PLANNER_RESPONSE_FORMAT)
    out = provider.complete({"prompt": "q"})
    assert '"sub_tasks"' in out
    assert bodies[0]["response_format"] == PLANNER_RESPONSE_FORMAT


def test_request_body_omits_response_format_when_off(monkeypatch) -> None:
    bodies: list[dict] = []

    def fake_urlopen(req, timeout=None):
        bodies.append(json.loads(req.data.decode()))
        return _ok_response('{"program": "add(v1, v2)"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _make_provider()
    provider.complete({"prompt": "q"})
    assert "response_format" not in bodies[0]


# ----------------------------------------------------------------------
# 自动回退：端点 400 → 摘除 response_format 立即重试
# ----------------------------------------------------------------------

def test_fallback_on_400_strips_schema_and_retries(monkeypatch) -> None:
    bodies: list[dict] = []
    state = {"calls": 0}

    def fake_urlopen(req, timeout=None):
        state["calls"] += 1
        bodies.append(json.loads(req.data.decode()))
        if state["calls"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {}, io.BytesIO(b"{}")
            )
        return _ok_response('{"program": "add(v1, v2)"}')

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = _make_provider(response_format=PROGRAMMER_RESPONSE_FORMAT)
    out = provider.complete({"prompt": "q"})
    # 第二次请求成功返回内容
    assert '"program"' in out
    # 恰好两次调用：首战 400，摘除后立即重试（未走指数退避）
    assert state["calls"] == 2
    assert "response_format" in bodies[0]
    assert "response_format" not in bodies[1]
    # schema 已被永久摘除（后续调用不再携带）
    provider.complete({"prompt": "q2"})
    assert "response_format" not in bodies[2]


def test_500_still_raises_to_retry_loop(monkeypatch) -> None:
    """非 400/422（如 500）不在回退范围——由 _BaseProvider 重试契约处理。"""
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                req.full_url, 500, "Server Error", {}, io.BytesIO(b"{}")
            )
        return _ok_response("ok")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda *_: None)
    provider = _make_provider(response_format=PROGRAMMER_RESPONSE_FORMAT)
    # 500 → _request_chat 冒泡 → _BaseProvider 重试 → 第二次成功
    assert provider.complete({"prompt": "q"}) == "ok"
    # 500 未触发 schema 摘除：重试请求仍带 response_format
    assert provider.response_format == PROGRAMMER_RESPONSE_FORMAT


# ----------------------------------------------------------------------
# make_llm_stack 接线
# ----------------------------------------------------------------------

def test_make_llm_stack_json_schema_wiring() -> None:
    _, _, providers = make_llm_stack(
        mode="both", api_key="k", json_schema=True,
    )
    by_kind = {p.kind: p for p in providers}
    assert by_kind["planner"].response_format == PLANNER_RESPONSE_FORMAT
    assert by_kind["programmer"].response_format == PROGRAMMER_RESPONSE_FORMAT


def test_make_llm_stack_default_off() -> None:
    _, _, providers = make_llm_stack(mode="both", api_key="k")
    for p in providers:
        assert p.response_format is None


def test_make_llm_stack_schema_names_distinct() -> None:
    """两个 schema 的 name 不同（OpenAI 端按 name 区分用途）。"""
    p_name = PLANNER_RESPONSE_FORMAT["json_schema"]["name"]
    g_name = PROGRAMMER_RESPONSE_FORMAT["json_schema"]["name"]
    assert p_name == "financial_plan"
    assert g_name == "dsl_program"
    assert p_name != g_name


def test_schemas_are_valid_json_schema_shapes() -> None:
    """schema 自检：strict 形态的关键字段齐全。"""
    for rf in (PLANNER_RESPONSE_FORMAT, PROGRAMMER_RESPONSE_FORMAT):
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["strict"] is True
        schema = rf["json_schema"]["schema"]
        assert schema["type"] == "object"
        assert schema.get("additionalProperties") is False
