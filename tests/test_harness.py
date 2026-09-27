"""Phase 13 — Tool-calling harness 测试（mock provider，锁定循环语义）。

诚实边界：无 API key 不跑真实 provider；harness 的循环契约（工具执行回喂 /
三停机条件 / 协议失败降级）以 :class:`ScriptedLLM`（构造注入）锁定，
:meth:`OpenAICompatLLM.chat` 的请求构造 / 响应解析用 monkeypatch 替换
``urllib.request.urlopen`` 验证（W2 所练手法在项目里的真实应用）。
"""

from __future__ import annotations

import io
import json

import pytest

from verifin.harness import HarnessResult, ToolCallingHarness
from verifin.llm_provider import OpenAICompatLLM
from verifin.schemas import ToolCall
from verifin.tools.calculator import calc_expression
from verifin.tools.registry import ToolRegistry


# ----------------------------------------------------------------------
# 测试基建
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_registry_singleton():
    """ToolRegistry 为进程级单例：每个用例前后清空工具与历史，防跨用例泄漏。"""
    registry = ToolRegistry()
    registry.reset()
    yield
    registry.reset()


class ScriptedLLM:
    """按脚本顺序回放的假 provider（构造注入，不打补丁）。

    每项脚本为一个 dict（harness 协议形态）或 None（模拟 provider 失败）。
    记录每次收到的 (messages, tools) 供断言。
    """

    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.received: list[tuple[list, list | None]] = []

    def chat(self, messages: list, tools: list | None = None):
        self.received.append((messages, tools))
        if not self.responses:
            return {"content": "", "tool_calls": []}
        return self.responses.pop(0)


class _FakeHTTPResponse(io.BytesIO):
    """duck-typing 的 HTTP 响应（含上下文管理器协议）。"""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _registry_with_calc() -> ToolRegistry:
    """独立注册表：只挂 calc_expression（PoT 沙箱，零外部依赖）。"""
    registry = ToolRegistry()
    registry.register(
        name="calc_expression",
        func=calc_expression,
        description="Safely evaluate a numeric expression in a sandbox",
        schema={"type": "object", "properties": {"expr": {"type": "string"}}},
    )
    return registry


def _call(name: str, args: dict, call_id: str = "tc1") -> dict:
    """构造一条 harness 协议的 tool call。"""
    return {"id": call_id, "name": name, "args": args}


# ----------------------------------------------------------------------
# 1. 循环核心语义
# ----------------------------------------------------------------------

def test_direct_answer_no_tools() -> None:
    """模型首轮直接回答 → finished，无工具调用，turns=1。"""
    llm = ScriptedLLM([{"content": "42", "tool_calls": []}])
    harness = ToolCallingHarness(llm, registry=_registry_with_calc())
    result = harness.run("meaning of life?")
    assert result.answer == "42"
    assert result.stopped_reason == "finished"
    assert result.turns == 1
    assert result.tool_calls == []
    # 消息历史：system + user + assistant（无 tool 轮）
    roles = [m["role"] for m in result.messages]
    assert roles == ["system", "user", "assistant"]


def test_tool_call_then_answer() -> None:
    """先调工具、拿到结果再回答 → 工具被执行且回喂、最终答案正确。"""
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [_call("calc_expression", {"expr": "6 * 7"})]},
        {"content": "6 * 7 = 42", "tool_calls": []},
    ])
    harness = ToolCallingHarness(llm, registry=_registry_with_calc())
    result = harness.run("what is 6 * 7?")
    assert result.stopped_reason == "finished"
    assert result.answer == "6 * 7 = 42"
    assert result.turns == 2
    # 工具被真实执行（PoT 沙箱），调用形态入轨
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "calc_expression"
    assert result.tool_calls[0].args == {"expr": "6 * 7"}
    # 第二轮模型收到了 tool 消息（role=tool + tool_call_id 回绑）
    second_messages, _tools = llm.received[1]
    tool_msgs = [m for m in second_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc1"
    payload = json.loads(tool_msgs[0]["content"])
    assert payload["output"]["value"] == pytest.approx(42.0)


def test_parallel_tool_calls_in_one_turn() -> None:
    """单轮并行两个工具调用 → 依序执行、依序回喂。"""
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [
            _call("calc_expression", {"expr": "1 + 1"}, call_id="a"),
            _call("calc_expression", {"expr": "2 + 2"}, call_id="b"),
        ]},
        {"content": "done", "tool_calls": []},
    ])
    harness = ToolCallingHarness(llm, registry=_registry_with_calc())
    result = harness.run("compute")
    assert result.stopped_reason == "finished"
    assert [c.call_id for c in result.tool_calls] == ["a", "b"]
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["a", "b"]


def test_unknown_tool_error_fed_back() -> None:
    """未知工具不中断循环 → 错误以 tool 消息回喂，模型可自主纠正。"""
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [_call("nonexistent_tool", {"x": 1})]},
        {"content": "recovered", "tool_calls": []},
    ])
    harness = ToolCallingHarness(llm, registry=_registry_with_calc())
    result = harness.run("try something")
    assert result.stopped_reason == "finished"
    assert result.answer == "recovered"
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert "not found" in tool_msgs[0]["content"]


def test_tool_exception_fed_back() -> None:
    """工具执行抛异常 → error JSON 回喂（registry.execute 不抛出契约）。"""
    def _boom(expr: str):
        raise RuntimeError("boom")

    registry = ToolRegistry()
    registry.register("calc_expression", _boom, "explodes", {"type": "object"})
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [_call("calc_expression", {"expr": "1"})]},
        {"content": "survived", "tool_calls": []},
    ])
    result = ToolCallingHarness(llm, registry=registry).run("q")
    assert result.stopped_reason == "finished"
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert "boom" in json.loads(tool_msgs[0]["content"])["error"]


# ----------------------------------------------------------------------
# 2. 三停机条件
# ----------------------------------------------------------------------

def test_max_turns_stops_loop() -> None:
    """模型每轮都请求工具 → max_turns 截停，防死循环。"""
    endless = {"content": None, "tool_calls": [_call("calc_expression", {"expr": "1"})]}
    llm = ScriptedLLM([endless] * 50)
    harness = ToolCallingHarness(llm, registry=_registry_with_calc(), max_turns=3)
    result = harness.run("loop forever")
    assert result.stopped_reason == "max_turns"
    assert result.turns == 3
    assert result.answer == ""
    assert len(result.tool_calls) == 3


def test_llm_returns_none_stops_with_error() -> None:
    """provider 失败（chat 返回 None）→ llm_error，answer 空串由上层降级。"""
    llm = ScriptedLLM([None])
    harness = ToolCallingHarness(llm, registry=_registry_with_calc())
    result = harness.run("q")
    assert result.stopped_reason == "llm_error"
    assert result.answer == ""
    assert result.turns == 1


def test_llm_raises_is_contained() -> None:
    """协议外：provider 直接抛异常 → harness 兜底为 llm_error，不冒泡。"""
    class _Raises:
        def chat(self, messages, tools=None):
            raise RuntimeError("provider crashed")

    result = ToolCallingHarness(_Raises(), registry=_registry_with_calc()).run("q")
    assert result.stopped_reason == "llm_error"


# ----------------------------------------------------------------------
# 3. 上下文构造与工具 schema
# ----------------------------------------------------------------------

def test_tools_passed_to_llm_from_registry() -> None:
    """默认把注册表 schema 传给模型（function calling 协议形态）。"""
    llm = ScriptedLLM([{"content": "ok", "tool_calls": []}])
    registry = _registry_with_calc()
    ToolCallingHarness(llm, registry=registry).run("q")
    _messages, tools = llm.received[0]
    assert tools == registry.list_tools()
    assert tools[0]["name"] == "calc_expression"
    assert "properties" in tools[0]["parameters"]


def test_custom_system_prompt_and_none() -> None:
    """自定义 system prompt 进消息；None 时用内置默认（金融 QA）。"""
    llm = ScriptedLLM([{"content": "ok", "tool_calls": []}])
    ToolCallingHarness(
        llm, registry=_registry_with_calc(), system_prompt="be terse"
    ).run("q")
    messages, _ = llm.received[0]
    assert messages[0] == {"role": "system", "content": "be terse"}

    llm2 = ScriptedLLM([{"content": "ok", "tool_calls": []}])
    ToolCallingHarness(llm2, registry=_registry_with_calc(), system_prompt=None).run("q")
    messages2, _ = llm2.received[0]
    assert messages2[0]["role"] == "system"
    assert "financial" in messages2[0]["content"]


def test_tool_result_truncated() -> None:
    """超大工具输出按 MAX_TOOL_CONTENT_CHARS 截断，防上下文爆炸。"""
    from verifin.harness import MAX_TOOL_CONTENT_CHARS

    big = "x" * (MAX_TOOL_CONTENT_CHARS + 1000)
    registry = ToolRegistry()
    registry.register("spew", lambda: big, "spews", {"type": "object"})
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [_call("spew", {})]},
        {"content": "ok", "tool_calls": []},
    ])
    result = ToolCallingHarness(llm, registry=registry).run("q")
    tool_msgs = [m for m in result.messages if m["role"] == "tool"]
    assert len(tool_msgs[0]["content"]) <= MAX_TOOL_CONTENT_CHARS + 10


# ----------------------------------------------------------------------
# 4. OpenAICompatLLM.chat：请求构造 / 响应解析（monkeypatch urlopen）
# ----------------------------------------------------------------------

def test_openai_chat_request_and_parse(monkeypatch) -> None:
    """chat 构造带 tools 的请求体、解析 tool_calls 响应为 harness 协议。"""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["auth"] = req.headers.get("Authorization")
        resp_payload = {
            "choices": [{
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "calc_expression",
                            "arguments": "{\"expr\": \"6 * 7\"}",
                        },
                    }],
                }
            }]
        }
        return _FakeHTTPResponse(json.dumps(resp_payload).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatLLM(api_key="sk-test", retries=0)
    tools = [{"type": "function", "function": {"name": "calc_expression"}}]
    resp = llm.chat(
        [{"role": "user", "content": "q"}], tools=tools
    )
    # 请求：端点 / 鉴权 / 工具随行
    assert captured["url"].endswith("/chat/completions")
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["tools"] == tools
    assert captured["body"]["tool_choice"] == "auto"
    # 响应：解析成 harness 协议（args 已从 JSON 字符串反序列化为 dict）
    assert resp == {
        "content": "",
        "tool_calls": [{"id": "call_1", "name": "calc_expression",
                        "args": {"expr": "6 * 7"}}],
    }
    # 调用统计落盘口径
    assert llm.stats[-1]["ok"] is True
    assert llm.stats[-1]["kind"] == "deepseek:chat"


def test_openai_chat_no_tools_omits_field(monkeypatch) -> None:
    """无 tools 时请求体不含 tools 字段（纯对话形态）。"""
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["body"] = json.loads(req.data.decode())
        return _FakeHTTPResponse(json.dumps({
            "choices": [{"message": {"content": "plain answer"}}]
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatLLM(api_key="sk-test", retries=0)
    resp = llm.chat([{"role": "user", "content": "hi"}])
    assert "tools" not in captured["body"]
    assert resp == {"content": "plain answer", "tool_calls": []}


def test_openai_chat_invalid_args_json_returns_none(monkeypatch) -> None:
    """tool_calls 的 arguments 非法 JSON → 整体失败返回 None（不半解析）。"""
    def fake_urlopen(req, timeout=None):
        return _FakeHTTPResponse(json.dumps({
            "choices": [{"message": {
                "content": None,
                "tool_calls": [{"id": "c", "type": "function", "function": {
                    "name": "f", "arguments": "not json",
                }}],
            }}]
        }).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatLLM(api_key="sk-test", retries=0)
    assert llm.chat([{"role": "user", "content": "q"}], tools=[{"x": 1}]) is None


def test_openai_chat_network_failure_returns_none(monkeypatch) -> None:
    """网络异常 → None（重试耗尽；retries=0 免等待），统计记 fail。"""
    def fake_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    llm = OpenAICompatLLM(api_key="sk-test", retries=0)
    assert llm.chat([{"role": "user", "content": "q"}]) is None
    assert llm.stats[-1]["ok"] is False
    assert "OSError" in llm.stats[-1]["error"]


# ----------------------------------------------------------------------
# 5. 端到端（mock LLM + 真实 registry + 真实工具）
# ----------------------------------------------------------------------

def test_e2e_mock_llm_real_tools_roundtrip() -> None:
    """模型驱动两次工具调用后作答：真实 calc_expression 全程参与。"""
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [_call("calc_expression",
                                               {"expr": "100 - 30"}, call_id="s1")]},
        {"content": None, "tool_calls": [_call("calc_expression",
                                               {"expr": "70 / 100"}, call_id="s2")]},
        {"content": "70%", "tool_calls": []},
    ])
    harness = ToolCallingHarness(llm, registry=_registry_with_calc())
    result = harness.run("what is the gross margin?")
    assert result.answer == "70%"
    assert result.stopped_reason == "finished"
    assert result.turns == 3
    # 第二轮的 user 视角消息里应含第一轮工具结果（上下文累积）
    turn2_messages, _ = llm.received[1]
    assert any(m["role"] == "tool" for m in turn2_messages)
    # registry 历史与 harness 轨迹一致
    assert len(harness.registry.history()) == 2


def test_harness_result_is_replayable_dataclass() -> None:
    """HarnessResult 为纯数据（messages 可直接 json 序列化回放）。"""
    llm = ScriptedLLM([
        {"content": None, "tool_calls": [_call("calc_expression", {"expr": "2 ** 3"})]},
        {"content": "8", "tool_calls": []},
    ])
    result = ToolCallingHarness(llm, registry=_registry_with_calc()).run("q")
    dumped = json.dumps(result.messages, ensure_ascii=False)
    assert "tool" in dumped
    assert isinstance(result, HarnessResult)
    assert isinstance(result.tool_calls[0], ToolCall)
