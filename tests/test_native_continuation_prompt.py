from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import claude_code_server as server


TOOLS = [{"type": "function", "function": {
    "name": "next_step", "parameters": {"type": "object", "properties": {},
                                    "additionalProperties": False},
}}]


def _post(body):
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as exc:
            return exc.code, json.load(exc)
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def test_native_result_turn_requests_continuation_without_changing_history(monkeypatch):
    prompts = []

    def select(prompt, model, tools, choice, overrides=None):
        prompts.append(prompt)
        return {"text": "Still working", "usage": server._zero_usage(), "error": "",
                "model": "claude-opus-5", "tool_calls": []}

    monkeypatch.setattr(server, "run_claude_mcp_bridge", select)
    monkeypatch.setenv("CLAUDE_CODE_CLI_ENGINE", "auto")
    monkeypatch.setenv("CLAUDE_CODE_CLI_NATIVE_TOOLS", "1")
    status, response = _post({"model": "opus", "tools": TOOLS, "tool_choice": "auto",
        "messages": [
            {"role": "user", "content": "Complete the remaining steps."},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1",
                "type": "function", "function": {"name": "next_step", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "receipt: café\nsecond line"},
        ]})
    assert status == 200
    assert response["choices"][0]["message"]["content"] == "Still working"
    prompt = prompts[0]
    assert prompt.index("USER: Complete") < prompt.index("[requested tools: mcp__hermes__next_step]") < prompt.index("TOOL RESULT: receipt: café\nsecond line")
    assert "provided via MCP" in prompt
    assert "Hermes executes it" in prompt
    assert "continue any pending work" in prompt
    assert "only when the work is done or genuinely blocked" in prompt
    assert "Do not infer a per-turn tool cap" in prompt
    assert "Return only the response text." not in prompt


def test_native_ordinary_final_and_required_tool_still_fails_closed(monkeypatch):
    prompts = []

    def cli(prompt, *args, **kwargs):
        prompts.append(prompt)
        return {"events": [{"type": "assistant", "message": {
            "model": "claude-opus-5", "content": [{"type": "text", "text": "HELLO"}]}},
            {"type": "result", "result": "HELLO", "modelUsage": {"claude-opus-5": {}}}],
            "returncode": 0, "stderr": "", "timed_out": False}

    monkeypatch.setattr(server, "_run_mcp_cli", cli)
    monkeypatch.setattr(server, "_read_bridge_events", lambda *args, **kwargs: [])
    monkeypatch.setenv("CLAUDE_CODE_CLI_ENGINE", "auto")
    monkeypatch.setenv("CLAUDE_CODE_CLI_NATIVE_TOOLS", "1")
    base = {"model": "opus", "tools": TOOLS,
            "messages": [{"role": "user", "content": "Reply exactly HELLO"}]}
    status, response = _post({**base, "tool_choice": "auto"})
    assert status == 200
    assert response["choices"][0]["finish_reason"] == "stop"
    assert response["choices"][0]["message"]["content"] == "HELLO"
    assert "Ordinary responses that need no tool" in prompts[0]
    status, response = _post({**base, "tool_choice": "required"})
    assert status == 400
    assert "required tool was not selected" in response["error"]["message"]


def test_none_and_no_tools_keep_plain_prompt_exactly(monkeypatch):
    prompts = []

    def plain(prompt, model, engine=False, overrides=None):
        prompts.append(prompt)
        return {"text": "HELLO", "usage": server._zero_usage(), "error": "",
                "model": "claude-opus-5"}

    monkeypatch.setattr(server, "run_claude", plain)
    monkeypatch.setenv("CLAUDE_CODE_CLI_ENGINE", "auto")
    messages = [{"role": "user", "content": "Reply exactly HELLO"}]
    expected = ("CONVERSATION:\nUSER: Reply exactly HELLO\n\n"
                "Respond as the assistant to the latest user turn. Return only the response text.")
    for extra in ({}, {"tools": TOOLS, "tool_choice": "none"}):
        status, response = _post({"model": "opus", "messages": messages, **extra})
        assert status == 200
        assert response["choices"][0]["message"]["content"] == "HELLO"
    assert prompts == [expected, expected]
