import json
import pytest
import claude_code_server as shim

TOOLS = [{"type": "function", "function": {"name": "terminal"}}]

@pytest.mark.parametrize("fenced", [False, True])
def test_malformed_tool_envelope_is_not_an_answer(fenced):
    # Actual failure shape: the call object is missing its closing brace.
    text = '{"tool_calls":[{"name":"terminal","arguments":{"command":"pwd","timeout":30}]}'
    if fenced:
        text = '```json\n' + text + '\n```'
    with pytest.raises(ValueError, match="tool-call envelope"):
        shim.extract_native_tool_calls(text, TOOLS)


def test_valid_fenced_call_and_plain_answer():
    text = json.dumps({"tool_calls": [{"name": "terminal", "arguments": {"command": "pwd"}}]})
    calls = shim.extract_native_tool_calls('```json\n' + text + '\n```', TOOLS)
    assert calls[0]["function"]["name"] == "terminal"
    assert json.loads(calls[0]["function"]["arguments"]) == {"command": "pwd"}
    assert shim.extract_native_tool_calls("Here is the answer.", TOOLS) == []


@pytest.mark.parametrize("stream", [False, True])
def test_http_rejects_malformed_envelope_before_success_headers(monkeypatch, stream):
    import threading
    import urllib.request
    import urllib.error
    from http.server import ThreadingHTTPServer
    monkeypatch.setenv("CLAUDE_CODE_CLI_API_KEY", "test-key")
    monkeypatch.setenv("CLAUDE_CODE_CLI_ENGINE", "never")
    monkeypatch.setenv("CLAUDE_CODE_CLI_NATIVE_TOOLS", "1")
    monkeypatch.setattr(shim, "_run_mcp_cli", lambda *a, **k: {
        "events": [{"type": "result", "result": '```json\n{"tool_calls":[{"name":"terminal","arguments":{"command":"pwd"}]}\n```',
                    "modelUsage": {"claude-fable-5-1": {}}, "usage": {}}],
        "returncode": 0, "stderr": "", "timed_out": False})
    server = ThreadingHTTPServer(("127.0.0.1", 0), shim.Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        body = {"model": "claude-fable-5-1", "messages": [{"role": "user", "content": "Use terminal"}],
                "tools": TOOLS, "stream": stream}
        req = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer test-key"})
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(req, timeout=5)
        assert caught.value.code == 503
        response = json.loads(caught.value.read())
        assert "invalid tool-call envelope" in response["error"]["message"]
        assert "choices" not in response
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
