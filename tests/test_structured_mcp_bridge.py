from __future__ import annotations

import json
import os
import pathlib
import stat
import subprocess
import sys
import tempfile

import claude_code_server as server


ROOT = pathlib.Path(__file__).resolve().parents[1]
BRIDGE = ROOT / "hermes_mcp_bridge.py"


def _rpc(proc: subprocess.Popen, request: dict) -> dict:
    assert proc.stdin is not None and proc.stdout is not None
    proc.stdin.write(json.dumps(request) + "\n")
    proc.stdin.flush()
    return json.loads(proc.stdout.readline())


def test_bridge_protocol_round_trips_multiline_unicode_and_never_executes():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        spec = root / "spec.json"
        events = root / "events.jsonl"
        spec.write_text(json.dumps({
            "request_id": "req-test",
            "tools": [{
                "name": "probe",
                "description": "Capture only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                    "additionalProperties": False,
                },
            }],
        }), encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(BRIDGE), str(spec), str(events)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            init = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
            listed = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
            text = 'line 1 "quoted"\nline 2 café'
            called = _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                "name": "probe", "arguments": {"text": text},
                "_meta": {"claudecode/toolUseId": "toolu_test"},
            }})
        finally:
            proc.terminate()
            proc.wait(timeout=5)

        assert init["result"]["serverInfo"]["name"] == "hermes-openai-tool-bridge"
        assert [tool["name"] for tool in listed["result"]["tools"]] == ["probe"]
        assert called["result"]["isError"] is False
        assert "deferred" in called["result"]["content"][0]["text"].lower()
        event = json.loads(events.read_text(encoding="utf-8").splitlines()[0])
        assert event == {
            "request_id": "req-test",
            "status": "accepted",
            "name": "probe",
            "arguments": {"text": text},
            "tool_use_id": "toolu_test",
        }


def test_bridge_rejects_unknown_tool_and_schema_invalid_arguments():
    with tempfile.TemporaryDirectory() as td:
        root = pathlib.Path(td)
        spec = root / "spec.json"
        events = root / "events.jsonl"
        spec.write_text(json.dumps({
            "request_id": "req-invalid",
            "tools": [{
                "name": "probe",
                "description": "Capture only.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer", "minimum": 1}},
                    "required": ["count"],
                    "additionalProperties": False,
                },
            }],
        }), encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(BRIDGE), str(spec), str(events)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        try:
            unknown = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
                "name": "missing", "arguments": {},
            }})
            invalid = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "probe", "arguments": {"count": 0, "extra": True},
            }})
        finally:
            proc.terminate()
            proc.wait(timeout=5)

        assert unknown["result"]["isError"] is True
        assert invalid["result"]["isError"] is True
        records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
        assert [record["status"] for record in records] == ["rejected", "rejected"]
        assert "unknown tool" in records[0]["error"]
        assert "arguments" in records[1]["error"]


def _make_fake_mcp_claude(directory: pathlib.Path) -> pathlib.Path:
    path = directory / "claude"
    path.write_text(r'''#!/usr/bin/env python3
import json, subprocess, sys

prompt = sys.stdin.read()
assert "Available Hermes-native tools" not in prompt
config_path = sys.argv[sys.argv.index("--mcp-config") + 1]
config = json.load(open(config_path, encoding="utf-8"))
entry = config["mcpServers"]["hermes"]
proc = subprocess.Popen([entry["command"], *entry["args"]], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def rpc(r):
    proc.stdin.write(json.dumps(r) + "\n"); proc.stdin.flush()
    return json.loads(proc.stdout.readline())
rpc({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})
listed = rpc({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
assert [t["name"] for t in listed["result"]["tools"]] == ["lookup_value"]
args = {"key": "alpha\n\"beta\""}
rpc({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
    "name":"lookup_value","arguments":args,"_meta":{"claudecode/toolUseId":"toolu_live"}}})
proc.terminate(); proc.wait()
def emit(obj): print(json.dumps(obj), flush=True)
emit({"type":"system","subtype":"init","tools":["mcp__hermes__lookup_value"],"mcp_servers":[{"name":"hermes","status":"connected"}],"model":"claude-opus-5"})
emit({"type":"assistant","message":{"model":"claude-opus-5","content":[{"type":"tool_use","id":"toolu_live","name":"mcp__hermes__lookup_value","input":args}]}})
emit({"type":"result","is_error":True,"terminal_reason":"max_turns","result":"Reached maximum number of turns (1)","usage":{"input_tokens":2,"output_tokens":3},"modelUsage":{"claude-opus-5":{"contextWindow":1000000}}})
sys.exit(1)
''', encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_mcp_selection_returns_openai_tool_call_with_billed_provenance():
    tools = [{
        "type": "function",
        "function": {
            "name": "lookup_value",
            "description": "Look up a harmless value.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
                "additionalProperties": False,
            },
        },
    }]
    with tempfile.TemporaryDirectory() as td:
        fake = _make_fake_mcp_claude(pathlib.Path(td))
        old = {key: os.environ.get(key) for key in ("CLAUDE_CODE_CLI_BIN", "CLAUDE_CODE_CLI_TIMEOUT")}
        os.environ["CLAUDE_CODE_CLI_BIN"] = str(fake)
        os.environ["CLAUDE_CODE_CLI_TIMEOUT"] = "5"
        try:
            outcome = server.run_claude_mcp_bridge(
                "USER: use the lookup", "opus", tools, "required", overrides={}
            )
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    assert outcome["error"] == ""
    assert outcome["model"] == "claude-opus-5"
    assert outcome["usage"]["total_tokens"] == 5
    assert len(outcome["tool_calls"]) == 1
    call = outcome["tool_calls"][0]
    assert call["id"] == "toolu_live"
    assert call["function"]["name"] == "lookup_value"
    assert json.loads(call["function"]["arguments"]) == {"key": 'alpha\n"beta"'}


def test_mcp_selection_rejects_capture_not_matching_model_event(monkeypatch):
    monkeypatch.setattr(server, "_run_mcp_cli", lambda *args, **kwargs: {
        "events": [{"type": "assistant", "message": {"model": "claude-opus-5", "content": [{
            "type": "tool_use", "id": "toolu_model", "name": "mcp__hermes__probe", "input": {"x": 1},
        }]}}, {"type": "result", "usage": {}, "modelUsage": {"claude-opus-5": {}}}],
        "returncode": 0, "stderr": "", "timed_out": False,
    })
    monkeypatch.setattr(server, "_read_bridge_events", lambda *args, **kwargs: [{
        "request_id": kwargs.get("request_id", "wrong"), "status": "accepted", "name": "probe",
        "arguments": {"x": 2}, "tool_use_id": "toolu_model",
    }])
    tools = [{"type": "function", "function": {"name": "probe", "parameters": {"type": "object"}}}]
    outcome = server.run_claude_mcp_bridge("prompt", "opus", tools, "required", overrides={})
    assert outcome["tool_calls"] == []
    assert "bridge capture mismatch" in outcome["error"]
