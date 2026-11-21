"""Structured CLI input keeps the conversation prompt-cacheable.

Before this, every request piped the whole flattened conversation to
``claude -p`` as ONE text block with the response instruction appended last.
A growing single block is never a block-aligned prefix of the next request,
so each agent-loop step re-wrote the full conversation as cache-creation
tokens (measured ~70-90K cache writes per call on the Personal shim).
"""
import importlib.util
import json
import os
import pathlib
import stat
import tempfile
import unittest
from contextlib import contextmanager

_spec = importlib.util.spec_from_file_location(
    "structured_cache_server",
    pathlib.Path(__file__).resolve().parents[1] / "claude_code_server.py",
)
assert _spec is not None and _spec.loader is not None
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


@contextmanager
def env(**values):
    old = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def fake_cli(directory: pathlib.Path, body: str) -> pathlib.Path:
    path = directory / "claude"
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def convo(steps: int) -> list[dict]:
    messages: list[dict] = [
        {"role": "system", "content": "You are Hermes."},
        {"role": "user", "content": "Do the task."},
    ]
    for i in range(steps):
        messages.append({"role": "assistant", "content": f"step {i}",
                         "tool_calls": [{"id": f"c{i}", "type": "function",
                                         "function": {"name": "terminal", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"result {i}"})
    return messages


class StructuredPromptTests(unittest.TestCase):
    def structured(self, messages, native=True):
        blocks, instruction = S.flatten_message_parts(messages, native_tools=native)
        return S.StructuredPrompt(blocks, instruction)

    def test_text_is_byte_identical_to_legacy_flatten(self):
        for native in (False, True):
            messages = convo(3)
            self.assertEqual(str(self.structured(messages, native)),
                             S.flatten_messages(messages, native_tools=native))

    def test_growth_only_appends_blocks(self):
        # The cache property: request N's blocks are an exact prefix of N+1's.
        for steps in range(0, 6):
            before = self.structured(convo(steps)).blocks
            after = self.structured(convo(steps + 1)).blocks
            self.assertEqual(after[:len(before)], before)
            self.assertEqual(len(after), len(before) + 2)

    def test_instruction_moves_to_system_prompt_not_conversation_tail(self):
        prompt = self.structured(convo(2))
        extra, stdin_text, structured = S._cli_input(prompt)
        self.assertTrue(structured)
        self.assertEqual(extra[:2], ["--input-format", "stream-json"])
        self.assertEqual(extra[extra.index("--append-system-prompt") + 1], prompt.instruction)
        message = json.loads(stdin_text)
        texts = [part["text"] for part in message["message"]["content"]]
        self.assertEqual(texts, list(prompt.blocks))
        self.assertNotIn(prompt.instruction, stdin_text)
        self.assertTrue(texts[0].startswith("SYSTEM INSTRUCTIONS:"))
        self.assertTrue(texts[1].startswith("CONVERSATION:\nUSER: Do the task."))

    def test_plain_string_and_kill_switch_use_legacy_stdin(self):
        self.assertEqual(S._cli_input("hello"), ([], "hello", False))
        prompt = self.structured(convo(1))
        with env(CLAUDE_CODE_CLI_STRUCTURED_INPUT="0"):
            self.assertEqual(S._cli_input(prompt), ([], str(prompt), False))
        empty = S.StructuredPrompt([], "Respond.")
        self.assertFalse(S._cli_input(empty)[2])

    def test_append_keeps_structure(self):
        prompt = self.structured(convo(1))
        forced = S._append_prompt_text(prompt, "You must use a tool.")
        self.assertIsInstance(forced, S.StructuredPrompt)
        self.assertEqual(forced.blocks[:-1], prompt.blocks)
        self.assertEqual(forced.blocks[-1], "You must use a tool.")
        self.assertTrue(str(forced).endswith("You must use a tool."))
        self.assertEqual(S._append_prompt_text("x", "y"), "x\n\ny")

    def test_block_cap_groups_are_stable_while_appending(self):
        blocks = [f"b{i}" for i in range(10)]
        self.assertEqual(S._group_prompt_blocks(blocks, 20), blocks)
        g11 = S._group_prompt_blocks([f"b{i}" for i in range(11)], 5)
        g12 = S._group_prompt_blocks([f"b{i}" for i in range(12)], 5)
        self.assertLessEqual(len(g12), 5)
        self.assertEqual(g12[:len(g11) - 1], g11[:-1])

    def test_buffered_run_uses_stream_json_input_and_parses_result(self):
        with tempfile.TemporaryDirectory() as td:
            capture = pathlib.Path(td) / "argv.json"
            cli = fake_cli(pathlib.Path(td), f"""
import json, sys
argv = sys.argv[1:]
message = json.loads(sys.stdin.readline())
json.dump({{"argv": argv, "message": message}}, open({str(capture)!r}, "w"))
print(json.dumps({{"type": "system", "subtype": "init"}}))
print(json.dumps({{"type": "assistant", "message": {{"model": "claude-opus-5-5", "content": [{{"type": "text", "text": "OK"}}]}}}}))
print(json.dumps({{"type": "result", "is_error": False, "result": "OK",
                   "usage": {{"input_tokens": 3, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 40, "output_tokens": 2}},
                   "modelUsage": {{"claude-opus-5-5": {{"inputTokens": 3}}}}}}))
""")
            prompt = self.structured(convo(2), native=False)
            with env(CLAUDE_CODE_CLI_BIN=str(cli), CLAUDE_CODE_CLI_TIMEOUT="10"):
                outcome = S.run_claude(prompt, "claude-opus-5-5")
            seen = json.loads(capture.read_text())
        self.assertEqual(outcome["error"], "")
        self.assertEqual(outcome["text"], "OK")
        self.assertEqual(outcome["model"], "claude-opus-5-5")
        self.assertEqual(outcome["usage"]["cache_read_input_tokens"], 900)
        argv = seen["argv"]
        self.assertEqual(argv[argv.index("--output-format") + 1], "stream-json")
        self.assertEqual(argv[argv.index("--input-format") + 1], "stream-json")
        self.assertNotIn("--include-partial-messages", argv)
        self.assertEqual(argv[argv.index("--append-system-prompt") + 1], prompt.instruction)
        self.assertEqual([p["text"] for p in seen["message"]["message"]["content"]], list(prompt.blocks))

    def test_structured_run_never_returns_raw_jsonl_as_answer(self):
        with tempfile.TemporaryDirectory() as td:
            cli = fake_cli(pathlib.Path(td), """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "system", "subtype": "init"}))
""")
            with env(CLAUDE_CODE_CLI_BIN=str(cli), CLAUDE_CODE_CLI_TIMEOUT="10"):
                outcome = S.run_claude(self.structured(convo(1), native=False), "claude-opus-5-5")
        self.assertTrue(outcome["error"])
        self.assertNotIn("subtype", outcome["text"])

    def test_structured_error_result_is_classified_failure(self):
        with tempfile.TemporaryDirectory() as td:
            cli = fake_cli(pathlib.Path(td), """
import json, sys
sys.stdin.read()
print(json.dumps({"type": "result", "is_error": True, "result": "You've hit your weekly limit"}))
sys.exit(1)
""")
            with env(CLAUDE_CODE_CLI_BIN=str(cli), CLAUDE_CODE_CLI_TIMEOUT="10"):
                outcome = S.run_claude(self.structured(convo(1), native=False), "claude-opus-5-5")
        self.assertIn("weekly limit", outcome["error"])
        status, _kind, _msg = S.classify_cli_error(outcome["error"])
        self.assertEqual(status, 429)


if __name__ == "__main__":
    unittest.main()
