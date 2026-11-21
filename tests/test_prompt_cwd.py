"""Non-engine CLI calls run from a fixed directory outside any git repository.

Claude Code adds its working directory's git status to every prompt. The shim
used to inherit its own checkout as the CLI cwd, so any edit or commit there
changed the prompt prefix of every active session and forced a full
prompt-cache rewrite. Engine mode keeps its explicit ``CLAUDE_CODE_CLI_CWD``.
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
    "prompt_cwd_server",
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


def cwd_recording_cli(directory: pathlib.Path, capture: pathlib.Path) -> pathlib.Path:
    """Fake ``claude`` that records its cwd and returns one stream-json result."""
    path = directory / "claude"
    path.write_text(f"""#!/usr/bin/env python3
import json, os, sys
sys.stdin.read()
open({str(capture)!r}, "w").write(os.getcwd())
print(json.dumps({{"type": "system", "subtype": "init"}}))
print(json.dumps({{"type": "assistant", "message": {{"model": "claude-opus-5-5",
                   "content": [{{"type": "text", "text": "OK"}}]}}}}))
print(json.dumps({{"type": "result", "is_error": False, "result": "OK",
                   "usage": {{"input_tokens": 1, "output_tokens": 1}},
                   "modelUsage": {{"claude-opus-5-5": {{"inputTokens": 1}}}}}}))
""", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def is_inside_git_repo(path: str) -> bool:
    p = pathlib.Path(path).resolve()
    return any((parent / ".git").exists() for parent in (p, *p.parents))


class PromptCwdTests(unittest.TestCase):
    def test_explicit_override_is_created_and_used(self):
        with tempfile.TemporaryDirectory() as td:
            target = pathlib.Path(td) / "nested" / "workdir"
            with env(CLAUDE_CODE_CLI_PROMPT_CWD=str(target)):
                self.assertEqual(S._prompt_cwd(), str(target))
            self.assertTrue(target.is_dir())

    def test_default_lives_under_hermes_home_state(self):
        with tempfile.TemporaryDirectory() as td:
            with env(CLAUDE_CODE_CLI_PROMPT_CWD=None, HERMES_HOME=td):
                got = S._prompt_cwd()
            self.assertEqual(got, str(pathlib.Path(td) / "state" / "claude-cli-workdir"))
            self.assertTrue(pathlib.Path(got).is_dir())

    def test_default_is_stable_across_calls(self):
        with tempfile.TemporaryDirectory() as td:
            with env(CLAUDE_CODE_CLI_PROMPT_CWD=None, HERMES_HOME=td):
                self.assertEqual(S._prompt_cwd(), S._prompt_cwd())

    def _run_and_capture(self, invoke) -> str:
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            capture = root / "cwd.txt"
            workdir = root / "workdir"
            cli = cwd_recording_cli(root, capture)
            with env(CLAUDE_CODE_CLI_BIN=str(cli), CLAUDE_CODE_CLI_TIMEOUT="10",
                     CLAUDE_CODE_CLI_PROMPT_CWD=str(workdir)):
                invoke(root)
            seen = capture.read_text()
            self.assertEqual(os.path.realpath(seen), os.path.realpath(str(workdir)))
            self.assertFalse(is_inside_git_repo(seen))
            return seen

    def test_buffered_text_call_uses_prompt_cwd(self):
        def invoke(_root):
            outcome = S.run_claude("hello", "claude-opus-5-5")
            self.assertEqual(outcome["error"], "")
        self._run_and_capture(invoke)

    def test_streaming_text_call_uses_prompt_cwd(self):
        def invoke(_root):
            events = list(S.stream_claude("hello", "claude-opus-5-5"))
            self.assertEqual(events[-1][0], "final")
            self.assertEqual(events[-1][1]["error"], "")
        self._run_and_capture(invoke)

    def test_mcp_bridge_call_uses_prompt_cwd(self):
        def invoke(root):
            config = root / "mcp.json"
            config.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
            result = S._run_mcp_cli("hello", "claude-opus-5-5", config, "mcp__hermes__probe")
            self.assertEqual(result["returncode"], 0)
        self._run_and_capture(invoke)

    def test_engine_mode_keeps_engine_cwd(self):
        with tempfile.TemporaryDirectory() as td:
            root = pathlib.Path(td)
            capture = root / "cwd.txt"
            engine_dir = root / "engine"
            engine_dir.mkdir()
            cli = cwd_recording_cli(root, capture)
            with env(CLAUDE_CODE_CLI_BIN=str(cli), CLAUDE_CODE_CLI_TIMEOUT="10",
                     CLAUDE_CODE_CLI_CWD=str(engine_dir),
                     CLAUDE_CODE_CLI_PROMPT_CWD=str(root / "workdir")):
                S.run_claude("hello", "claude-opus-5-5", engine=True)
            self.assertEqual(os.path.realpath(capture.read_text()),
                             os.path.realpath(str(engine_dir)))


if __name__ == "__main__":
    unittest.main()
