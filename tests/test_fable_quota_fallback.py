import importlib
import importlib.util
import os
import pathlib
import tempfile
import unittest
from unittest import mock


def _load_server():
    configured = os.environ.get("SHIM_MODULE")
    if configured:
        path = pathlib.Path(configured)
        if path.is_file():
            spec = importlib.util.spec_from_file_location("shim_under_test", path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return importlib.import_module(configured)
    path = pathlib.Path(__file__).with_name("claude_code_server.py")
    if not path.is_file():
        path = pathlib.Path(__file__).resolve().parents[1] / "claude_code_server.py"
    spec = importlib.util.spec_from_file_location("shim_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S = _load_server()
LIMIT = "You've reached your Fable limit. Switch to another model, or manage usage credits."
ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def result(text="ok", error=""):
    return {"text": text, "usage": dict(ZERO), "error": error}


class FableFallbackTests(unittest.TestCase):
    def setUp(self):
        root = os.environ.get("TMPDIR", tempfile.gettempdir())
        self.tmp = tempfile.TemporaryDirectory(dir=root)
        self.personal = pathlib.Path(self.tmp.name, "personal")
        self.hedra = pathlib.Path(self.tmp.name, "hedra")
        self.personal.mkdir()
        self.hedra.mkdir()
        self.env = {
            "PATH": os.environ.get("PATH", ""),
            "TMPDIR": root,
            "CLAUDE_CONFIG_DIR": str(self.personal),
            "CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR": str(self.hedra),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def run_mocked(self, outcomes, model="fable", engine=False, env=None):
        merged = dict(self.env if env is None else env)
        with mock.patch.dict(os.environ, merged, clear=True), \
             mock.patch.object(S, "_model_family", side_effect=lambda m: "fable" if m == "fable" else "sonnet"), \
             mock.patch.object(S, "_fable_provenance_retries", return_value=0), \
             mock.patch.object(S, "_is_fable_provenance_substitution", return_value=False), \
             mock.patch.object(S, "_run_claude_once", side_effect=outcomes) as once:
            value = S.run_claude("same prompt", model, engine, {"x": 1})
            return value, once

    def test_quota_routes_to_hedra_with_same_request(self):
        value, once = self.run_mocked([result(error=LIMIT), result("hedra")])
        self.assertEqual(value["text"], "hedra")
        self.assertEqual(value["account_source"], "hedra")
        self.assertIs(value["account_fallback"], True)
        self.assertEqual(once.call_count, 2)
        for call in once.call_args_list:
            self.assertEqual(call.args[:4], ("same prompt", "fable", False, {"x": 1}))
        self.assertIsNone(once.call_args_list[0].kwargs["subprocess_env"])
        self.assertEqual(once.call_args_list[1].kwargs["subprocess_env"]["CLAUDE_CONFIG_DIR"], str(self.hedra.resolve()))

    def test_healthy_personal_does_not_fallback(self):
        value, once = self.run_mocked([result("personal")])
        self.assertEqual(value["text"], "personal")
        self.assertEqual(once.call_count, 1)

    def test_non_fable_quota_does_not_fallback(self):
        value, once = self.run_mocked([result(error=LIMIT)], model="sonnet")
        self.assertEqual(value["error"], LIMIT)
        self.assertEqual(once.call_count, 1)

    def test_missing_opt_in_does_not_fallback(self):
        env = dict(self.env)
        env.pop("CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR")
        _, once = self.run_mocked([result(error=LIMIT)], env=env)
        self.assertEqual(once.call_count, 1)

    def test_generic_and_auth_errors_do_not_fallback(self):
        for error in ("HTTP 429", "HTTP 503", "authentication failed", "request timed out", "rate limit exceeded", "request quota exceeded"):
            with self.subTest(error=error):
                _, once = self.run_mocked([result(error=error)])
                self.assertEqual(once.call_count, 1)

    def test_successful_text_mentioning_quota_does_not_fallback(self):
        value, once = self.run_mocked([result(LIMIT, "")])
        self.assertEqual(value["text"], LIMIT)
        self.assertEqual(once.call_count, 1)

    def test_missing_or_same_directory_does_not_hop(self):
        cases = [pathlib.Path(self.tmp.name, "missing"), self.personal]
        for target in cases:
            with self.subTest(target=target):
                env = dict(self.env)
                env["CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR"] = str(target)
                _, once = self.run_mocked([result(error=LIMIT)], env=env)
                self.assertEqual(once.call_count, 1)

    def test_hedra_failure_is_returned_without_extra_hop(self):
        failure = result(error="Hedra unavailable")
        value, once = self.run_mocked([result(error=LIMIT), failure])
        self.assertEqual(once.call_count, 2)
        self.assertEqual(value["error"], "Hedra unavailable")
        self.assertEqual(value["account_source"], "hedra")
        self.assertIs(value["account_fallback"], True)

    def test_fallback_env_is_local_and_removes_sensitive_overrides(self):
        env = dict(self.env)
        sensitive = (
            "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_BASE_URL",
        )
        env.update({name: "secret" for name in sensitive})
        before = dict(os.environ)
        _, once = self.run_mocked([result(error=LIMIT), result("ok")], env=env)
        fallback_env = once.call_args_list[1].kwargs["subprocess_env"]
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(fallback_env["PATH"], S.build_subprocess_env()["PATH"])
        for name in sensitive:
            self.assertNotIn(name, fallback_env)
        self.assertEqual(dict(os.environ), before)

    def test_engine_request_is_not_replayed(self):
        _, once = self.run_mocked([result(error=LIMIT)], engine=True)
        self.assertEqual(once.call_count, 1)

    def test_provenance_retries_apply_on_both_accounts(self):
        substitutions = iter([True, False, True, False])
        outcomes = [result(error="substitution"), result(error=LIMIT),
                    result(error="substitution"), result("hedra")]
        with mock.patch.dict(os.environ, self.env, clear=True), \
             mock.patch.object(S, "_model_family", return_value="fable"), \
             mock.patch.object(S, "_fable_provenance_retries", return_value=1), \
             mock.patch.object(S, "_is_fable_provenance_substitution", side_effect=lambda m, o: next(substitutions)), \
             mock.patch.object(S, "_run_claude_once", side_effect=outcomes) as once:
            value = S.run_claude("p", "fable")
        self.assertEqual(once.call_count, 4)
        self.assertEqual(value["text"], "hedra")
        self.assertEqual(value["account_source"], "hedra")


if __name__ == "__main__":
    unittest.main()
