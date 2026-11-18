"""Account fallback is a routing decision, not a transport one.

The shim used to replay a Fable-exhausted request against a second Claude
config dir. That hid the account switch from Hermes' session provenance and
cost one extra OAuth login per shim wanting the fallback. The shim now returns
a classified error and Hermes' own fallback_providers chain picks the next
route.

Classification matters as much as the deletion: a 503 maps to
FailoverReason.overloaded, which carries no should_fallback and only reaches
the chain after the retry budget is spent. Account exhaustion is deterministic
for this route, so it must classify billing-class for an immediate cascade.
"""

import importlib
import importlib.util
import os
import pathlib
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

FABLE_LIMIT = (
    "You've reached your Fable limit. Switch to another model, "
    "or manage usage credits."
)
SESSION_LIMIT = (
    "You've hit your session limit \u00b7 resets 7:20pm (America/Los_Angeles)"
)
ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def result(text="ok", error=""):
    return {"text": text, "usage": dict(ZERO), "error": error}


class NoAccountReplayTests(unittest.TestCase):
    def test_exhaustion_returns_error_without_a_second_invocation(self):
        env = {
            "PATH": os.environ.get("PATH", ""),
            # Deliberately still set: the retired opt-in must be inert, so a
            # stale env file cannot resurrect the replay.
            "CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR": "/tmp",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(S, "_fable_provenance_retries", return_value=0), \
             mock.patch.object(S, "_is_fable_provenance_substitution", return_value=False), \
             mock.patch.object(S, "_run_claude_once",
                               side_effect=[result(error=FABLE_LIMIT)]) as once:
            outcome = S.run_claude("p", "claude-fable-5-1")
        self.assertEqual(once.call_count, 1)
        self.assertEqual(outcome["error"], FABLE_LIMIT)
        self.assertNotIn("account_fallback", outcome)
        self.assertNotIn("account_source", outcome)

    def test_account_fallback_helpers_are_gone(self):
        for name in ("_fable_fallback_env", "_is_fable_exhaustion",
                     "_FABLE_EXHAUSTION_RE"):
            self.assertFalse(hasattr(S, name), f"{name} should be deleted")

    def test_provenance_retry_still_runs(self):
        """Served-model provenance is unrelated to account routing; keep it."""
        subs = iter([True, False])
        with mock.patch.dict(os.environ, {"PATH": os.environ.get("PATH", "")}, clear=True), \
             mock.patch.object(S, "_fable_provenance_retries", return_value=1), \
             mock.patch.object(S, "_is_fable_provenance_substitution",
                               side_effect=lambda m, o: next(subs)), \
             mock.patch.object(S, "_run_claude_once",
                               side_effect=[result(error="sub"), result("good")]) as once:
            outcome = S.run_claude("p", "claude-fable-5-1")
        self.assertEqual(once.call_count, 2)
        self.assertEqual(outcome["text"], "good")


class ExhaustionClassificationTests(unittest.TestCase):
    def test_fable_limit_is_billing_class_429(self):
        status, etype, msg = S.classify_cli_error(FABLE_LIMIT)
        self.assertEqual(status, 429)
        self.assertEqual(etype, "rate_limit_error")
        self.assertIn("credits exhausted", msg.lower())
        # The original wording survives for the operator reading the log.
        self.assertIn("Fable limit", msg)

    def test_session_limit_is_billing_class_429(self):
        status, _etype, msg = S.classify_cli_error(SESSION_LIMIT)
        self.assertEqual(status, 429)
        self.assertIn("credits exhausted", msg.lower())
        self.assertIn("session limit", msg)

    def test_weekly_limit_is_billing_class_429(self):
        for text in (
            "You've hit your weekly limit · resets Oct 1 at 5am (America/Los_Angeles)",
            "You've hit your weekly usage limit",
        ):
            with self.subTest(text=text):
                status, etype, message = S.classify_cli_error(text)
                self.assertEqual((status, etype), (429, "rate_limit_error"))
                self.assertEqual(message, "credits exhausted: " + text)

    def test_normalization_is_idempotent(self):
        once = S.classify_cli_error(FABLE_LIMIT)[2]
        twice = S.classify_cli_error(once)[2]
        self.assertEqual(once, twice)

    def test_ordinary_rate_limit_stays_a_plain_429(self):
        status, _etype, msg = S.classify_cli_error("rate limit exceeded")
        self.assertEqual(status, 429)
        self.assertNotIn("credits exhausted", msg.lower())

    def test_model_auth_and_transport_paths_are_untouched(self):
        self.assertEqual(S.classify_cli_error("It may not exist")[0], 404)
        self.assertEqual(S.classify_cli_error("not logged in")[0], 401)
        self.assertEqual(S.classify_cli_error("request timed out")[0], 503)


if __name__ == "__main__":
    unittest.main()
