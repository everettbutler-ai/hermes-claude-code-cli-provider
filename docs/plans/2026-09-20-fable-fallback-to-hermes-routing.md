# Move Fable account fallback from the shim into Hermes routing

> **For Hermes:** implement task-by-task. Every task has its own verification step. Do not batch config edits across profiles.

**Goal:** Delete the in-shim Personal→Hedra Fable replay and let Hermes' own fallback chain route Fable quota exhaustion to the existing `claude-hedra` provider on port 8768.

**Architecture:** The personal shims (8765, 8767) stop spawning a second `claude -p` against a private Hedra config dir. On Fable quota exhaustion they return a classified non-2xx, which `agent/error_classifier.py` already turns into a `should_fallback` verdict. The `claude-hedra` provider (already registered in both configs, already pointed at the supervised 8768 shim) becomes the next chain entry. Account switches then appear in Hermes session metadata instead of being invisible inside the transport.

**Outcome:** 5 Claude config dirs → 3. 3 Hedra OAuth sessions → 1. One code path deleted plus its test file. Account provenance becomes visible in `billing_provider` / `model`.

---

## Current state (verified 2026-09-20)

Shims:

| Port | Service label | `CLAUDE_CONFIG_DIR` | Account |
|---|---|---|---|
| 8765 | `com.hermes.claude-code-cli-shim` | `~/.hermes/claude-code-automation` | personal Max |
| 8767 | `com.hermes.zeus-hedra-shim` (misnamed; it is Zeus personal) | `~/.hermes/profiles/zeus/claude-code-automation` | personal Max |
| 8768 | `com.hermes.claude-code-cli-shim-hedra` | `~/.hermes/profiles/zeus/claude-hedra` | Hedra team |

Fallback-only login dirs to be retired:

- `~/.hermes/claude-hedra-fallback` (Atlas fallback target, Keychain `Claude Code-credentials-e59422f1`)
- `~/.hermes/profiles/zeus/claude-hedra-fallback` (Zeus fallback target, Keychain `Claude Code-credentials-259b231a`)

Providers already present in BOTH `~/.hermes/config.yaml` and `~/.hermes/profiles/zeus/config.yaml`:

```yaml
claude-hedra:
  base_url: http://127.0.0.1:8768/v1
  key_cmd: ... credential_helper.py claude-hedra
  models: [claude-sonnet-5, opus, haiku, claude-fable-5-1]
  api_mode: chat_completions
```

Chains today:

- **Zeus** primary `claude-personal/claude-fable-5-1`; chain `claude-personal/opus` → `shared-codex/gpt-6-astra` → `shared-codex/gpt-5.6-sol`
- **Atlas (root)** primary `custom:shared-claude/opus`; chain `claude-personal/claude-fable-5-1` → `claude-personal/opus` → `shared-codex/gpt-6-astra` → `shared-codex/gpt-5.6-sol`

---

## Two problems this plan must NOT silently introduce

### A. The 8768 shim does not set `CLAUDE_CODE_CLI_TEXT_ONLY_MODELS`

`~/.hermes/profiles/zeus/claude-code-cli-hedra-shim.env` has `CLAUDE_CODE_CLI_NATIVE_TOOLS=1` and **no** `CLAUDE_CODE_CLI_TEXT_ONLY_MODELS` line. Both personal shims set `claude-fable-5,claude-fable-5-1`.

Today the in-shim replay runs inside the personal process, so it inherits that setting. Routing Fable to 8768 loses it, and Anthropic's Fable safeguard trips on native tool schemas (`Fable 5's safeguards flagged this message`) — a 503 that looks like an outage. **Task 1 fixes this and must land before any config change.**

### B. A 429 burns the retry budget before cascading

`classify_cli_error()` maps `usage limit` / `quota` / `credit balance` to **429**. In `agent/error_classifier.py`, 429 → `_V_RATE_LIMIT = _v(rate_limit, should_rotate_credential=True, should_fallback=True)`, and `rate_limit` is a member of `RETRYABLE_CLIENT_REASONS` — "the cascade for these runs after the backoff budget, not immediately." With `agent.api_max_retries: 3` that is several seconds of pointless backoff against a bucket known to be empty.

By contrast `_V_BILLING = _v(billing, retryable=False, should_rotate_credential=True, should_fallback=True)` and `billing` is **not** in `RETRYABLE_CLIENT_REASONS` — immediate cascade.

**Task 3 resolves this by inspection, not assumption.** Read `_BILLING_PATTERNS` in `agent/error_classifier.py` and decide whether a Fable-exhaustion message already matches billing, or whether `classify_cli_error` should route the Fable-exhaustion case to a status/wording that lands on the billing verdict. Do not guess which patterns are in the tuple.

---

## Tasks

### Task 1: Give the Hedra shim the same Fable text-only guard

**Objective:** 8768 stops sending native tool schemas for Fable models.

**Files:**
- Modify: `/Users/everettbutler/.hermes/profiles/zeus/claude-code-cli-hedra-shim.env`

**Step 1:** Add the line after `CLAUDE_CODE_CLI_NATIVE_TOOLS=1`:

```
CLAUDE_CODE_CLI_TEXT_ONLY_MODELS=claude-fable-5,claude-fable-5-1
```

**Step 2:** Restart and confirm the live process carries it.

```bash
launchctl kickstart -k gui/$(id -u)/com.hermes.claude-code-cli-shim-hedra
sleep 4
curl -s -o /dev/null -w '8768 %{http_code}\n' http://127.0.0.1:8768/healthz
PID=$(pgrep -f 'claude_code_server.py' | while read p; do ps -o command= -p $p >/dev/null && echo $p; done)
for p in $(pgrep -f claude_code_server.py); do ps eww -o command= -p $p | tr ' ' '\n' | grep -E 'CLAUDE_CODE_CLI_(PORT|TEXT_ONLY_MODELS)='; done
```

Expected: the 8768 process shows `CLAUDE_CODE_CLI_TEXT_ONLY_MODELS=claude-fable-5,claude-fable-5-1`.

**Step 3: gate.** Direct Fable canary through 8768 with a tool-free request.

```bash
KEY=$(grep -E '^CLAUDE_CODE_CLI_API_KEY=' ~/.hermes/profiles/zeus/claude-code-cli-hedra-shim.env | cut -d= -f2-)
curl -s http://127.0.0.1:8768/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"claude-fable-5-1","messages":[{"role":"user","content":"Reply with exactly: HEDRA-FABLE-OK"}]}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print(d.get("model"), "|", (d.get("choices") or [{}])[0].get("message",{}).get("content") or d.get("error"))'
```

Expected: `claude-fable-5-1 | HEDRA-FABLE-OK`. If Hedra is session-limited, **stop here and wait for reset** — do not proceed to config changes on an unverified route. A session-limit error means the route is correct but unusable now; record it and resume later.

**Step 4: commit nothing** (env files are not in the repo). Note the change in the fleet README at Task 7.

---

### Task 2: Write the failing test for the new shim behavior

**Objective:** Pin that Fable exhaustion leaves the shim as a classified error, with no second CLI invocation.

**Files:**
- Create: `tests/test_fable_quota_no_account_replay.py`
- Test-only; do not touch `claude_code_server.py` yet.

**Step 1:** Write the test.

```python
import os
import unittest
from unittest import mock

from tests.test_fable_quota_fallback import _load_server  # reuse the loader

S = _load_server()
LIMIT = "You've reached your Fable limit. Switch to another model, or manage usage credits."
ZERO = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def result(text="ok", error=""):
    return {"text": text, "usage": dict(ZERO), "error": error}


class NoAccountReplayTests(unittest.TestCase):
    def test_fable_exhaustion_returns_error_without_second_invocation(self):
        env = {
            "PATH": os.environ.get("PATH", ""),
            # Deliberately still set: the opt-in must be inert after removal.
            "CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR": "/tmp",
        }
        with mock.patch.dict(os.environ, env, clear=True), \
             mock.patch.object(S, "_fable_provenance_retries", return_value=0), \
             mock.patch.object(S, "_is_fable_provenance_substitution", return_value=False), \
             mock.patch.object(S, "_run_claude_once",
                               side_effect=[result(error=LIMIT)]) as once:
            outcome = S.run_claude("p", "claude-fable-5-1")
        self.assertEqual(once.call_count, 1)
        self.assertEqual(outcome["error"], LIMIT)
        self.assertNotIn("account_fallback", outcome)
        self.assertNotIn("account_source", outcome)

    def test_exhaustion_classifies_for_failover(self):
        status, etype, _ = S.classify_cli_error(LIMIT)
        self.assertIn(status, (402, 429))
        self.assertNotEqual(status, 200)


if __name__ == "__main__":
    unittest.main()
```

If importing the loader from the sibling test file is awkward under the current runner, copy `_load_server` verbatim rather than restructuring the existing tests.

**Step 2:** Run it, expect FAIL.

```bash
cd ~/.hermes/vendor/hermes-claude-code-cli-provider
python3 -m pytest tests/test_fable_quota_no_account_replay.py -v
```

Expected: the first test fails (`call_count == 2`, `account_fallback` present).

---

### Task 3: Decide the classification status by inspection

**Objective:** Pick 429 vs a billing-class status with evidence, not assumption.

**Files:**
- Read: `/Users/everettbutler/.hermes/hermes-agent/agent/error_classifier.py`

**Step 1:** Print the relevant tuples and the verdict a Fable-limit message produces.

```bash
cd ~/.hermes/hermes-agent
python3 - <<'PY'
import agent.error_classifier as ec
print("BILLING:", ec._BILLING_PATTERNS)
print("RATE:", ec._RATE_LIMIT_PATTERNS)
print("USAGE:", getattr(ec, "_USAGE_LIMIT_PATTERNS", None))
print("RETRYABLE:", sorted(r.value for r in ec.RETRYABLE_CLIENT_REASONS))
PY
```

**Step 2:** Decide.

- If `"usage limit"` / `"manage usage credits"` already sits in `_BILLING_PATTERNS` and the 429 handler prefers billing when the body is not itself a rate-limit phrase, **keep 429**. The message already lands on the non-retryable billing verdict.
- Otherwise, add a Fable-exhaustion branch to `classify_cli_error()` ahead of `_CLI_RATE_PATTERNS` that returns the billing-class status/wording the classifier actually recognizes.

**Step 3:** Record the decision and the evidence in the commit message. Do not proceed on "probably fine."

---

### Task 4: Delete the in-shim account fallback

**Objective:** `run_claude` becomes a thin wrapper over the provenance retry.

**Files:**
- Modify: `claude_code_server.py` — remove `_fable_fallback_env()` (~769-795), `_is_fable_exhaustion()` (~759-766), `_FABLE_EXHAUSTION_RE` (~749-756), and the fallback body of `run_claude()` (~817-837)
- Delete: `tests/test_fable_quota_fallback.py`
- Modify: the `CLAUDE_CODE_CLI_*` env documentation block near the top of `claude_code_server.py`

**Step 1:** Reduce `run_claude` to:

```python
def run_claude(prompt: str, model: str, engine: bool = False,
               overrides: dict | None = None) -> dict:
    return _run_claude_with_provenance(prompt, model, engine, overrides)
```

Keep `_run_claude_with_provenance` and its `subprocess_env` parameter — the provenance retry still uses it and `_run_claude_once` signature stays stable.

**Step 2:** Apply any `classify_cli_error` change decided in Task 3.

**Step 3:** Remove the now-dead `CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR` mention from the module docstring/env table.

**Step 4:** Run the full suite.

```bash
cd ~/.hermes/vendor/hermes-claude-code-cli-provider
python3 -c "import ast; ast.parse(open('claude_code_server.py').read())"
python3 -m pytest tests/ -q
```

Expected: all pass, including the two new tests. Test count drops by the 11 deleted fallback tests.

**Step 5:** Restart both personal shims and health-check all three.

```bash
launchctl kickstart -k gui/$(id -u)/com.hermes.claude-code-cli-shim
launchctl kickstart -k gui/$(id -u)/com.hermes.zeus-hedra-shim
sleep 5
for p in 8765 8767 8768; do curl -s -o /dev/null -w "$p %{http_code}\n" http://127.0.0.1:$p/healthz; done
```

**Step 6:** Prove the shim now returns a classified error rather than a Hedra answer.

```bash
KEY=$(grep -E '^CLAUDE_CODE_CLI_API_KEY=' ~/.hermes/profiles/zeus/claude-code-cli-shim.env | cut -d= -f2-)
curl -s -w '\nHTTP %{http_code}\n' http://127.0.0.1:8767/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"claude-fable-5-1","messages":[{"role":"user","content":"ping"}]}'
```

Expected while personal Fable is exhausted: non-2xx with the Fable limit message, and **no** `account fallback` line in `~/.hermes/profiles/zeus/logs/hedra-shim.stderr.log`.

**Step 7:** Commit.

```bash
git add claude_code_server.py tests/
git commit -m "refactor: remove in-shim Fable account fallback; classify exhaustion for Hermes failover"
```

---

### Task 5: Insert `claude-hedra` Fable into the Zeus chain

**Objective:** Zeus Fable work continues on Fable via Hedra instead of dropping to Opus.

**Files:**
- Modify: `/Users/everettbutler/.hermes/profiles/zeus/config.yaml` (`fallback_providers`)

**Step 1:** Snapshot.

```bash
cp ~/.hermes/profiles/zeus/config.yaml ~/.hermes/profiles/zeus/config.yaml.bak-$(date +%Y%m%dT%H%M%S)
hermes --profile zeus fallback list
```

**Step 2:** Direct canary for the exact route BEFORE editing.

```bash
hermes --profile zeus chat -q 'Reply with exactly: ZEUS_HEDRA_FABLE_OK' \
  --provider claude-hedra --model claude-fable-5-1 --no-restore-cwd
```

Omit `--safe-mode`: `claude-hedra` is a profile-local custom provider and safe mode would strip it.

**Step 3:** Target chain (order matters — Fable-on-Hedra first, then the existing entries unchanged):

```yaml
fallback_providers:
  - provider: claude-hedra
    model: claude-fable-5-1
    reasoning_effort: medium
  - provider: claude-personal
    model: opus
    reasoning_effort: medium
  - provider: shared-codex
    model: gpt-6-astra
    reasoning_effort: medium
  - provider: shared-codex
    model: gpt-5.6-sol
    reasoning_effort: medium
```

Build it with `hermes --profile zeus fallback add` (never a JSON scalar through `config set`, and never an indexed setter before a real list exists).

**Step 4:** Verify parsed, two independent reads.

```bash
hermes --profile zeus fallback list
python3 - <<'PY'
import yaml
c = yaml.safe_load(open('/Users/everettbutler/.hermes/profiles/zeus/config.yaml'))
fb = c['fallback_providers']
assert isinstance(fb, list), type(fb)
print([(e['provider'], e['model']) for e in fb])
assert (fb[0]['provider'], fb[0]['model']) == ('claude-hedra', 'claude-fable-5-1')
assert len(fb) == 4
print("OK")
PY
```

---

### Task 6: Insert `claude-hedra` Fable into the Atlas (root) chain

**Objective:** Same coverage for the root profile. Its primary is Opus, so Hedra Fable goes **after** the existing personal-Fable entry.

**Files:**
- Modify: `/Users/everettbutler/.hermes/config.yaml` (`fallback_providers`)

**Step 1:** Snapshot and canary from the root profile explicitly.

```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak-$(date +%Y%m%dT%H%M%S)
env HERMES_HOME=/Users/everettbutler/.hermes hermes fallback list
env HERMES_HOME=/Users/everettbutler/.hermes hermes chat \
  -q 'Reply with exactly: ATLAS_HEDRA_FABLE_OK' \
  --provider claude-hedra --model claude-fable-5-1 --no-restore-cwd
```

Qualify with an explicit root `HERMES_HOME`: this shell's environment may resolve a profile even for `-p default`.

**Step 2:** Target chain — one insertion at position 2, everything else preserved:

```yaml
fallback_providers:
  - provider: claude-personal
    model: claude-fable-5-1
    reasoning_effort: medium
  - provider: claude-hedra          # new
    model: claude-fable-5-1
    reasoning_effort: medium
  - provider: claude-personal
    model: opus
    reasoning_effort: medium
  - provider: shared-codex
    model: gpt-6-astra
    reasoning_effort: medium
  - provider: shared-codex
    model: gpt-5.6-sol
    reasoning_effort: medium
```

**Step 3:** Verify programmatically that this is an insertion, not a rebuild.

```bash
python3 - <<'PY'
import glob, yaml
bak = sorted(glob.glob('/Users/everettbutler/.hermes/config.yaml.bak-*'))[-1]
before = [(e['provider'], e['model']) for e in yaml.safe_load(open(bak))['fallback_providers']]
after  = [(e['provider'], e['model']) for e in yaml.safe_load(open('/Users/everettbutler/.hermes/config.yaml'))['fallback_providers']]
new = ('claude-hedra', 'claude-fable-5-1')
assert [e for e in after if e != new] == before, (before, after)
assert after.index(new) == 1
print("insertion verified:", after)
PY
env HERMES_HOME=/Users/everettbutler/.hermes hermes fallback list
```

---

### Task 7: Behavioral failover test (the real acceptance gate)

**Objective:** Prove Hermes — not the shim — moves the request to Hedra, and that the session record says so.

**Step 1:** With personal Fable still quota-exhausted, run a no-override Zeus canary.

```bash
hermes --profile zeus chat -q 'Reply with exactly: FAILOVER_ROUTED_OK' --no-restore-cwd
```

**Step 2:** Read provenance from the session record, not from the answer.

```bash
hermes --profile zeus sessions list --limit 1
# then inspect the latest session's billing_provider / model fields
```

**Acceptance — all four must hold:**

1. The answer is correct.
2. The session records `billing_provider` = `claude-hedra` (or the equivalent provider field), `model` = `claude-fable-5-1`.
3. `~/.hermes/logs/claude-code-cli-shim.err.log` and `~/.hermes/profiles/zeus/logs/hedra-shim.stderr.log` contain **no** `account fallback` lines for this request.
4. The 8768 shim log shows the request arriving there.

**Step 3:** Repeat for Atlas.

```bash
env HERMES_HOME=/Users/everettbutler/.hermes hermes chat -q 'Reply with exactly: ATLAS_FAILOVER_OK' --no-restore-cwd
```

**If personal Fable has reset** by the time this runs, the primary will simply succeed and no failover occurs. That is not a pass. Either wait for the next exhaustion window, or force it by temporarily pointing the Zeus primary at a nonexistent Claude model (backed up, restored byte-for-byte afterward) to prove the cascade mechanism — and say explicitly in the report which of the two you did.

**Rollback trigger:** if the chain does not activate, restore both `config.yaml.bak-*` files and `git revert` the Task 4 commit. Re-run a clean canary on the restored route before reporting the profiles healthy.

---

### Task 8: Retire the two fallback logins

**Objective:** 3 Hedra OAuth sessions → 1. Only after Task 7 passes.

**Step 1:** Remove the dead opt-in from both personal shim env files.

- `/Users/everettbutler/.hermes/claude-code-cli-shim.env`: delete the `CLAUDE_CODE_CLI_FABLE_FALLBACK_CONFIG_DIR` line and its two comment lines
- `/Users/everettbutler/.hermes/profiles/zeus/claude-code-cli-shim.env`: same

**Step 2:** Log out each fallback dir so the session is released server-side, then delete the dirs.

```bash
for d in /Users/everettbutler/.hermes/claude-hedra-fallback \
         /Users/everettbutler/.hermes/profiles/zeus/claude-hedra-fallback; do
  CLAUDE_CONFIG_DIR=$d /Users/everettbutler/.local/bin/claude auth logout || true
done
```

**Step 3:** Confirm the three surviving dirs are still logged in BEFORE deleting anything.

```bash
for d in ~/.hermes/claude-code-automation \
         ~/.hermes/profiles/zeus/claude-code-automation \
         ~/.hermes/profiles/zeus/claude-hedra; do
  echo "$d :: $(CLAUDE_CONFIG_DIR=$d /Users/everettbutler/.local/bin/claude auth status 2>&1 | tr -d '\n')"
done
```

Expected: personal/personal/Hedra, all `loggedIn: true`. **If any is false, stop — do not delete.**

**Step 4:** Delete the two retired dirs and their Keychain entries.

```bash
rm -rf /Users/everettbutler/.hermes/claude-hedra-fallback \
       /Users/everettbutler/.hermes/profiles/zeus/claude-hedra-fallback
security delete-generic-password -s 'Claude Code-credentials-e59422f1' || true
security delete-generic-password -s 'Claude Code-credentials-259b231a' || true
```

Confirm the suffixes against `security dump-keychain | grep 'Claude Code-credentials'` first — do not delete by memory of the hash.

**Step 5:** Restart both personal shims, health-check all three, re-run the Task 7 Zeus canary once.

---

### Task 9: Update the docs

**Files:**
- Modify: `/Users/everettbutler/.hermes/shared/fleet-model-access/README.md` — replace the "Preserve the approved Fable account fallback" section. Everett's approval for Personal→Hedra Fable fallback **still stands**; only the mechanism changes. State that it is now a Hermes chain entry (`claude-hedra/claude-fable-5-1`) rather than an in-transport replay, that the account switch is visible in session `billing_provider`, and that 8768 now sets `CLAUDE_CODE_CLI_TEXT_ONLY_MODELS`.
- Modify: skill `claude-cli-shim-failover` — the "Never let two long-running shims share one `CLAUDE_CONFIG_DIR`" pitfall stays. Add the lesson: account fallback is a routing decision; implementing it in the transport makes the account switch invisible to session provenance and forces one extra OAuth login per shim wanting the fallback.

---

## Out of scope (noted, not touched)

- Every `auxiliary.*.fallback_chain` in both configs lists `provider: claude-code-cli`, which is **not** in either `providers:` block. Those entries are already dead. Worth a separate cleanup pass; not part of this change.
- The 8767 launchd label is `com.hermes.zeus-hedra-shim` but the service is Zeus **personal**. Misleading, harmless, rename separately.
- Hedra session limits are per account. Consolidating to one session reduces pressure but does not add quota.
