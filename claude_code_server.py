#!/usr/bin/env python3
"""OpenAI-compatible shim for the local Claude Code CLI (`claude -p`).

This is the backend for the ``claude-code-cli`` Hermes provider. It exposes a
minimal OpenAI Chat Completions surface and, for each request, shells out to::

    claude -p --output-format json --model <model> [--effort ..] [--tools ..] ...

then returns the CLI's ``result`` field as the assistant message - the same way
the ``fusion-consult`` skill drives Claude Code as an advisory worker. No
Anthropic API key and no network egress are involved; everything runs through
your local, already-authenticated Claude Code login.

Endpoints
---------
* ``GET  /healthz``               liveness probe
* ``GET  /v1/models``             advertises the model ids the CLI accepts
* ``POST /v1/chat/completions``   chat completion (supports ``stream: true``)

Tool behavior
-------------
When Hermes sends OpenAI ``tools``, the shim defaults to Hermes-native
tool-call emulation: Claude Code is invoked in text mode, returns a strict JSON
request, and the shim reshapes it into OpenAI-style ``tool_calls`` for Hermes
to execute. Set ``CLAUDE_CODE_CLI_ENGINE=always`` when you deliberately want
Claude Code's own tool loop instead.

Configuration (environment variables)
-------------------------------------
    CLAUDE_CODE_CLI_HOST            bind host           (default 127.0.0.1)
    CLAUDE_CODE_CLI_PORT            bind port           (default 8765)
    CLAUDE_CODE_CLI_BIN            path to claude       (default: autodetect)
    CLAUDE_CODE_CLI_MODEL          fallback model       (default sonnet)
    CLAUDE_CODE_CLI_EFFORT         reasoning effort     (default high; empty=omit)
    CLAUDE_CODE_CLI_TOOLS          --tools value        (default ""  => no tools)
    CLAUDE_CODE_CLI_DISALLOWED_TOOLS  --disallowedTools (default unset => omit)
    CLAUDE_CODE_CLI_MAX_TURNS      --max-turns          (default 12)
    CLAUDE_CODE_CLI_TIMEOUT        per-request seconds  (default 600)
    CLAUDE_CODE_CLI_EXTRA_ARGS     extra argv, shlex    (default unset)
    CLAUDE_CODE_CLI_MCP            account MCP servers  (default off => isolate;
                                   'account' inherits the login's claude.ai
                                   connectors; any other value is a --mcp-config
                                   path/JSON loaded with --strict-mcp-config)
    CLAUDE_CODE_CLI_STREAM         real token streaming (default 1; see #3)
    CLAUDE_CODE_CLI_NATIVE_TOOLS   emit OpenAI tool_calls (default on)
    CLAUDE_CODE_CLI_VISION         pass images to claude(default on;  see #4)
    CLAUDE_CODE_CLI_MAX_IMAGES     images per request   (default 8)
    CLAUDE_CODE_CLI_MAX_IMAGE_MB   per-image size cap MB (default 20)

Per-request overrides (OpenAI request body)
-------------------------------------------
A single request may tune effort / max-turns / the tool allowlist without
restarting the server; each field falls back to the env default above when the
request omits it (see ``extract_overrides``):
    extra_body.reasoning.effort   -> --effort   (or top-level reasoning_effort)
    extra_body.max_turns          -> --max-turns
    extra_body.allowed_tools      -> --allowedTools   (engine mode; CSV or list)
    extra_body.disallowed_tools   -> --disallowedTools
"""
from __future__ import annotations

import base64
import hmac
import json
import os
import pathlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Model ids the shim advertises and forwards verbatim to `claude --model`.
# Hermes validates `/model` switches against this listing (models_validate
# live-listing branch), so the exact id Hermes is configured with must appear
# here or every slash-command model check rejects it even though inference
# works. `claude-fable-5-2` is the current Fable id (needs Claude Code >= 2.1.290);
# `claude-fable-5-1` stays advertised so older pinned jobs and rollback still validate.
ADVERTISED_MODELS = ('opus', 'sonnet', 'haiku', 'fable', 'claude-fable-5-2', 'claude-fable-5-1', 'claude-sonnet-5-5', 'claude-opus-5-5')

# Inline `--mcp-config` payload that, combined with `--strict-mcp-config`,
# makes the CLI load zero MCP servers regardless of what the account has.
EMPTY_MCP_CONFIG = '{"mcpServers":{}}'

# Model strings the CLI accepts syntactically but does NOT resolve to the model
# they name -- it silently serves a fallback instead. `claude-fable-5` is the
# worst case: the CLI accepts it, returns 200, and bills Haiku 4.5 + Opus 4.8.
# The public aliases that actually resolve to Fable are `fable` and the pinned
# 1M form `claude-fable-5[1m]`, so normalize the phantom spelling before invoking.
_MODEL_ALIASES = {
    "claude-fable-5": "fable",
    "claude-fable5": "fable",
    "fable-5": "fable",
    "claude-fable-5-mythos-5": "fable",
    "mythos": "fable",
    "claude-mythos-5": "fable",
}


def normalize_model(model: str) -> str:
    """Map a known-phantom model spelling onto the alias the CLI truly resolves."""
    key = (model or "").strip()
    return _MODEL_ALIASES.get(key.lower(), key)


# --------------------------------------------------------------------------- #
# Configuration helpers
# --------------------------------------------------------------------------- #
def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    return default if val is None else val


def build_subprocess_env() -> dict[str, str]:
    """Return an env with user-local CLI paths restored.

    Mirrors fusion_runner.build_subprocess_env: cron/gateway-launched parents
    can have a minimal PATH that lacks ~/.local/npm/bin and nvm's node, which
    breaks `claude`'s `/usr/bin/env node` shebang even when the binary resolves.
    """
    env = os.environ.copy()
    home = pathlib.Path.home()
    candidates: list[pathlib.Path] = [home / ".local/npm/bin", home / ".local/bin"]

    nvm_versions = home / ".nvm/versions/node"
    if nvm_versions.exists():
        try:
            node_bins = [p for p in nvm_versions.glob("*/bin") if (p / "node").exists()]
            node_bins.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            candidates.extend(node_bins)
        except OSError:
            pass

    candidates.extend(
        pathlib.Path(p)
        for p in ("/usr/local/bin", "/usr/bin", "/bin", "/usr/local/sbin", "/usr/sbin", "/sbin")
    )

    parts: list[str] = []
    for p in candidates:
        s = str(p)
        if p.exists() and s not in parts:
            parts.append(s)
    for s in (env.get("PATH") or "").split(os.pathsep):
        if s and s not in parts:
            parts.append(s)
    env["PATH"] = os.pathsep.join(parts)
    return env


def resolve_claude_bin() -> str:
    """Resolve the claude binary: explicit env > PATH > common locations."""
    explicit = _env("CLAUDE_CODE_CLI_BIN").strip()
    if explicit:
        return explicit
    found = shutil.which("claude", path=build_subprocess_env().get("PATH"))
    if found:
        return found
    for cand in ("/usr/bin/claude", "/usr/local/bin/claude",
                 str(pathlib.Path.home() / ".local/npm/bin/claude")):
        if pathlib.Path(cand).exists():
            return cand
    return "claude"  # last resort; will surface a clear "not found" error


# --------------------------------------------------------------------------- #
# Prompt construction + CLI invocation
# --------------------------------------------------------------------------- #
# Vision passthrough (#4): claude -p is blind to inline OpenAI image parts, so
# we materialize base64 `data:` images to temp files and reference their paths
# in the prompt (Claude Code's Read tool can open image files in engine mode).
# Remote URLs are passed through as text references - the shim intentionally
# performs no network egress, so it never downloads them itself.
_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w.+-]+/[\w.+-]+)?(?P<b64>;base64)?,(?P<data>.*)$", re.DOTALL
)
_MIME_EXT = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "image/svg+xml": ".svg",
    "image/heic": ".heic",
    "image/heif": ".heif",
}


def _vision_enabled() -> bool:
    return _env("CLAUDE_CODE_CLI_VISION", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _max_images() -> int:
    try:
        return max(0, int(_env("CLAUDE_CODE_CLI_MAX_IMAGES", "8")))
    except ValueError:
        return 8


def _max_image_bytes() -> int:
    try:
        mb = float(_env("CLAUDE_CODE_CLI_MAX_IMAGE_MB", "20"))
    except ValueError:
        mb = 20.0
    return int(max(0.0, mb) * 1024 * 1024)


def _image_url_from_part(part: dict) -> str | None:
    """Extract the URL string from an OpenAI ``image_url`` content part.

    Handles both the object form ``{"image_url": {"url": "..."}}`` and the
    shorthand ``{"image_url": "..."}``.
    """
    iu = part.get("image_url")
    if isinstance(iu, dict):
        url = iu.get("url")
        return url if isinstance(url, str) and url.strip() else None
    if isinstance(iu, str) and iu.strip():
        return iu
    return None


class _ImageCollector:
    """Materialize image parts to temp files, bounded by count and size.

    ``add(url)`` returns the prompt text that should stand in for the image and,
    for decodable ``data:`` URLs, writes a temp file whose path is referenced in
    that text. Call :meth:`cleanup` after the CLI invocation returns to delete
    the temp files. When vision is disabled it degrades to ``[image omitted]``,
    preserving the pre-#4 behavior.
    """

    def __init__(self) -> None:
        self.enabled = _vision_enabled()
        self.max_images = _max_images()
        self.max_bytes = _max_image_bytes()
        self.count = 0
        self.paths: list[str] = []
        self._dir: str | None = None

    def _tmpdir(self) -> str:
        if self._dir is None:
            self._dir = tempfile.mkdtemp(prefix="claude-code-cli-img-")
        return self._dir

    def add(self, url: str) -> str:
        if not self.enabled:
            return "[image omitted]"
        if self.count >= self.max_images:
            return f"[image omitted: over the {self.max_images}-image limit]"
        url = url.strip()
        m = _DATA_URL_RE.match(url)
        if m and m.group("b64"):
            try:
                raw = base64.b64decode(m.group("data"), validate=False)
            except Exception:
                return "[image omitted: undecodable data URL]"
            if not raw:
                return "[image omitted: empty image]"
            if self.max_bytes and len(raw) > self.max_bytes:
                return f"[image omitted: exceeds {self.max_bytes // (1024 * 1024)}MB cap]"
            mime = (m.group("mime") or "image/png").lower()
            ext = _MIME_EXT.get(mime, ".img")
            path = os.path.join(self._tmpdir(), f"image_{self.count:02d}{ext}")
            try:
                with open(path, "wb") as fh:
                    fh.write(raw)
            except OSError as exc:
                return f"[image omitted: {exc}]"
            self.count += 1
            self.paths.append(path)
            return f"[image saved to {path} - read this file to view the image]"
        # Non-data reference: pass it through as text so an engine-mode agent can
        # fetch it (WebFetch) if it chooses. The shim itself does not download.
        if url.startswith(("http://", "https://", "file://", "/")):
            self.count += 1
            return f"[image at {url} - fetch this URL/path to view the image]"
        return "[image omitted: unsupported image reference]"

    def cleanup(self) -> None:
        for p in self.paths:
            try:
                os.remove(p)
            except OSError:
                pass
        if self._dir:
            try:
                os.rmdir(self._dir)
            except OSError:
                pass
        self.paths = []
        self._dir = None


def _content_to_text(content, images: _ImageCollector | None = None) -> str:
    """Flatten OpenAI message content (str or list-of-parts) to plain text.

    When ``images`` is provided, ``image_url`` parts are materialized/referenced
    through the collector (#4); otherwise they degrade to ``[image omitted]`` so
    text-only callers keep their original behavior.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    chunks.append(part["text"])
                elif part.get("type") == "image_url" or "image_url" in part:
                    url = _image_url_from_part(part) if images is not None else None
                    if images is not None and url:
                        chunks.append(images.add(url))
                    else:
                        chunks.append("[image omitted]")
            elif isinstance(part, str):
                chunks.append(part)
        return "\n".join(chunks)
    if content is None:
        return ""
    return str(content)


def flatten_messages(messages: list[dict], images: _ImageCollector | None = None,
                     native_tools: bool = False) -> str:
    """Render an OpenAI message list into a single prompt for `claude -p`.

    System/developer messages become a leading SYSTEM block; the conversation
    is rendered as labelled turns. Tool messages are folded in as context so
    nothing in the history is silently dropped. When an ``images`` collector is
    passed, inline images are materialized/referenced instead of dropped (#4).
    """
    system_blocks: list[str] = []
    convo: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = (msg.get("role") or "user").lower()
        text = _content_to_text(msg.get("content"), images)
        if role in ("system", "developer"):
            if text.strip():
                system_blocks.append(text)
        elif role == "assistant":
            # Surface any tool calls the assistant previously made as context.
            tool_calls = msg.get("tool_calls") or []
            call_desc = ""
            if tool_calls:
                names = [
                    (tc.get("function") or {}).get("name", "tool")
                    for tc in tool_calls if isinstance(tc, dict)
                ]
                if native_tools:
                    names = [n if n.startswith('mcp__hermes__') else 'mcp__hermes__' + n
                             for n in names if isinstance(n, str) and n]
                call_desc = f" [requested tools: {', '.join(n for n in names if n)}]"
            convo.append(f"ASSISTANT: {text}{call_desc}".rstrip())
        elif role == "tool":
            convo.append(f"TOOL RESULT: {text}")
        else:
            convo.append(f"USER: {text}")

    parts: list[str] = []
    if system_blocks:
        parts.append("SYSTEM INSTRUCTIONS:\n" + "\n\n".join(system_blocks))
    if convo:
        parts.append("CONVERSATION:\n" + "\n\n".join(convo))
    if native_tools:
        parts.append(
            "Respond as the assistant to the latest user turn. Tools available for this "
            "request are provided via MCP. When you select one, Hermes executes it outside "
            "this CLI turn and supplies its real result in a subsequent conversation turn; "
            "do not simulate execution. After a tool result, continue any pending work using "
            "the provided tools when needed. Give a final answer only when the work is done "
            "or genuinely blocked. Do not infer a per-turn tool cap or ask the user to restart "
            "already-approved work. Actual callable tool names start with mcp__hermes__; "
            "use the exact listed names and argument types, not bare labels from conversation history. "
            "Ordinary responses that need no tool can be answered normally."
        )
    else:
        parts.append(
            "Respond as the assistant to the latest user turn. "
            "Return only the response text."
        )
    return "\n\n".join(parts).strip()


def _native_tools_enabled() -> bool:
    """Whether the shim may emulate OpenAI tool_calls for Hermes-native tools."""
    return _env("CLAUDE_CODE_CLI_NATIVE_TOOLS", "1").strip().lower() not in {
        "0", "false", "no", "off", "disabled",
    }


def _tool_choice_is_none(tool_choice) -> bool:
    return isinstance(tool_choice, str) and tool_choice.strip().lower() == "none"


def _tool_choice_name(tool_choice) -> str:
    if not isinstance(tool_choice, dict):
        return ""
    fn = tool_choice.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        return name.strip() if isinstance(name, str) else ""
    return ""


def _openai_tool_defs(tools) -> list[dict]:
    """Return sanitized OpenAI function-tool definitions from a request body."""
    if not isinstance(tools, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for item in tools:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        if not isinstance(fn, dict):
            continue
        name = fn.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        if name in seen:
            continue
        seen.add(name)
        rec = {"type": "function", "function": {"name": name}}
        for key in ("description", "parameters"):
            if key in fn:
                rec["function"][key] = fn[key]
        out.append(rec)
    return out


def build_native_tool_prompt(prompt: str, tools: list[dict], tool_choice=None) -> str:
    """Append instructions that let Claude request Hermes-native tool calls.

    Claude Code is still invoked in text mode. If it needs a Hermes tool, it
    returns a strict JSON envelope; the shim reshapes that envelope into the
    OpenAI ``message.tool_calls`` form that Hermes already knows how to execute.
    """
    tool_json = json.dumps(tools, ensure_ascii=False, indent=2)
    required = isinstance(tool_choice, str) and tool_choice.strip().lower() == "required"
    specific = _tool_choice_name(tool_choice)
    if specific:
        choice_line = f"You must call the Hermes-native tool named {specific!r}."
    elif required:
        choice_line = "You must call at least one Hermes-native tool."
    else:
        choice_line = "Call a Hermes-native tool only when it is needed; otherwise answer normally as text."
    return (
        prompt.rstrip()
        + "\n\nHermes-native tool calling is available. "
        + "Do not run these tools yourself and do not simulate their results. "
        + "When you request a tool, Hermes will execute it and send the result back in a later turn.\n\n"
        + "Available Hermes-native tools:\n"
        + tool_json
        + "\n\n"
        + choice_line
        + "\nIf you need one or more tools, respond with ONLY strict JSON in this exact shape:\n"
        + '{"tool_calls":[{"name":"tool_name","arguments":{"key":"value"}}]}'
        + "\nUse only tool names from the list above. Arguments must be a JSON object. "
        + "Do not include prose, Markdown fences, or extra keys around the JSON. "
        + "After Hermes sends tool results, use them to answer normally unless another tool call is needed."
    )


def _tool_arguments_json(arguments) -> str:
    if arguments is None:
        return "{}"
    if isinstance(arguments, str):
        raw = arguments.strip()
        if not raw:
            return "{}"
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = {"input": arguments}
    elif isinstance(arguments, dict):
        parsed = arguments
    else:
        parsed = {"value": arguments}
    try:
        return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps({"value": str(arguments)}, ensure_ascii=False, separators=(",", ":"))


def extract_native_tool_calls(text: str, tools: list[dict]) -> list[dict]:
    """Extract OpenAI-shaped tool calls from Claude's strict JSON envelope."""
    parsed = _extract_json_object(text)
    if not isinstance(parsed, dict):
        # A malformed transport envelope is not a user-facing answer. Do not
        # guess/repair executable arguments; let the client retry or fail over.
        if re.match(r'^\s*(?:```(?:json)?\s*)?\{\s*"tool_calls?"\s*:',
                    text or "", flags=re.IGNORECASE):
            raise ValueError("invalid tool-call envelope: malformed JSON")
        return []
    raw_calls = parsed.get("tool_calls")
    if raw_calls is None and "tool_call" in parsed:
        raw_calls = [parsed.get("tool_call")]
    if raw_calls is None and ("name" in parsed or "function" in parsed):
        raw_calls = [parsed]
    if not isinstance(raw_calls, list):
        return []

    allowed = {
        tool.get("function", {}).get("name")
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("function"), dict)
    }
    out: list[dict] = []
    for call in raw_calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") if isinstance(call.get("function"), dict) else {}
        name = call.get("name") or fn.get("name")
        if not isinstance(name, str) or name not in allowed:
            continue
        arguments = call.get("arguments")
        if arguments is None:
            arguments = fn.get("arguments")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id.strip():
            call_id = "call_" + uuid.uuid4().hex
        out.append({
            "id": call_id.strip(),
            "type": "function",
            "function": {
                "name": name,
                "arguments": _tool_arguments_json(arguments),
            },
        })
    return out


# Tools pre-approved in engine mode so `claude -p` can run them without an
# interactive permission prompt (it can't prompt in print mode). Override with
# CLAUDE_CODE_CLI_ENGINE_TOOLS.
DEFAULT_ENGINE_TOOLS = "Read,Write,Edit,Bash,Glob,Grep,WebFetch,WebSearch,TodoWrite"


def _coerce_tool_list(value) -> str:
    """Coerce an allow/deny tools value into claude's CSV form.

    Accepts a CSV string (returned trimmed) or a list. List items may be plain
    tool-name strings or OpenAI ``{"type":"function","function":{"name":...}}``
    tool objects; the function name is extracted from the latter. Returns ``""``
    when nothing usable is present.
    """
    if isinstance(value, str):
        return ",".join(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, list):
        names: list[str] = []
        for item in value:
            if isinstance(item, str):
                name = item.strip()
            elif isinstance(item, dict):
                fn = item.get("function")
                if isinstance(fn, dict):
                    name = str(fn.get("name") or "").strip()
                else:
                    name = str(item.get("name") or "").strip()
            else:
                name = ""
            if name and name not in names:
                names.append(name)
        return ",".join(names)
    return ""


def extract_overrides(body: object) -> dict:
    """Pull per-request CLI overrides out of an OpenAI request body.

    Only keys that are actually present (and non-empty) are returned, so callers
    can treat a missing key as "fall back to the env default". This is parsed
    defensively - a malformed field is ignored rather than raising, keeping a
    single bad request from breaking the shim. Recognised inputs:

    * ``reasoning_effort`` (top-level) or ``extra_body.reasoning.effort``
      or ``extra_body.effort``                       -> ``effort``
    * ``extra_body.max_turns`` / top-level ``max_turns`` (int or str) -> ``max_turns``
    * ``extra_body.allowed_tools`` (CSV/list/tool-objects)  -> ``allowed_tools``
    * ``extra_body.disallowed_tools`` (CSV/list)            -> ``disallowed_tools``
    """
    if not isinstance(body, dict):
        return {}
    extra = body.get("extra_body")
    if not isinstance(extra, dict):
        extra = {}

    out: dict = {}

    # --- effort: top-level reasoning_effort, then extra_body.reasoning.effort ---
    effort = body.get("reasoning_effort")
    if not (isinstance(effort, str) and effort.strip()):
        reasoning = extra.get("reasoning")
        if isinstance(reasoning, dict):
            effort = reasoning.get("effort")
        else:
            effort = extra.get("reasoning_effort") or extra.get("effort")
    if isinstance(effort, str) and effort.strip():
        out["effort"] = effort.strip()

    # --- max_turns: extra_body first, then top-level; accept int or str ---
    max_turns = extra.get("max_turns")
    if max_turns is None:
        max_turns = body.get("max_turns")
    if isinstance(max_turns, bool):
        max_turns = None  # guard: bool is an int subclass
    if isinstance(max_turns, int) and max_turns > 0:
        out["max_turns"] = str(max_turns)
    elif isinstance(max_turns, str) and max_turns.strip().isdigit() and int(max_turns.strip()) > 0:
        out["max_turns"] = max_turns.strip()

    # --- tool allow / deny lists (engine mode) ---
    allowed = _coerce_tool_list(extra.get("allowed_tools"))
    if allowed:
        out["allowed_tools"] = allowed
    disallowed = _coerce_tool_list(extra.get("disallowed_tools"))
    if disallowed:
        out["disallowed_tools"] = disallowed

    return out


def build_claude_argv(model: str, engine: bool, overrides: dict | None = None,
                      stream: bool = False) -> list[str]:
    """Assemble the `claude -p` argv from config. Prompt is piped via stdin.

    engine=True  → Claude Code uses its OWN tools to actually do the work
                   (read/edit files, run bash) - "use Claude Code as an engine".
    engine=False → no tools, single turn - a plain text model (aux tasks).

    ``overrides`` (from :func:`extract_overrides`) lets a single request tune
    effort / max-turns / tool lists; any key it omits falls back to the env
    default, so behavior is unchanged when the request carries no overrides.

    ``stream=True`` (issue #3) switches the output format to ``stream-json`` so
    the CLI emits incremental JSONL events instead of one final JSON blob;
    ``--verbose`` is required by the CLI in that mode, and partial (token-level)
    messages are requested unless ``CLAUDE_CODE_CLI_PARTIAL_MESSAGES`` disables
    them (an escape hatch for CLI versions that lack the flag).
    """
    overrides = overrides or {}
    argv = [
        resolve_claude_bin(),
        "-p",
        "--model", model,
    ]
    if stream:
        argv += ["--output-format", "stream-json", "--verbose"]
        if _env("CLAUDE_CODE_CLI_PARTIAL_MESSAGES", "1").strip().lower() not in {
            "0", "false", "no", "off",
        }:
            argv += ["--include-partial-messages"]
    else:
        argv += ["--output-format", "json"]
    argv += ["--no-session-persistence"]

    # Claude CLI auto-loads every claude.ai MCP connector attached to the
    # logged-in account (Gmail, Drive, Linear, Figma, ...) and injects all of
    # their tool schemas into EVERY request. With a dozen connectors live that
    # is ~250K tokens against a 200K window, so even a 5-token probe 503s with
    # "Prompt is too long". Hermes supplies its own tools, so isolate the CLI
    # from account MCP unless the operator opts in.
    raw_mcp = str(overrides.get("_mcp_config") or _env("CLAUDE_CODE_CLI_MCP", "")).strip()
    mcp_mode = raw_mcp.lower()
    if overrides.get("_mcp_config"):
        # Internal per-request bridge config. This key is never accepted from an
        # OpenAI request body; only run_claude_mcp_bridge sets it.
        argv += ["--strict-mcp-config", "--mcp-config", raw_mcp]
    elif mcp_mode in {"", "0", "off", "false", "no", "none"}:
        argv += ["--strict-mcp-config", "--mcp-config", EMPTY_MCP_CONFIG]
    elif mcp_mode in {"1", "on", "true", "yes", "account", "inherit"}:
        pass  # let the CLI load the account's connectors as it normally would
    else:
        argv += ["--strict-mcp-config", "--mcp-config", raw_mcp]

    effort = str(overrides.get("effort") or _env("CLAUDE_CODE_CLI_EFFORT", "high")).strip()
    if effort:
        argv += ["--effort", effort]

    if engine:
        # Let the CLI's own agentic tool loop run. Pre-approve a capable tool set
        # so it executes autonomously instead of blocking on permission prompts.
        tools = str(
            overrides.get("allowed_tools")
            or _env("CLAUDE_CODE_CLI_ENGINE_TOOLS", DEFAULT_ENGINE_TOOLS)
        ).strip()
        if tools:
            argv += ["--allowedTools", tools]
        if _env("CLAUDE_CODE_CLI_ENGINE_PERMISSION", "").strip().lower() in {
            "bypass", "skip", "dangerous", "yolo",
        }:
            argv += ["--dangerously-skip-permissions"]
        disallowed = str(
            overrides.get("disallowed_tools")
            or _env("CLAUDE_CODE_CLI_DISALLOWED_TOOLS", "")
        ).strip()
        if disallowed:
            argv += ["--disallowedTools", disallowed]
        add_dir = _env("CLAUDE_CODE_CLI_ADD_DIR", "").strip()
        if add_dir:
            argv += ["--add-dir", *add_dir.split(os.pathsep)]
        max_turns = str(
            overrides.get("max_turns")
            or _env("CLAUDE_CODE_CLI_ENGINE_MAX_TURNS", "40")
        ).strip()
        if max_turns:
            argv += ["--max-turns", max_turns]
    else:
        # Disable the CLI's own tool loop so it behaves as a pure text model
        # (Hermes auxiliary tasks: title-gen, compression, etc.).
        argv += ["--tools", _env("CLAUDE_CODE_CLI_TOOLS", "")]
        allowed_mcp_tools = str(overrides.get("_allowed_mcp_tools") or "").strip()
        if allowed_mcp_tools:
            argv += ["--allowedTools", allowed_mcp_tools]
        disallowed = str(
            overrides.get("disallowed_tools")
            or _env("CLAUDE_CODE_CLI_DISALLOWED_TOOLS", "")
        ).strip()
        if disallowed:
            argv += ["--disallowedTools", disallowed]
        max_turns = str(
            overrides.get("max_turns")
            or _env("CLAUDE_CODE_CLI_MAX_TURNS", "12")
        ).strip()
        if max_turns:
            argv += ["--max-turns", max_turns]

    extra = _env("CLAUDE_CODE_CLI_EXTRA_ARGS", "").strip()
    if extra:
        try:
            argv += shlex.split(extra)
        except ValueError:
            pass
    return argv


def _engine_cwd() -> str:
    """Working directory Claude Code operates in during engine mode."""
    raw = _env("CLAUDE_CODE_CLI_CWD", "").strip()
    cwd = pathlib.Path(raw).expanduser() if raw else pathlib.Path.home()
    return str(cwd) if cwd.is_dir() else str(pathlib.Path.home())


def _extract_json_object(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


# --------------------------------------------------------------------------- #
# Structured MCP bridge for Hermes-owned tools
# --------------------------------------------------------------------------- #
def _mcp_tool_specs(tools: list[dict]) -> list[dict]:
    specs: list[dict] = []
    for item in tools:
        fn = item.get("function") if isinstance(item, dict) else None
        if not isinstance(fn, dict):
            raise ValueError("invalid function tool specification")
        name = fn.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", name):
            raise ValueError("invalid tool name")
        if any(spec["name"] == name for spec in specs):
            raise ValueError("duplicate tool name")
        schema = fn.get("parameters", {"type": "object"})
        if not isinstance(schema, (dict, bool)):
            raise ValueError("invalid tool parameter schema")
        specs.append({
            "name": name,
            "description": str(fn.get("description") or ""),
            "inputSchema": schema,
        })
    return specs


def _write_private_json(path: pathlib.Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    path.chmod(0o600)


def _run_mcp_cli(prompt: str, model: str, config_path: pathlib.Path,
                 allowed_tools: str, overrides: dict | None = None) -> dict:
    """Run one bounded CLI turn and collect stream-json without leaking children."""
    internal = dict(overrides or {})
    internal.update({
        "_mcp_config": str(config_path),
        "_allowed_mcp_tools": allowed_tools,
        "max_turns": "1",
    })
    argv = build_claude_argv(model, engine=False, overrides=internal, stream=True)
    try:
        timeout = int(_env("CLAUDE_CODE_CLI_TIMEOUT", "600"))
    except ValueError:
        timeout = 600
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=build_subprocess_env(),
            start_new_session=True,
        )
    except FileNotFoundError:
        return {"events": [], "returncode": 127, "stderr": "claude binary not found", "timed_out": False}
    except Exception as exc:
        return {"events": [], "returncode": 126, "stderr": f"claude invocation failed: {exc!r}", "timed_out": False}
    timed_out = False
    try:
        stdout, stderr = proc.communicate(prompt, timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        stdout, stderr = proc.communicate()
    events = list(iter_stream_json((stdout or "").splitlines()))
    return {
        "events": events,
        "returncode": proc.returncode,
        "stderr": (stderr or "")[-2000:],
        "timed_out": timed_out,
    }


def _read_bridge_events(path: pathlib.Path, *, request_id: str) -> list[dict]:
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if isinstance(value, dict) and value.get("request_id") == request_id:
                records.append(value)
    except (OSError, json.JSONDecodeError):
        return []
    return records


def _assistant_tool_uses(events: list[dict]) -> list[dict]:
    calls: list[dict] = []
    for event in events:
        if event.get("type") != "assistant":
            continue
        message = event.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                calls.append(block)
    return calls


def _recover_unprefixed_capture(events: list[dict], result: dict, model: str,
                                specs: list[dict], captured: list[dict],
                                spec_path: pathlib.Path, events_path: pathlib.Path,
                                bridge_script: pathlib.Path, request_id: str,
                                invocation: dict) -> list[dict]:
    """Capture only exact CLI-rejected bare calls; never execute Hermes tools."""
    calls = _assistant_tool_uses(events)
    if captured or not calls or invocation.get('timed_out'):
        return captured
    if invocation.get('returncode') != 0 and not (
        invocation.get('returncode') == 1 and result.get('is_error') is True
        and result.get('terminal_reason') == 'max_turns'
    ):
        return captured
    if not result or result.get('permission_denials') or (
        result.get('is_error') and result.get('terminal_reason') != 'max_turns'
    ):
        return captured
    billed = result.get('modelUsage')
    if not isinstance(billed, dict) or not billed or not _extract_served_model(result, model):
        return captured
    if check_model_provenance(model, _extract_served_model(result, model)):
        return captured
    allowed = {spec['name'] for spec in specs}
    ids: set[str] = set()
    for call in calls:
        name, cid = call.get('name'), call.get('id')
        if (not isinstance(name, str) or name not in allowed or
            not isinstance(cid, str) or not cid.strip() or cid in ids or
            not isinstance(call.get('input'), dict)):
            return captured
        ids.add(cid)
    rejected: dict[str, list[str]] = {}
    for event in events:
        if event.get('type') != 'user':
            continue
        message = event.get('message')
        content = message.get('content') if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get('type') != 'tool_result':
                continue
            cid = block.get('tool_use_id')
            if not isinstance(cid, str) or cid not in ids or block.get('is_error') is not True:
                return captured
            text = block.get('content')
            if isinstance(text, list) and len(text) == 1 and isinstance(text[0], dict):
                text = text[0].get('text')
            if not isinstance(text, str):
                return captured
            rejected.setdefault(cid, []).append(text)
    for call in calls:
        text = f"Error: No such tool available: {call['name']}"
        actual = rejected.get(call['id'], [])
        if len(actual) != 1 or actual[0] not in (text, f'<tool_use_error>{text}</tool_use_error>'):
            return captured
    requests = [json.dumps({
        'jsonrpc': '2.0', 'id': index + 1, 'method': 'tools/call',
        'params': {'name': call['name'], 'arguments': call['input'],
                   '_meta': {'claudecode/toolUseId': call['id']}},
    }, ensure_ascii=False) for index, call in enumerate(calls)]
    try:
        proc = subprocess.run(
            [sys.executable, str(bridge_script), str(spec_path), str(events_path)],
            input='\n'.join(requests) + '\n', capture_output=True, text=True,
            encoding='utf-8', timeout=10, check=False,
        )
        if proc.returncode != 0:
            return captured
        replies = [json.loads(line) for line in proc.stdout.splitlines()]
        if (len(replies) != len(calls) or any(
            not isinstance(reply, dict) or type(reply.get('id')) is not int or
            reply['id'] != index + 1 or not isinstance(reply.get('result'), dict) or
            type(reply['result'].get('isError')) is not bool
            for index, reply in enumerate(replies)
        )):
            return captured
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return captured
    records = _read_bridge_events(events_path, request_id=request_id)
    if (len(records) != len(calls) or any(
        record.get('status') not in ('accepted', 'rejected') or
        record.get('name') != call['name'] or
        record.get('arguments') != call['input'] or record.get('tool_use_id') != call['id'] or
        reply['result']['isError'] is not (record['status'] == 'rejected')
        for call, record, reply in zip(calls, records, replies)
    )):
        return captured
    return records


def _last_result_event(events: list[dict]) -> dict:
    for event in reversed(events):
        if event.get("type") == "result" or (
            event.get("type") is None and isinstance(event.get("result"), str)
        ):
            return event
    return {}


def run_claude_mcp_bridge(prompt: str, model: str, tools: list[dict], tool_choice,
                          overrides: dict | None = None) -> dict:
    """Ask Claude to select Hermes tools through a request-scoped MCP server.

    The MCP process advertises schemas and captures validated calls only. It
    never executes a tool. This function cross-checks the capture against the
    model's structured ``tool_use`` event before emitting OpenAI ``tool_calls``.
    """
    try:
        specs = _mcp_tool_specs(tools)
    except ValueError as exc:
        return {"text": "", "usage": _zero_usage(), "error": str(exc),
                "http_status": 400, "model": "", "tool_calls": []}
    specific = _tool_choice_name(tool_choice)
    if specific:
        specs = [spec for spec in specs if spec["name"] == specific]
        if not specs:
            return {"text": "", "usage": _zero_usage(), "error": "unknown required tool",
                    "http_status": 400, "model": "", "tool_calls": []}
        prompt = prompt.rstrip() + f"\n\nYou must use the available Hermes tool {'mcp__hermes__' + specific!r} for this turn."
    elif isinstance(tool_choice, str) and tool_choice.strip().lower() == "required":
        prompt = prompt.rstrip() + "\n\nYou must use one of the available Hermes tools for this turn."
    if not specs:
        return {"text": "", "usage": _zero_usage(), "error": "no valid tool schemas",
                "http_status": 400, "model": "", "tool_calls": []}

    request_id = uuid.uuid4().hex
    bridge_script = pathlib.Path(__file__).with_name("hermes_mcp_bridge.py")
    with tempfile.TemporaryDirectory(prefix="claude-hermes-mcp-") as td:
        root = pathlib.Path(td)
        root.chmod(0o700)
        spec_path = root / "tools.json"
        events_path = root / "events.jsonl"
        config_path = root / "mcp.json"
        _write_private_json(spec_path, {"request_id": request_id, "tools": specs})
        _write_private_json(config_path, {"mcpServers": {"hermes": {
            "command": sys.executable,
            "args": [str(bridge_script), str(spec_path), str(events_path)],
        }}})
        allowed = ",".join(f"mcp__hermes__{spec['name']}" for spec in specs)
        invocation = _run_mcp_cli(prompt, model, config_path, allowed, overrides)
        captured = _read_bridge_events(events_path, request_id=request_id)
        raw = invocation.get('events')
        if isinstance(raw, list):
            captured = _recover_unprefixed_capture(
                raw, _last_result_event(raw), model, specs, captured,
                spec_path, events_path, bridge_script, request_id, invocation,
            )

    if invocation.get("timed_out"):
        return {"text": "", "usage": _zero_usage(), "error": "claude MCP tool selection timed out",
                "model": "", "tool_calls": []}
    raw_events = invocation.get("events")
    events: list[dict] = raw_events if isinstance(raw_events, list) else []
    result = _last_result_event(events)
    usage = _map_usage(result.get("usage"))
    served = _extract_served_model(result, model)
    substitution = check_model_provenance(model, served)
    if substitution:
        return {"text": "", "usage": usage, "error": substitution,
                "model": served, "tool_calls": []}
    # A bounded selection ends at max_turns after capture. Every other
    # terminal failure must remain a failure, even if partial calls exist.
    if result.get("is_error") and result.get("terminal_reason") != "max_turns":
        message = str(result.get("result") or result.get("error") or invocation.get("stderr") or "claude reported is_error")
        return {"text": "", "usage": usage, "error": message, "model": served, "tool_calls": []}
    if not result or not served:
        return {"text": "", "usage": usage, "error": "missing terminal result or billed model provenance",
                "http_status": 503, "model": served, "tool_calls": []}
    rejected = [record for record in captured if record.get("status") != "accepted"]
    if rejected:
        detail = str(rejected[0].get("error") or "invalid tool request")
        return {"text": "", "usage": usage, "error": detail,
                "http_status": 503, "model": served, "tool_calls": []}

    model_calls = _assistant_tool_uses(events)
    if model_calls:
        if len(model_calls) != len(captured):
            allowed_names = {spec["name"] for spec in specs}
            def safe_name(value):
                name = value if isinstance(value, str) else ""
                if name.startswith("mcp__hermes__"):
                    name = name[len("mcp__hermes__"):]
                return name if name in allowed_names else "<unknown>"
            detail = {
                "model_calls": len(model_calls), "captured": len(captured),
                "model_tool_names": [safe_name(block.get("name")) for block in model_calls],
                "captured_tool_names": [safe_name(record.get("name")) for record in captured],
                "permission_denials": len(result.get("permission_denials") or []),
            }
            return {"text": "", "usage": usage, "error": "bridge capture mismatch: call count; " + json.dumps(detail),
                    "http_status": 503, "model": served, "tool_calls": []}
        openai_calls: list[dict] = []
        for block, record in zip(model_calls, captured):
            model_name = str(block.get("name") or "")
            raw_name = model_name[len("mcp__hermes__"):] if model_name.startswith("mcp__hermes__") else model_name
            model_input = block.get("input")
            call_id = str(block.get("id") or "")
            if (
                record.get("status") != "accepted"
                or record.get("name") != raw_name
                or record.get("arguments") != model_input
                or (record.get("tool_use_id") and record.get("tool_use_id") != call_id)
            ):
                return {"text": "", "usage": usage, "error": "bridge capture mismatch: structured call differs",
                        "http_status": 503, "model": served, "tool_calls": []}
            openai_calls.append({
                "id": call_id or "call_" + uuid.uuid4().hex,
                "type": "function",
                "function": {
                    "name": raw_name,
                    "arguments": json.dumps(model_input, ensure_ascii=False, separators=(",", ":")),
                },
            })
        return {"text": "", "usage": usage, "error": "", "model": served,
                "tool_calls": openai_calls}

    text = ""
    for event in reversed(events):
        if event.get("type") == "assistant":
            text = _extract_assistant_message_text(event)
            if text:
                break
    if not text:
        result_text = result.get("result")
        text = result_text if isinstance(result_text, str) else ""
    if re.search(r'"tool_calls"\s*:', text):
        return {"text": "", "usage": usage, "error": "invalid tool-call envelope: text is not a structured MCP request",
                "http_status": 503, "model": served, "tool_calls": []}
    required = isinstance(tool_choice, str) and tool_choice.strip().lower() == "required"
    if required or specific:
        return {"text": "", "usage": usage, "error": "required tool was not selected",
                "http_status": 400, "model": served, "tool_calls": []}
    if result.get("is_error"):
        message = text or str(result.get("error") or invocation.get("stderr") or "claude reported is_error")
        return {"text": "", "usage": usage, "error": message, "model": served, "tool_calls": []}
    if not text:
        message = str(invocation.get("stderr") or "claude returned no output")
        return {"text": "", "usage": usage, "error": message, "model": served, "tool_calls": []}
    return {"text": text, "usage": usage, "error": "", "model": served, "tool_calls": []}


def _run_claude_with_provenance(prompt: str, model: str, engine: bool,
                                overrides: dict | None,
                                subprocess_env: dict | None = None) -> dict:
    retries = _fable_provenance_retries(model)
    outcome: dict = {}
    for attempt in range(retries + 1):
        outcome = _run_claude_once(
            prompt, model, engine, overrides, subprocess_env=subprocess_env
        )
        if not _is_fable_provenance_substitution(model, outcome):
            return outcome
        if attempt < retries:
            sys.stderr.write(
                "[claude-code-cli] transient Fable model substitution; "
                f"retrying exact route ({attempt + 1}/{retries})\n"
            )
    return outcome


def run_claude(prompt: str, model: str, engine: bool = False,
               overrides: dict | None = None) -> dict:
    """Run one request. Account selection is NOT done here.

    An exhausted account used to be replayed against a second CLAUDE_CONFIG_DIR
    from inside this process. That hid the account switch from Hermes' session
    provenance and cost one extra OAuth login per shim wanting the fallback.
    Exhaustion now surfaces as a classified error and Hermes' own
    fallback_providers chain picks the next route.
    """
    return _run_claude_with_provenance(prompt, model, engine, overrides)


def _run_claude_once(prompt: str, model: str, engine: bool = False,
                     overrides: dict | None = None, *,
                     subprocess_env: dict | None = None) -> dict:
    """Invoke `claude -p` and return {text, usage, error}.

    `usage` is {prompt_tokens, completion_tokens, total_tokens}. On any failure
    `error` is a human-readable string and `text` carries the same message so
    streaming clients still see something. In engine mode the CLI runs its own
    tools in CLAUDE_CODE_CLI_CWD and may take much longer.
    """
    argv = build_claude_argv(model, engine, overrides)
    default_timeout = "1200" if engine else "600"
    try:
        timeout = int(_env("CLAUDE_CODE_CLI_TIMEOUT", default_timeout))
    except ValueError:
        timeout = int(default_timeout)
    cwd = _engine_cwd() if engine else None
    run_env = dict(subprocess_env) if subprocess_env is not None else build_subprocess_env()

    try:
        proc = subprocess.run(
            argv,
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            env=run_env,
            cwd=cwd,
            check=False,
        )
    except FileNotFoundError:
        msg = f"claude binary not found (tried {argv[0]!r}); set CLAUDE_CODE_CLI_BIN"
        return {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg}
    except subprocess.TimeoutExpired:
        msg = f"claude timed out after {timeout}s"
        return {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg}
    except Exception as exc:  # defensive: never crash the request thread
        msg = f"claude invocation failed: {exc!r}"
        return {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg}

    stdout, stderr = proc.stdout or "", proc.stderr or ""
    parsed = _extract_json_object(stdout)

    if parsed is not None:
        if parsed.get("is_error"):
            msg = str(parsed.get("result") or parsed.get("error") or "claude reported is_error")
            return {"text": f"[claude-code-cli error] {msg}", "usage": _map_usage(parsed.get("usage")), "error": msg}
        served = _extract_served_model(parsed, model)
        substitution = check_model_provenance(model, served)
        if substitution:
            return {
                "text": f"[claude-code-cli error] {substitution}",
                "usage": _map_usage(parsed.get("usage")),
                "error": substitution,
                "model": served,
            }
        result = parsed.get("result")
        if not isinstance(result, str):
            result = parsed.get("content")
        if isinstance(result, str) and result.strip():
            return {
                "text": result.strip(),
                "usage": _map_usage(parsed.get("usage")),
                "error": "",
                "model": served,
            }

    # Non-zero exit with no parseable result → surface stderr/stdout tail.
    if proc.returncode != 0:
        tail = (stderr.strip() or stdout.strip())[-1000:]
        msg = f"claude exited {proc.returncode}: {tail}"
        return {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg}

    # Exit 0 but unparseable → degrade to raw stdout as the completion text.
    fallback = stdout.strip()
    if fallback:
        return {"text": fallback, "usage": _zero_usage(), "error": ""}
    msg = "claude returned no output"
    return {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg}


# --------------------------------------------------------------------------- #
# Served-model provenance
# --------------------------------------------------------------------------- #
# The shim used to echo the *requested* model string straight back in the
# OpenAI `model` response field. That is a provenance hole: the Claude CLI can
# accept an unknown `--model` and quietly serve its default, and the caller has
# no way to tell. Any policy that depends on "which model wrote this" was
# trusting a label the shim invented rather than anything the CLI reported.
#
# The CLI's JSON envelope carries `modelUsage`, keyed by the model id actually
# billed. That is ground truth, so surface it instead of the request string.

_MODEL_FAMILIES = ("opus", "sonnet", "haiku", "fable")


def _strict_model_enabled() -> bool:
    return _env("CLAUDE_CODE_CLI_STRICT_MODEL", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _extract_served_model(parsed, requested: str = "") -> str:
    """Return the billed model matching the requested family when available.

    Claude Code may bill more than one model in one run (for example an
    auxiliary Haiku plus the requested Opus). Taking the first `modelUsage` key
    creates a false substitution alarm. Prefer a key in the requested family;
    only fall back to the first billed model when that family is absent.
    """
    if not isinstance(parsed, dict):
        return ""
    usage = parsed.get("modelUsage")
    if isinstance(usage, dict):
        models = [key.strip() for key in usage if isinstance(key, str) and key.strip()]
        want = _model_family(requested)
        if want:
            for model in models:
                if _model_family(model) == want:
                    return model
        if models:
            return models[0]
    for key in ("model", "modelId"):
        value = parsed.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    message = parsed.get("message")
    if isinstance(message, dict):
        value = message.get("model")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _model_family(name: str) -> str:
    low = (name or "").lower()
    for family in _MODEL_FAMILIES:
        if family in low:
            return family
    return ""


def check_model_provenance(requested: str, served: str) -> str:
    """Return an error string when the CLI silently served a different family.

    Empty string means no detectable substitution. Compare *families*, not exact
    ids: legitimate aliases exist -- `opus` resolves to `claude-opus-5-<date>`,
    and that is not a substitution. A request for `claude-fable-5` answered by
    `claude-opus-5-...` is, and it must not be reported as Fable.

    The message is prefixed `model not found:` so classify_cli_error() maps it
    to a 404 and Hermes' failover chain engages, rather than the caller silently
    receiving copy from a model it did not ask for.
    """
    if not _strict_model_enabled():
        return ""
    want, got = _model_family(requested), _model_family(served)
    if not want or not got or want == got:
        return ""
    return (
        f"model not found: requested {requested!r} but the Claude CLI served "
        f"{served!r}; the requested model is unavailable on this account"
    )


def _fable_provenance_retries(model: str) -> int:
    """Return the bounded retry count for a Fable request served by another family."""
    if _model_family(model) != "fable" or not _strict_model_enabled():
        return 0
    raw = _env("CLAUDE_CODE_CLI_FABLE_PROVENANCE_RETRIES", "1").strip()
    try:
        value = int(raw)
    except ValueError:
        value = 1
    return max(0, min(value, 2))


def _is_fable_provenance_substitution(model: str, outcome: dict) -> bool:
    """True only for a strict Fable request that completed under another model family."""
    if _model_family(model) != "fable" or not isinstance(outcome, dict):
        return False
    error = str(outcome.get("error") or "").lower()
    served = str(outcome.get("model") or "")
    return error.startswith("model not found:") and _model_family(served) not in {"", "fable"}


def _requires_provenance_buffer(model: str) -> bool:
    """Hold Fable output until terminal modelUsage proves the requested family ran.

    A transient wrong-family completion cannot be retracted after SSE headers
    and deltas are sent. Buffering Fable lets the bounded retry above discard
    substituted output and return only a provenance-checked completion.
    """
    return _strict_model_enabled() and _model_family(model) == "fable"


def _zero_usage() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def _map_usage(usage) -> dict:
    import efficiency_guard
    return efficiency_guard.map_claude_usage(usage)


# --------------------------------------------------------------------------- #
# Real token streaming (#3): `claude -p --output-format stream-json`
# --------------------------------------------------------------------------- #
def _real_streaming_enabled() -> bool:
    """Whether ``stream:true`` requests use incremental stream-json (#3).

    Default on ("auto"); set CLAUDE_CODE_CLI_STREAM=off to force the legacy
    run-to-completion-then-one-chunk behavior.
    """
    return _env("CLAUDE_CODE_CLI_STREAM", "auto").strip().lower() not in {
        "0", "false", "no", "off", "buffered", "sync",
    }


def iter_stream_json(lines):
    """Yield JSON object events from a Claude Code stream-json line source.

    The CLI writes one JSON object per line, but stderr/progress wrappers or
    future CLI versions may add blank or non-JSON lines. Invalid lines are
    ignored instead of failing the request, matching the shim's defensive JSON
    handling elsewhere.
    """
    for raw in lines:
        line = str(raw).strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            yield event


def _extract_stream_delta(event: dict) -> str:
    """Pull an assistant *text* delta out of a stream-json event.

    Defensive against schema drift: accepts Claude Code's wrapped
    ``{"type":"stream_event","event":...}`` records as well as direct
    Anthropic-style ``content_block_delta`` records. Only text deltas are
    forwarded; tool-argument ``input_json_delta`` events and everything else
    yield ``""``.
    """
    inner = event.get("event") if event.get("type") == "stream_event" else event
    if not isinstance(inner, dict):
        return ""
    itype = inner.get("type")
    if itype == "content_block_delta":
        delta = inner.get("delta")
        if isinstance(delta, dict):
            dtype = delta.get("type")
            text = delta.get("text")
            if isinstance(text, str) and dtype in ("text_delta", None):
                return text
    elif itype == "content_block_start":
        block = inner.get("content_block")
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                return text
    return ""


def _extract_assistant_message_text(event: dict) -> str:
    """Concatenate text blocks from a complete ``assistant`` message event."""
    msg = event.get("message")
    if not isinstance(msg, dict):
        return ""
    content = msg.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return ""


def _extract_result_event(event: dict):
    """Return ``(text, usage, error)`` for a terminal event, else ``None``.

    Recognizes the stream-json ``{"type":"result",...}`` event *and*, defensively,
    a bare ``{"result": ..., "usage": ...}`` blob - some CLI versions ignore
    ``--output-format stream-json`` and emit a single JSON object, and we still
    want a usable final result + usage in that case.
    """
    is_result_type = event.get("type") == "result"
    bare_result = event.get("type") is None and isinstance(event.get("result"), str)
    if not (is_result_type or bare_result):
        return None
    if event.get("is_error"):
        msg = str(event.get("result") or event.get("error") or "claude reported is_error")
        return ("", _map_usage(event.get("usage")), msg)
    text = event.get("result")
    text = text if isinstance(text, str) else str(event.get("content") or "")
    return (text, _map_usage(event.get("usage")), "")


def _kill_proc(proc) -> None:
    if proc.poll() is None:
        try:
            proc.kill()
        except Exception:
            pass


def stream_claude(prompt: str, model: str, engine: bool = False,
                  overrides: dict | None = None):
    """Generator yielding ``("delta", text)`` as tokens arrive, then exactly one
    ``("final", {"text","usage","error"})``.

    Invokes ``claude -p --output-format stream-json`` and forwards assistant-text
    deltas the moment the CLI emits them, so time-to-first-token is the first
    token rather than the whole generation (issue #3). It never raises: spawn
    failures, timeouts, and unparseable output are surfaced through the terminal
    ``final`` outcome, mirroring :func:`run_claude`. The prompt is fed on a
    thread and stderr drained on a thread to avoid pipe deadlocks.
    """
    argv = build_claude_argv(model, engine, overrides, stream=True)
    default_timeout = "1200" if engine else "600"
    try:
        timeout = int(_env("CLAUDE_CODE_CLI_TIMEOUT", default_timeout))
    except ValueError:
        timeout = int(default_timeout)
    cwd = _engine_cwd() if engine else None

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=build_subprocess_env(),
            cwd=cwd,
            bufsize=1,
        )
    except FileNotFoundError:
        msg = f"claude binary not found (tried {argv[0]!r}); set CLAUDE_CODE_CLI_BIN"
        yield ("final", {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg})
        return
    except Exception as exc:  # defensive: never crash the request thread
        msg = f"claude invocation failed: {exc!r}"
        yield ("final", {"text": f"[claude-code-cli error] {msg}", "usage": _zero_usage(), "error": msg})
        return

    def _feed():
        try:
            if proc.stdin:
                proc.stdin.write(prompt)
                proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    stderr_chunks: list[str] = []

    def _drain_stderr():
        try:
            if proc.stderr:
                for line in proc.stderr:
                    stderr_chunks.append(line)
        except Exception:
            pass

    timed_out = {"flag": False}

    def _watchdog():
        deadline = time.time() + timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)
        if proc.poll() is None:
            timed_out["flag"] = True
            _kill_proc(proc)

    feeder = threading.Thread(target=_feed, daemon=True)
    err_thread = threading.Thread(target=_drain_stderr, daemon=True)
    watch = threading.Thread(target=_watchdog, daemon=True)
    feeder.start()
    err_thread.start()
    watch.start()

    emitted_any = False
    emitted_parts: list[str] = []
    assistant_text = ""
    result_text = ""
    result_usage = _zero_usage()
    result_error = ""
    saw_result = False
    served_model = ""

    try:
        for event in iter_stream_json(proc.stdout or ()):  # watchdog kills on timeout
            etype = event.get("type")
            if etype in {"stream_event", "content_block_delta", "content_block_start"}:
                delta = _extract_stream_delta(event)
                if delta:
                    emitted_any = True
                    emitted_parts.append(delta)
                    yield ("delta", delta)
            elif etype == "assistant":
                served_model = _extract_served_model(event, model) or served_model
                assistant_text = _extract_assistant_message_text(event)
                if assistant_text and not emitted_any:
                    # No partial deltas (older CLI / flag off): emit the whole
                    # assistant message once so the client still streams.
                    emitted_any = True
                    emitted_parts.append(assistant_text)
                    yield ("delta", assistant_text)
            else:
                served_model = _extract_served_model(event, model) or served_model
                parsed = _extract_result_event(event)
                if parsed is not None:
                    result_text, result_usage, result_error = parsed
                    saw_result = True
    finally:
        try:
            proc.wait(timeout=5)
        except Exception:
            _kill_proc(proc)
        feeder.join(timeout=1)
        err_thread.join(timeout=1)
        watch.join(timeout=1)
        for pipe in (proc.stdout, proc.stderr):
            try:
                if pipe:
                    pipe.close()
            except Exception:
                pass

    emitted_text = "".join(emitted_parts)
    stderr_tail = "".join(stderr_chunks).strip()[-1000:]

    if timed_out["flag"]:
        msg = f"claude timed out after {timeout}s"
        text = emitted_text or f"[claude-code-cli error] {msg}"
        yield ("final", {"text": text, "usage": result_usage, "error": msg})
        return

    if result_error:
        if not emitted_any:
            err_line = f"[claude-code-cli error] {result_error}"
            emitted_text = err_line
            yield ("delta", err_line)
        yield ("final", {"text": emitted_text, "usage": result_usage, "error": result_error})
        return

    final_text = result_text or emitted_text or assistant_text
    if not emitted_any and final_text:
        # Never streamed (e.g. a bare-json CLI) but we have text → emit once.
        yield ("delta", final_text)

    if not final_text and not saw_result:
        rc = proc.returncode
        if rc not in (0, None):
            msg = f"claude exited {rc}: {stderr_tail}" if stderr_tail else f"claude exited {rc}"
        else:
            msg = "claude returned no output"
        err_line = f"[claude-code-cli error] {msg}"
        yield ("delta", err_line)
        yield ("final", {"text": err_line, "usage": _zero_usage(), "error": msg})
        return

    substitution = check_model_provenance(model, served_model)
    if substitution:
        yield ("final", {
            "text": f"[claude-code-cli error] {substitution}",
            "usage": result_usage,
            "error": substitution,
            "model": served_model,
        })
        return

    yield ("final", {
        "text": final_text,
        "usage": result_usage,
        "error": "",
        "model": served_model,
    })


# --------------------------------------------------------------------------- #
# OpenAI response shaping
# --------------------------------------------------------------------------- #
def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def build_completion(model: str, text: str, usage: dict) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": usage,
    }


def build_tool_call_completion(model: str, tool_calls: list[dict], usage: dict) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
            "finish_reason": "tool_calls",
        }],
        "usage": usage,
    }


def build_stream_chunks(model: str, text: str, usage: dict, include_usage: bool):
    cid, created = _completion_id(), int(time.time())

    def chunk(delta, finish=None, usage_obj=None):
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage_obj is not None:
            payload["usage"] = usage_obj
        return payload

    yield chunk({"role": "assistant"})
    if text:
        yield chunk({"content": text})
    yield chunk({}, finish="stop")
    if include_usage:
        # OpenAI emits a trailing usage-only chunk with an empty choices list.
        yield {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [], "usage": usage,
        }


def build_tool_call_stream_chunks(model: str, tool_calls: list[dict], usage: dict, include_usage: bool):
    cid, created = _completion_id(), int(time.time())

    def chunk(delta, finish=None, usage_obj=None):
        payload = {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage_obj is not None:
            payload["usage"] = usage_obj
        return payload

    yield chunk({"role": "assistant"})
    for idx, tc in enumerate(tool_calls):
        yield chunk({
            "tool_calls": [{
                "index": idx,
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": tc["function"]["arguments"],
                },
            }]
        })
    yield chunk({}, finish="tool_calls")
    if include_usage:
        yield {
            "id": cid, "object": "chat.completion.chunk", "created": created,
            "model": model, "choices": [], "usage": usage,
        }


# --------------------------------------------------------------------------- #
# CLI failure -> HTTP status classification
# --------------------------------------------------------------------------- #
# The Claude Code CLI reports every failure mode -- expired subscription auth,
# usage limits, a missing/invalid model, a missing binary, a hung invocation --
# as ordinary result text with exit code 0. If the shim forwards that text
# inside a 200 completion envelope, Hermes never raises: agent/error_classifier
# is never consulted, should_fallback is never set, and the configured fallback
# chain (e.g. Opus 5 -> Sol 5.6) is silently skipped. The agent just prints the
# error string as its answer.
#
# Mapping these to real HTTP statuses restores failover parity with the
# openai-codex provider, which already returns 401/404/429/5xx natively.

_CLI_MODEL_PATTERNS = (
    "issue with the selected model",
    "model not found",
    "invalid model",
    "unknown model",
    "unsupported model",
    "may not exist",
    "does not exist",
    "you may not have access",
    "pick a different model",
)

_CLI_AUTH_PATTERNS = (
    "invalid api key",
    "authentication_error",
    "unauthorized",
    "not authenticated",
    "not logged in",
    "please run /login",
    "claude login",
    "log in to claude",
    "session expired",
    "token expired",
    "token has expired",
    "oauth token",
    "revoked",
)

_CLI_RATE_PATTERNS = (
    "rate limit",
    "rate_limit",
    "too many requests",
    "usage limit",
    "quota",
    "overloaded",
    "at capacity",
    "credit balance",
    "insufficient credits",
    "upgrade to",
)


# An exhausted account is deterministic for this route: no amount of retrying
# refills the bucket, and the only real recovery is a different account. Hermes
# maps a bare 429 to FailoverReason.rate_limit (retryable -- burns the whole
# backoff budget before cascading) and a 503 to FailoverReason.overloaded
# (carries no should_fallback at all). Prefixing with a phrase from Hermes'
# _BILLING_PATTERNS lands it on the billing verdict instead: retryable=False,
# should_fallback=True, immediate cascade to the next fallback_providers entry.
_CLI_EXHAUSTION_PREFIX = "credits exhausted: "

_CLI_EXHAUSTION_PATTERNS = (
    "manage usage credits",
    "usage limit reached",
    "usage limit exceeded",
    "weekly limit",
    "weekly usage limit",
    "session limit",
    "fable limit",
    "credit balance",
    "insufficient credits",
    "out of usage credits",
)


def classify_cli_error(msg: str) -> tuple:
    """Map a claude-code-cli failure string to (status, error_type, message).

    The message is normalized where the CLI's wording would miss Hermes'
    pattern lists -- the CLI says "It may not exist", but Hermes'
    _MODEL_NOT_FOUND_PATTERNS looks for "model not found" / "does not exist".
    Without the prefix a 404 falls through to the generic bucket (unknown,
    retryable, no fallback) and failover still never fires.
    """
    raw = (msg or "").strip() or "claude-code-cli failed"
    low = raw.lower()
    if any(p in low for p in _CLI_MODEL_PATTERNS):
        normalized = raw if low.startswith("model not found:") else f"model not found: {raw}"
        return 404, "model_not_found", normalized
    if any(p in low for p in _CLI_AUTH_PATTERNS):
        return 401, "authentication_error", raw
    if low.startswith(_CLI_EXHAUSTION_PREFIX):
        return 429, "rate_limit_error", raw
    if any(p in low for p in _CLI_EXHAUSTION_PATTERNS):
        # Keep the CLI's own wording after the prefix: the operator reading the
        # log still needs to know WHICH limit was hit and when it resets.
        return 429, "rate_limit_error", f"{_CLI_EXHAUSTION_PREFIX}{raw}"
    if any(p in low for p in _CLI_RATE_PATTERNS):
        return 429, "rate_limit_error", raw
    # Timeouts, missing binary, non-zero exit, empty output: the backend is
    # unusable right now. 503 -> FailoverReason.overloaded (retryable, then
    # fallback) instead of a silent 200.
    return 503, "service_unavailable", raw


# --------------------------------------------------------------------------- #
# HTTP handler
# --------------------------------------------------------------------------- #
class Handler(BaseHTTPRequestHandler):
    server_version = "claude-code-cli-shim/1.0"

    # ---- low-level write helpers ----
    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str, etype: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": etype}})

    def _send_cli_failure(self, error_msg: str) -> None:
        """Return a claude-code-cli failure as a real HTTP error, not a 200.

        A 200 completion carrying error text is invisible to Hermes' failover
        machinery: nothing raises, error_classifier never runs, should_fallback
        is never set, and the configured fallback chain is skipped entirely.
        """
        status, etype, msg = classify_cli_error(error_msg)
        self.log_message("cli failure -> HTTP %s (%s): %s", status, etype, msg[:200])
        self._send_error(status, msg, etype)

    def log_message(self, fmt, *args):  # quieter, single-line logs to stderr
        sys.stderr.write("[claude-code-cli] " + (fmt % args) + "\n")

    def _authorized(self) -> bool:
        """Require the configured localhost bearer for inference requests."""
        expected = _env("CLAUDE_CODE_CLI_API_KEY").strip()
        if not expected:
            return True
        value = (self.headers.get("Authorization") or "").strip()
        token = value[7:].strip() if value.lower().startswith("bearer ") else ""
        if not token or not hmac.compare_digest(token, expected):
            self._send_error(401, "invalid or missing bearer token", "authentication_error")
            return False
        return True

    # ---- routing ----
    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path in ("/healthz", "/health", "/v1/healthz"):
            self._send_json(200, {"status": "ok", "bin": resolve_claude_bin()})
        elif path == "/v1/models":
            now = int(time.time())
            self._send_json(200, {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "created": now, "owned_by": "claude-code-cli"}
                    for m in ADVERTISED_MODELS
                ],
            })
        else:
            self._send_error(404, f"unknown path: {self.path}", "not_found")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path != "/v1/chat/completions":
            self._send_error(404, f"unknown path: {self.path}", "not_found")
            return
        if not self._authorized():
            return

        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw.decode("utf-8")) if raw else {}
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_error(400, f"invalid JSON body: {exc}")
            return

        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            self._send_error(400, "'messages' must be a non-empty array")
            return

        model = str(body.get("model") or _env("CLAUDE_CODE_CLI_MODEL", "sonnet")).strip() or "sonnet"
        model = normalize_model(model)
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))

        raw_tools = body.get("tools")
        if raw_tools:
            if not isinstance(raw_tools, list):
                self._send_error(400, "tools must be an array", "invalid_request_error")
                return
            try:
                _mcp_tool_specs([item if isinstance(item, dict) and "function" in item
                                 else {"function": item} for item in raw_tools])
            except ValueError as exc:
                self._send_error(400, str(exc), "invalid_request_error")
                return
        tools = _openai_tool_defs(raw_tools)
        tool_choice = body.get("tool_choice", "auto")

        # Engine mode: let Claude Code use its own tools to do real work, but
        # only when explicitly requested. In the default/auto setting, OpenAI
        # request tools mean Hermes-native tool calls, matching normal providers.
        mode = _env("CLAUDE_CODE_CLI_ENGINE", "auto").strip().lower()
        if mode in {"always", "on", "1", "true", "yes"}:
            engine = True
        else:
            engine = False
        native_tools = (
            bool(tools)
            and not engine
            and _native_tools_enabled()
            and not _tool_choice_is_none(tool_choice)
        )

        images = _ImageCollector()
        try:
            prompt = flatten_messages(messages, images, native_tools=native_tools)
            if engine:
                prompt += (
                    "\n\nYou are operating as an autonomous coding engine with your own tools "
                    "(Read, Write, Edit, Bash, Glob, Grep, etc.) in your working directory. Use "
                    "them to actually carry out the task - read/modify files, run commands - then "
                    "report what you did. Do not ask for permission; act."
                )
            overrides = extract_overrides(body)
            started = time.time()

            if native_tools:
                # Advertise Hermes-owned tools through a request-scoped MCP
                # server. The server captures validated selections but never
                # executes them; Hermes receives ordinary OpenAI tool_calls and
                # retains its normal approval and execution boundary.
                outcome = run_claude_mcp_bridge(
                    prompt, model, tools, tool_choice, overrides=overrides
                )
                if outcome["error"]:
                    if outcome.get("http_status"):
                        self._send_error(
                            int(outcome["http_status"]), outcome["error"],
                            "invalid_request_error" if int(outcome["http_status"]) < 500
                            else "service_unavailable",
                        )
                    else:
                        self._send_cli_failure(outcome["error"])
                else:
                    served = outcome.get("model") or model
                    tool_calls = outcome.get("tool_calls") or []
                    if tool_calls:
                        if stream:
                            self._stream_tool_call_response(served, tool_calls, outcome["usage"], include_usage)
                        else:
                            self._send_json(200, build_tool_call_completion(served, tool_calls, outcome["usage"]))
                    else:
                        if stream:
                            self._stream_response(served, outcome, include_usage)
                        else:
                            self._send_json(200, build_completion(served, outcome["text"], outcome["usage"]))
            elif stream and _real_streaming_enabled() and not _requires_provenance_buffer(model):
                outcome = self._stream_live_response(model, prompt, engine, overrides, include_usage)
            else:
                outcome = run_claude(prompt, model, engine, overrides)
                if outcome["error"]:
                    self._send_cli_failure(outcome["error"])
                else:
                    served = outcome.get("model") or model
                    if stream:
                        self._stream_response(served, outcome, include_usage)
                    else:
                        self._send_json(200, build_completion(served, outcome["text"], outcome["usage"]))

            try:
                import efficiency_guard
                _atlas_log = _env('CLAUDE_CODE_CLI_USAGE_LOG', '')
                if isinstance(_atlas_log, str) and _atlas_log.strip():
                    import uuid
                    from datetime import datetime, timezone
                    _atlas_metadata = body.get('metadata') if isinstance(body, dict) else None
                    _atlas_session = _atlas_metadata.get('session_id') if isinstance(_atlas_metadata, dict) else None
                    if not (isinstance(_atlas_session, str) and 0 < len(_atlas_session) <= 128 and _atlas_session.isprintable()):
                        _atlas_session = 'unattributed'
                    native_tool_count = len(tools) if isinstance(tools, (list, tuple)) else 0
                    _atlas_model = outcome.get('model')
                    if not (isinstance(_atlas_model, str) and 0 < len(_atlas_model) <= 128 and _atlas_model.isprintable()):
                        _atlas_model = model if isinstance(model, str) and 0 < len(model) <= 128 and model.isprintable() else 'unknown'
                    _atlas_port = int(_env('CLAUDE_CODE_CLI_PORT', '8765'))
                    if not 1 <= _atlas_port <= 65535:
                        raise ValueError
                    _atlas_provider = {8765: 'Personal', 8767: 'Personal', 8768: 'Hedra', 8769: 'XBuild', 8770: 'Genesis'}.get(_atlas_port, 'claude-code-cli')
                    efficiency_guard.append_request(_atlas_log, {
                        'timestamp': time.time(),
                        'request_id': uuid.uuid4().hex,
                        'session_id': _atlas_session,
                        'profile': 'default',
                        'provider': _atlas_provider,
                        'endpoint_port': _atlas_port,
                        'model': _atlas_model,
                        'task': 'shim',
                        'prompt_chars': len(prompt) if isinstance(prompt, str) else 0,
                        'tool_count': native_tool_count,
                        'latency_seconds': max(0.0, time.time() - started),
                        'usage': outcome.get('usage') or {},
                        'status': 'error' if outcome.get('error') else 'ok',
                        'retry_count': 0,
                    })
            except Exception:
                try:
                    import sys
                    sys.stderr.write('Atlas efficiency telemetry unavailable.\n')
                except Exception:
                    pass

            self.log_message(
                "model=%s engine=%s native_tools=%s stream=%s live_stream=%s images=%s effort=%s %.1fs %s",
                model, engine, native_tools, stream,
                bool(
                    stream and _real_streaming_enabled() and not native_tools
                    and not _requires_provenance_buffer(model)
                ), images.count,
                overrides.get("effort") or _env("CLAUDE_CODE_CLI_EFFORT", "high"),
                time.time() - started, "error" if outcome["error"] else "ok",
            )
        finally:
            images.cleanup()

    def _write_sse(self, payload: dict | str) -> None:
        if isinstance(payload, str):
            data = payload.encode("utf-8")
        else:
            data = json.dumps(payload).encode("utf-8")
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def _stream_response(self, model: str, outcome: dict, include_usage: bool) -> None:
        """Legacy buffered stream response: one content chunk after completion."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for chunk in build_stream_chunks(model, outcome["text"], outcome["usage"], include_usage):
                self._write_sse(chunk)
            self._write_sse("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up mid-stream

    def _stream_tool_call_response(self, model: str, tool_calls: list[dict],
                                   usage: dict, include_usage: bool) -> None:
        """Buffered SSE response for native Hermes tool calls."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            for chunk in build_tool_call_stream_chunks(model, tool_calls, usage, include_usage):
                self._write_sse(chunk)
            self._write_sse("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_live_response(self, model: str, prompt: str, engine: bool,
                              overrides: dict, include_usage: bool) -> dict:
        """SSE bridge for real `stream-json` deltas (#3)."""
        cid, created = _completion_id(), int(time.time())

        def chunk(delta, finish=None, usage_obj=None):
            payload = {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage_obj is not None:
                payload["usage"] = usage_obj
            return payload

        outcome = {"text": "", "usage": _zero_usage(), "error": "stream ended before final event"}
        headers_sent = False
        prelude = ""
        hold_limit = 128

        def open_stream() -> None:
            # Deferred so a CLI failure in the prelude can still be returned
            # as an HTTP error. Once the 200 status line is on the wire it
            # cannot be taken back, and Hermes' failover never fires.
            nonlocal headers_sent
            if headers_sent:
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            headers_sent = True
            self._write_sse(chunk({"role": "assistant"}))

        try:
            for kind, payload in stream_claude(prompt, model, engine, overrides):
                if kind == "delta":
                    if headers_sent:
                        self._write_sse(chunk({"content": payload}))
                        continue
                    while payload and not headers_sent:
                        room = hold_limit - len(prelude)
                        prelude += payload[:room]
                        payload = payload[room:]
                        if len(prelude) == hold_limit:
                            patterns = (_CLI_MODEL_PATTERNS + _CLI_AUTH_PATTERNS
                                        + _CLI_RATE_PATTERNS + _CLI_EXHAUSTION_PATTERNS)
                            if hold_limit == 128 and any(
                                pattern in prelude.lower()
                                for pattern in patterns
                            ):
                                hold_limit = 4096
                            else:
                                open_stream()
                                self._write_sse(chunk({"content": prelude}))
                                prelude = ""
                    if headers_sent and payload:
                        self._write_sse(chunk({"content": payload}))
                elif kind == "final" and isinstance(payload, dict):
                    outcome = payload
                    break
            if outcome["error"] and not headers_sent:
                # Discard the held error text so Hermes can classify the HTTP failure.
                self._send_cli_failure(outcome["error"])
                return outcome
            open_stream()
            if prelude:
                self._write_sse(chunk({"content": prelude}))
            self._write_sse(chunk({}, finish="stop"))
            if include_usage:
                self._write_sse({
                    "id": cid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model,
                    "choices": [],
                    "usage": outcome["usage"],
                })
            self._write_sse("[DONE]")
        except (BrokenPipeError, ConnectionResetError):
            pass  # client hung up mid-stream
        return outcome


def main() -> int:
    host = _env("CLAUDE_CODE_CLI_HOST", "127.0.0.1")
    try:
        port = int(_env("CLAUDE_CODE_CLI_PORT", "8765"))
    except ValueError:
        port = 8765

    bin_path = resolve_claude_bin()
    if shutil.which(bin_path) is None and not pathlib.Path(bin_path).exists():
        sys.stderr.write(
            f"[claude-code-cli] WARNING: claude binary {bin_path!r} not found on PATH; "
            "requests will fail until CLAUDE_CODE_CLI_BIN points at a valid `claude`.\n"
        )

    httpd = ThreadingHTTPServer((host, port), Handler)
    sys.stderr.write(
        f"[claude-code-cli] serving OpenAI-compatible Claude Code CLI on "
        f"http://{host}:{port}/v1  (claude={bin_path})\n"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[claude-code-cli] shutting down\n")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
