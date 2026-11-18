#!/usr/bin/env python3
"""Per-request stdio MCP bridge for Hermes-owned OpenAI tools.

The bridge advertises a request's tool schemas and captures structured tool
selections. It never imports or executes Hermes tools. The HTTP shim reads the
capture only after the Claude CLI exits, converts it to OpenAI ``tool_calls``,
and leaves execution and approval enforcement to Hermes.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import re
import sys
from typing import Any


_PROTOCOL_VERSION = "2024-11-05"
_TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_ANNOTATION_KEYS = {
    "$id", "$schema", "$comment", "title", "description", "default",
    "examples", "deprecated", "readOnly", "writeOnly", "format",
}
_STRUCTURAL_KEYS = {
    "$ref", "$defs", "definitions", "type", "enum", "const", "nullable",
    "allOf", "anyOf", "oneOf", "not", "if", "then", "else",
    "properties", "patternProperties", "additionalProperties", "required",
    "minProperties", "maxProperties", "dependentRequired", "dependencies",
    "propertyNames", "items", "prefixItems", "additionalItems", "contains",
    "minContains", "maxContains", "minItems", "maxItems", "uniqueItems",
    "minLength", "maxLength", "pattern", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
}


def _json_equal(left: Any, right: Any) -> bool:
    return type(left) is type(right) and left == right


def _type_matches(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool)
                  and math.isfinite(value),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(expected, False)


def _resolve_ref(root: dict, ref: str) -> Any:
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported non-local schema reference {ref!r}")
    current: Any = root
    for raw in ref[2:].split("/"):
        key = raw.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or key not in current:
            raise ValueError(f"unresolved schema reference {ref!r}")
        current = current[key]
    return current


def validate_json_schema(value: Any, schema: Any, *, root: dict | None = None,
                         path: str = "$", depth: int = 0) -> list[str]:
    """Validate common Draft 7/2020-12 constraints without third-party deps.

    Unsupported assertion keywords fail closed. Annotation keywords are safely
    ignored. Local ``$ref`` values are supported; remote refs are rejected.
    """
    if schema is True:
        return []
    if schema is False:
        return [f"{path}: schema rejects every value"]
    if not isinstance(schema, dict):
        return [f"{path}: schema must be an object or boolean"]
    if depth > 64:
        return [f"{path}: schema nesting exceeds 64 levels"]
    root = schema if root is None else root
    unknown = set(schema) - _ANNOTATION_KEYS - _STRUCTURAL_KEYS
    if unknown:
        return [f"{path}: unsupported schema keyword(s): {', '.join(sorted(unknown))}"]
    if "$ref" in schema:
        try:
            target = _resolve_ref(root, str(schema["$ref"]))
        except ValueError as exc:
            return [f"{path}: {exc}"]
        return validate_json_schema(value, target, root=root, path=path, depth=depth + 1)

    errors: list[str] = []
    if schema.get("nullable") is True and value is None:
        return []
    expected = schema.get("type")
    if isinstance(expected, str):
        if not _type_matches(value, expected):
            return [f"{path}: expected {expected}"]
    elif isinstance(expected, list):
        types = [item for item in expected if isinstance(item, str)]
        if not types or not any(_type_matches(value, item) for item in types):
            return [f"{path}: expected one of {types!r}"]
    elif expected is not None:
        return [f"{path}: schema type must be a string or array"]

    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not any(_json_equal(value, item) for item in choices):
            errors.append(f"{path}: value is not in enum")
    if "const" in schema and not _json_equal(value, schema["const"]):
        errors.append(f"{path}: value does not equal const")

    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword not in schema:
            continue
        branches = schema[keyword]
        if not isinstance(branches, list) or not branches:
            errors.append(f"{path}: {keyword} must be a non-empty array")
            continue
        branch_errors = [validate_json_schema(value, branch, root=root, path=path, depth=depth + 1)
                         for branch in branches]
        passes = sum(not result for result in branch_errors)
        if keyword == "allOf" and passes != len(branches):
            errors.append(f"{path}: allOf constraint failed")
        elif keyword == "anyOf" and passes == 0:
            errors.append(f"{path}: anyOf constraint failed")
        elif keyword == "oneOf" and passes != 1:
            errors.append(f"{path}: oneOf constraint matched {passes} branches")
    if "not" in schema and not validate_json_schema(
        value, schema["not"], root=root, path=path, depth=depth + 1
    ):
        errors.append(f"{path}: not constraint failed")
    if "if" in schema:
        condition_passes = not validate_json_schema(
            value, schema["if"], root=root, path=path, depth=depth + 1
        )
        branch_key = "then" if condition_passes else "else"
        if branch_key in schema:
            errors.extend(validate_json_schema(
                value, schema[branch_key], root=root, path=path, depth=depth + 1
            ))

    if isinstance(value, dict):
        required = schema.get("required", [])
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            errors.append(f"{path}: required must be an array of strings")
            required = []
        for key in required:
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            errors.append(f"{path}: properties must be an object")
            properties = {}
        patterns = schema.get("patternProperties", {})
        if not isinstance(patterns, dict):
            errors.append(f"{path}: patternProperties must be an object")
            patterns = {}
        compiled: list[tuple[re.Pattern, Any]] = []
        for pattern, subschema in patterns.items():
            try:
                compiled.append((re.compile(pattern), subschema))
            except (TypeError, re.error):
                errors.append(f"{path}: invalid patternProperties expression {pattern!r}")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            matched = False
            if key in properties:
                matched = True
                errors.extend(validate_json_schema(
                    item, properties[key], root=root, path=f"{path}.{key}", depth=depth + 1
                ))
            for pattern, subschema in compiled:
                if pattern.search(key):
                    matched = True
                    errors.extend(validate_json_schema(
                        item, subschema, root=root, path=f"{path}.{key}", depth=depth + 1
                    ))
            if not matched:
                if additional is False:
                    errors.append(f"{path}: unexpected property {key!r}")
                elif isinstance(additional, (dict, bool)):
                    errors.extend(validate_json_schema(
                        item, additional, root=root, path=f"{path}.{key}", depth=depth + 1
                    ))
        for keyword, comparison in (("minProperties", lambda n, x: n < x),
                                    ("maxProperties", lambda n, x: n > x)):
            if keyword in schema and comparison(len(value), schema[keyword]):
                errors.append(f"{path}: {keyword} constraint failed")
        dependent = schema.get("dependentRequired", schema.get("dependencies", {}))
        if isinstance(dependent, dict):
            for key, needs in dependent.items():
                if key in value and isinstance(needs, list):
                    for need in needs:
                        if need not in value:
                            errors.append(f"{path}: property {key!r} requires {need!r}")
        if "propertyNames" in schema:
            for key in value:
                errors.extend(validate_json_schema(
                    key, schema["propertyNames"], root=root, path=f"{path}.<property>", depth=depth + 1
                ))

    if isinstance(value, list):
        prefix = schema.get("prefixItems")
        if isinstance(prefix, list):
            for index, subschema in enumerate(prefix[:len(value)]):
                errors.extend(validate_json_schema(
                    value[index], subschema, root=root, path=f"{path}[{index}]", depth=depth + 1
                ))
            item_schema = schema.get("items", schema.get("additionalItems", True))
            start = len(prefix)
        else:
            item_schema = schema.get("items", True)
            start = 0
        if isinstance(item_schema, list):
            for index, subschema in enumerate(item_schema[:len(value)]):
                errors.extend(validate_json_schema(
                    value[index], subschema, root=root, path=f"{path}[{index}]", depth=depth + 1
                ))
        else:
            for index in range(start, len(value)):
                errors.extend(validate_json_schema(
                    value[index], item_schema, root=root, path=f"{path}[{index}]", depth=depth + 1
                ))
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: minItems constraint failed")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: maxItems constraint failed")
        if schema.get("uniqueItems") is True:
            encoded = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]
            if len(encoded) != len(set(encoded)):
                errors.append(f"{path}: uniqueItems constraint failed")
        if "contains" in schema:
            matches = sum(not validate_json_schema(
                item, schema["contains"], root=root, path=f"{path}[{index}]", depth=depth + 1
            ) for index, item in enumerate(value))
            minimum = schema.get("minContains", 1)
            maximum = schema.get("maxContains")
            if matches < minimum or (maximum is not None and matches > maximum):
                errors.append(f"{path}: contains constraint failed")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: minLength constraint failed")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: maxLength constraint failed")
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], value) is None:
                    errors.append(f"{path}: pattern constraint failed")
            except (TypeError, re.error):
                errors.append(f"{path}: invalid pattern expression")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        for keyword, failed in (
            ("minimum", lambda limit: value < limit),
            ("maximum", lambda limit: value > limit),
            ("exclusiveMinimum", lambda limit: value <= limit),
            ("exclusiveMaximum", lambda limit: value >= limit),
        ):
            if keyword in schema:
                try:
                    if failed(schema[keyword]):
                        errors.append(f"{path}: {keyword} constraint failed")
                except TypeError:
                    errors.append(f"{path}: invalid {keyword} constraint")
        if "multipleOf" in schema:
            try:
                divisor = schema["multipleOf"]
                quotient = value / divisor
                if divisor <= 0 or not math.isclose(quotient, round(quotient), abs_tol=1e-9):
                    errors.append(f"{path}: multipleOf constraint failed")
            except (TypeError, ZeroDivisionError):
                errors.append(f"{path}: invalid multipleOf constraint")
    return errors


def _send(value: dict) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _append_event(path: pathlib.Path, event: dict) -> None:
    data = (json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)


def _tool_result(message: str, *, error: bool) -> dict:
    return {"content": [{"type": "text", "text": message}], "isError": error}


def main() -> int:
    if len(sys.argv) != 3:
        sys.stderr.write("usage: hermes_mcp_bridge.py SPEC_JSON EVENTS_JSONL\n")
        return 2
    spec_path, events_path = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        request_id = str(spec["request_id"])
        raw_tools = spec["tools"]
        if not isinstance(raw_tools, list) or not raw_tools:
            raise ValueError("tools must be a non-empty array")
        tools: dict[str, dict] = {}
        for tool in raw_tools:
            name = tool.get("name") if isinstance(tool, dict) else None
            if not isinstance(name, str) or not _TOOL_NAME_RE.fullmatch(name) or name in tools:
                raise ValueError(f"invalid or duplicate tool name {name!r}")
            input_schema = tool.get("inputSchema", {"type": "object"})
            if not isinstance(input_schema, (dict, bool)):
                raise ValueError(f"tool {name!r} has an invalid input schema")
            tools[name] = {
                "name": name,
                "description": str(tool.get("description") or ""),
                "inputSchema": input_schema,
            }
    except Exception as exc:
        sys.stderr.write(f"invalid bridge spec: {exc}\n")
        return 2

    for raw in sys.stdin:
        try:
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise ValueError("request must be an object")
        except Exception as exc:
            _send({"jsonrpc": "2.0", "id": None,
                   "error": {"code": -32700, "message": f"Parse error: {exc}"}})
            continue
        rid = request.get("id")
        method = request.get("method")
        if rid is None:
            continue
        if method == "initialize":
            result = {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "hermes-openai-tool-bridge", "version": "1.0"},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": list(tools.values())}
        elif method == "tools/call":
            params = request.get("params")
            params = params if isinstance(params, dict) else {}
            name = params.get("name")
            arguments = params.get("arguments")
            raw_meta = params.get("_meta")
            meta: dict = raw_meta if isinstance(raw_meta, dict) else {}
            tool_use_id = meta.get("claudecode/toolUseId")
            event = {
                "request_id": request_id,
                "status": "accepted",
                "name": name,
                "arguments": arguments,
                "tool_use_id": tool_use_id if isinstance(tool_use_id, str) else "",
            }
            error = ""
            if not isinstance(name, str) or name not in tools:
                error = f"unknown tool {name!r}"
            elif not isinstance(arguments, dict):
                error = "arguments must be an object"
            else:
                validation = validate_json_schema(arguments, tools[name]["inputSchema"])
                if validation:
                    error = "arguments failed schema validation: " + "; ".join(validation[:8])
            if error:
                event["status"] = "rejected"
                event["error"] = error
                result = _tool_result("Tool request rejected by Hermes bridge: " + error, error=True)
            else:
                result = _tool_result(
                    "DEFERRED: Hermes has not executed this tool. Stop this turn; the host will execute it after approval checks.",
                    error=False,
                )
            _append_event(events_path, event)
        else:
            _send({"jsonrpc": "2.0", "id": rid,
                   "error": {"code": -32601, "message": "Method not found"}})
            continue
        _send({"jsonrpc": "2.0", "id": rid, "result": result})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
