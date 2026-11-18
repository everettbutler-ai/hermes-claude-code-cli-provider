import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import fcntl

LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.:@/\-]{0,95}\Z')
FIELDS = {'timestamp', 'request_id', 'session_id', 'profile', 'provider', 'endpoint_port', 'model', 'task', 'effort', 'prompt_chars', 'tool_count', 'latency_seconds', 'usage', 'status', 'retry_count'}
COUNTERS = {'prompt_tokens', 'completion_tokens', 'total_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'}
MAX_SCAN = 10 * 1024 * 1024


def _count(value):
    return type(value) is int and 0 <= value <= 2**63 - 1


def _number(value):
    return type(value) in (int, float) and 0 <= value <= 1e20 and math.isfinite(value)


def _label(value):
    return isinstance(value, str) and LABEL.fullmatch(value) is not None


def map_claude_usage(usage):
    """Claude input_tokens excludes cache reads/writes; sum each bucket once.

    Missing cache measurements are omitted, rather than reported as measured zero.
    Hermes reads prompt_tokens_details.cached_tokens and the top-level
    cache_creation_input_tokens field.
    """
    source = usage if isinstance(usage, dict) else {}
    keys = ('input_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens', 'output_tokens')
    values = {k: source[k] if _count(source.get(k)) else 0 for k in keys}
    available = any(k in source for k in keys) and all(_count(source[k]) for k in keys if k in source)
    prompt = sum(values[k] for k in keys[:3])
    result = {'prompt_tokens': prompt, 'completion_tokens': values['output_tokens'],
              'total_tokens': prompt + values['output_tokens'],
              'prompt_tokens_details': {}, 'usage_available': available}
    for source_key, detail in (('cache_read_input_tokens', 'cached_tokens'),
                              ('cache_creation_input_tokens', 'cache_write_tokens')):
        if source_key in source and _count(source[source_key]):
            result[source_key] = source[source_key]
            result['prompt_tokens_details'][detail] = source[source_key]
    return result


def _usage(value):
    if not isinstance(value, dict):
        return {}
    out = {k: value[k] for k in COUNTERS if _count(value.get(k))}
    details = value.get('prompt_tokens_details')
    if isinstance(details, dict):
        out['prompt_tokens_details'] = {k: details[k] for k in ('cached_tokens', 'cache_write_tokens') if _count(details.get(k))}
    if type(value.get('usage_available')) is bool:
        out['usage_available'] = value['usage_available']
    return out


def _clean(record):
    if not isinstance(record, dict):
        raise ValueError('record must be a mapping')
    out = {}
    for key in FIELDS:
        if key not in record:
            continue
        value = record[key]
        if key == 'usage':
            out[key] = _usage(value)
        elif key in ('timestamp', 'latency_seconds'):
            if _number(value):
                out[key] = value
        elif key == 'endpoint_port':
            if type(value) is int and 1 <= value <= 65535:
                out[key] = value
        elif key in ('prompt_chars', 'tool_count', 'retry_count'):
            if _count(value):
                out[key] = value
        elif _label(value):
            out[key] = value
    return out


@contextlib.contextmanager
def _private_file(path, flags):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # O_NOFOLLOW is required: do not silently weaken symlink protection.
    fd = os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK, 0o600)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('regular file required')
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.fchmod(fd, 0o600)
        yield fd
    finally:
        os.close(fd)


def append_request(path, record):
    """Append only approved metadata; never pass prompts or raw errors here."""
    data = (json.dumps(_clean(record), separators=(',', ':'), allow_nan=False) + '\n').encode()
    with _private_file(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND) as fd:
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise OSError('append unavailable')
            data = data[written:]


def check_brief(prompt, max_chars=60000):
    """Conservative character guard, not a tokenizer.

    Normal target is approximately 20K tokens. The 60,000-character default is
    never silently increased; a reviewed exceptional task may pass an override.
    Call this before invoking any inference/request function.
    """
    if type(max_chars) is not int or max_chars <= 0:
        raise ValueError('max_chars must be a positive integer')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('brief must be a non-empty string; provide a compact packet')
    if len(prompt) > max_chars:
        raise ValueError('brief too large; provide a compact packet')
    return prompt


def _message(session, conditions):
    descriptions = {'large_prompt': 'request prompt_tokens >150000',
                    'high_calls': 'successful calls >=40 in 30 minutes'}
    return 'Session ' + session + ': ' + '; '.join(descriptions[c] for c in conditions)


def summarize_burn(rows, now):
    """Return warnings only; never cancel work or estimate provider quotas."""
    if not _number(now):
        return []
    sessions, seen = {}, set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        ts, session = row.get('timestamp'), row.get('session_id')
        if not _number(ts) or not now - 1800 <= ts <= now or not _label(session):
            continue
        clean = _clean(row)
        request_id = clean.get('request_id')
        key = ('id', request_id) if request_id else ('fallback', hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
        if key in seen:
            continue
        seen.add(key)
        counts = sessions.setdefault(session, {'calls': 0, 'large': False})
        if clean.get('status') in ('success', 'ok', 'completed'):
            counts['calls'] += 1
        usage = clean.get('usage', {})
        if usage.get('usage_available') is not False and usage.get('prompt_tokens', 0) > 150000:
            counts['large'] = True
    warnings = []
    for session, counts in sorted(sessions.items()):
        conditions = []
        if counts['large']:
            conditions.append('large_prompt')
        if counts['calls'] >= 40:
            conditions.append('high_calls')
        if conditions:
            warnings.append({'session_id': session, 'conditions': conditions,
                             'message': _message(session, conditions)})
    return warnings


def _tail(path):
    # A bounded fresh tail scan avoids all inode/offset reuse after rotation.
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('regular telemetry file required')
        start = max(0, info.st_size - MAX_SCAN)
        stream.seek(start)
        data = stream.read(MAX_SCAN)
    if start:
        data = data.partition(b'\n')[2]
    rows = []
    for line in data.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            continue
        try:
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
        except (ValueError, UnicodeError, RecursionError):
            pass
    return rows


def _read_state(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC | os.O_NONBLOCK)
    except FileNotFoundError:
        return {}
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('regular state file required')
        try:
            value = json.loads(stream.read(MAX_SCAN))
        except (ValueError, UnicodeError, RecursionError):
            return {}
    return value if isinstance(value, dict) else {}


def _save_state(path, state):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('symlink state refused')
    fd, name = tempfile.mkstemp(prefix='.guard-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(state, stream, separators=(',', ':'), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def watch_tick(events, state_path, now=None):
    """One watchdog tick; stdout contains at most one aggregate JSON line.

    The caller may invoke --watch repeatedly. No scheduler is installed here.
    Unavailable telemetry produces no estimates and no output.
    """
    now = time.time() if now is None else now
    if not _number(now):
        return []
    with _private_file(str(state_path) + '.lock', os.O_RDWR | os.O_CREAT):
        state = _read_state(state_path)
        rows = _tail(events)
        previous = state.get('alerts', {})
        if not isinstance(previous, dict):
            previous = {}
        alerts = {}
        for session, conditions in previous.items():
            if _label(session) and isinstance(conditions, dict):
                recent = {c: ts for c, ts in conditions.items() if c in ('large_prompt', 'high_calls') and _number(ts) and now - 3600 < ts <= now}
                if recent:
                    alerts[session] = recent
        fresh = []
        for warning in summarize_burn(rows, now):
            session = warning['session_id']
            remembered = alerts.setdefault(session, {})
            conditions = [c for c in warning['conditions'] if c not in remembered]
            if conditions:
                for condition in conditions:
                    remembered[condition] = now
                fresh.append({'session_id': session, 'conditions': conditions,
                              'message': _message(session, conditions)})
        _save_state(state_path, {'alerts': alerts})
        if fresh:
            print(json.dumps({'warnings': fresh}, separators=(',', ':')))
        return fresh


def main(argv=None):
    parser = argparse.ArgumentParser(description='Private metadata-only efficiency watchdog; one tick per invocation.')
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--events', help='test-only telemetry path override')
    parser.add_argument('--state', help='test-only state path override')
    args = parser.parse_args(argv)
    if not args.watch:
        parser.print_help()
        return 0
    root = Path(os.environ.get('HERMES_HOME', '~/.hermes')).expanduser()
    directory = root / 'artifacts' / 'atlas-efficiency-20261109'
    try:
        watch_tick(args.events or directory / 'requests.jsonl', args.state or directory / 'watch-state.json')
    except (OSError, ValueError, RecursionError):
        # Never print raw exceptions, file contents, paths, or synthetic usage.
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
