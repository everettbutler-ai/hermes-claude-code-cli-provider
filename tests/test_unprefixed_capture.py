import copy
import json
import unittest
from unittest.mock import patch

import claude_code_server as server


MODEL = 'claude-opus-5-5'
TOOL = {'type': 'function', 'function': {'name': 'terminal', 'parameters': {'type': 'object', 'properties': {'command': {'type': 'string'}}, 'required': ['command'], 'additionalProperties': False}}}


def fixture(name='terminal', arguments=None, cid='toolu_1', rejection=True, billed=True, denial=False, terminal_error=False, timeout=False, prefix=False, duplicates=False):
    arguments = {'command': 'pwd'} if arguments is None else arguments
    call = {'type': 'tool_use', 'id': cid, 'name': 'mcp__hermes__' + name if prefix else name, 'input': arguments}
    calls = [call, copy.deepcopy(call)] if duplicates else [call]
    events = [{'type': 'assistant', 'message': {'content': calls}}]
    if rejection:
        events.append({'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': cid, 'is_error': True, 'content': f'<tool_use_error>Error: No such tool available: {name}</tool_use_error>'}]}})
    events.append({'type': 'result', 'is_error': True, 'terminal_reason': 'authentication_error' if terminal_error else 'max_turns', 'result': 'authentication_error: quota exceeded' if terminal_error else '', 'permission_denials': ['denied'] if denial else [], 'modelUsage': {MODEL: {'inputTokens': 1}} if billed else {}, 'usage': {'input_tokens': 2, 'output_tokens': 3}})
    return {'events': events, 'returncode': 1, 'stderr': '', 'timed_out': timeout}


class UnprefixedCaptureTests(unittest.TestCase):
    def run_fixture(self, data, *, ordinary_capture=False):
        def cli(prompt, model, config, allowed, overrides):
            if ordinary_capture:
                import subprocess
                import sys
                from pathlib import Path
                spec = Path(config).with_name('tools.json')
                output = Path(config).with_name('events.jsonl')
                call = data['events'][0]['message']['content'][0]
                request = {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call', 'params': {'name': 'terminal', 'arguments': call['input'], '_meta': {'claudecode/toolUseId': call['id']}}}
                bridge = Path(server.__file__).with_name('hermes_mcp_bridge.py')
                completed = subprocess.run([sys.executable, str(bridge), str(spec), str(output)], input=json.dumps(request) + '\n', text=True, capture_output=True, timeout=10)
                self.assertEqual(completed.returncode, 0, completed.stderr)
            return data
        with patch.object(server, '_run_mcp_cli', side_effect=cli):
            return server.run_claude_mcp_bridge('test', 'opus', [TOOL], 'auto')

    def test_bare_capture_uses_real_bridge(self):
        outcome = self.run_fixture(fixture())
        self.assertEqual(outcome['error'], '')
        completion = server.build_tool_call_completion(outcome['model'], outcome['tool_calls'], outcome['usage'])
        call = completion['choices'][0]['message']['tool_calls'][0]
        self.assertEqual((call['id'], call['function']['name'], json.loads(call['function']['arguments'])), ('toolu_1', 'terminal', {'command': 'pwd'}))
        self.assertEqual(completion['model'], MODEL)

    def test_prefixed_ordinary_capture(self):
        outcome = self.run_fixture(fixture(prefix=True, rejection=False), ordinary_capture=True)
        self.assertEqual(outcome['error'], '')
        self.assertEqual(outcome['tool_calls'][0]['function']['name'], 'terminal')

    def test_nonzero_exit_without_max_turns_fails_closed(self):
        data = fixture()
        data['events'][-1]['is_error'] = False
        data['events'][-1]['terminal_reason'] = 'completed'
        outcome = self.run_fixture(data)
        self.assertTrue(outcome['error'])
        self.assertFalse(outcome['tool_calls'])

    def test_fail_closed(self):
        cases = [fixture(name='unknown'), fixture(rejection=False), fixture(arguments={'command': 3}), fixture(duplicates=True), fixture(terminal_error=True), fixture(timeout=True), fixture(denial=True), fixture(billed=False), fixture(prefix=True)]
        for case in cases:
            with self.subTest(case=case):
                outcome = self.run_fixture(case)
                self.assertFalse(outcome['tool_calls'])
                self.assertTrue(outcome['error'])


if __name__ == '__main__':
    unittest.main()
