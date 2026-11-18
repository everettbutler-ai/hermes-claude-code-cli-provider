import copy
import json
import unittest
from unittest.mock import patch

import claude_code_server as server


# Synthetic CLI events: these are not retained provider output or performance data.
MODEL = 'claude-opus-5-5'
TOOL = {'type': 'function', 'function': {'name': 'terminal', 'parameters': {
    'type': 'object', 'properties': {'command': {'type': 'string'},
                                     'timeout': {'type': 'integer'}},
    'required': ['command'], 'additionalProperties': False}}}


def selection(arguments=None, cid='toolu_1'):
    return {'type': 'tool_use', 'id': cid, 'name': 'terminal',
            'input': {'command': 'pwd'} if arguments is None else arguments}


def cli_events(calls, *, reject=True):
    events = [{'type': 'assistant', 'message': {'content': copy.deepcopy(calls)}}]
    if reject:
        events.append({'type': 'user', 'message': {'content': [
            {'type': 'tool_result', 'tool_use_id': call['id'], 'is_error': True,
             'content': '<tool_use_error>Error: No such tool available: '
                        + call['name'] + '</tool_use_error>'} for call in calls]}})
    events.append({'type': 'result', 'is_error': True, 'terminal_reason': 'max_turns',
                   'result': '', 'permission_denials': [],
                   'modelUsage': {MODEL: {'inputTokens': 1}},
                   'usage': {'input_tokens': 2, 'output_tokens': 3}})
    return {'events': events, 'returncode': 1, 'stderr': '', 'timed_out': False}


class ResidualCaptureTests(unittest.TestCase):
    def invoke(self, data, choice: object = 'auto'):
        prompts = []

        def cli(prompt, model, config, allowed, overrides):
            prompts.append(prompt)
            return data

        with patch.object(server, '_run_mcp_cli', side_effect=cli):
            result = server.run_claude_mcp_bridge('synthetic prompt', 'opus',
                                                  [TOOL], choice)
        return result, prompts

    def test_valid_bare_call_uses_actual_capture_subprocess(self):
        result, _ = self.invoke(cli_events([selection()]))
        self.assertEqual(result['error'], '')
        self.assertEqual(result['model'], MODEL)
        self.assertEqual(len(result['tool_calls']), 1)
        self.assertEqual(result['tool_calls'][0]['function']['name'], 'terminal')
        self.assertEqual(json.loads(result['tool_calls'][0]['function']['arguments']),
                         {'command': 'pwd'})

    def test_string_timeout_preserves_schema_error_and_input(self):
        args = {'command': 'pwd', 'timeout': '60'}
        data = cli_events([selection(args)])
        before = copy.deepcopy(data)
        result, _ = self.invoke(data)
        self.assertEqual(data, before)
        self.assertEqual(result['tool_calls'], [])
        self.assertEqual(result['http_status'], 503)
        self.assertNotIn('bridge capture mismatch: call count', result['error'])
        self.assertIn('timeout', result['error'].lower())
        self.assertTrue(any(word in result['error'].lower()
                            for word in ('integer', 'type', 'schema')))

    def test_mixed_capture_rejects_entire_batch(self):
        calls = [selection(cid='toolu_1'),
                 selection({'command': 'pwd', 'timeout': '60'}, 'toolu_2')]
        result, _ = self.invoke(cli_events(calls))
        self.assertEqual(result['tool_calls'], [])
        self.assertIn('timeout', result['error'].lower())
        self.assertNotIn('call count', result['error'])

    def test_transport_mismatches_remain_blocked(self):
        call = selection()
        cases = [cli_events([call], reject=False),
                 cli_events([selection(cid='toolu_1'), selection(cid='toolu_1')]),
                 cli_events([selection({'command': 3})])]
        wrong_id = cli_events([call])
        wrong_id['events'][1]['message']['content'][0]['tool_use_id'] = 'other'
        cases.append(wrong_id)
        for data in cases:
            with self.subTest(data=data):
                result, _ = self.invoke(data)
                self.assertFalse(result['tool_calls'])
                self.assertTrue(result['error'])

    def test_history_names_only_change_in_native_mode(self):
        messages = [{'role': 'system', 'content': 'keep terminal literal'},
                    {'role': 'assistant', 'content': 'terminal', 'tool_calls': [
                        {'function': {'name': 'terminal', 'arguments': '{}'}},
                        {'function': {'name': 'mcp__hermes__terminal'}}]},
                    {'role': 'tool', 'content': 'terminal result'},
                    {'role': 'user', 'content': 'terminal request'}]
        plain = server.flatten_messages(messages)
        self.assertIn('[requested tools: terminal, mcp__hermes__terminal]', plain)
        self.assertEqual(plain, server.flatten_messages(messages, native_tools=False))
        native = server.flatten_messages(messages, native_tools=True)
        self.assertIn('[requested tools: mcp__hermes__terminal, mcp__hermes__terminal]', native)
        for text in ('keep terminal literal', 'ASSISTANT: terminal',
                     'TOOL RESULT: terminal result', 'USER: terminal request'):
            self.assertIn(text, plain)
            self.assertIn(text, native)

    def test_forced_specific_hint_names_callable(self):
        result, prompts = self.invoke(cli_events([selection()]),
                                      {'type': 'function', 'function': {'name': 'terminal'}})
        self.assertEqual(result['error'], '')
        self.assertIn("available Hermes tool 'mcp__hermes__terminal'", prompts[0])
        self.assertNotIn("available Hermes tool 'terminal'", prompts[0])


if __name__ == '__main__':
    unittest.main()
