import unittest
from unittest import mock
import importlib.util
from pathlib import Path
_spec = importlib.util.spec_from_file_location('prelude_server', Path(__file__).resolve().parents[1] / 'claude_code_server.py')
assert _spec is not None and _spec.loader is not None
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)
ZERO = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

class StreamingPreludeTests(unittest.TestCase):
    def handler(self):
        h = object.__new__(S.Handler)
        for name in ('send_response','send_header','end_headers','_write_sse','_send_cli_failure'):
            setattr(h,name,mock.Mock())
        return h

    def test_quota_delta_then_error_is_not_success(self):
        for msg in ("You've hit your weekly limit", "You've hit your session limit" + ' x' * 100):
            with self.subTest(msg=msg):
                h=self.handler()
                events=[('delta',msg),('final',{'text':msg,'usage':ZERO,'error':msg})]
                with mock.patch.object(S,'stream_claude',return_value=iter(events)):
                    h._stream_live_response('claude-opus-5-5','p',False,{},True)
                h._send_cli_failure.assert_called_once_with(msg)
                h.send_response.assert_not_called()
                h._write_sse.assert_not_called()

    def test_healthy_long_response_opens_before_final(self):
        h=self.handler()
        def events(*args):
            yield 'delta', 'a' * 128
            h.send_response.assert_called_once_with(200)
            yield 'delta', 'b' * 200
            yield 'final', {'text':'a'*128+'b'*200,'usage':ZERO,'error':''}
        with mock.patch.object(S,'stream_claude',side_effect=events):
            h._stream_live_response('claude-opus-5-5','p',False,{},True)
        h._send_cli_failure.assert_not_called()
        chunks=[c.args[0] for c in h._write_sse.call_args_list]
        content=''.join(c['choices'][0]['delta'].get('content','') for c in chunks if isinstance(c,dict) and c.get('choices'))
        self.assertEqual(content,'a'*128+'b'*200)

    def test_short_success_is_preserved(self):
        h=self.handler()
        with mock.patch.object(S,'stream_claude',return_value=iter([('delta','ok'),('final',{'text':'ok','usage':ZERO,'error':''})])):
            h._stream_live_response('claude-opus-5-5','p',False,{},False)
        h.send_response.assert_called_once_with(200)
        h._send_cli_failure.assert_not_called()
