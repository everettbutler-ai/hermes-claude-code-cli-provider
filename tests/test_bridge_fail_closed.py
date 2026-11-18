import pytest
import claude_code_server as s
T=[{'type':'function','function':{'name':'probe','parameters':{'type':'object'}}}]
@pytest.mark.parametrize('fn',[None,{'name':' bad '},{'name':'probe','parameters':[] }])
def test_bad_specs_fail_closed(fn):
 r=s.run_claude_mcp_bridge('', 'opus',[{'function':fn}],'auto');assert r['http_status']==400;assert not r['tool_calls']
@pytest.mark.parametrize('reason',['quota','authentication_error'])
def test_terminal_failure_discards_partial_call(monkeypatch,reason):
 monkeypatch.setattr(s,'_run_mcp_cli',lambda *a,**k:{'events':[{'type':'assistant','message':{'content':[{'type':'tool_use','id':'x','name':'mcp__hermes__probe','input':{}}]}},{'type':'result','is_error':True,'terminal_reason':reason,'result':reason,'modelUsage':{'claude-opus-5':{}}}]})
 monkeypatch.setattr(s,'_read_bridge_events',lambda *a,**k:[{'status':'accepted','name':'probe','arguments':{},'tool_use_id':'x'}])
 r=s.run_claude_mcp_bridge('', 'opus',T,'auto');assert r['error']==reason;assert not r['tool_calls']
def test_multiple_calls_preserve_order_and_arguments(monkeypatch):
 args=[{'text':'a\n"café"'}, {'text':'two'}]
 blocks=[{'type':'tool_use','id':str(i),'name':'mcp__hermes__probe','input':a} for i,a in enumerate(args)]
 monkeypatch.setattr(s,'_run_mcp_cli',lambda *a,**k:{'events':[{'type':'assistant','message':{'content':blocks}},{'type':'result','is_error':True,'terminal_reason':'max_turns','modelUsage':{'claude-opus-5':{}}}]})
 monkeypatch.setattr(s,'_read_bridge_events',lambda *a,**k:[{'status':'accepted','name':'probe','arguments':b['input'],'tool_use_id':b['id']} for b in blocks])
 import json
 r=s.run_claude_mcp_bridge('', 'opus',T,'auto');assert not r['error'];assert [json.loads(c['function']['arguments']) for c in r['tool_calls']]==args
