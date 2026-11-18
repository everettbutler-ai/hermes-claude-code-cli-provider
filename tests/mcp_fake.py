import json
from tests.test_claude_code_server import make_fake_claude

def make(directory, value, usage):
    source = '''#!/usr/bin/env python3
import json,subprocess,sys
sys.stdin.read()
c=json.load(open(sys.argv[sys.argv.index('--mcp-config')+1]))['mcpServers']['hermes']
p=subprocess.Popen([c['command'],*c['args']],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
a={'name':VALUE}
p.stdin.write(json.dumps({'jsonrpc':'2.0','id':1,'method':'tools/call','params':{'name':'skill_view','arguments':a}})+'\\n');p.stdin.flush()
assert not json.loads(p.stdout.readline())['result']['isError']
p.terminate();p.wait()
print(json.dumps({'type':'assistant','message':{'content':[{'type':'tool_use','id':'call_test','name':'mcp__hermes__skill_view','input':a}]}}))
print(json.dumps({'type':'result','usage':USAGE,'modelUsage':{'claude-haiku-4-5':{}},'is_error':True,'terminal_reason':'max_turns'}))
'''.replace('VALUE',repr(value)).replace('USAGE',repr(usage))
    return make_fake_claude(directory,source)
