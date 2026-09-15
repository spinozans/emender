import json,shutil
from pathlib import Path
import pytest,tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import native_turn
from scripts.e97_pi_native_tool_bridge import MODEL,PROVIDER,NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
ENC=tiktoken.get_encoding('p50k_base')

def manifest_tools(*names):
 m=json.load(open('configs/pi/e97-active-tool-surface-v1.json'));return [t for t in m['model_visible_tools'] if t['name'] in names]
def read_tool():return manifest_tools('read')[0]
def frame(tools,name,args):
 return native_turn({'role':'assistant','content':None,'reasoning_content':'private','think':None,'tool_calls':[{'type':'function','function':{'name':name,'arguments':compact(args)}}]},tools)
def request(b,messages):return {'systemPrompt':b.panel['system'],'messages':messages,'tools':b.tools,'model':MODEL,'provider':PROVIDER}

def test_bridge_ingests_exact_pi_result_before_next_generation():
 tools=[read_tool()];turns=iter([frame(tools,'read',{'path':'sample.txt'}),frame(tools,'finish',{'message':'done'})]);prompts=[]
 def generate(prompt,budget,deadline):prompts.append(prompt);text=next(turns);return text,ENC.encode_ordinary(text),'valid'
 panel={'system':'system','tools':tools,'max_turns':4,'generation_budget':2048,'episode_generation_budget':4096,'episode_seconds':30}
 b=NativePiToolBridge(panel,'read it',ENC,generate);first=b.next(request(b,b.history));call=first['message']['content'][-1]
 result={'role':'toolResult','toolCallId':call['id'],'toolName':'read','content':[{'type':'text','text':'1: opaque'}],'isError':False}
 second=b.next(request(b,b.history+[result]));assert second['stop_reason']=='stop' and b.final=='done' and '1: opaque' in prompts[1]
 assert b.close(b.history)['verified'] and b.closed and len(b.generations)==2

def test_bridge_rejects_mismatched_result_identity():
 tools=[read_tool()];text=frame(tools,'read',{'path':'a'})
 b=NativePiToolBridge({'system':'s','tools':tools,'max_turns':2,'generation_budget':1000,'episode_generation_budget':2000,'episode_seconds':30},'q',ENC,lambda p,b,d:(text,ENC.encode_ordinary(text),'valid'))
 call=b.next(request(b,b.history))['message']['content'][-1]
 bad={'role':'toolResult','toolCallId':call['id'],'toolName':'bash','content':[{'type':'text','text':'x'}],'isError':False}
 with pytest.raises(ValueError,match='mismatch'):b.next(request(b,b.history+[bad]))

def test_real_pi_retains_model_failure_without_retry(tmp_path):
 pi=shutil.which('pi')
 if not pi:return
 tools=[read_tool()];calls=0
 def generate(prompt,budget,deadline):
  nonlocal calls
  calls+=1;return None,[1],'invalid_opening'
 panel={'system':'Pi native failure control','tools':tools,'max_turns':2,'generation_budget':2048,'episode_generation_budget':4096,'episode_seconds':30}
 b=NativePiToolBridge(panel,'Fail once.',ENC,generate);cwd=tmp_path/'failure-cwd';cwd.mkdir()
 terminal=serve_pi_native_tools(b,tmp_path/'failure-pi',pi_bin=pi,provider_extension=Path('configs/pi/e97-pi-native.ts'),cwd=cwd,seconds=60,allow_model_failure=True)
 assert calls==1 and terminal['model_failure_verified'] and not terminal['close_verified']
 assert b.failed and b.closed and b.reason=='invalid_opening' and len(b.generations)==1


def test_real_pi_stage_a_exact_schema_safe_read(tmp_path):
 pi=shutil.which('pi')
 if not pi:return
 tools=manifest_tools('read','bash','edit','write','process');cwd=tmp_path/'safe-cwd';cwd.mkdir();(cwd/'sample.txt').write_text('safe-value\n')
 turns=iter([frame(tools,'read',{'path':'sample.txt'}),frame(tools,'finish',{'message':'done'})])
 def generate(prompt,budget,deadline):text=next(turns);return text,ENC.encode_ordinary(text),'valid'
 panel={'system':'safe stage A','tools':tools,'max_turns':4,'generation_budget':2048,'episode_generation_budget':4096,'episode_seconds':30};b=NativePiToolBridge(panel,'read safely',ENC,generate)
 terminal=serve_pi_native_tools(b,tmp_path/'safe-pi',pi_bin=pi,provider_extension=Path('configs/pi/e97-pi-native.ts'),pi_extensions=[Path('configs/pi/e97-pi-native-stage-a-tools.ts')],cwd=cwd,seconds=60,no_builtin_tools=True,extra_env={'E97_PI_TOOL_MANIFEST':str(Path('configs/pi/e97-active-tool-surface-v1.json').resolve())})
 assert terminal['close_verified'] and 'safe-value' in b.episode.text()


def test_real_pi_executes_builtin_read_and_returns_exact_result(tmp_path):
 pi=shutil.which('pi')
 if not pi:return
 cwd=tmp_path/'cwd';cwd.mkdir();(cwd/'sample.txt').write_text('opaque-value\n')
 tools=[read_tool()];turns=iter([frame(tools,'read',{'path':'sample.txt'}),frame(tools,'finish',{'message':'done'})])
 def generate(prompt,budget,deadline):text=next(turns);return text,ENC.encode_ordinary(text),'valid'
 panel={'system':'Pi native control','tools':tools,'max_turns':4,'generation_budget':2048,'episode_generation_budget':4096,'episode_seconds':30}
 b=NativePiToolBridge(panel,'Read sample.txt and finish.',ENC,generate)
 terminal=serve_pi_native_tools(b,tmp_path/'pi',pi_bin=pi,provider_extension=Path('configs/pi/e97-pi-native.ts'),cwd=cwd,seconds=60)
 assert terminal['close_verified'] and [r['op'] for r in terminal['requests']]==['next','next','close']
 assert b.final=='done' and 'opaque-value' in b.episode.text() and len(b.generations)==2
