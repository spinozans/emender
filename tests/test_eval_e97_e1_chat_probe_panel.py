import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import tiktoken
from scripts.eval_e97_e1_chat_probe_panel import (check_greeting,extract_frame,semantic_frame,
 canonical_frame,decode_emission,ProbeEpisode,run_case,PANEL_SYSTEM,EOT_ID)
from scripts.e97_pi_native_codec import tool_map,compact

ENC=tiktoken.get_encoding('p50k_base')
TOOLS=json.loads(Path('configs/pi/e97-active-tool-surface-v1.json').read_text())['model_visible_tools']

def frame(name,args,analysis=None,commentary=None):
 return '\n'.join(('Analysis: '+compact(analysis),'Commentary: '+compact(commentary),
  'Think: '+compact(None),'Action: '+name,'Arguments: '+compact(args)))

PANEL=json.loads(Path('configs/pi/e97-e1-chat-probe-panel-v1.json').read_text())

def case(kind):
 return next(c for c in PANEL['cases'] if c['kind']==kind)

def test_extract_frame_and_semantics():
 f=frame('finish',{'message':'Hello! Doing well, thanks for asking.'})
 text=f+'\n\nUser:\n'+compact({'role':'user','content':'next'})
 assert extract_frame(text)==f
 assert extract_frame('Hello! Doing well.')is None
 m=semantic_frame(f,TOOLS)
 assert m['name']=='finish' and m['arguments']['message'].startswith('Hello')
 assert canonical_frame(f,TOOLS,ENC)

def test_greeting_verdicts():
 ok_finish=frame('finish',{'message':'Hi! I am doing well, thanks.'})
 assert check_greeting(ok_finish,TOOLS)==(True,'finish-frame')
 tool_call=frame('bash',{'command':'date'})
 passed,path=check_greeting(tool_call,TOOLS)
 assert not passed and path.startswith('tool-call-frame:bash')
 assert check_greeting('Hi! Doing well, thanks for asking.',TOOLS)==(True,'text-only')
 assert check_greeting('Hi!\nAction: read\npath: x',TOOLS)==(False,'text-only-with-action-line')
 assert check_greeting('   ',TOOLS)==(False,'empty-response')
 empty_finish=frame('finish',{'message':'   '})
 assert check_greeting(empty_finish,TOOLS)==(False,'invalid-frame:invalid_finish')

def test_decode_emission_stops_at_rs_and_eot():
 rs=ENC.encode('\x1e')[0]
 text=decode_emission(ENC.encode('He')+[rs]+ENC.encode('ll'),ENC)
 assert text=='He'

def test_probe_episode_prompt_matches_codec_byte_for_byte():
 from scripts.e97_pi_native_codec import PiNativeEpisode
 pe=ProbeEpisode(TOOLS,ENC,PANEL_SYSTEM)
 canonical=PiNativeEpisode(TOOLS,ENC,65536)
 canonical.append_context({'role':'system','content':PANEL_SYSTEM})
 prompt=pe.open(case('date-tool')['user_message'])
 canonical.append_context({'role':'user','content':case('date-tool')['user_message']})
 assert prompt==canonical.prompt()
 f=frame('bash',{'command':'date'})
 assert pe.accept(f)
 canonical.accept_generated_turn(f)
 prompt2=pe.feed_tool_result('bash','Sat Sep 20 12:00:00 UTC 2026')
 canonical.append_context({'role':'toolResult','toolCallId':'probe-bash','toolName':'bash',
  'content':[{'type':'text','text':'Sat Sep 20 12:00:00 UTC 2026'}],'isError':False})
 assert prompt2==canonical.prompt()

def test_date_case_pass_and_fail_paths():
 import datetime
 fixtures={'pass':None}
 def make_sampler(script):
  def sampler(cid,prompt,turn):
   return script[turn-1]
  return sampler
 good=[frame('bash',{'command':'date'}),frame('finish',{'message':'Today is '+datetime.datetime.now().strftime('%A')+'.'})]
 r=run_case(case('date-tool'),TOOLS,ENC,make_sampler(good),4)
 assert r['pass'] and r['path']=='bash-date-then-finish-weekday'
 today=datetime.datetime.now().strftime('%A')
 wrong_day='Monday' if today!='Monday' else 'Tuesday'
 wrong=[frame('bash',{'command':'date'}),frame('finish',{'message':'It is '+wrong_day+'.'})]
 r=run_case(case('date-tool'),TOOLS,ENC,make_sampler(wrong),4)
 assert not r['pass'] and r['reason'].startswith('finish-missing-weekday')
 noweekday=[frame('finish',{'message':'Today is a nice day.'})]
 r=run_case(case('date-tool'),TOOLS,ENC,make_sampler(noweekday),4)
 assert not r['pass'] and r['reason']=='finish-without-bash-date'
 ls=[frame('bash',{'command':'ls'}),frame('finish',{'message':'x'})]
 r=run_case(case('date-tool'),TOOLS,ENC,make_sampler(ls),4)
 assert not r['pass'] and r['reason'].startswith('bash-without-date')

def test_two_tool_case_pass_and_fail_paths():
 def make_sampler(script):
  return lambda cid,prompt,turn: script[turn-1]
 good=[frame('read',{'path':'data/ledger_e1probe.txt'}),
  frame('bash',{'command':"printf '%s\\n' $((157+284))"}),
  frame('finish',{'message':'441'})]
 r=run_case(case('two-tool-mini-task'),TOOLS,ENC,make_sampler(good),6)
 assert r['pass'] and r['path']=='read-then-bash-then-finish-sum'
 skip_read=[frame('bash',{'command':'echo $((157+284))'}),frame('finish',{'message':'441'})]
 r=run_case(case('two-tool-mini-task'),TOOLS,ENC,make_sampler(skip_read),6)
 assert not r['pass'] and r['reason']=='finish-without-read'
 wrong_sum=[frame('read',{'path':'data/ledger_e1probe.txt'}),
  frame('bash',{'command':'echo $((157+284))'}),frame('finish',{'message':'442'})]
 r=run_case(case('two-tool-mini-task'),TOOLS,ENC,make_sampler(wrong_sum),6)
 assert not r['pass'] and r['reason'].startswith('finish-missing-sum')

def test_greeting_case_runner():
 r=run_case(case('greeting'),TOOLS,ENC,lambda cid,prompt,turn:frame('finish',{'message':'Hi! Doing well.'}),4)
 assert r['pass'] and r['path']=='finish-frame'
