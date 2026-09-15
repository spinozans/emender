import json
import pytest
import tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import PiNativeEpisode,native_turn,normalize_context_message,validate_generated_turn

TOOLS=[dict(name='read',label='Read',description='read',parameters={'type':'object'}),
       dict(name='web_search',label='Web Search',description='search',parameters={'type':'object'})]
ENC=tiktoken.get_encoding('p50k_base')

def turn(name,args,analysis='private',commentary=None):
 return native_turn(dict(role='assistant',content=commentary,reasoning_content=analysis,think=None,
  tool_calls=[{'type':'function','function':{'name':name,'arguments':compact(args)}}]),TOOLS)


def test_external_call_result_and_finish_are_causal_and_exact():
 e=PiNativeEpisode(TOOLS,ENC);e.append_context({'role':'system','content':'system'});e.append_context({'role':'user','content':'question'})
 first=e.prompt();t=e.accept_generated_turn(turn('web_search',{'query':'weather now'}))
 assert t.semantic(TOOLS)['arguments']=={'query':'weather now'} and e.pending_tool=='web_search'
 with pytest.raises(ValueError,match='mismatched'):e.append_context({'role':'toolResult','toolCallId':'x','toolName':'read','content':[{'type':'text','text':'wrong'}],'isError':False})
 result={'role':'toolResult','toolCallId':'x','toolName':'web_search','content':[{'type':'text','text':'13 C'}],'isError':False}
 e.append_context(result);second=e.prompt();assert first!=second and compact(result) in second
 e.accept_generated_turn(turn('finish',{'message':'13 C'}));assert e.finished and e.pending_tool is None
 assert 'private' in e.text() and e.source_messages()[-1]['role']=='assistant'


def test_private_think_requires_fixed_internal_observation():
 e=PiNativeEpisode(TOOLS,ENC);e.append_context({'role':'user','content':'q'});e.accept_generated_turn(turn('think',{'thought':'secret'}))
 assert e.pending_tool=='think' and 'secret' not in repr(validate_generated_turn(turn('think',{'thought':'secret'}),TOOLS,ENC))
 e.resolve_think();assert e.pending_tool is None and 'Your thought has been logged.' in e.prompt()


def test_rejects_undeclared_actions_noncanonical_and_bad_results():
 with pytest.raises(ValueError,match='undeclared'):turn('bash',{'command':'pwd'})
 text=turn('read',{'path':'a'})
 with pytest.raises(ValueError,match='noncanonical'):validate_generated_turn(text.replace('{"path":"a"}','{"path": "a"}'),TOOLS,ENC)
 with pytest.raises(ValueError,match='tool_result'):normalize_context_message({'role':'toolResult','toolName':'read'})


def test_manifest_tool_surface_constructs_without_mutating_authority():
 manifest=json.load(open('configs/pi/e97-active-tool-surface-v1.json'))
 tools=manifest['model_visible_tools'];e=PiNativeEpisode(tools,ENC)
 assert len(tools)==11 and e.pending_tool is None
 assert [t['name'] for t in tools]==manifest['tool_order']
