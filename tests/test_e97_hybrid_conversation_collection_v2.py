import json
import tiktoken
import pytest
from scripts.e97_open_swe_native_codec import compact
from scripts.build_e97_hybrid_conversation_collection_v2 import (
	edit_case,write_case,resolve_dynamic_v2,parse_date_observation,make_all_cases)
from scripts.audit_e97_pi_native_curriculum import (
	audit_date_values,verify_case,verify_hybrid_v2_case)

AFFECTED_FAMILIES={'hybrid-write-verify','hybrid-edit-verify','hybrid-chat-into-tool-write','hybrid-chat-into-tool-edit'}

def frame(name,arguments,analysis=None,commentary=None):
	return '\n'.join(('Analysis: '+compact(analysis),'Commentary: '+compact(commentary),
		'Think: '+compact(None),'Action: '+name,'Arguments: '+compact(arguments)))

def frame_arguments(text):
	return json.loads(text.split('\n')[4][len('Arguments: '):])

def result(text):
	return {'role':'toolResult','toolCallId':'c','toolName':'bash','content':[{'type':'text','text':text}],'isError':False}

def assistant(name,args):
	return {'role':'assistant','content':None,'reasoning_content':None,'think':None,
		'tool_calls':[{'type':'function','function':{'name':name,'arguments':compact(args)}}]}

# --------------------------------------------------------- builder oracle fix
def test_edit_case_oracle_expects_post_edit_file():
	for seam in (False,True):
		case=edit_case(70,seam=seam)
		(path,fixture),=case['files'].items()
		assert fixture.startswith('mode = alpha')
		assert case['expected_files'][path].startswith('mode = beta')
		assert case['expected_files'][path].endswith(fixture.split('zone = ')[1])

def test_write_case_oracle_expects_created_file():
	for seam in (False,True):
		case=write_case(90,seam=seam)
		assert case['files']=={}
		write=case['steps'][0]['arguments']
		assert case['expected_files']=={write['path']:write['content']}

def test_make_all_cases_write_and_edit_oracles():
	cases,_=make_all_cases()
	assert len(cases)==5200 and len({c['id'] for c in cases})==len(cases)
	affected=[c for c in cases if c['family'] in AFFECTED_FAMILIES]
	assert len(affected)==300
	for case in affected:
		if case['family'] in ('hybrid-write-verify','hybrid-chat-into-tool-write'):
			write=case['steps'][0]['arguments']
			assert case['expected_files']=={write['path']:write['content']}
		else:
			edit=case['steps'][0]['arguments']
			assert edit['edits']==[{'oldText':'mode = alpha','newText':'mode = beta'}]
			assert all(v.startswith('mode = beta') and v.endswith('\n') for v in case['expected_files'].values())

# ---------------------------------------------- auditor dynamic reconstruction
OBSERVATION='Fri Sep 25 09:15:42 UTC 2026'
def date_case():
	return {'id':'v2-date','category':'hybrid','family':'hybrid-date-question','prompt':'What day is it?',
		'files':{},'expected_files':{},'supervise_from':0,'requires_error':False,'pure_chat':False,
		'date_fmt':'bare','answer_must_be_observed':False,
		'steps':[
			{'name':'bash','arguments':{'command':'date'},'analysis':'The clock is environment state.','analysis_requires':[],'commentary':'Checking the clock.'},
			{'name':'finish','dynamic':'observation-date','fmt':'bare',
			 'template':"It's {weekday} — today is {date}.",
			 'analysis_template':'The observation shows {weekday}, {date}.',
			 'commentary':"Got it from the clock."}]}

def resolved_date_turns():
	resolved=resolve_dynamic_v2(date_case()['steps'][1],OBSERVATION)
	assert resolved['arguments']=={'message':"It's Friday — today is September 25, 2026."}
	return [frame('bash',{'command':'date'},'The clock is environment state.','Checking the clock.'),
		frame('finish',resolved['arguments'],resolved['analysis'],resolved['commentary'])]

def test_audit_date_values_matches_builder_parse():
	assert audit_date_values(OBSERVATION,'bare')==parse_date_observation(OBSERVATION,'bare')
	with pytest.raises(ValueError,match='unparsed'):audit_date_values('nonsense','bare')

def test_verify_case_grounds_observation_date_final():
	case=date_case()
	turns=resolved_date_turns()
	message=dict(role='assistant',content=None,reasoning_content=None,think=None,
		tool_calls=[{'type':'function','function':{'name':'finish','arguments':compact(frame_arguments(turns[1]))}}])
	private={'source_messages':[assistant('bash',{'command':'date'}),result(OBSERVATION),message],'snapshot':{}}
	tools=[{'name':'bash','label':'b','description':'b','parameters':{}}]
	results,actions=verify_case(case,private,tools)
	assert actions[-1]['arguments']=={'message':"It's Friday — today is September 25, 2026."}
	verify_hybrid_v2_case(case,turns,results,tiktoken.get_encoding('p50k_base'))

def test_verify_hybrid_v2_case_rejects_ungrounded_date_final():
	enc=tiktoken.get_encoding('p50k_base');case=date_case()
	turns=resolved_date_turns()
	bad=[turns[0],frame('finish',{'message':"It's Monday — today is September 21, 2026."},'The observation shows Monday.','Got it from the clock.')]
	with pytest.raises(ValueError,match='v2 date finish'):verify_hybrid_v2_case(case,bad,[result(OBSERVATION)],enc)

def test_verify_hybrid_v2_case_rejects_off_spec_commentary():
	enc=tiktoken.get_encoding('p50k_base');case=date_case()
	turns=resolved_date_turns()
	spec=case['steps'][1];values=audit_date_values(OBSERVATION,'bare')
	bad=[turns[0],frame('finish',frame_arguments(turns[1]),spec['analysis_template'].format(**values),'Off-spec commentary.')]
	with pytest.raises(ValueError,match='v2 date finish commentary'):verify_hybrid_v2_case(case,bad,[result(OBSERVATION)],enc)

def test_verify_hybrid_v2_case_accepts_observation_exact_finish():
	enc=tiktoken.get_encoding('p50k_base')
	case={'id':'v2-exact','category':'hybrid','family':'hybrid-date-question','steps':[
		{'name':'bash','arguments':{'command':"date '+%A'"},'analysis':'Clock state.','analysis_requires':[],'commentary':'Checking.'},
		{'name':'finish','dynamic':'observation-exact','analysis_dynamic':'observation','commentary_dynamic':'observation'}]}
	resolved=resolve_dynamic_v2(case['steps'][1],'Friday')
	assert resolved=={'name':'finish','arguments':{'message':'Friday'},
		'analysis':'The tool observed Friday; the user-facing answer is exactly that observed value.',
		'commentary':'The observed answer is Friday.'}
	turns=[frame('bash',{'command':"date '+%A'"},'Clock state.','Checking.'),
		frame('finish',{'message':'Friday'},resolved['analysis'],resolved['commentary'])]
	verify_hybrid_v2_case(case,turns,[result('Friday')],enc)
	bad=[turns[0],frame('finish',{'message':'Saturday'},resolved['analysis'],resolved['commentary'])]
	with pytest.raises(ValueError,match='v2 exact finish'):verify_hybrid_v2_case(case,bad,[result('Friday')],enc)
