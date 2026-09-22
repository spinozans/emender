import json
import datetime
import tiktoken
import pytest
from scripts.e97_open_swe_native_codec import compact
from scripts.build_e97_hybrid_conversation_collection_v2 import (
	edit_case,write_case,resolve_dynamic_v2,parse_date_observation,make_all_cases,
	utc_ymd_span,check_clock,case_kind,pilot_sample)
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

# ------------------------------------------- u-ymd clock-check regression
# The stride-100 pilot never sampled a u-ymd case, so a span seeded with a
# weekday string (not a date) falsely rejected every u-ymd case at collect
# time. These tests exercise a u-ymd case explicitly and pin the pilot rule:
# a pilot sample must cover every case kind.
def test_utc_ymd_span_is_date_strings():
	now=datetime.datetime(2026,9,22,23,59,tzinfo=datetime.timezone.utc)
	assert utc_ymd_span(now)=={'2026-09-21','2026-09-22','2026-09-23'}
	assert not any(any(ch.isalpha() for ch in d) for d in utc_ymd_span(now))
	live=datetime.datetime.now(datetime.timezone.utc)
	assert live.strftime('%Y-%m-%d') in utc_ymd_span(live)

def u_ymd_case_turns(case,observation):
	"""Render a u-ymd case's emitted frames exactly as execute_case_v2 does at
	collect time: the static bash step as authored, the dynamic finish resolved
	against the real observation."""
	turns=[]
	for spec in case['steps']:
		resolved=resolve_dynamic_v2(spec,observation) if 'dynamic' in spec else spec
		turns.append(frame(spec['name'],resolved['arguments'],resolved.get('analysis'),resolved.get('commentary')))
	return turns

def verify_u_ymd_case_end_to_end(case,observation,enc):
	"""Drive one u-ymd case through the collect-time clock check and the
	independent auditor verification path: verify_case re-derives the
	date-observation final from the plan spec, verify_hybrid_v2_case re-derives
	every emitted frame."""
	now=datetime.datetime.strptime(observation,'%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
	check_clock('u-ymd',parse_date_observation(observation,'u-ymd'),now.strftime('%A'),now.strftime('%A'))
	turns=u_ymd_case_turns(case,observation)
	private={'source_messages':[assistant('bash',frame_arguments(turns[0])),result(observation),
		{'role':'assistant','content':None,'reasoning_content':None,'think':None,
		 'tool_calls':[{'type':'function','function':{'name':'finish','arguments':compact(frame_arguments(turns[-1]))}}]}],
		'snapshot':{}}
	tools=[{'name':'bash','label':'b','description':'b','parameters':{}}]
	verify_results,actions=verify_case(case,private,tools)
	verify_hybrid_v2_case(case,turns,verify_results,enc)
	return actions[-1]

def test_u_ymd_case_clock_check_and_finish():
	cases,_=make_all_cases()
	uymd=[c for c in cases if c['family']=='hybrid-date-question' and c.get('date_fmt')=='u-ymd']
	assert len(uymd)==125
	dated=[c for c in uymd if c['steps'][-1]['dynamic']=='observation-date']
	exact=[c for c in uymd if c['steps'][-1]['dynamic']=='observation-exact']
	assert (len(dated),len(exact))==(107,18)  # the exact-observation slice keeps its own finish kind
	for case in (dated[0],exact[0]):
		bash=[s for s in case['steps'] if s.get('name')=='bash'][0]
		assert bash['arguments']=={'command':"date -u '+%Y-%m-%d'"}
	now=datetime.datetime.now(datetime.timezone.utc);observation=now.strftime('%Y-%m-%d')
	values=parse_date_observation(observation,'u-ymd')
	assert values=={'ymd':observation}
	# the fixed clock check accepts the live UTC date (the pre-fix span was
	# seeded with a weekday string and rejected every u-ymd observation) and
	# rejects a stale date
	check_clock('u-ymd',values,now.strftime('%A'),now.strftime('%A'))
	stale=parse_date_observation((now-datetime.timedelta(days=3)).strftime('%Y-%m-%d'),'u-ymd')
	with pytest.raises(ValueError,match='clock date mismatch'):
		check_clock('u-ymd',stale,now.strftime('%A'),now.strftime('%A'))
	enc=tiktoken.get_encoding('p50k_base')
	# end-to-end through the verification path, one case per rejected group:
	# date-question observation-date (107), date-question observation-exact
	# (18), and chat-into-tool-date seam (60) — all 185 falsely rejected at
	# full-collect time by the pre-fix span.
	action=verify_u_ymd_case_end_to_end(dated[0],observation,enc)
	assert action['arguments']=={'message':dated[0]['steps'][-1]['template'].format(**values)}
	action=verify_u_ymd_case_end_to_end(exact[0],observation,enc)
	assert action['arguments']=={'message':observation}
	seam=[c for c in cases if c['family']=='hybrid-chat-into-tool-date' and c.get('date_fmt')=='u-ymd']
	assert len(seam)==60
	action=verify_u_ymd_case_end_to_end(seam[0],observation,enc)
	assert action['arguments']=={'message':seam[0]['steps'][-1]['template'].format(**values)}

def test_pilot_sample_covers_every_case_kind():
	cases,_=make_all_cases()
	kinds={case_kind(c) for c in cases}
	assert len(kinds)==31
	stride_only={case_kind(c) for c in cases[::100]}
	assert len(stride_only)<len(kinds)  # the sampling gap that hid the u-ymd bug
	sample=pilot_sample(cases,100)
	assert {case_kind(c) for c in sample}==kinds
	assert pilot_sample(cases,1)==cases
