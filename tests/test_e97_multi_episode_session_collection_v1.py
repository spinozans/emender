import json
import datetime
import tiktoken
import pytest
from scripts.e97_open_swe_native_codec import compact
from scripts.build_e97_multi_episode_session_collection_v1 import (
	make_all_cases,stitch_record,resolve_dynamic_session,case_kind,pilot_sample,
	DISTRIBUTION,lifecycle_session,derive_accumulate_session,edit_chain_session,
	chat_then_task_session,date_followup_session,DATE_RECALL_TEMPLATES)
from scripts.build_e97_hybrid_conversation_collection_v2 import parse_date_observation,check_text

# ----------------------------------------------------------- plan invariants
def test_plan_shape_and_distribution():
	cases,distribution=make_all_cases()
	assert distribution=={f:c for f,c,_ in DISTRIBUTION}
	assert sum(distribution.values())==1100
	assert len(cases)==1100 and len({c['id'] for c in cases})==len(cases)
	assert all(c['category']=='session' for c in cases)
	assert all(c['session_reset_policy']=='record-start-only' for c in cases)

def test_every_session_is_multi_episode_with_recall_or_date_carry():
	cases,_=make_all_cases()
	for c in cases:
		assert len(c['episodes'])>=3
		recalls=[ep for ep in c['episodes'] if ep.get('recall') is not None]
		date_carries=[ep for ep in c['episodes'] if any(s.get('dynamic')=='prior-observation' for s in ep['steps'])]
		assert recalls or date_carries, c['id']
		# every session crosses at least one finish->user boundary inside the record
		assert any(len(ep['steps'])>=1 for ep in c['episodes'])

def test_recall_episode_oracles():
	cases,_=make_all_cases()
	checked=0
	for c in cases:
		for ep in c['episodes']:
			r=ep.get('recall')
			if r is None:continue
			checked+=1
			assert ep['pure_chat'] and all(s['name']=='finish' for s in ep['steps'])
			prior=c['episodes'][r['episode']]
			assert not prior.get('pure_chat'), 'recall must reference an observed episode'
			assert r['episode']<c['episodes'].index(ep)
			# recall must reference an episode with a real tool observation (the
			# collect-time oracle then checks the value against the RECORDED results)
			assert any(s['name']!='finish' for s in prior['steps']), c['id']
			# and the authored finish must restate the recall value
			assert r['value'] in ep['steps'][0]['arguments']['message']
	assert checked>=1100, 'every session must carry at least one recall oracle'

def test_workspace_state_accumulates_across_episodes():
	cases,_=make_all_cases()
	for c in cases:
		seen=set()
		for ep in c['episodes']:
			assert set(ep.get('expected_files',{}))>=seen, 'episode oracle must be cumulative'
			seen|=set(ep.get('expected_files',{}))
	life=[c for c in cases if c['family']=='session-file-lifecycle'][0]
	assert any('out/note_' in p for ep in life['episodes'] for p in ep.get('expected_files',{}))

def test_panel_literal_guards_apply_to_session_authoring():
	with pytest.raises(ValueError):
		check_text('ledger_e1probe')
	cases,_=make_all_cases()
	for c in cases[:60]:
		for ep in c['episodes']:
			check_text(ep['prompt'])

def test_pilot_sample_covers_every_case_kind():
	cases,_=make_all_cases()
	kinds={case_kind(c) for c in cases}
	pilot=pilot_sample(cases,100)
	assert len({case_kind(c) for c in pilot})==len(kinds)
	assert len(pilot)<len(cases)

# --------------------------------------------------- record stitch semantics
def episode_text_stub(header,system,user,turns):
	return header+'\n\nSystem:\n'+system+'\n\nUser:\n'+user+''.join('\n\nAssistant:\n'+t for t in turns)

def test_stitch_record_states_header_and_system_once():
	header='Protocol:\nheader-payload'
	eps=[episode_text_stub(header,'be-system',f'prompt-{k}',[f'turn-{k}']) for k in range(3)]
	record=stitch_record(eps)
	assert record.count('Protocol:')==1
	assert record.count('System:')==1
	assert record.count('User:')==3
	assert record.count('Assistant:')==3
	assert record.endswith('turn-2')

def test_stitch_record_preserves_episode_order_and_boundaries():
	eps=[episode_text_stub('Protocol:\nh','s',f'u{k}',['a']) for k in range(2)]
	record=stitch_record(eps)
	assert record.index('u0')<record.index('a')<record.index('u1')<record.index('a',record.index('u1'))

def test_stitch_record_rejects_headerless_episode():
	with pytest.raises(ValueError):
		stitch_record(['no user marker here','also none'])

# ------------------------------------------------- prior-observation dynamics
OBS={'bare':'Fri Sep 25 09:15:42 UTC 2026','A':'Friday','A-ymd':'Friday 2026-09-25',
 'u-ymd':'2026-09-25','u-HM':'09:15','A-BdY':'Friday, September 25, 2026'}

def test_prior_observation_finish_resolves_from_recorded_observation():
	for fmt,obs in OBS.items():
		spec={'name':'finish','dynamic':'prior-observation','episode':0,'fmt':fmt,
		 'template':DATE_RECALL_TEMPLATES[fmt][0],'analysis_template':'The clock was observed earlier; it showed {summary}.','commentary':None}
		resolved=resolve_dynamic_session(spec,[[obs]])
		values=parse_date_observation(obs,fmt)
		assert resolved['name']=='finish'
		assert resolved['arguments']['message']==DATE_RECALL_TEMPLATES[fmt][0].format(**values)
		assert '{' not in resolved['analysis']

def test_prior_observation_finish_requires_earlier_observation():
	spec={'name':'finish','dynamic':'prior-observation','episode':0,'fmt':'A',
	 'template':'It said {weekday}.','analysis_template':'{summary}'}
	with pytest.raises(ValueError):
		resolve_dynamic_session(spec,[[]])

def test_date_followup_sessions_carry_the_observation_across_boundaries():
	cases=[date_followup_session(i) for i in range(12)]
	for c in cases:
		first,recall=c['episodes'][0],c['episodes'][2]
		assert first.get('date_fmt') is not None
		assert any(s.get('dynamic')=='observation-date' for s in first['steps'])
		assert recall['pure_chat']
		dyn=[s for s in recall['steps'] if s.get('dynamic')=='prior-observation']
		assert len(dyn)==1 and dyn[0]['episode']==0 and dyn[0]['fmt']==first['date_fmt']
		if len(c['episodes'])==4:
			assert c['episodes'][3].get('date_fmt') is not None

# ------------------------------------------------------- encode compatibility
def test_session_record_encodes_with_supervised_assistant_spans():
	import sys
	sys.path.insert(0,'.')
	from scripts.build_e97_pi_native_curriculum import encode_candidate
	from scripts.e97_pi_native_codec import PiNativeEpisode
	import json as _json
	enc=tiktoken.get_encoding('p50k_base')
	tools=[{'name':'bash','label':'bash','description':'run a shell command','parameters':{'type':'object','properties':{'command':{'type':'string'}},'required':['command'],'additionalProperties':False}}]
	episodes=[];generations=[]
	for k in range(2):
		ep=PiNativeEpisode(tools,enc)
		ep.append_context({'role':'system','content':'sys'})
		ep.append_context({'role':'user','content':f'prompt {k}'})
		turn='Analysis: '+compact('grounded analysis here')+'\nCommentary: '+compact('sure')+'\nThink: '+compact(None)+'\nAction: finish\nArguments: '+compact({'message':f'done {k}'})
		ep.accept_generated_turn(turn)
		episodes.append(ep.text());generations.append({'turn':k,'token_ids':enc.encode_ordinary(turn),'reason':'valid','prompt_sha256':'x'})
	record=stitch_record(episodes)
	ids,mask,units=encode_candidate(record,generations,0,enc)
	assert units==2 and sum(mask)>0 and mask[0]==0
	assert len(ids)<65536
