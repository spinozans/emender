import hashlib
from scripts.build_e97_pi_native_curriculum import make_extracterror_cases,make_longcopy_cases
from scripts.build_e97_pi_native_curriculum import make_pointerchase_cases
import json
from pathlib import Path
import tiktoken
from scripts.build_e97_pi_native_curriculum import encode_candidate,make_cases,sha

def test_case_mix_and_identity_are_deterministic():
 a,counts=make_cases(40,18765);b,_=make_cases(40,18765)
 assert a==b and counts=={'local':14,'web':10,'exact':10,'recovery':6}
 assert len({x['id'] for x in a})==40
 assert all(not Path(p).is_absolute() and '..' not in Path(p).parts for x in a for p in x['files'])
 assert sum(x['repository_discovery'] for x in a)>=10
 assert all('127.0.0.1' not in x['prompt'] for x in a if x['category']=='web')
 assert all(not x.get('web_page') for x in a if x['category']=='web')
 process=[x for x in a if x['family']=='process-lifecycle']
 assert len(process)==1 and [s['name'] for s in process[0]['steps']]==['process','process','process','finish']
 assert process[0]['steps'][0]['arguments']['notify']['onSuccess']=='ignore'
 assert not process[0]['requires_error'] and process[0]['supervise_from']==0

def test_repository_discovery_has_eight_families_at_program_scale():
 cases,_=make_cases(2000,18765)
 families={x['family'] for x in cases if x['repository_discovery']}
 assert len(families)>=8 and sum(x['repository_discovery'] for x in cases)>=256
 assert 'repo-edit-error-recovery' not in families
 assert len({x['prompt'] for x in cases})==len(cases)

def test_case_generation_requires_multiple_of_twenty():
 for count in (0,19,21):
  try:make_cases(count,18765)
  except ValueError:pass
  else:raise AssertionError(count)

def test_failure_prefix_mask_excludes_first_assistant_turn():
 enc=tiktoken.get_encoding('p50k_base');one='one';two='two'
 text='System:\ns\n\nUser:\nu\n\nAssistant:\n'+one+'\n\nAssistant:\n'+two
 generations=[{'token_ids':enc.encode_ordinary(one)},{'token_ids':enc.encode_ordinary(two)}]
 ids,mask,units=encode_candidate(text,generations,1,enc)
 decoded=[enc.decode([token]) for token,keep in zip(ids,mask) if keep]
 assert units==1 and sum(mask)==len(enc.encode_ordinary(two)) and ''.join(decoded)==two

def test_generated_cases_do_not_authorize_training_or_updates(tmp_path):
 cases,counts=make_cases(20,18765)
 plan={'cases':cases,'mix_counts':counts,'training_eligible':False,'optimizer_updates':0,'packing_authorized':False}
 path=tmp_path/'plan.json';path.write_text(json.dumps(plan,sort_keys=True))
 loaded=json.loads(path.read_text())
 assert not loaded['training_eligible'] and not loaded['packing_authorized'] and loaded['optimizer_updates']==0
 assert sha(path)==hashlib.sha256(path.read_bytes()).hexdigest()

def test_pointerchase_family_shapes_and_invariants():
 cases,counts=make_pointerchase_cases(9)
 assert counts=={'pointerchase':9} and len({c['id'] for c in cases})==9
 for c in cases:
  assert c['family']=='tool-error-pointer-chase' and c['requires_error'] and c['supervise_from']==1
  first=c['steps'][0]
  assert first['name']=='read' and first['arguments']['path'].startswith('data/entry_')
  second=c['steps'][1]
  assert second['name'] in ('read','bash','ffgrep')
  pointer=[s for s in c['steps'] if s['name']=='read' and 'catalog_' in s['arguments'].get('path','') or 'registry_' in s['arguments'].get('path','') or 'index_' in s['arguments'].get('path','')]
  assert pointer, c['id']
  assert 'active_path' in json.dumps([f for f in c['files'].values()])
  assert c['steps'][-1]['name']=='finish'
  assert not c['repository_discovery'] and not c.get('training_eligible',False)
 names={c['steps'][1]['arguments'].get('path') or c['steps'][1]['name'] for c in cases}
 assert len({c['id'] for c in make_pointerchase_cases(3)[0]})==3

def test_extracterror_family_shapes_and_invariants():
 cases,counts=make_extracterror_cases(9)
 assert counts=={'extracterror':9}
 for c in cases:
  assert c['family']=='tool-error-finish-recovery' and c['requires_error'] and c['supervise_from']==2
  assert c['steps'][0]['name']=='read'
  assert c['steps'][1]['name']=='bash'
  assert c['steps'][-1]['name']=='finish'
  assert 'never finish with an error' in c['prompt']
  # the correct value must be observable in the read result (file contents include it)
  assert any(c['steps'][-1]['arguments']['message'] in v for v in c['files'].values())

def test_longcopy_family_shapes_and_invariants():
 cases,counts=make_longcopy_cases(10)
 assert counts=={'longcopy':10}
 tiers={c['delay_tokens'] for c in cases}
 assert tiers=={1024,4096,8192,16384,32768}
 for c in cases:
  assert c['family']=='long-delay-copy' and not c['requires_error'] and c['supervise_from']==0
  assert len(c['steps'])==1 and c['steps'][0]['name']=='finish'
  assert not c['files'] and not c['repository_discovery']
  assert c['steps'][0]['arguments']['message'].startswith('COPY_')
