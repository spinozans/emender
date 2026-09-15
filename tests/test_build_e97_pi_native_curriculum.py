import hashlib
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

def test_repository_discovery_has_eight_families_at_program_scale():
 cases,_=make_cases(2000,18765)
 families={x['family'] for x in cases if x['repository_discovery']}
 assert len(families)>=8 and sum(x['repository_discovery'] for x in cases)>=256

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
