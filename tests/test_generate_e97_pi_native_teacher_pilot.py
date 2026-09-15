import json
from pathlib import Path
import pytest
from scripts.generate_e97_pi_native_teacher_pilot import render,validate

def config():return json.load(open('configs/pi/e97-pi-native-teacher-pilot-v1.json'))
def pool():
 c=config();tasks=[]
 for lane in c['lanes']:
  for i in range(20):
   tasks.append({'id':f"teacher-{lane['name']}-{i:03d}",'lane':lane['name'],'family':lane['families'][i%len(lane['families'])],'user_goal':f"Complete synthetic {lane['name']} task {i} with value Z{i}.",'workspace_files':{f'data/{lane["name"]}-{i}.txt':f'Z{i}\n'},'expected_first_action':'read','required_actions':['read','finish'],'oracle':f'Finish with Z{i} after observing the file.','failure_prefix':False,'diversity_tags':['family-'+str(i%3),'depth-'+str(i%4),'shape-'+str(i%5)]})
 return {'schema':'emender-e97-pi-native-teacher-task-pool-v1','tasks':tasks}

def test_render_is_bounded_and_requests_one_parallel_subagent_workflow():
 text=render(config());assert 'exactly 80' in text and 'subagent tool exactly once' in text and 'four parallel' in text

def test_validate_accepts_exact_stratified_pool(tmp_path):
 raw=tmp_path/'raw.json';raw.write_text(json.dumps(pool()));result,counts=validate(config(),raw);assert len(result['tasks'])==80;assert set(counts)=={'frame-copy','local-repository','recovery-composition','web-routing'}

def test_validate_does_not_treat_secretless_as_a_secret(tmp_path):
 data=pool();data['tasks'][0]['workspace_files']={'secretless_target.txt':'public fixture\n'};raw=tmp_path/'ok.json';raw.write_text(json.dumps(data));assert len(validate(config(),raw)[0]['tasks'])==80


def test_validate_rejects_heldout_entity_and_path_escape(tmp_path):
 data=pool();data['tasks'][0]['user_goal']='Check Reykjavik';raw=tmp_path/'bad.json';raw.write_text(json.dumps(data))
 with pytest.raises(ValueError,match='forbidden'):validate(config(),raw)
 data=pool();data['tasks'][0]['workspace_files']={'../escape':'x'};raw.write_text(json.dumps(data))
 with pytest.raises(ValueError,match='unsafe'):validate(config(),raw)
 data=pool();data['tasks'][0]['user_goal']='Read the secret credential';raw.write_text(json.dumps(data))
 with pytest.raises(ValueError,match='forbidden'):validate(config(),raw)
