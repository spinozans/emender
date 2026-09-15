#!/usr/bin/env python3
"""Independent audit of the fixed Stage-A Pi-native baseline artifacts."""
import argparse,hashlib,json
from pathlib import Path
import tiktoken
from scripts.e97_pi_native_codec import PiNativeEpisode
from scripts.eval_e97_native_execution import sha
PANEL_SHA='07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef';PLAN_SHA='f205876aaf19e81edabe6d81121b731d423c1bfa765f1c53a7a3fcb784416809'
PREFIX_IDS=['copy-00128','bind-00128','path-00128','copy-08192','bind-08192','path-08192']
SUFFIX_IDS=['copy-58000','bind-58000','path-58000','choice-direct-supplied','choice-fffind-local','choice-web-current']

def lines(path):return [json.loads(x) for x in path.read_text().splitlines()]
def initial_record(system,tools,prompt,encoding):
 episode=PiNativeEpisode(tools,encoding);episode.append_context({'role':'system','content':system});episode.append_context({'role':'user','content':prompt});return episode
def check_model(root):
 before=json.loads((root/'model-before-private.json').read_text());after=json.loads((root/'model-after-private.json').read_text())
 assert before['fingerprint']==after['fingerprint'];assert after['unchanged'] and after['no_gradients'] and after['all_parameters_bf16'];assert after['peak_hbm_allocated']<40*1024**3
 return after['peak_hbm_allocated']
def check_control(root,exit_code,budget):
 assert json.loads((root/'terminal.json').read_text())=={'original_exit':exit_code,'audited_exit':exit_code,'model_episode_budget':budget,'automatic_retries':0,'optimizer_updates':0}
 for name in ('authority-before.log','authority-after.log','source-before.log','source-after.log','pi-before.log','pi-after.log','interpreters-before.log','interpreters-after.log'):
  assert 'FAILED' not in (root/name).read_text()
 assert (root/'lease-release.log').exists() and (root/'lease-release.log').read_text()==''
def main():
 p=argparse.ArgumentParser();p.add_argument('--panel',type=Path,required=True);p.add_argument('--prefix-root',type=Path,required=True);p.add_argument('--prefix-control',type=Path,required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--suffix-root',type=Path,required=True);p.add_argument('--suffix-control',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert sha(a.panel)==PANEL_SHA;assert sha(a.plan)==PLAN_SHA
 panel=json.loads(a.panel.read_text());cases=[c for c in panel['cases'] if c['stage']=='A'];assert [c['id'] for c in cases]==PREFIX_IDS+SUFFIX_IDS;case_map={c['id']:c for c in cases}
 plan=json.loads(a.plan.read_text());assert plan['episode_order']==SUFFIX_IDS and plan['max_model_episodes']==6 and [r['id'] for r in plan['prefix_rows']]==PREFIX_IDS;assert len(plan['prefix_receipts'])==6
 for receipt in plan['prefix_receipts']:assert sha(a.prefix_root/receipt['id']/'episode-private.json')==receipt['episode_sha256']
 assert sha(plan['prefix']['terminal'])==plan['prefix']['terminal_sha256'];assert sha(a.prefix_root/'model-before-private.json')==plan['prefix']['model_before_sha256'];assert sha(a.prefix_root/'model-after-private.json')==plan['prefix']['model_after_sha256']
 check_control(a.prefix_control,1,12);check_control(a.suffix_control,0,6);peaks=[check_model(a.prefix_root),check_model(a.suffix_root)]
 summary=json.loads((a.suffix_root/'summary.json').read_text());assert summary['status']=='measurements-complete' and summary['model_episodes']==12 and summary['new_model_episodes']==6 and summary['retained_prefix_episodes']==6;assert summary['automatic_retries']==summary['optimizer_updates']==0 and not summary['training_eligible'] and not summary['checkpoint_promotion'];assert summary['plan_sha256']==PLAN_SHA
 encoding=tiktoken.get_encoding('p50k_base');full=[];missing=[]
 for index,case in enumerate(cases):
  root=a.prefix_root if index<6 else a.suffix_root;episode_path=root/case['id']/'episode-private.json';row=json.loads(episode_path.read_text());full.append(row)
  assert row['id']==case['id'] and row['family']==case['family'] and row['delay_tokens']==case['delay_tokens'];assert row['model_generations']==1 and row['reason'] in ('invalid_frame','generation_budget');assert row['failed'] and row['closed'] and not row['close_verified'] and row['model_failure_verified'];assert row['final'] is None and row['actions']==[None] and not row['grade']['success'];assert len(row['public_history'])==1 and row['public_history'][0]['role']=='user' and row['public_history'][0]['content'][0]['text']==case['prompt'];assert row['source_messages'][1]=={'role':'user','content':case['prompt']};assert not any(m['role']=='toolResult' for m in row['public_history'])
  record=initial_record(panel['system'],panel['tools'],case['prompt'],encoding);assert row['native_record']==record.text() and row['source_messages']==record.source_messages();assert len(encoding.encode_ordinary(record.prompt()))==case['initial_prompt_tokens']==row['initial_prompt_tokens']
  for name,text in case['files'].items():assert (root/case['id']/'workspace'/name).read_text()==text and row['snapshot'][name]==text
  terminal=row['terminal'];assert terminal['bridge_failed'] and terminal['closed'] and terminal['model_failure_verified'] and terminal['reason']==row['reason'] and terminal['pi_exit']==0;assert [x['op'] for x in terminal['requests']]==['next','close']
  reqs=lines(root/case['id']/'pi/requests-private.jsonl');assert [x['op'] for x in reqs]==['next','close'];assert reqs[0]['systemPrompt']==panel['system'] and reqs[0]['model']=='e97-4b-pi-native' and reqs[0]['provider']=='e97-pi-native';assert sorted(reqs[0]['tools'],key=lambda t:t['name'])==sorted(panel['tools'],key=lambda t:t['name']);assert reqs[0]['messages'][0]['content'][0]['text']==case['prompt'];assert reqs[1]['messages'][-1]['stopReason']=='error' and reqs[1]['messages'][-1]['errorMessage']=='Pi-native transport stopped; inspect private owner receipt.'
  events=lines(root/case['id']/'pi/pi-events-private.jsonl');assert len(events)==10 and [e['message']['role'] for e in events if e['type']=='message_end']==['user','assistant']
  generation=root/case['id']/'generations-private.json'
  if not generation.exists():missing.append(case['id'])
 assert [r['id'] for r in summary['rows']]==[r['id'] for r in full]
 assert all(not r['grade']['success'] for r in full);assert sum(r['grade']['checks']['first_action_correct'] for r in full)==0;assert sum(r['grade']['checks']['tool_results_observed'] for r in full)==0
 assert missing==PREFIX_IDS+SUFFIX_IDS
 result={'schema':'emender-e97-pi-native-stage-a-audit-v1','verdict':'qualified-negative-outcomes-with-incomplete-generation-receipts','panel_sha256':PANEL_SHA,'continuation_plan_sha256':PLAN_SHA,'episodes':12,'successes':0,'copy_bind_path_successes':0,'tool_choice_successes':0,'first_actions_correct':0,'tool_results_observed':0,'fixed_model_failures':12,'automatic_retries':0,'optimizer_updates':0,'weights_unchanged':True,'peak_hbm_allocated':peaks,'source_files_unchanged':True,'pi_histories_verified':True,'generation_token_receipts_complete':False,'missing_generation_token_receipts':missing,'stage_b_authorized':False,'training_eligible':False,'checkpoint_promotion':False,'residual_risk':'Rejected free-generation token IDs were not persisted, so an independent audit cannot reconstruct invalid_frame/generation_budget classifications from token bytes. The exact Pi errors and causal histories are retained. Do not rerun or reinterpret these episodes.'}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print('PI_NATIVE_STAGE_A_AUDIT',result['verdict'],result['episodes'],result['successes'])
if __name__=='__main__':main()
