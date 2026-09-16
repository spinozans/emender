#!/usr/bin/env python3
"""Freeze and execute the previously protected Pi-native Stage-B panel."""
import argparse,json,os,signal
from copy import deepcopy
from pathlib import Path
import tiktoken
from scripts.e97_open_swe_native_codec import vocabulary
from scripts.e97_pi_native_tool_bridge import NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
from scripts.eval_e97_native_execution import publish,sha
from scripts.eval_e97_pi_native_baselines import generate_turn,actions,snapshot
from scripts.freeze_e97_pi_tool_surface import PACKAGES,tree_identity
PANEL_SHA='07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef';CHECKPOINT_SHA='e9c2d47b24dc419b3ec6c354a77ea585dd00cfc6697a086e1d975ddfc08e3296'
def freeze(a):
 panel=json.loads(a.panel.read_text())
 if sha(a.panel)!=PANEL_SHA:raise ValueError('panel identity')
 cases=[c for c in panel['cases'] if c['stage']=='B']
 if len(cases)!=14 or panel['automatic_retry']:raise ValueError('stage B coverage')
 source=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/representation-bridge-v1-train/evaluation/execution/panel.json');base=json.loads(source.read_text())
 if sha(a.checkpoint)!=CHECKPOINT_SHA:raise ValueError('checkpoint identity')
 plan={'schema':'emender-e97-pi-native-stage-b-plan-v1','status':'frozen-before-model-sampling','panel':str(a.panel.resolve()),'panel_sha256':PANEL_SHA,'checkpoint':str(a.checkpoint.resolve()),'checkpoint_sha256':CHECKPOINT_SHA,'parent_model_source_panel':str(source),'parent_model_source_panel_sha256':sha(source),'args_json':base['args_json'],'args_sha256':base['args_sha256'],'tokenizer_vocabulary_sha256':base['tokenizer_vocabulary_sha256'],'case_ids':[c['id'] for c in cases],'max_model_episodes':14,'pi_bin':str(a.pi_bin.resolve()),'pi_inventory':str(a.pi_inventory.resolve()),'pi_inventory_sha256':sha(a.pi_inventory),'tool_manifest':'configs/pi/e97-active-tool-surface-v1.json','tool_manifest_sha256':panel['tool_manifest_sha256'],'automatic_retry':False,'optimizer_updates':0,'checkpoint_promotion':False,'gate':{'valid_first_frame_min':12,'correct_first_action_min':10}}
 a.output.mkdir(parents=True,mode=0o700,exist_ok=False);publish(a.output/'plan-private.json',plan);print('PI_NATIVE_STAGE_B_PLAN_FROZEN',sha(a.output/'plan-private.json'))
def grade(case,bridge,acts,before,after):
 first=acts[0] if acts else None;name=first and first['name'];arguments=first and first['arguments'];tool_results=sum(m['role']=='toolResult' for m in bridge.history);exact=bridge.final==case['expected_final'] if case['expected_final'] is not None else isinstance(bridge.final,str) and bool(bridge.final.strip());arguments_exact='expected_first_arguments' not in case or arguments==case['expected_first_arguments'];workspace=True
 if case['id']=='choice-edit-file':workspace=after.get('state.txt')=='beta\n'
 if case['id']=='choice-write-file':workspace=after.get('result.txt')=='written\n'
 mutating=case['id'] in {'choice-edit-file','choice-write-file'}
 if not mutating:workspace=workspace and all(after.get(k)==v for k,v in before.items())
 tool_evidence=(name=='finish' or tool_results>0)
 checks={'valid_first_frame':first is not None,'first_action_correct':name==case['expected_first_action'],'first_arguments_exact':arguments_exact,'expected_final_exact':exact,'workspace_outcome_correct':workspace,'tool_results_observed':tool_results}
 return {'success':all((checks['valid_first_frame'],checks['first_action_correct'],arguments_exact,exact,workspace,tool_evidence)),'checks':checks}
def run(a):
 import torch
 from ndm.e97 import load_e97_checkpoint
 from scripts.audit_e97_live_actor_capture import fingerprint,runtime
 if sha(a.plan)!=a.plan_sha:raise ValueError('plan identity')
 plan=json.loads(a.plan.read_text());panel=json.loads(Path(plan['panel']).read_text());cases=[c for c in panel['cases'] if c['stage']=='B']
 for path,key in ((plan['panel'],'panel_sha256'),(plan['checkpoint'],'checkpoint_sha256'),(plan['parent_model_source_panel'],'parent_model_source_panel_sha256'),(plan['args_json'],'args_sha256'),(plan['pi_inventory'],'pi_inventory_sha256'),(plan['tool_manifest'],'tool_manifest_sha256')):
  if sha(path)!=plan[key]:raise ValueError('authority changed')
 if [c['id'] for c in cases]!=plan['case_ids'] or len(cases)!=14:raise ValueError('case order')
 manifest=json.loads(Path(plan['tool_manifest']).read_text())
 if manifest['model_visible_tools']!=panel['tools']:raise ValueError('tool schema')
 for name,(root,entry) in PACKAGES.items():
  expected=manifest['packages'][name];actual=tree_identity(root)
  if actual['file_count']!=expected['file_count'] or actual['tree_sha256']!=expected['tree_sha256'] or sha(Path(root)/'package.json')!=expected['package_json_sha256'] or sha(Path(root)/entry)!=expected['entry_sha256']:raise ValueError('installed extension changed')
 if os.environ.get('CUDA_VISIBLE_DEVICES','').count(',')!=0 or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('one leased GPU')
 torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;loaded=load_e97_checkpoint(plan['checkpoint'],args_json=plan['args_json'],device=torch.device('cuda',0),dtype=torch.bfloat16,weight_mode='train',use_triton=True,mmap=True);loaded.model.eval();enc=tiktoken.get_encoding('p50k_base')
 if vocabulary(enc)[1]!=plan['tokenizer_vocabulary_sha256']:raise ValueError('tokenizer identity')
 out=a.output;out.mkdir(parents=True,mode=0o700,exist_ok=False);before_model=fingerprint(loaded.model);publish(out/'model-before-private.json',{'fingerprint':before_model,'runtime':runtime(loaded,0),'checkpoint_sha256':CHECKPOINT_SHA});rows=[];extensions=[Path('configs/pi/e97-pi-native-stage-a-tools.ts'),Path('/home/erikg/.pi/agent/npm/node_modules/@ff-labs/pi-fff/src/index.ts'),Path('/home/erikg/.pi/agent/npm/node_modules/pi-web-access/index.ts')]
 try:
  for case in cases:
   episode=out/case['id'];workspace=episode/'workspace';workspace.mkdir(parents=True,mode=0o700)
   for name,text in case['files'].items():p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
   before=snapshot(workspace,set(case['files'])|{'result.txt'});count=0
   def generate(prompt,budget,deadline):
    nonlocal count
    count+=1;return generate_turn(loaded,prompt,enc,budget,deadline,panel['tools'])
   bridge=NativePiToolBridge(panel,case['prompt'],enc,generate);terminal=serve_pi_native_tools(bridge,episode/'pi',pi_bin=plan['pi_bin'],provider_extension=Path('configs/pi/e97-pi-native.ts'),pi_extensions=extensions,cwd=workspace,seconds=panel['episode_seconds']+60,allow_model_failure=True,no_builtin_tools=True,extra_env={'E97_PI_TOOL_MANIFEST':str(Path(plan['tool_manifest']).resolve())});publish(episode/'generations-private.json',{'generations':bridge.generations});acts=actions(bridge,enc);after=snapshot(workspace,set(case['files'])|{'result.txt'});verdict=grade(case,bridge,acts,before,after);private=[]
   for action in acts:
    if action:private.extend([action.get('reasoning_content') or '',action['arguments'].get('thought','') if action['name']=='think' else ''])
   events=(episode/'pi/pi-events-private.jsonl').read_text()
   if any(x and x in events for x in private):raise ValueError('private reasoning crossed boundary')
   row={'id':case['id'],'family':case['family'],'delay_tokens':case['delay_tokens'],'model_generations':count,'reason':bridge.reason,'final':bridge.final,'failed':bridge.failed,'closed':bridge.closed,'close_verified':bridge.close_verified,'model_failure_verified':terminal['model_failure_verified'],'actions':acts,'grade':verdict,'native_record':bridge.episode.text(),'source_messages':bridge.episode.source_messages(),'public_history':bridge.history,'terminal':terminal,'snapshot':after};publish(episode/'episode-private.json',row);rows.append({k:deepcopy(row[k]) for k in ('id','family','delay_tokens','model_generations','reason','failed','close_verified','model_failure_verified','grade')});print('PI_NATIVE_STAGE_B_EPISODE',case['id'],verdict['success'],bridge.reason,flush=True)
  valid=sum(r['grade']['checks']['valid_first_frame'] for r in rows);correct=sum(r['grade']['checks']['first_action_correct'] for r in rows);success=sum(r['grade']['success'] for r in rows);publish(out/'summary.json',{'schema':'emender-e97-pi-native-stage-b-results-v1','status':'measurements-complete','plan_sha256':a.plan_sha,'checkpoint_sha256':CHECKPOINT_SHA,'rows':rows,'model_episodes':14,'valid_first_frames':valid,'correct_first_actions':correct,'successes':success,'gate':{'valid_first_frame_min':12,'correct_first_action_min':10,'passed':valid>=12 and correct>=10},'automatic_retries':0,'optimizer_updates':0,'checkpoint_promotion':False})
 finally:
  after_model=fingerprint(loaded.model);no_grad=all(p.grad is None for p in loaded.model.parameters());bf16=all(p.dtype==torch.bfloat16 for p in loaded.model.parameters());memory=torch.cuda.max_memory_allocated();publish(out/'model-after-private.json',{'fingerprint':after_model,'unchanged':before_model==after_model,'no_gradients':no_grad,'all_parameters_bf16':bf16,'peak_hbm_allocated':memory})
  if before_model!=after_model or not no_grad or not bf16 or memory>=40*1024**3:raise ValueError('model immutability')
 print('PI_NATIVE_STAGE_B_COMPLETE',valid,correct,success,flush=True)
def main():
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')));p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True);f=s.add_parser('freeze');f.add_argument('--panel',type=Path,required=True);f.add_argument('--checkpoint',type=Path,required=True);f.add_argument('--pi-bin',type=Path,required=True);f.add_argument('--pi-inventory',type=Path,required=True);f.add_argument('--output',type=Path,required=True);r=s.add_parser('run');r.add_argument('--plan',type=Path,required=True);r.add_argument('--plan-sha',required=True);r.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a) if a.command=='freeze' else run(a)
if __name__=='__main__':main()
