#!/usr/bin/env python3
"""Run frozen Stage-A Pi-native copy/tool baseline with unchanged live-y weights."""
import argparse,hashlib,json,os,shutil,signal,time
from pathlib import Path
import tiktoken
from scripts.e97_open_swe_native_codec import compact
from scripts.e97_pi_native_codec import validate_generated_turn
from scripts.e97_pi_native_tool_bridge import NativePiToolBridge
from scripts.e97_pi_native_tool_transport import serve_pi_native_tools
from scripts.eval_e97_native_execution import publish,sha
from scripts.freeze_e97_pi_tool_surface import PACKAGES,tree_identity
R=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining');PANEL_SHA='07cc1d5843ee4060503ecdcf7d2bb813dce51fa0aeba89df367b7696e47994ef';CHECKPOINT_SHA='9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa'

def freeze(args):
 if sha(args.panel)!=PANEL_SHA:raise ValueError('panel identity')
 panel=json.loads(args.panel.read_text());cases=[c for c in panel['cases'] if c['stage']=='A']
 if len(cases)!=12 or panel['generation_budget']!=1024 or panel['automatic_retry'] is not False:raise ValueError('stage A authority')
 source=R/'representation-bridge-v1-train/evaluation/execution/panel.json';model_panel=json.loads(source.read_text());target=next(m for m in model_panel['models'] if m['name']=='bridge-y')
 if target['sha256']!=CHECKPOINT_SHA or sha(target['checkpoint'])!=CHECKPOINT_SHA or target['mode']!='train':raise ValueError('checkpoint identity')
 plan=dict(schema='emender-e97-pi-native-stage-a-plan-v1',panel=str(args.panel.resolve()),panel_sha256=sha(args.panel),model_source_panel=str(source),model_source_panel_sha256=sha(source),target=target,args_json=model_panel['args_json'],args_sha256=model_panel['args_sha256'],tokenizer_vocabulary_sha256=model_panel['tokenizer_vocabulary_sha256'],case_ids=[c['id'] for c in cases],max_model_episodes=12,episode_order=[c['id'] for c in cases],pi_bin=str(args.pi_bin.resolve()),pi_inventory=str(args.pi_inventory.resolve()),pi_inventory_sha256=sha(args.pi_inventory),tool_manifest='configs/pi/e97-active-tool-surface-v1.json',tool_manifest_sha256=panel['tool_manifest_sha256'],stage_a_safe_local_tools=True,real_fff=True,real_web_search=True,automatic_retry=False,optimizer_updates=0,training_eligible=False,checkpoint_promotion=False,gate='measurement completeness, exact identities, honest model failures and unchanged weights; model scores reported without threshold')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);publish(args.output/'plan-private.json',plan);print('PI_NATIVE_STAGE_A_PLAN_FROZEN',sha(args.output/'plan-private.json'))

def generate_turn(loaded,prompt,encoding,budget,deadline,tools):
 import torch
 from ndm.e97 import advance_e97_cache_segment,generate_e97_from_cache
 prefix=encoding.encode_ordinary(prompt)
 if len(prefix)+budget>65536:return None,[],'context_budget'
 ids=[];reason='generation_budget'
 with torch.no_grad():
  cache=advance_e97_cache_segment(loaded,prefix)
  for _ in range(budget):
   if time.monotonic()>=deadline:reason='episode_deadline';break
   new,cache=generate_e97_from_cache(loaded,cache,max_new_tokens=1,temperature=0.,top_k=0,top_p=0.,stop_token_ids=(218,))
   if not new:reason='empty';break
   ids.extend(new);text=encoding.decode(ids)
   if not ('Analysis: '.startswith(text) or text.startswith('Analysis: ')):reason='invalid_opening';break
   if text.count('\n')>4:reason='invalid_frame';break
   if text.count('\n')==4 and text.endswith('}'):
    try:validate_generated_turn(text,tools,encoding)
    except ValueError:pass
    else:return text,ids,'valid'
   if 218 in new:reason='separator_before_valid_turn';break
 return None,ids,reason

def actions(bridge,encoding):
 result=[]
 for g in bridge.generations:
  if g['reason']!='valid':result.append(None);continue
  try:result.append(validate_generated_turn(encoding.decode(g['token_ids']),bridge.tools,encoding).semantic(bridge.tools))
  except ValueError:result.append(None)
 return result

def snapshot(workspace,files):return {p:(workspace/p).read_text() if (workspace/p).exists() else None for p in files}
def grade(case,bridge,acts,before,after):
 first=acts[0] if acts else None;first_action=first and first['name'];first_args=first and first['arguments'];unchanged=before==after
 exact=True
 if case['expected_final'] is not None:exact=bridge.final==case['expected_final']
 path_args=True
 if 'expected_first_arguments' in case:path_args=first_args==case['expected_first_arguments']
 tool_results=sum(m['role']=='toolResult' for m in bridge.history)
 outcome=exact and unchanged
 if case['id']=='choice-fffind-local':outcome=outcome and bridge.final=='weather_adapter_73c91.py' and tool_results>0
 if case['id']=='choice-web-current':outcome=first_action=='web_search' and tool_results>0 and isinstance(bridge.final,str) and bool(bridge.final.strip())
 checks={'first_action_correct':first_action==case['expected_first_action'],'first_arguments_exact':path_args,'expected_final_exact':exact,'source_files_unchanged':unchanged,'tool_results_observed':tool_results}
 return {'success':outcome and checks['first_action_correct'] and path_args,'checks':checks}

def run(args):
 import torch
 from ndm.e97 import load_e97_checkpoint
 from scripts.audit_e97_live_actor_capture import fingerprint,runtime
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 plan=json.loads(args.plan.read_text());panel=json.loads(Path(plan['panel']).read_text());cases=[c for c in panel['cases'] if c['stage']=='A']
 if (sha(plan['panel'])!=plan['panel_sha256'] or plan['panel_sha256']!=PANEL_SHA or
  sha(plan['model_source_panel'])!=plan['model_source_panel_sha256'] or sha(plan['pi_inventory'])!=plan['pi_inventory_sha256'] or
  sha(plan['tool_manifest'])!=plan['tool_manifest_sha256']):raise ValueError('authority changed')
 manifest=json.loads(Path(plan['tool_manifest']).read_text())
 if manifest['model_visible_tools']!=panel['tools']:raise ValueError('panel/tool schema mismatch')
 for name,(package_root,entry) in PACKAGES.items():
  expected=manifest['packages'][name];actual=tree_identity(package_root)
  if (actual['file_count']!=expected['file_count'] or actual['tree_sha256']!=expected['tree_sha256'] or
   sha(Path(package_root)/'package.json')!=expected['package_json_sha256'] or sha(Path(package_root)/entry)!=expected['entry_sha256']):raise ValueError('installed Pi extension changed')
 if len(cases)!=plan['max_model_episodes'] or [c['id'] for c in cases]!=plan['episode_order']:raise ValueError('case budget/order')
 if not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1:raise ValueError('one leased GPU')
 torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False;target=plan['target'];loaded=load_e97_checkpoint(target['checkpoint'],args_json=plan['args_json'],device=torch.device('cuda',0),dtype=torch.bfloat16,weight_mode='train',use_triton=True,mmap=True);loaded.model.eval();encoding=tiktoken.get_encoding('p50k_base')
 root=Path(args.output);root.mkdir(parents=True,mode=0o700,exist_ok=False)
 before_model=fingerprint(loaded.model);publish(root/'model-before-private.json',{'fingerprint':before_model,'runtime':runtime(loaded,0),'target':target})
 rows=[];extensions=[Path('configs/pi/e97-pi-native-stage-a-tools.ts'),Path('/home/erikg/.pi/agent/npm/node_modules/@ff-labs/pi-fff/src/index.ts'),Path('/home/erikg/.pi/agent/npm/node_modules/pi-web-access/index.ts')]
 try:
  for case in cases:
   out=root/case['id'];workspace=out/'workspace';workspace.mkdir(parents=True,mode=0o700)
   for name,text in case['files'].items():p=workspace/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)
   original=snapshot(workspace,case['files']);count=0
   def generate(prompt,budget,deadline):
    nonlocal count
    count+=1;return generate_turn(loaded,prompt,encoding,budget,deadline,panel['tools'])
   bridge=NativePiToolBridge(panel,case['prompt'],encoding,generate)
   terminal=serve_pi_native_tools(bridge,out/'pi',pi_bin=plan['pi_bin'],provider_extension=Path('configs/pi/e97-pi-native.ts'),pi_extensions=extensions,cwd=workspace,seconds=panel['episode_seconds']+60,allow_model_failure=True,no_builtin_tools=True,extra_env={'E97_PI_TOOL_MANIFEST':str(Path(plan['tool_manifest']).resolve())})
   acts=actions(bridge,encoding);final_snapshot=snapshot(workspace,case['files']);verdict=grade(case,bridge,acts,original,final_snapshot)
   private=[]
   for a in acts:
    if a:private.extend([a.get('reasoning_content') or '',a['arguments'].get('thought','') if a['name']=='think' else ''])
   events=(out/'pi/pi-events-private.jsonl').read_text()
   if any(x and x in events for x in private):raise ValueError('private reasoning crossed Pi boundary')
   row={'id':case['id'],'family':case['family'],'delay_tokens':case['delay_tokens'],'initial_prompt_tokens':case['initial_prompt_tokens'],'model_generations':count,'reason':bridge.reason,'final':bridge.final,'failed':bridge.failed,'closed':bridge.closed,'close_verified':bridge.close_verified,'model_failure_verified':terminal['model_failure_verified'],'actions':acts,'grade':verdict,'native_record':bridge.episode.text(),'source_messages':bridge.episode.source_messages(),'public_history':bridge.history,'terminal':terminal,'snapshot':final_snapshot}
   publish(out/'episode-private.json',row);rows.append({k:row[k] for k in ('id','family','delay_tokens','initial_prompt_tokens','model_generations','reason','failed','close_verified','model_failure_verified','grade')});print('PI_NATIVE_STAGE_A_EPISODE',case['id'],verdict['success'],bridge.reason,flush=True)
  aggregates={}
  for family in sorted({r['family'] for r in rows}):
   selected=[r for r in rows if r['family']==family];aggregates[family]={'episodes':len(selected),'successes':sum(r['grade']['success'] for r in selected),'first_action_correct':sum(r['grade']['checks']['first_action_correct'] for r in selected)}
  summary={'schema':'emender-e97-pi-native-stage-a-results-v1','plan_sha256':args.plan_sha,'status':'measurements-complete','rows':rows,'aggregates':aggregates,'model_episodes':len(rows),'automatic_retries':0,'optimizer_updates':0,'training_eligible':False,'checkpoint_promotion':False};publish(root/'summary.json',summary)
 finally:
  after=fingerprint(loaded.model);no_grad=all(p.grad is None for p in loaded.model.parameters());bf16=all(p.dtype==torch.bfloat16 for p in loaded.model.parameters());memory=torch.cuda.max_memory_allocated();publish(root/'model-after-private.json',{'fingerprint':after,'unchanged':before_model==after,'no_gradients':no_grad,'all_parameters_bf16':bf16,'peak_hbm_allocated':memory})
  if before_model!=after or not no_grad or not bf16 or memory>=40*1024**3:raise ValueError('model immutability')
 if len(rows)!=12:raise ValueError('measurement coverage')
 print('PI_NATIVE_STAGE_A_COMPLETE',len(rows),flush=True)

def main():
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')));p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True);f=sp.add_parser('freeze');f.add_argument('--panel',type=Path,required=True);f.add_argument('--pi-bin',type=Path,required=True);f.add_argument('--pi-inventory',type=Path,required=True);f.add_argument('--output',type=Path,required=True);r=sp.add_parser('run');r.add_argument('--plan',type=Path,required=True);r.add_argument('--plan-sha',required=True);r.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a) if a.command=='freeze' else run(a)
if __name__=='__main__':main()
