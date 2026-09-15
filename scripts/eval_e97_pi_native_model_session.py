#!/usr/bin/env python3
"""Two unchanged-weight model tasks in one explicit Pi/OpenHands session."""
import argparse,hashlib,json,os,signal,time
from copy import deepcopy
from pathlib import Path
from scripts.e97_native_execution_cases import context,grade
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_codec import vocabulary
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode
from scripts.e97_pi_native_session import NativeTaskSession
from scripts.e97_pi_native_session_transport import serve_pi_session
from scripts.eval_e97_native_execution import generate_turn,publish,sha
R=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
CHECKPOINT_SHA='9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa'
CASE_IDS=('bridge-fresh-lookup-0000-world-0','bridge-fresh-edit-0000-world-0')


def select_cases(panel):
 cases=[deepcopy(next(c for c in panel['cases'] if c['id']==cid and c['cohort']=='fresh')) for cid in CASE_IDS]
 if any(c.get('supplied_calls') for c in cases) or {c['family'] for c in cases}!={'lookup','edit'}:
  raise ValueError('fixed cases')
 return cases


def freeze(args):
 source=R/'representation-bridge-v1-train/evaluation/execution/panel.json'
 if sha(source)!='8e70382d1e941e4c03d3c137699d8a1b6f76dfb9170619a5e308a6346d7d40eb':raise ValueError('source panel')
 panel=json.loads(source.read_text());target=next(m for m in panel['models'] if m['name']=='bridge-y')
 if target['sha256']!=CHECKPOINT_SHA or target['mode']!='train' or sha(target['checkpoint'])!=CHECKPOINT_SHA:raise ValueError('live-y checkpoint')
 cases=select_cases(panel)
 prior=R/'pi-native-compatibility-v1-evaluation/summary.json'; ps=json.loads(prior.read_text()); rows={r['id']:r for r in ps['rows']}
 for cid in CASE_IDS:
  r=rows[cid]
  if not all(r[k] for k in ('direct_success','pi_success','pi_close_verified','first_prompt_equal','pi_prompt_replay_exact','native_transcript_equal','generated_tokens_equal')):raise ValueError('cases not previously route-qualified')
 session=R/'pi-native-session-v2-r2'; control=R/'pi-native-session-v2-r2-control'
 if sha(session/'summary.json')!='a9370fda914020a8d3d3d3cb6c48db1e7667dfbb19eb80da1f0217bbbaecfca3':raise ValueError('scripted session summary')
 audit=json.loads((session/'independent-audit.json').read_text()); terminal=json.loads((control/'terminal.json').read_text())
 if not audit['passed'] or terminal!=dict(original_exit=0,audited_exit=0,model_generations=0,optimizer_updates=0):raise ValueError('session qualification')
 for name in ('configs/pi/e97-openhands-compat.ts','scripts/e97_pi_native_bridge.py','scripts/e97_pi_native_failure.py','scripts/e97_pi_native_session.py','scripts/e97_pi_native_session_transport.py','scripts/e97_open_swe_native_runtime_protocol.py','scripts/e97_openhands_native_backend.py'):
  if sha(name)!=sha(control/'worktree'/name):raise ValueError('source differs from session qualification')
 panel={**panel,'models':[target],'cases':cases,'max_turns':16,'episode_generation_budget':32768,'episode_seconds':180}
 plan=dict(schema='emender-e97-pi-native-model-session-v1',panel=panel,source_panel=str(source),source_panel_sha256=sha(source),
  prior_compatibility=str(prior),prior_compatibility_sha256=sha(prior),scripted_session_summary=str(session/'summary.json'),
  scripted_session_summary_sha256=sha(session/'summary.json'),scripted_session_audit=str(session/'independent-audit.json'),
  scripted_session_audit_sha256=sha(session/'independent-audit.json'),pi_bin=str(R/'pi-native-compatibility-v1-control/bin/pi'),
  pi_inventory=str(R/'pi-native-compatibility-v1-control/pi-files.sha256'),pi_inventory_sha256=sha(R/'pi-native-compatibility-v1-control/pi-files.sha256'),
  task_order=list(CASE_IDS),max_model_episodes=2,max_tasks=2,session_tokens=65536,session_seconds=600,gpu_count=1,
  automatic_retry=False,optimizer_updates=0,training_eligible=False,checkpoint_promotion=False,independent_generalization=False,
  conversational_memory_claim=False,pi_native_tools=False,gate='two previously-qualified tasks succeed; exact prompt replay; closures/session close; unchanged weights')
 args.output.mkdir(parents=True,mode=0o700,exist_ok=False);publish(args.output/'plan-private.json',plan)
 print('PI_NATIVE_MODEL_SESSION_PLAN_FROZEN',sha(args.output/'plan-private.json'),flush=True)


def run(args):
 import torch,tiktoken
 from ndm.e97 import load_e97_checkpoint
 from scripts.audit_e97_live_actor_capture import fingerprint,runtime
 if sha(args.plan)!=args.plan_sha:raise ValueError('plan identity')
 plan=json.loads(args.plan.read_text());panel=plan['panel'];target=panel['models'][0]
 if plan['max_model_episodes']!=2 or target['sha256']!=CHECKPOINT_SHA or sha(target['checkpoint'])!=CHECKPOINT_SHA:raise ValueError('fixed model budget')
 if not os.environ.get('CUDA_VISIBLE_DEVICES') or len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))!=1:raise ValueError('one GPU')
 for a,b in [('source_panel','source_panel_sha256'),('prior_compatibility','prior_compatibility_sha256'),('scripted_session_summary','scripted_session_summary_sha256'),('scripted_session_audit','scripted_session_audit_sha256'),('pi_inventory','pi_inventory_sha256')]:
  if sha(plan[a])!=plan[b]:raise ValueError('authority changed')
 torch.cuda.set_device(0);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
 loaded=load_e97_checkpoint(target['checkpoint'],args_json=panel['args_json'],device=torch.device('cuda',0),dtype=torch.bfloat16,weight_mode='train',use_triton=True,mmap=True);loaded.model.eval()
 enc=tiktoken.get_encoding('p50k_base')
 if vocabulary(enc)[1]!=panel['tokenizer_vocabulary_sha256']:raise ValueError('tokenizer')
 before=fingerprint(loaded.model);publish(args.output/'model-before-private.json',dict(fingerprint=before,runtime=runtime(loaded,0),target=target))
 cases=panel['cases'];files={}
 for c in cases:
  if set(files)&set(c['files']):raise ValueError('case path collision')
  files.update(c['files'])
 tasks=[dict(task_id=c['id'],prompt=c['prompt']) for c in cases]; began=time.monotonic();session=None
 try:
  with NativeSandbox(panel,args.output/'sandbox') as sandbox:
   sandbox.request('setup',files=files)
   def generate(prompt,budget,deadline):return generate_turn(loaded,prompt,enc,budget,deadline)
   def execute(call):return sandbox.request('execute',call=call)
   session=NativeTaskSession(panel,enc,generate,execute,max_tasks=2,session_tokens=plan['session_tokens'],session_seconds=plan['session_seconds'])
   terminal=serve_pi_session(session,tasks,args.output/'pi',pi_bin=plan['pi_bin'],extension=Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts',seconds=plan['session_seconds']+60,require_successful_tasks=False)
   names=list(files)
   for c in cases:
    if c['expected_output'] is not None:names.append(c.get('output_path','result.json'))
   snapshot=sandbox.snapshot(names)
  rows=[]
  for c,t in zip(cases,session.tasks):
   verdict=grade(c,t.final,t.calls,snapshot); replay=NativeEpisode(panel['tools'],enc);j=0
   for m in t.episode.source_messages():
    if m['role']=='assistant':
     if hashlib.sha256(replay.prompt().encode()).hexdigest()!=t.generations[j]['prompt_sha256']:raise ValueError('prompt replay')
     j+=1
    replay.append_source_message(m)
   rows.append(dict(id=c['id'],family=c['family'],success=verdict['success'],grade=verdict,final=t.final,reason=t.reason,
     closed=t.closed,close_verified=t.close_verified,failed=t.failed,generations=t.generations,calls=t.calls,prompt_replay_exact=j==len(t.generations),native_record_sha256=hashlib.sha256(t.episode.text().encode()).hexdigest()))
  checks=dict(two_task_results=len(rows)==2,two_successes=all(r['success'] for r in rows),two_verified_finishes=all(r['close_verified'] for r in rows),
   exact_prompt_replay=all(r['prompt_replay_exact'] for r in rows),session_close_verified=terminal['session_close_verified'],
   prior_task_absent_from_second_native_context=cases[0]['prompt'] not in session.tasks[1].episode.text(),
   public_history_retained=len(session.completed_history)>0)
  publish(args.output/'session-private.json',dict(boundaries=session.boundaries,completed_public_history=session.completed_history,rows=rows,snapshot=snapshot,terminal=terminal))
  publish(args.output/'summary.json',dict(schema=plan['schema'],plan_sha256=args.plan_sha,checks=checks,passed=all(checks.values()),outcomes=[{k:r[k] for k in ('id','family','success','reason')} for r in rows],seconds=time.monotonic()-began,model_episodes=2,optimizer_updates=0,checkpoint_promotion=False,independent_generalization=False,conversational_memory_claim=False))
  if not all(checks.values()):raise ValueError('two-task model session gate failed')
 finally:
  after=fingerprint(loaded.model);no_grad=all(p.grad is None for p in loaded.model.parameters());bf16=all(p.dtype==torch.bfloat16 for p in loaded.model.parameters());memory=torch.cuda.max_memory_allocated()
  publish(args.output/'model-after-private.json',dict(fingerprint=after,unchanged=before==after,no_gradients=no_grad,all_parameters_bf16=bf16,peak_hbm_allocated=memory))
  if before!=after or not no_grad or not bf16 or memory>=40*1024**3:raise ValueError('model immutability guard')
 print('PI_NATIVE_MODEL_SESSION_COMPLETE',*(r['success'] for r in rows),flush=True)


def main():
 os.umask(0o077);signal.signal(signal.SIGTERM,lambda s,f:(_ for _ in ()).throw(TimeoutError('interrupted')))
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='command',required=True);f=sp.add_parser('freeze');f.add_argument('--output',type=Path,required=True);r=sp.add_parser('run');r.add_argument('--plan',type=Path,required=True);r.add_argument('--plan-sha',required=True);r.add_argument('--output',type=Path,required=True);a=p.parse_args();freeze(a) if a.command=='freeze' else run(a)
if __name__=='__main__':main()
