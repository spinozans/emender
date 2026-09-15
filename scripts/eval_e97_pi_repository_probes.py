#!/usr/bin/env python3
"""Bounded repository diagnostic, direct owner loop versus actual Pi.

Both routes use the already-qualified native bridge/codec/generator/executor.
This direct route is a bridge-owner control, not the older standalone harness.
"""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import time

from scripts.e97_pi_native_bridge import NativePiBridge, BridgeStopped, MODEL, PROVIDER
from scripts.e97_pi_native_failure import MODEL_STOP_REASONS, verify_model_failure
from scripts.e97_pi_native_transport import serve_pi
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.qualify_e97_pi_repository_probes import verifier, TEST_COMMAND
from scripts.e97_pi_repository_probes import inspect_snapshot
from scripts.eval_e97_native_execution import generate_turn, publish, sha
from scripts.e97_open_swe_native_codec import vocabulary

R=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')
CHECKPOINT=R/'representation-bridge-v1-train/checkpoints/checkpoint_agent_sft_u000032_loss_0.5302.pt'
CHECKPOINT_SHA='9b78628d47c48c304c14de50399fe1853d8c83386c7cf5d9bd4a0e878bf679fa'


def freeze(args):
    preflight=R/'pi-repository-workflow-v1-preflight'
    control=R/'pi-repository-workflow-v1-control'
    terminal=json.loads((control/'terminal.json').read_text())
    summary=json.loads((preflight/'summary.json').read_text())
    audit=json.loads((preflight/'independent-audit.json').read_text())
    if terminal!=dict(original_exit=0,audited_exit=0,model_generations=0,optimizer_updates=0) or not audit['passed']:
        raise ValueError('audited repository preflight required')
    if summary['status']!='authored-repository-workflow-preflight-passed' or len(summary['results'])!=4:
        raise ValueError('four authored controls required')
    source_plan=json.loads((preflight/'plan-private.json').read_text())
    panel_path=R/'native-grounding-diagnostic-v1/panel.json'
    if sha(panel_path)!=source_plan['panel_sha256'] or sha(CHECKPOINT)!=CHECKPOINT_SHA:
        raise ValueError('panel/checkpoint identity')
    for name in ('scripts/e97_pi_repository_probes.py','scripts/qualify_e97_pi_repository_probes.py',
                 'scripts/e97_pi_native_bridge.py','scripts/e97_pi_native_transport.py','configs/pi/e97-openhands-compat.ts'):
        if sha(name)!=sha(control/'worktree'/name): raise ValueError('preflight source changed')
    cases=source_plan['cases']
    if len(cases)!=4 or len({c['id'] for c in cases})!=4: raise ValueError('case coverage')
    plan=dict(schema='emender-e97-pi-repository-model-probe-v1',cases=cases,panel=json.loads(panel_path.read_text()),
        checkpoint=str(CHECKPOINT),checkpoint_sha256=CHECKPOINT_SHA,weight_mode='train',
        preflight_summary=str(preflight/'summary.json'),preflight_summary_sha256=sha(preflight/'summary.json'),
        preflight_audit=str(preflight/'independent-audit.json'),preflight_audit_sha256=sha(preflight/'independent-audit.json'),
        pi_bin=str(R/'pi-native-compatibility-v1-control/bin/pi'),max_model_episodes=8,gpu_count=1,
        run_seconds=2400,model_updates=0,automatic_retry=False,training_eligible=False,checkpoint_promotion=False,
        independent_repository_claim=False,direct_route='qualified-native-bridge-owner-loop',
        routes=[['direct','pi'] if i%2==0 else ['pi','direct'] for i in range(4)],
        capability_gate='Pi >=2/4 successes and >=1 in each family',
        transport_gate='all episode histories/failure terminations valid; per-case success parity')
    args.output.mkdir(parents=True,mode=0o700,exist_ok=False)
    publish(args.output/'plan-private.json',plan)
    print('PI_REPOSITORY_MODEL_PLAN_FROZEN',sha(args.output/'plan-private.json'),flush=True)


def episode(loaded,case,panel,encoding,output,route,pi_bin,extension):
    output.mkdir(mode=0o700)
    began=time.monotonic(); transport=None
    with NativeSandbox(panel,output/'sandbox') as sandbox:
        sandbox.request('setup',files=case['files'])
        def generate(prompt,budget,deadline): return generate_turn(loaded,prompt,encoding,budget,deadline)
        def execute(call): return sandbox.request('execute',call=call)
        b=NativePiBridge(panel,case['prompt'],encoding,generate,execute)
        try:
            if route=='pi':
                try:
                    serve_pi(b,output/'pi',pi_bin=pi_bin,extension=extension,seconds=panel['episode_seconds']+60)
                    transport=dict(native_finish_verified=True,model_failure_transport_verified=False)
                except BridgeStopped:
                    transport=verify_model_failure(b,output/'pi')
            else:
                try:
                    while not b.episode.finished:
                        response=b.next(dict(systemPrompt=panel['system'],tools=b.tools,model=MODEL,provider=PROVIDER,
                                             messages=deepcopy(b.history)))
                        call=response['message']['content'][-1]
                        b.execute(dict(id=call['id'],name=call['name'],arguments=call['arguments']))
                    b.close(deepcopy(b.history))
                    transport=dict(native_finish_verified=True,model_failure_transport_verified=False)
                except BridgeStopped:
                    if b.reason not in MODEL_STOP_REASONS or b.pending is not None or b.final is not None: raise
                    b.close([])
                    transport=dict(native_finish_verified=False,model_failure_transport_verified=True,reason=b.reason)
            snapshot=sandbox.snapshot(list(case['files']))
        finally:
            publish(output/'bridge-private.json',dict(source_messages=b.episode.source_messages(),generations=b.generations,
                calls=b.calls,public_history=b.history,prompt_and_history=b.episode.text(),reason=b.reason,final=b.final,
                closed=b.closed,close_verified=b.close_verified,transport=transport))
    # The AST guard performs no host execution. Semantics run only in a fresh,
    # isolated verifier with original tests and no extra agent-created files.
    unsupported=None; verifier_exit=None
    try:
        inspect_snapshot(case,snapshot)
    except (ValueError,SyntaxError) as exc:
        unsupported=type(exc).__name__+': '+str(exc)
    else:
        verifier_exit=verifier(panel,case,snapshot,output/'verifier')
        if verifier_exit not in (0,1): raise ValueError('verifier did not terminate with a test verdict')
    tested=any(c['request']['name']=='execute_bash' and c['request']['arguments'].get('command','').strip()==TEST_COMMAND
               and c.get('result',{}).get('exit_code')==0 for c in b.calls)
    success=b.final is not None and verifier_exit==0 and tested and unsupported is None
    result=dict(id=case['id'],family=case['family'],route=route,reason=b.reason,final=b.final,
        success=success,model_ran_passing_test_command=tested,verifier_exit=verifier_exit,
        verifier_unsupported=unsupported,snapshot=snapshot,transport=transport,
        seconds=time.monotonic()-began,assisted=False)
    publish(output/'result-private.json',result)
    return result


def measures(results, case_ids):
    expected={(route,c) for route in ('direct','pi') for c in case_ids}
    if len(case_ids)!=4 or len(set(case_ids))!=4 or len(results)!=8 or {(r['route'],r['id']) for r in results}!=expected:
        raise ValueError('repository result coverage')
    if any(type(r['success']) is not bool or r['family'] not in ('boundary','selection') for r in results):
        raise ValueError('repository result type')
    pi=[r for r in results if r['route']=='pi']; direct=[r for r in results if r['route']=='direct']
    if any(sum(r['family']==f for r in pi)!=2 for f in ('boundary','selection')):
        raise ValueError('repository family coverage')
    parity=all(r['success']==next(x['success'] for x in direct if x['id']==r['id']) for r in pi)
    capability=sum(r['success'] for r in pi)>=2 and all(any(r['success'] and r['family']==f for r in pi) for f in ('boundary','selection'))
    return dict(per_case_parity=parity,capability_gate_passed=capability)


def run(args):
    import torch
    import tiktoken
    from ndm.e97 import load_e97_checkpoint
    from scripts.audit_e97_live_actor_capture import fingerprint,runtime
    if sha(args.plan)!=args.plan_sha: raise ValueError('plan identity')
    p=json.loads(args.plan.read_text()); panel=p['panel']
    if p['max_model_episodes']!=8 or p['weight_mode']!='train' or sha(p['checkpoint'])!=CHECKPOINT_SHA:
        raise ValueError('fixed model budget/identity')
    if sha(panel['args_json'])!=panel['args_sha256']: raise ValueError('architecture args identity')
    for name in ('preflight_summary','preflight_audit'):
        if sha(p[name])!=p[name+'_sha256']: raise ValueError('preflight changed')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not visible or len(visible.split(','))!=1: raise ValueError('one leased GPU required')
    torch.cuda.set_device(0); torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded=load_e97_checkpoint(p['checkpoint'],args_json=panel['args_json'],device=torch.device('cuda',0),
                              dtype=torch.bfloat16,weight_mode='train',use_triton=True,mmap=True)
    loaded.model.eval(); encoding=tiktoken.get_encoding('p50k_base')
    if vocabulary(encoding)[1]!=panel['tokenizer_vocabulary_sha256']: raise ValueError('tokenizer identity')
    before=fingerprint(loaded.model)
    publish(args.output/'model-before-private.json',dict(fingerprint=before,runtime=runtime(loaded,0),checkpoint_sha256=CHECKPOINT_SHA))
    results=[]
    try:
        for index,case in enumerate(p['cases']):
            for route in p['routes'][index]:
                result=episode(loaded,case,panel,encoding,args.output/(route+'-'+case['id']),route,p['pi_bin'],
                               Path(__file__).resolve().parents[1]/'configs/pi/e97-openhands-compat.ts')
                results.append(result)
                print('PI_REPOSITORY_MODEL_EPISODE',route,case['id'],result['success'],result['reason'],flush=True)
        checks=measures(results,[c['id'] for c in p['cases']])
        publish(args.output/'summary.json',dict(schema=p['schema'],plan_sha256=args.plan_sha,episodes=len(results),
            results=[{k:r[k] for k in ('id','route','family','success','reason','verifier_exit','verifier_unsupported')} for r in results],
            **checks,measurements_complete=True,
            optimizer_updates=0,checkpoint_promotion=False,independent_repository_claim=False))
    finally:
        after=fingerprint(loaded.model); no_gradients=all(v.grad is None for v in loaded.model.parameters())
        bf16=all(v.dtype==torch.bfloat16 for v in loaded.model.parameters()); peak=torch.cuda.max_memory_allocated()
        publish(args.output/'model-after-private.json',dict(fingerprint=after,unchanged=before==after,no_gradients=no_gradients,
            all_parameters_bf16=bf16,peak_hbm_allocated=peak))
        if before!=after or not no_gradients or not bf16 or peak>=40*1024**3: raise ValueError('model state guard failed')


def main():
    os.umask(0o077)
    def interrupted(signum,frame): raise TimeoutError('repository model probe interrupted')
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze'); f.add_argument('--output',type=Path,required=True)
    r=sub.add_parser('run'); r.add_argument('--plan',type=Path,required=True); r.add_argument('--plan-sha',required=True); r.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    try: globals()[a.command](a)
    except BaseException as exc:
        if a.output.exists(): publish(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),optimizer_updates=0))
        raise


if __name__=='__main__': main()
