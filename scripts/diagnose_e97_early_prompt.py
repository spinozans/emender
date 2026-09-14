#!/usr/bin/env python3
"""Reference-bound six-row localization of record0's early-prompt discrepancy."""
import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import torch
from scripts.e97_first_divergence import Rows,compare,digest
from scripts.e97_packed_forward_probe import assemble
from scripts.qualify_e97_packed_forward import actor,teacher,exact,save_private
from scripts.eval_e97_native_execution import publish,sha
from scripts.audit_e97_live_actor_capture import fingerprint

BASE=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/packed-forward-v1')
BASE_SHA='af9339afe3819cd495ec1a052b77c5d46761bb060ac0d910ac0f73b2240d1fbf'
INPUT_SHA='7340c9d573acce2b1f9e79c77cc25a3130f4759c80726cab72f13ca00e9da7b9'
ACTOR_SHA='aa16f806993d852e1e0a8fd5b4478e0ea852965d119abf5a7fb65a959e1db99b'
TEACHER_SHA='e5eae759930f2bd1bae3b1368297ef0f1526c0b39acb3a37802f9465f769c491'
PARAM_SHA='f57eaff2882ce2914413f501a93454e8e0a9bfcd4f408b91ed962405c7ad16bd'
CAUSAL_ROWS=6


def reference():
    if sha(BASE/'plan.json')!=BASE_SHA or sha(BASE/'inputs-private.pt')!=INPUT_SHA:raise ValueError('packed baseline authority')
    for repeat in (0,1):
        a=BASE/f'actor-0-boundary-{repeat}-private.json';t=BASE/f'standalone-0-{repeat}-private.json'
        if sha(a)!=ACTOR_SHA or sha(t)!=TEACHER_SHA:raise ValueError('repeated native reference')
    plan=json.loads((BASE/'plan.json').read_text());data=torch.load(BASE/'inputs-private.pt',map_location='cpu',weights_only=True)
    actor_ref=json.loads(a.read_text())['logprobs'];teacher_ref=json.loads(t.read_text());teacher_ref.pop('plan_sha256')
    if data['probes'][0]['boundary']!=list(range(8)) or len(data['records'][0]['tokens'])!=2657:raise ValueError('record0 selection')
    return plan,data,actor_ref,teacher_ref


def freeze(args):
    base,_,_,_=reference();args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    publish(args.output/'plan.json',dict(schema='e97-early-prompt-trace-v1',base_plan_sha256=BASE_SHA,
        inputs_sha256=INPUT_SHA,actor_sha256=ACTOR_SHA,teacher_sha256=TEACHER_SHA,
        record=0,prediction_row=5,causal_rows=CAUSAL_ROWS,numerical_policy=base['numerical_policy'],
        teacher_tokens=2657,teacher_input_steps=2656,actor_scores=8,
        phases=['coarse','first-differing-layer'],profiles=['actor','teacher'],repetitions=2,maximum_evaluations=8,
        maximum_fp64_selected_dots=3,trace_bytes_per_bank=16*1024**2,maximum_live_trace_bytes=64*1024**2,
        max_hbm_allocated=16*1024**3,worker_seconds=900,outer_seconds=1500,teardown_seconds=30,
        binding='exact8 actor scores and full original teacher receipt:40 scores, hidden/logit hashes, CE and18 state hashes',
        terminal_safety='parameter/buffer fingerprint, gradients, tuner and HBM on ordinary success/failure; SIGTERM attempts orderly unwinding; SIGKILL cannot be audited in-memory',
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope='Early tokenwise prompt diagnostic; unchanged complete native executions. Not packed64K, new precision policy, or acceptance-threshold change.'))
    print('EARLY_PROMPT_FROZEN',sha(args.output/'plan.json'),flush=True)


@contextmanager
def terminal_guard(model,output,before,peak,tuner,limit):
    try:yield
    finally:
        report=dict(before=before,passed=False)
        try:
            after=fingerprint(model);used=peak()
            checks=dict(parameters_buffers_unchanged=before==after,no_gradients=all(p.grad is None for p in model.parameters()),
                no_autotune=not tuner,memory=used<=limit)
            report.update(after=after,checks=checks,peak_hbm_allocated=used,passed=all(checks.values()))
        except Exception as exc:report['audit_error']=dict(type=type(exc).__name__,message=str(exc))
        publish(output/'terminal-safety.json',report)
        if not report['passed']:raise RuntimeError('terminal forward safety audit failed')


def ordered_first(rows):
    differing=[(i,r) for i,r in enumerate(rows) if not r['exact']]
    first=differing[0][1] if differing else None
    temporal=min(differing,key=lambda pair:(pair[1]['first_token'] if pair[1]['first_token'] is not None else -1,pair[0]))[1] if differing else None
    return first,temporal


def evaluate(loaded,data,profile,layer,budget):
    trace=Rows(loaded.model,CAUSAL_ROWS,layer,max_bytes=budget)
    try:
        if profile=='actor':native=actor(loaded,data['records'][0],data['probes'][0]['boundary'])
        else:
            batch=assemble(data['records'],[0],2656)
            native=teacher(loaded.model,batch,data['probes'],[0],next(loaded.model.parameters()).device)
        bank=trace.finish()
        receipt=dict(native=native,tensor_sha256={k:digest(v) for k,v in bank.items()},
            calls=trace.calls,site_rows=trace.counts,trace_payload_bytes=trace.bytes)
        return bank,receipt
    finally:trace.close()


def round_bf16(value):
    # Avoid a double->float32->BF16 intermediate rounding in framework casts.
    if not math.isfinite(value) or abs(value)>=2.**127:raise ValueError('selected reference magnitude')
    if value==0:return value
    exponent=math.frexp(abs(value))[1]
    unit=math.ldexp(1.,max(exponent-8,-133))
    return math.copysign(round(abs(value)/unit)*unit,value)


def selected_dot(module,x,actor_value,teacher_value,column):
    if x.dtype!=torch.bfloat16 or module.weight.dtype!=torch.bfloat16 or x.numel()>11520:raise ValueError('selected dot dtype/bounds')
    w=module.weight[column].detach().cpu();x=x.detach().cpu()
    bias=None if module.bias is None else module.bias[column].detach().cpu()
    value=float((x.double()*w.double()).sum().item())+(0. if bias is None else float(bias.double().item()))
    rounded=round_bf16(value)
    return dict(fp64_dot=value,bf16_rounded=rounded,actor_value=float(actor_value),teacher_value=float(teacher_value),
        actor_matches_rounded=float(actor_value)==rounded,teacher_matches_rounded=float(teacher_value)==rounded,
        input_sha256=digest(x),weight_row_sha256=digest(w),input_features=x.numel()),dict(input=x,weight=w,bias=bias)


def dot_audit(model,layer,a,b):
    wrapper=model.layers[layer];m=wrapper.mixer
    names=['qkv_proj','a_proj','g_proj','erase_gate_proj','value_write_gate_proj','o_proj','mlp.w1','mlp.w2','mlp.w3']
    candidates=[]
    for order,name in enumerate(names):
        ins=name+'.input';outs=name+'.output'
        for token in range(CAUSAL_ROWS):
            if digest(a[ins][token])!=digest(b[ins][token]):continue
            u=a[outs][token];v=b[outs][token]
            bits=(u.contiguous().view(torch.uint8)!=v.contiguous().view(torch.uint8)).reshape(u.numel(),u.element_size()).any(-1)
            if bool(bits.any()):
                column=int(bits.nonzero()[0,0]);candidates.append((token,order,name,column));break
    reports=[];operands={}
    for token,_,name,column in sorted(candidates)[:3]:
        module=getattr(wrapper.mlp,name.split('.')[1]) if name.startswith('mlp.') else getattr(m,name)
        result,values=selected_dot(module,a[name+'.input'][token],a[name+'.output'][token,column],b[name+'.output'][token,column],column)
        result.update(site=name,token=token,column=column);reports.append(result);operands[f'{name}:{token}:{column}']=values
    return reports,operands


def run(args):
    from ndm.e97 import load_e97_checkpoint
    from ndm.numerical_policy import configure_numerical_policy
    from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
    if sha(args.output/'plan.json')!=args.plan_sha:raise ValueError('plan identity')
    plan=json.loads((args.output/'plan.json').read_text());base,data,actor_ref,teacher_ref=reference()
    if plan['schema']!='e97-early-prompt-trace-v1' or plan['causal_rows']!=6:raise ValueError('trace scope')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('one leased worker')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);target=base['model']
    for path,expected in ((target['checkpoint'],target['sha256']),(base['args_json'],base['args_sha256'])):
        if sha(path)!=expected:raise ValueError('model/argument identity')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=base['args_json'],device=torch.device('cuda',local),
        dtype=torch.bfloat16,weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;configure_numerical_policy(model,base['numerical_policy']);before=fingerprint(model)
    if before['parameters']!=PARAM_SHA or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('persistent parameters')
    results=[];layer=None;cross_phase={};dots=[]
    with terminal_guard(model,args.output,before,lambda:torch.cuda.max_memory_allocated(local),_AUTOTUNE_CACHE,plan['max_hbm_allocated']):
        if len(model.layers)!=18 or any(m.mixer.projection_chunk_size!=512 for m in model.layers):raise ValueError('geometry')
        for phase in ('coarse','fine'):
            banks={}
            for profile in ('actor','teacher'):
                for repeat in range(2):
                    bank,receipt=evaluate(loaded,data,profile,layer,plan['trace_bytes_per_bank'])
                    receipt.update(phase=phase,profile=profile,repeat=repeat,plan_sha256=args.plan_sha,
                        reference_bound=exact(receipt['native'],actor_ref if profile=='actor' else teacher_ref))
                    publish(args.output/f'{phase}-{profile}-{repeat}-private.json',receipt)
                    if not receipt['reference_bound']:raise ValueError('native reference binding failed; trace not interpretable')
                    if repeat==0:
                        save_private(args.output/f'{phase}-{profile}-tensors-private.pt',dict(plan_sha256=args.plan_sha,bank=bank));banks[profile]=bank
                    else:
                        if receipt['tensor_sha256']!={k:digest(v) for k,v in banks[profile].items()}:raise ValueError('trace repeat mismatch')
                        del bank
            rows=compare(banks['actor'],banks['teacher']);first,temporal=ordered_first(rows)
            result=dict(phase=phase,layer=layer,first_site=first,earliest_token_site=temporal,sites=rows,references_exact=True,repeats_exact=True)
            publish(args.output/f'{phase}-comparison.json',result);results.append(result)
            if phase=='coarse' and first is not None and '.block.' in first['site']:
                layer=int(first['site'].split('.')[0])
                cross_phase={p:{io:digest(bank[f'{layer:02d}.block.{io}']) for io in ('input','output')} for p,bank in banks.items()}
            elif phase=='fine':
                if any(digest(banks[p][f'block.{io}'])!=cross_phase[p][io] for p in banks for io in ('input','output')):raise ValueError('cross-phase block binding')
                dots,operands=dot_audit(model,layer,banks['actor'],banks['teacher'])
                publish(args.output/'selected-dot-private.json',dict(plan_sha256=args.plan_sha,results=dots,scope='CPU FP64 selected-dot reference after exact binding; not full FP64 model or a repair'))
                save_private(args.output/'selected-dot-operands-private.pt',operands)
            print('EARLY_PROMPT_BOUND',json.dumps(dict(phase=phase,first_site=None if first is None else first['site'],earliest_token=None if temporal is None else temporal['first_token'])),flush=True)
            del banks
            if phase=='coarse' and (first is None or '.block.' not in first['site']):break
    publish(args.output/'summary.json',dict(schema='e97-early-prompt-result-v1',plan_sha256=args.plan_sha,
        localization_ready=True,all_references_exact=True,all_trace_repeats_exact=True,
        phases=[{k:r[k] for k in ('phase','layer','first_site','earliest_token_site')} for r in results],
        fp64_selected_dot_cases=len(dots),terminal_safety_sha256=sha(args.output/'terminal-safety.json'),
        numerical_gate_still_failed=True,optimizer_updates=0,training_eligible=False,scope=plan['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-sha')
    a=p.parse_args()
    def terminate(_signum,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    try:globals()[a.command](a)
    except BaseException as exc:
        if a.output.exists():publish(a.output/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),localization_ready=False,training_eligible=False))
        raise
