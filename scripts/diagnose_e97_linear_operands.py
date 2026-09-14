#!/usr/bin/env python3
"""Capture the actual first QKV FP32 GEMM, then replay without a model."""
import argparse
import json
import os
from pathlib import Path
import signal
import torch
import torch.nn.functional as F
from scripts.e97_tensor_capsule import pack,unpack
from scripts.e97_first_divergence import digest
from scripts.diagnose_e97_early_prompt import reference,terminal_guard,PARAM_SHA
from scripts.qualify_e97_packed_forward import actor,teacher,exact,save_private
from scripts.e97_packed_forward_probe import assemble
from scripts.eval_e97_native_execution import publish,sha
from scripts.audit_e97_live_actor_capture import fingerprint

TRACE=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/early-prompt-trace-v1')
TRACE_SHA='406b8896763c337572ba2681b09c693a3466236adbf97623696a99f2d27eee02'
BANK_SHA={'actor':'441d3800b2e7f72960507bf5bf7e82d937442843684c07ec6eca61f11d9c284a',
          'teacher':'51e0d67f21bf9671d10e725004c55c5845c3480186bc61c5c4e868da9d5a1040'}
CAP=256*1024**2


def hashes(values):return {k:digest(None if v is None else v.detach().cpu()) for k,v in values.items()}


class LinearCapture:
    """Observe one selected module's actual F.linear call; no result substitution."""
    def __init__(self,module):
        self.module=module;self.active=False;self.complete=False;self.handles=[];self.capsule=None;self.fp32=None;self.body=None
        self.original=F.linear;self.observer=self.linear
    def __enter__(self):
        self.handles=[self.module.register_forward_pre_hook(self.before),self.module.register_forward_hook(self.after)]
        F.linear=self.observer;return self
    def before(self,_module,args):
        if self.complete:return None
        if self.active:raise ValueError('nested selected Linear')
        if args[0].dtype!=torch.bfloat16 or self.module.weight.dtype!=torch.bfloat16:raise ValueError('stored Linear dtype')
        self.active=True;return None
    def linear(self,x,w,bias=None):
        if not self.active:return self.original(x,w,bias)
        if self.capsule is not None:raise ValueError('multiple selected functional calls')
        if x.dtype!=torch.float32 or w.dtype!=torch.float32 or (bias is not None and bias.dtype!=torch.float32):raise ValueError('actual GEMM dtype')
        self.capsule=pack(dict(x=x,w=w,b=bias),CAP)
        inputs=unpack(self.capsule,'cpu',CAP);self.input_hashes=hashes(inputs);del inputs
        result=self.original(x,w,bias)
        if result.dtype!=torch.float32:raise ValueError('actual GEMM result dtype')
        if result.numel()*result.element_size()>CAP:raise ValueError('captured result byte limit')
        self.fp32=result.detach().cpu().clone()
        if hashes(dict(x=x,w=w,b=bias))!=self.input_hashes:raise ValueError('GEMM mutated inputs')
        return result
    def after(self,_module,_args,result):
        if not self.active:return None
        if self.fp32 is None or result.dtype!=torch.bfloat16:raise ValueError('selected Linear result')
        self.body=result.detach().cpu().clone();self.active=False;self.complete=True
        if digest(self.fp32.bfloat16())!=digest(self.body):raise ValueError('native BF16 store differs from FP32 conversion')
        return None
    def __exit__(self,*_args):
        for handle in self.handles:handle.remove()
        if F.linear is not self.observer:raise ValueError('functional observer ownership')
        F.linear=self.original
        self.observer=None;self.handles=[]  # break the bound-method self-cycle promptly


def references():
    base,data,a,t=reference()
    if sha(TRACE/'summary.json')!=TRACE_SHA:raise ValueError('trace reference summary')
    refs={}
    for profile in ('actor','teacher'):
        p=TRACE/f'fine-{profile}-tensors-private.pt'
        if sha(p)!=BANK_SHA[profile]:raise ValueError('trace bank identity')
        bank=torch.load(p,map_location='cpu',weights_only=True)['bank']
        refs[profile]={io:bank[f'qkv_proj.{io}'] for io in ('input','output')}
    return base,data,a,t,refs


def freeze(args):
    base,_,_,_,_=references();args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    publish(args.output/'plan.json',dict(schema='e97-linear-operands-v1',trace_summary_sha256=TRACE_SHA,trace_bank_sha256=BANK_SHA,
        numerical_policy=base['numerical_policy'],module='layer0.qkv_proj',geometries=[1,512],in_features=3840,out_features=11520,
        full_model_evaluations=4,repetitions=2,model_free_gemms=8,capsule_bytes=CAP,max_capture_cpu_bytes=2*1024**3,
        capture_hbm=16*1024**3,replay_hbm=2*1024**3,capture_seconds=900,replay_seconds=300,outer_seconds=1500,teardown_seconds=30,
        binding='exact original actor8 scores/full teacher receipt and sampled BF16 QKV trace; repeats exact for all FP32 operands/results and BF16 stores',
        controls='original capsules twice each; then one shared teacher input/weight allocation, row counts1/512 twice each, no changed numeric entries',
        optimizer_updates=0,training_eligible=False,automatic_expansion=False,
        scope='Actual transient FP32 Linear operands and pre-store outputs; no new precision policy, full FP32 model, backward, or packed64K work.'))
    print('LINEAR_OPERANDS_FROZEN',sha(args.output/'plan.json'),flush=True)


def plan(args):
    if sha(args.output/'plan.json')!=args.plan_sha:raise ValueError('plan identity')
    result=json.loads((args.output/'plan.json').read_text())
    if result['schema']!='e97-linear-operands-v1':raise ValueError('plan schema')
    if int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('one worker')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local)
    return result,local


def capture(args):
    from ndm.e97 import load_e97_checkpoint
    from ndm.numerical_policy import configure_numerical_policy
    from ndm.triton.e88_triton_forward import _AUTOTUNE_CACHE
    recipe,local=plan(args);base,data,actor_ref,teacher_ref,refs=references();target=base['model']
    for path,expected in ((target['checkpoint'],target['sha256']),(base['args_json'],base['args_sha256'])):
        if sha(path)!=expected:raise ValueError('checkpoint/args identity')
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=base['args_json'],device=torch.device('cuda',local),dtype=torch.bfloat16,
        weight_mode=target['mode'],use_triton=True,mmap=True)
    model=loaded.model;configure_numerical_policy(model,base['numerical_policy']);before=fingerprint(model)
    if before['parameters']!=PARAM_SHA or any(p.dtype!=torch.bfloat16 for p in model.parameters()):raise ValueError('persistent parameter identity')
    receipts={};artifacts={}
    with terminal_guard(model,args.output,before,lambda:torch.cuda.max_memory_allocated(local),_AUTOTUNE_CACHE,recipe['capture_hbm']):
        module=model.layers[0].mixer.qkv_proj
        if module.weight.shape!=(11520,3840) or len(model.layers)!=18:raise ValueError('production geometry')
        for profile in ('actor','teacher'):
            for repeat in (0,1):
                with LinearCapture(module) as observer:
                    if profile=='actor':native=actor(loaded,data['records'][0],data['probes'][0]['boundary'])
                    else:native=teacher(model,assemble(data['records'],[0],2656),data['probes'],[0],torch.device('cuda',local))
                if not observer.complete:raise ValueError('no selected operation captured')
                v=unpack(observer.capsule,'cpu',CAP);height=1 if profile=='actor' else 512
                if v['x'].shape!=(1,height,3840) or v['w'].shape!=(11520,3840):raise ValueError('captured geometry')
                if any(not bool(torch.isfinite(z).all()) for z in v.values() if z is not None):raise ValueError('nonfinite operand')
                # Bind transient cast to the unchanged BF16 model parameter and source input trace.
                if digest(v['w'].bfloat16())!=digest(module.weight.detach().cpu()) or digest(v['w'].bfloat16().float())!=digest(v['w']):raise ValueError('weight cast binding')
                count=min(height,6)
                if digest(v['x'][0,:count])!=digest(refs[profile]['input'][:count].float()):raise ValueError('actual FP32 input cast/trace binding')
                del v
                row=dict(native=native,reference_bound=exact(native,actor_ref if profile=='actor' else teacher_ref),
                    trace_bound=digest(observer.body[0,:count])==digest(refs[profile]['output'][:count]),input_hashes=observer.input_hashes,
                    fp32_sha256=digest(observer.fp32),bf16_sha256=digest(observer.body),
                    fp32_finite=bool(torch.isfinite(observer.fp32).all()),bf16_store_exact=True,plan_sha256=args.plan_sha)
                publish(args.output/f'{profile}-{repeat}-private.json',row)
                if repeat==0:
                    capfile=args.output/f'{profile}-capsule-private.pt';out=args.output/f'{profile}-outputs-private.pt'
                    save_private(capfile,observer.capsule);save_private(out,dict(fp32=observer.fp32,bf16=observer.body))
                    artifacts[profile]=dict(capsule_sha256=sha(capfile),outputs_sha256=sha(out));receipts[profile]=row
                elif not exact(row,receipts[profile]):raise ValueError('captured operation or native repeat differs')
                if not all(row[k] for k in ('reference_bound','trace_bound','fp32_finite')):raise ValueError('capture binding failed; uninterpretable')
                del observer
            print('LINEAR_CAPTURE_BOUND',profile,flush=True)
    publish(args.output/'capture-summary.json',dict(plan_sha256=args.plan_sha,artifacts=artifacts,all_native_and_trace_bindings=True,
        all_repeats_exact=True,bf16_stores_exact=True,terminal_safety_sha256=sha(args.output/'terminal-safety.json'),optimizer_updates=0,training_eligible=False))


def replay(args):
    recipe,local=plan(args);summary=json.loads((args.output/'capture-summary.json').read_text())
    if summary['plan_sha256']!=args.plan_sha or not all(summary[k] for k in ('all_native_and_trace_bindings','all_repeats_exact','bf16_stores_exact')):raise ValueError('capture precondition')
    safety=args.output/'terminal-safety.json'
    if sha(safety)!=summary['terminal_safety_sha256'] or not json.loads(safety.read_text())['passed']:raise ValueError('capture safety')
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    expected={};capsules={};receipts=[]
    try:
        for profile in ('actor','teacher'):
            capfile=args.output/f'{profile}-capsule-private.pt';out=args.output/f'{profile}-outputs-private.pt'
            if sha(capfile)!=summary['artifacts'][profile]['capsule_sha256'] or sha(out)!=summary['artifacts'][profile]['outputs_sha256']:raise ValueError('captured artifacts identity')
            capsules[profile]=torch.load(capfile,map_location='cpu',weights_only=True)
            expected[profile]=torch.load(out,map_location='cpu',weights_only=True)
        def measure(v,profile,phase,repeat):
            before=hashes(v)
            with torch.no_grad(),torch.autocast('cuda',enabled=False):
                result=F.linear(v['x'],v['w'],v['b'])
                bf16=result.bfloat16()
            row=dict(phase=phase,profile=profile,repeat=repeat,input_hashes=before,
                fp32_sha256=digest(result.cpu()),bf16_sha256=digest(bf16.cpu()),
                x_address=v['x'].data_ptr(),w_address=v['w'].data_ptr(),x_shape=list(v['x'].shape),x_stride=list(v['x'].stride()))
            row['fp32_bound']=row['fp32_sha256']==digest(expected[profile]['fp32']);row['bf16_bound']=row['bf16_sha256']==digest(expected[profile]['bf16'])
            publish(args.output/f'replay-{phase}-{profile}-{repeat}.json',row);receipts.append(row)
            if hashes(v)!=before:raise ValueError('micro GEMM input mutation')
            if not row['fp32_bound'] or not row['bf16_bound']:raise ValueError('micro result not bound to actual captured operation')
        for profile in ('actor','teacher'):
            v=unpack(capsules[profile],torch.device('cuda',local),CAP)
            for repeat in (0,1):measure(v,profile,'original',repeat)
            del v
        a=unpack(capsules['actor'],'cpu',CAP);b=unpack(capsules['teacher'],'cpu',CAP)
        if digest(a['w'])!=digest(b['w']) or digest(a['b'])!=digest(b['b']) or digest(a['x'])!=digest(b['x'][:,:1]):raise ValueError('common allocation eligibility')
        del a,b
        common=unpack(capsules['teacher'],torch.device('cuda',local),CAP)
        for profile in ('actor','teacher'):
            v=dict(common);v['x']=common['x'][:,:1] if profile=='actor' else common['x']
            for repeat in (0,1):measure(v,profile,'common',repeat)
        common_rows=[r for r in receipts if r['phase']=='common']
        if len({r['x_address'] for r in common_rows})!=1 or len({r['w_address'] for r in common_rows})!=1 or len({tuple(r['x_stride']) for r in common_rows})!=1:raise ValueError('common allocations/strides changed')
        fp32_diff=expected['actor']['fp32'][0,0]-expected['teacher']['fp32'][0,0]
        if torch.cuda.max_memory_allocated(local)>recipe['replay_hbm']:raise ValueError('micro HBM bound')
        publish(args.output/'replay-summary.json',dict(plan_sha256=args.plan_sha,all_original_replays_exact=True,common_allocation_replays_exact=True,
            fp32_first_row_differing_elements=int((fp32_diff!=0).sum()),fp32_first_row_absolute_max=float(fp32_diff.abs().max()),
            bf16_stores_exact=True,peak_hbm_allocated=torch.cuda.max_memory_allocated(local),optimizer_updates=0,training_eligible=False,
            scope='Same numeric input row, weights, addresses and strides; matrix height controls pre-store arithmetic including backend/output-allocation choices. No instruction-level cause or repair claim.'))
    finally:
        peak=torch.cuda.max_memory_allocated(local)
        publish(args.output/'replay-safety.json',dict(peak_hbm_allocated=peak,memory_passed=peak<=recipe['replay_hbm'],model_loaded=False,optimizer_updates=0))
        if peak>recipe['replay_hbm']:raise ValueError('micro HBM bound')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','capture','replay'));p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-sha');a=p.parse_args()
    def terminate(_signum,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    try:globals()[a.command](a)
    except BaseException as exc:
        if a.output.exists():publish(a.output/f'{a.command}-failure.json',dict(type=type(exc).__name__,message=str(exc),training_eligible=False))
        raise
