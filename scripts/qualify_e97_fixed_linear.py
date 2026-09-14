#!/usr/bin/env python3
"""One fixed-schedule candidate on captured QKV operands; no model dispatch."""
import argparse
import json
import os
from pathlib import Path
import statistics
import signal
import torch
import torch.nn.functional as F
from ndm.triton.fixed_fp32_linear import fixed_fp32_linear,_fixed_linear,KERNEL_ID
from scripts.e97_tensor_capsule import unpack
from scripts.e97_first_divergence import digest
from scripts.eval_e97_native_execution import publish,sha
from scripts.qualify_e97_packed_forward import save_private

ROOT=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining/linear-operands-v1')
CAPTURE_SHA='f53513d58ef4b6b12ee7b8f39dcbf950753feff7334fe46c695a1b68bd8cb47c'
REPLAY_SHA='80e2ff8a7a9ed5eb915f77c0951bf4122e370019ff7e1bedd828d4a0e2e9eb5d'


def reference():
    if sha(ROOT/'capture-summary.json')!=CAPTURE_SHA or sha(ROOT/'replay-summary.json')!=REPLAY_SHA:raise ValueError('actual replay authority')
    summary=json.loads((ROOT/'capture-summary.json').read_text())
    for profile,identity in summary['artifacts'].items():
        if sha(ROOT/f'{profile}-capsule-private.pt')!=identity['capsule_sha256'] or sha(ROOT/f'{profile}-outputs-private.pt')!=identity['outputs_sha256']:raise ValueError('capsule/results identity')
    return summary


def freeze(args):
    reference();args.output.mkdir(mode=0o700,parents=True,exist_ok=True)
    publish(args.output/'plan.json',dict(schema='e97-fixed-linear-micro-v1',kernel=KERNEL_ID,capture_sha256=CAPTURE_SHA,replay_sha256=REPLAY_SHA,
        heights=[1,512],rotations=[3,12],repetitions=2,maximum_gemms=100,maximum_compiled_actual_geometry_variants=1,
        fp32_atol=1e-5,fp32_rtol=1e-5,exact_height_and_row_permutation=True,maximum_median_slowdown=4.,
        warmup_pairs=3,timed_pairs=10,fp64_rows=[0,1,5,31,511],fp64_columns=[0,36,1023,5759,11519],
        maximum_hbm=2*1024**3,worker_seconds=300,outer_seconds=600,teardown_seconds=30,
        model_policy_integration=False,optimizer_updates=0,training_eligible=False,automatic_sweep=False,
        scope='One non-dispatched IEEE FP32 fixed-row kernel; captured QKV and small bias/tail cases. Not full-model or correctly rounded dot-product qualification.'))
    print('FIXED_LINEAR_FROZEN',sha(args.output/'plan.json'),flush=True)


def close(a,b):return bool(torch.all((a-b).abs()<=1e-5+1e-5*b.abs()))


def small_cuda_checks(device):
    generator=torch.Generator(device=device).manual_seed(4321)
    x=torch.randn(1,17,33,device=device,generator=generator).bfloat16().float()
    w=torch.randn(65,33,device=device,generator=generator).bfloat16().float()
    b=torch.randn(65,device=device,generator=generator).bfloat16().float()
    with torch.no_grad():
        full=fixed_fp32_linear(x,w,b)
        if not close(full,F.linear(x,w,b)):raise ValueError('bias/tail accuracy')
        for row in (0,3,12,16):
            one=fixed_fp32_linear(x[:,row:row+1],w,b)
            if digest(one.cpu())!=digest(full[:,row:row+1].cpu()):raise ValueError('bias/tail exact row independence')
    return True


def benchmark(x,w,b):
    functions={'native':F.linear,'fixed':fixed_fp32_linear};samples={name:[] for name in functions}
    with torch.no_grad():
        for _ in range(3):
            for fn in functions.values():fn(x,w,b)
        torch.cuda.synchronize()
        for repeat in range(10):
            order=('native','fixed') if repeat%2==0 else ('fixed','native')
            for name in order:
                start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                start.record();result=functions[name](x,w,b);end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end));del result
    medians={k:statistics.median(v) for k,v in samples.items()}
    return dict(samples_ms=samples,median_ms=medians,slowdown=medians['fixed']/medians['native'])


def run(args):
    if sha(args.output/'plan.json')!=args.plan_sha:raise ValueError('plan identity')
    recipe=json.loads((args.output/'plan.json').read_text());reference()
    if recipe['kernel']!=KERNEL_ID or int(os.environ.get('WORLD_SIZE','1'))!=1:raise ValueError('single candidate/worker')
    local=int(os.environ.get('LOCAL_RANK','0'));torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    capsule=torch.load(ROOT/'teacher-capsule-private.pt',map_location='cpu',weights_only=True)
    values=unpack(capsule,device);cpu=unpack(capsule,'cpu');del capsule
    x,w,b=values['x'],values['w'],values['b'];before={k:digest(None if v is None else v.cpu()) for k,v in values.items()}
    expected={p:torch.load(ROOT/f'{p}-outputs-private.pt',map_location='cpu',weights_only=True)['fp32'] for p in ('actor','teacher')}
    checks={};results={};timings={}
    try:
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            # Current native operation must still bind before candidate evaluation.
            for profile in ('actor','teacher'):
                inp=x[:,:1] if profile=='actor' else x
                native=F.linear(inp,w,b).cpu()
                if digest(native)!=digest(expected[profile]):raise ValueError('native precondition changed')
            for profile in ('actor','teacher'):
                inp=x[:,:1] if profile=='actor' else x;first=None
                for repeat in (0,1):
                    result=fixed_fp32_linear(inp,w,b).cpu()
                    row=dict(profile=profile,repeat=repeat,sha256=digest(result),finite=bool(torch.isfinite(result).all()),
                        close_to_native=close(result,expected[profile]),absolute_max=float((result-expected[profile]).abs().max()))
                    publish(args.output/f'{profile}-{repeat}.json',row)
                    if not row['finite'] or not row['close_to_native']:raise ValueError('candidate FP32 accuracy')
                    if first is not None and digest(first)!=digest(result):raise ValueError('candidate repeat differs')
                    first=result
                results[profile]=first;save_private(args.output/f'{profile}-candidate-private.pt',first)
            checks['height_exact']=digest(results['actor'])==digest(results['teacher'][:,:1])
            for rotation in (3,12):
                moved=torch.roll(x,rotation,1);expected_rotated=torch.roll(results['teacher'],rotation,1)
                for repeat in (0,1):
                    result=fixed_fp32_linear(moved,w,b).cpu();bound=digest(result)==digest(expected_rotated)
                    publish(args.output/f'rotation-{rotation}-{repeat}.json',dict(rotation=rotation,repeat=repeat,exact=bound,sha256=digest(result)))
                    if repeat==0:save_private(args.output/f'rotation-{rotation}-candidate-private.pt',result)
                    if not bound:raise ValueError('candidate row permutation differs')
                del moved,result
            checks['permutations_exact']=True
            cache=_fixed_linear.device_caches[local][0]
            checks['one_actual_kernel_variant']=len(cache)==1
            if not all(checks.values()):raise ValueError('height or compiled-kernel identity')
            kernel=next(iter(cache.values()))
            ptx=args.output/'actual-kernel.ptx'
            with ptx.open('x') as f:f.write(kernel.asm['ptx'])
            dots=[]
            for row in (0,1,5,31,511):
                for col in (0,36,1023,5759,11519):
                    target=float((cpu['x'][0,row].double()*cpu['w'][col].double()).sum())+(0. if cpu['b'] is None else float(cpu['b'][col]))
                    measured=float(results['teacher'][0,row,col]);error=abs(measured-target)
                    dots.append(dict(row=row,column=col,fp64=target,candidate=measured,error=error,passed=error<=1e-5+1e-5*abs(target)))
            publish(args.output/'fp64-dots-private.json',dict(rows=dots))
            if not all(d['passed'] for d in dots):raise ValueError('selected FP64 reference')
            checks['fp64_selected_dots']=True;checks['bias_tail']=small_cuda_checks(device)
            for height in (1,512):timings[str(height)]=benchmark(x[:,:height],w,b)
        after={k:digest(None if v is None else v.cpu()) for k,v in values.items()}
        checks['inputs_unchanged']=before==after
        checks['performance']=all(t['slowdown']<=4. for t in timings.values())
        checks['memory']=torch.cuda.max_memory_allocated(local)<=recipe['maximum_hbm']
        publish(args.output/'summary.json',dict(plan_sha256=args.plan_sha,kernel=KERNEL_ID,checks=checks,passed=all(checks.values()),
            timings=timings,peak_hbm_allocated=torch.cuda.max_memory_allocated(local),ptx_sha256=sha(ptx),
            optimizer_updates=0,training_eligible=False,model_policy_integration=False,scope=recipe['scope']))
        if not all(checks.values()):raise ValueError('fixed Linear micro gate failed')
        print('FIXED_LINEAR_MICRO_PASSED',flush=True)
    finally:
        publish(args.output/'terminal-safety.json',dict(peak_hbm_allocated=torch.cuda.max_memory_allocated(local),
            input_hashes_before=before,input_hashes_after={k:digest(None if v is None else v.cpu()) for k,v in values.items()},model_loaded=False,optimizer_updates=0))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('freeze','run'));p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-sha');a=p.parse_args()
    def terminate(_signum,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    try:globals()[a.command](a)
    except BaseException as exc:
        if a.output.exists():publish(a.output/f'{a.command}-failure.json',dict(type=type(exc).__name__,message=str(exc),training_eligible=False))
        raise
