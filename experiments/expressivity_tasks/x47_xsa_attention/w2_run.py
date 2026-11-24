"""Gate CPU/GPU replay before measuring real native-branch workbench vs fixed join."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import subprocess
from time import perf_counter_ns

import torch

from .workbench import Meter
from .workbench_model import W,D
from .workbench_native import EXE_SHA, Live
from .workbench_run import check_gpu_owner, dump, log, quantiles_ns, sha
from .workbench_witness import close, digest_state, require
from .w2_join import Episode, FixedJoin, Neural, ReplicatedMount, array
from .w2_panel import TAGS, WORLDS, analytic, initial_model, panel, parity, prefill, prompt


def sync(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def sample(policy,bank,world,tag,diagnostic):
    stream=prompt(tag); src=policy.device
    start=perf_counter_ns()
    with Live() as live:
        e=Episode(policy,live,bank); before=Meter(src); current=Meter(src,profile=diagnostic)
        sync(src); t0=perf_counter_ns(); prefill(e,stream,before); sync(src); t1=perf_counter_ns()
        p,c=e.predict(current); e.observe(c,current); sync(src); t2=perf_counter_ns()
        expected,gold=analytic(WORLDS[world],tag,True); close(array(p),expected)
        require(c==gold and before.reads==0 and current.reads==3,'unmatched timed episode')
        require((before.passes,current.passes)==((len(stream),3) if isinstance(policy,Neural) else (0,0)), 'call count')
        return dict(world=world,tag=tag,byte=c,setup_ns=t0-start,prefill_ns=t1-t0,current_ns=t2-t1,
                    prefill_passes=before.passes,current_passes=current.passes,current_rpcs=current.reads,
                    stages=current.stages)


@torch.inference_mode()
def bench(model,out):
    rows=[]; opens=[]; arms={'full':Neural(model,'full'),'fixed':FixedJoin()}
    with ExitStack() as stack, (out/'samples.jsonl').open('x') as f:
        banks=[]
        for docs in WORLDS:
            t=perf_counter_ns(); banks.append(stack.enter_context(ReplicatedMount(docs))); opens.append(perf_counter_ns()-t)
        for block,repeats in (('warmup',1),('primary',8),('diagnostic',1)):
            for rep in range(repeats):
                for w in range(2):
                    for j,tag in enumerate(TAGS):
                        order=('full','fixed') if (rep+j+w)%2==0 else ('fixed','full')
                        for name in order:
                            r=sample(arms[name],banks[w],w,tag,block=='diagnostic')
                            r.update(arm=name,block=block,rep=rep); rows.append(r); f.write(json.dumps(r)+'\n')
            f.flush(); log(f'{block}: {repeats*4} complete cases per arm')
    summary={}
    for name in arms:
        prim=[r for r in rows if r['arm']==name and r['block']=='primary']
        diag=[r for r in rows if r['arm']==name and r['block']=='diagnostic']
        summary[name]={phase:quantiles_ns([r[phase+'_ns'] for r in prim]) for phase in ('setup','prefill','current')}
        stages=sorted({s for r in diag for s,_ in r['stages']})
        summary[name]['diagnostic_stage_total']={s:quantiles_ns([sum(v for k,v in r['stages'] if k==s)
                                                      for r in diag]) for s in stages}
    return dict(arms=summary,open_two_replicas_ns=opens)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('stage',choices=('cpu','gpu'))
    ap.add_argument('out',type=Path); ap.add_argument('--cpu-from',type=Path); a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    require(os.environ.get('OMP_NUM_THREADS')=='4','set OMP_NUM_THREADS=4')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    ref=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    model=initial_model()
    meta=dict(ref=ref,exe_sha=EXE_SHA,state=digest_state(model),host=platform.node(),
              torch=str(torch.__version__),slots=W,width=D,params=sum(p.numel() for p in model.parameters()))
    if a.stage=='cpu':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','hide CUDA for CPU gate')
        trace=panel(model); path=a.out/'initial.pt'; torch.save(model.state_dict(),path)
        dump(a.out/'cpu.json',dict(**meta,trace=trace,file_sha=sha(path))); log('CPU panel passed: 16 crossed cases')
    else:
        meta['lease']=check_gpu_owner(); meta['card']=torch.cuda.get_device_name(0); meta['cuda']=torch.version.cuda
        require(a.cpu_from is not None,'need CPU replay'); cpu=json.loads((a.cpu_from/'cpu.json').read_text())
        path=a.cpu_from/'initial.pt'; require(cpu['ref']==ref and sha(path)==cpu['file_sha'],'different CPU run')
        model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True)); model.to('cuda')
        require(digest_state(model)==cpu['state']==meta['state'],'device weights differ')
        trace=panel(model); err=parity(cpu['trace'],trace)
        dump(a.out/'gpu.json',dict(**meta,trace=trace,cpu_gpu_max_error=err)); log(f'GPU panel passed; max CPU/GPU error {err:.3g}')
        summary=bench(model,a.out); dump(a.out/'timing.json',dict(**meta,**summary)); log('all clocks complete')


if __name__=='__main__':
    main()
