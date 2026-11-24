"""W1 repeatable fixture gates and paired clocks. No optimizer or Pile access."""
import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns

import numpy as np
import torch

from .workbench import Decoder, Direct, Meter
from .workbench_model import W, D
from .workbench_native import Archive, Live, EXE_SHA
from .workbench_witness import (PREFIX, WORLDS, close, compare_traces, digest_state,
                                initial_model, oracle, require, witness)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj,allow_nan=False)+'\n'); tmp.replace(path)


def log(s):
    print(s, file=sys.stderr, flush=True)


def check_gpu_owner():
    card = os.environ.get('GPU_LEASE_HELD')
    require(card in ('0','1') and os.environ.get('CUDA_VISIBLE_DEVICES') == card, 'need physical 0/1 lease')
    f = Path(os.environ.get('GPU_LEASE_DIR','.wg/gpu_leases')) / card
    lease=dict(line.split('=',1) for line in f.read_text().splitlines())
    require(lease['adopted']=='1' and lease['gpu']==card and lease['host']==platform.node(), 'invalid lease')
    ancestor=os.getpid()
    while ancestor > 1:
        fields=Path(f'/proc/{ancestor}/stat').read_text().rsplit(') ',1)[1].split()
        if ancestor==int(lease['pid']):
            require(fields[19]==lease['starttime'],'owner starttime changed'); break
        ancestor=int(fields[1])
    else:
        raise RuntimeError('not a child of live lease owner')
    require(torch.cuda.device_count()==1,'expected one logical device')
    return lease


ARMS=('rebuilt','static','direct_cpu')


def pair(model, archive, world, arm, profile):
    start=perf_counter_ns()
    with Live(PREFIX) as live:
        engine = Direct(live,archive) if arm=='direct_cpu' else Decoder(model,live,archive,speculate=arm=='rebuilt')
        dev = torch.device('cpu') if arm=='direct_cpu' else next(model.parameters()).device
        meters=[Meter(dev,profile=profile) for _ in range(2)]
        if dev.type=='cuda':
            torch.cuda.synchronize(dev)
        t0=perf_counter_ns(); setup=t0-start
        outputs=[]; walls=[]
        for meter in meters:
            t=perf_counter_ns()
            p,byte=engine.predict(meter)
            engine.observe(byte,meter)  # actual greedy generation, never supplied by oracle
            if dev.type=='cuda':
                torch.cuda.synchronize(dev)
            walls.append(perf_counter_ns()-t); outputs.append((p,byte))
        total=perf_counter_ns()-t0
        generated=bytes(byte for _,byte in outputs)
        require(generated==(b'de' if world==0 else b'ce'),'generation mismatch')
        for i,(p,_) in enumerate(outputs):
            close(p.cpu().numpy() if torch.is_tensor(p) else p, oracle(PREFIX+generated[:i],WORLDS[world]))
        return dict(world=world,arm=arm,generated=generated.hex(),setup_ns=setup,pair_ns=total,
                    byte_ns=walls, rpcs=[m.reads for m in meters], passes=[m.passes for m in meters],
                    stages=[m.stages for m in meters] if profile else [])


def quantiles_ns(values):
    v=np.asarray(values)*1e-6
    return dict(n=len(v), p50_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),mean_ms=float(v.mean()))


@torch.inference_mode()
def bench(model, out):
    samples=[]; opens=[]
    with ExitStack() as stack, (out/'samples.jsonl').open('x') as sink:
        banks=[]
        for docs in WORLDS:
            t=perf_counter_ns(); banks.append(stack.enter_context(Archive(docs))); opens.append(perf_counter_ns()-t)
        for block,n in (('warmup',4),('primary',32),('diagnostic',8)):
            for i in range(n):
                world=i%2
                for arm in ARMS[i%3:]+ARMS[:i%3]:
                    sample=pair(model,banks[world],world,arm,block=='diagnostic')
                    sample.update(block=block,pair_index=i)
                    sink.write(json.dumps(sample)+'\n'); samples.append(sample)
            sink.flush(); log(f'{block}: {n} complete pairs per arm')
    report={}
    for arm in ARMS:
        rows=[r for r in samples if r['block']=='primary' and r['arm']==arm]
        diag=[r for r in samples if r['block']=='diagnostic' and r['arm']==arm]
        names=sorted({k for r in diag for b in r['stages'] for k,_ in b})
        # Diagnostic numbers are per emitted byte including zeros for a stage not taken.
        report[arm]=dict(byte=quantiles_ns([v for r in rows for v in r['byte_ns']]),
                         first=quantiles_ns([r['byte_ns'][0] for r in rows]),
                         second=quantiles_ns([r['byte_ns'][1] for r in rows]),
                         pair=quantiles_ns([r['pair_ns'] for r in rows]),
                         setup=quantiles_ns([r['setup_ns'] for r in rows]),
                         rpcs_per_pair=sum(sum(r['rpcs']) for r in rows)/len(rows),
                         passes_per_pair=sum(sum(r['passes']) for r in rows)/len(rows),
                         diagnostic_stages={k:quantiles_ns([sum(ns for key,ns in b if key==k)
                             for r in diag for b in r['stages']]) for k in names})
    return dict(arms=report,archive_open_seal_ns=opens)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=('cpu','gpu')); ap.add_argument('out',type=Path)
    ap.add_argument('--cpu-from',type=Path); args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    require(os.environ.get('OMP_NUM_THREADS')=='4','require OMP_NUM_THREADS=4')
    torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    ref=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    meta=dict(ref=ref,torch=str(torch.__version__),python=sys.version.split()[0],host=platform.node(),
              exe_sha=EXE_SHA,slots=W,width=D,threads=torch.get_num_threads(),dtype='float32')
    if args.stage=='cpu':
        require(os.environ.get('CUDA_VISIBLE_DEVICES')=='', 'hide CUDA for reference')
        model=initial_model(); log('CPU independent oracle + native trace')
        trace=witness(model); weights=args.out/'initial.pt'; torch.save(model.state_dict(),weights)
        dump(args.out/'cpu.json',dict(**meta,params=sum(p.numel() for p in model.parameters()),
             state=digest_state(model),file_sha=sha(weights),trace=trace))
        log(f"CPU gate: 8 cases, max oracle error {trace['max_oracle_error']:.3g}")
    else:
        require(args.cpu_from is not None,'CPU receipt required')
        cpu=json.loads((args.cpu_from/'cpu.json').read_text()); weights=args.cpu_from/'initial.pt'
        require(cpu['ref']==ref and cpu['file_sha']==sha(weights),'CPU run/weights mismatch')
        meta['lease']=check_gpu_owner(); meta['card']=torch.cuda.get_device_name(0); meta['cuda']=torch.version.cuda
        model=initial_model(); model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
        model.to('cuda'); require(digest_state(model)==cpu['state'],'different device weights')
        log('leased GPU: checking actual trace before timing')
        trace=witness(model); error=compare_traces(cpu['trace'],trace)
        dump(args.out/'parity.json',dict(**meta,state=cpu['state'],cpu_gpu_max_error=error,trace=trace))
        log(f'GPU gate passed, max CPU/GPU difference {error:.3g}')
        report=bench(model,args.out); dump(args.out/'timing.json',dict(**meta,**report))
        log('all paired timings complete')


if __name__=='__main__':
    main()
