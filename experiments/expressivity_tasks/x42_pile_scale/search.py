"""Gated sequential CMA-ES driver. Evaluator must train the decoded candidate."""
import argparse
import importlib
import json
import os
import time
from pathlib import Path
import numpy as np
from .cmaes import CMAES
from .train import gate

# Four lane policies at maximum; inactive lanes remain in the vector.
BOUNDS = ([100,1000,10000,100000,-4,-4,.01,64,64,64,64,0,32,1,0,1e-4,100] ,
          [10000,100000,1000000,10000000,4,1,.9,2944,2944,2944,2944,.1,256,4,1,1e-3,2000])
CENTER = [1000,10000,100000,1000000,0,-1,.1,64,256,736,1472,.01,64,1,.5,3e-4,1000]
NAMES = ['horizon0','horizon1','horizon2','horizon3','write_gate_bias','match_coef',
         'pop_threshold','budget0','budget1','budget2','budget3','gate_cost_lambda',
         'w_in_width','lane_count','lane_placement_fraction','lr','warmup_steps']
LOG = {0,1,2,3,7,8,9,10,12,15,16}


def encode(values):
    lo,hi = map(lambda x:np.array(x,dtype=float),BOUNDS)
    v = np.array(values,dtype=float)
    for i in LOG:
        lo[i],hi[i],v[i]=np.log(lo[i]),np.log(hi[i]),np.log(v[i])
    return (v-lo)/(hi-lo)


def decode(unit):
    lo,hi = map(lambda x:np.array(x,dtype=float),BOUNDS)
    for i in LOG:
        lo[i],hi[i]=np.log(lo[i]),np.log(hi[i])
    # Reflection is deterministic and avoids piles at hard clipped bounds.
    unit = 1-np.abs(np.mod(unit,2)-1)
    values=lo+unit*(hi-lo)
    for i in LOG:
        values[i]=np.exp(values[i])
    result=dict(zip(NAMES,values.tolist()))
    for name in ['budget0','budget1','budget2','budget3','w_in_width','lane_count','warmup_steps']:
        result[name]=round(result[name])
    return result


def run(config, evaluator, output, verdict):
    gate(verdict,config)
    if os.environ.get("GPU_LEASE_HELD") not in ("0","1","5","6"):
        raise RuntimeError("search must use run_leased.sh")
    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'registration.json').write_text(json.dumps(dict(config=config,evaluator=evaluator,
        protocol_sha256=__import__('hashlib').sha256(Path(__file__).with_name('RECEIPTS.md').read_bytes()).hexdigest()),indent=2))
    config = dict(config, verdict=verdict)
    module,name=evaluator.split(':')
    evaluate=getattr(importlib.import_module(module),name)
    es=CMAES(encode(CENTER),population=12,seed=4201)
    start=time.monotonic()
    with open(out/'candidates.jsonl','w',buffering=1) as log:
        for generation in range(8):
            scored=[]
            for i,x in enumerate(es.ask()):
                if time.monotonic()-start >= 96*3600:
                    raise RuntimeError('96 GPU-hour budget exhausted')
                candidate=decode(x)
                # Evaluator contract: both paired seeds, identical fixed 150M-token prefix,
                # 24 batches/validation, no early selection or architecture mutation.
                result=evaluate(candidate,config,out/f'g{generation}_c{i}',
                                seeds=[3401,3402],tokens=150_208_512,
                                wall_seconds=min(3600,96*3600-(time.monotonic()-start)))
                if result.get('tokens_per_seed') != [150_208_512]*2:
                    raise RuntimeError('unmatched/incomplete proxy exposure')
                nll=float(np.mean(result['valid_nll']))
                slope=float(np.clip(result['occupancy_length_slope'],0,1))
                if not np.isfinite([nll,slope]).all():
                    raise RuntimeError('nonfinite objective')
                scored.append((nll,slope))
                log.write(json.dumps(dict(generation=generation,index=i,candidate=candidate,result=result))+'\n')
            # Lexicographic tie at quantized 1e-4 nat precision; slope cannot
            # compensate a loss in another NLL bucket.
            order=sorted(range(12),key=lambda i:(round(scored[i][0]/1e-4),scored[i][1]))
            ranks=np.empty(12)
            ranks[order]=np.arange(12)
            es.tell(ranks)
    (out/'result.json').write_text(json.dumps(dict(status='completed',mean=decode(es.mean))))

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('config','evaluator','output','verdict'):
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    run(json.loads(Path(a.config).read_text()),a.evaluator,a.output,a.verdict)
