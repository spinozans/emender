"""Controlled 128-live-slot diagnostic; no trained artifact or kernel writes.

Zero stream values make adjacent residuals exactly zero, so the standing stream
law appends each token. This isolates high occupancy at the unchanged 257 cap;
it is deliberately synthetic, not sustained training throughput.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace
import torch
from .c14_b_step_composition import bits
from . import c14_b_control_native as native
from . import c14_b_control_graph as graph
from . import c14_b_control_replay as batched
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--compare',type=Path)
    p.add_argument('--reference-slicing',action='store_true')
    a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5','6')
    assert not a.output.exists()
    a.output.mkdir(parents=True)
    if a.reference_slicing:
        source=inspect.getsource(batched.replay)
        source=source.replace('    token_columns=tuple(x.unbind(1) for x in (k,q,v,av,pop,append,logits))\n','')
        source=source.replace('tuple(column[i] for column in token_columns)','tuple(x[:,i] for x in (k,q,v,av,pop,append,logits))')
        ns=dict(vars(batched));exec(compile(source,'<reference slices>','exec'),ns)
        batched.replay=ns['replay'];graph.replay=ns['replay']
    torch.set_num_threads(1);torch.manual_seed(3401)
    policy=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    fields=dict(vars(policy.configs()[0]));fields.update(budget=257,slots=257,array_append_only=False)
    cfg=SimpleNamespace(**fields);gp=policy.gate_policy(0)
    b,t=16,128
    xs=[torch.randn(b,t,d,device='cuda') for d in (64,64)]
    xs += [torch.zeros(b,t,d,device='cuda') for d in (128,128)]
    xs += [torch.zeros(b,t,device='cuda'),torch.zeros(b,t,device='cuda'),torch.full((b,t),20.,device='cuda')]
    xs += [torch.zeros((),device='cuda') for _ in range(3)]
    metadata=([[None]*t for _ in range(b)],[b'a'*t]*b,[(j,0) for j in range(b)],[t]*b,[[False]*t for _ in range(b)],[[97]*t for _ in range(b)])
    up=torch.randn(b,t,128,device='cuda')
    def run():
        out=native.native_forward(xs,cfg,gp,metadata)
        replay,grads=graph.backward(xs,up,cfg,gp,metadata)
        torch.testing.assert_close(replay,out[0],atol=2e-5,rtol=2e-5)
        return out,replay,grads
    run();first=run();second=run()
    fingerprint=bits(first)
    exact=fingerprint==bits(second)
    comparison=None if a.compare is None else fingerprint==json.loads(a.compare.read_text())
    (a.output/'fingerprint.json').write_text(json.dumps(fingerprint,indent=2)+'\n')
    reports=native.native_reports(first[0][1][1],cfg,[t]*b)
    assert all(r['peak_occupancy']==128 and r['capacity_merges']==0 for r in reports)
    rows=[]
    for _ in range(30):
        torch.cuda.synchronize();start=time.perf_counter()
        native.native_forward(xs,cfg,gp,metadata)
        torch.cuda.synchronize();middle=time.perf_counter()
        graph.backward(xs,up,cfg,gp,metadata)
        torch.cuda.synchronize();end=time.perf_counter()
        rows.append([(middle-start)*1000,(end-middle)*1000])
    result=dict(scope='synthetic 16x128 high-occupancy native + replay; not training',
        peak_live=128,slots=257,padded_native_slots=512,reports=reports,
        run_to_run_exact=exact,comparison_exact=comparison,samples_native_replay_ms=rows,
        median_native_ms=statistics.median(x[0] for x in rows),median_replay_ms=statistics.median(x[1] for x in rows))
    (a.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('reports','samples_native_replay_ms')}),flush=True)
    assert exact and comparison is not False

if __name__=='__main__':main()
