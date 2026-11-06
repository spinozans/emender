"""Exact native/state/replay/gradient fingerprints at controlled live peaks."""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from . import c14_b_control_native as native
from . import c14_b_control_graph as graph
from .c14_launch_shape import install
from .c14_b_step_composition import bits
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','5','6')
    assert not a.output.exists()
    torch.set_num_threads(1);torch.manual_seed(8134)
    original=native.native_forward
    policy=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    rows=[]
    try:
        for cap,t in ((48,48),(257,110),(257,128),(257,257)):
            fields=dict(vars(policy.configs()[0]));fields.update(slots=cap,budget=cap,array_append_only=False,decay_rate=1.)
            cfg=SimpleNamespace(**fields);b=2
            xs=[torch.randn(b,t,d,device='cuda') for d in (64,64)]
            xs += [torch.randn(b,1,128,device='cuda').expand(b,t,128).contiguous(),torch.zeros(b,t,128,device='cuda')]
            xs += [torch.zeros(b,t,device='cuda'),torch.full((b,t),20.,device='cuda'),torch.full((b,t),20.,device='cuda')]
            xs += [torch.zeros((),device='cuda') for _ in range(3)]
            metadata=([[None]*t for _ in range(b)],[b'a'*t]*b,[(j,0) for j in range(b)],[t]*b,[[False]*t for _ in range(b)],[[97]*t for _ in range(b)])
            upstream=torch.randn(b,t,128,device='cuda')
            def run():
                out=native.native_forward(xs,cfg,policy.gate_policy(0),metadata)
                replay,grads=graph.backward(xs,upstream,cfg,policy.gate_policy(0),metadata)
                torch.testing.assert_close(replay,out[0],atol=2e-5,rtol=2e-5)
                return out,replay,grads
            native.native_forward=original
            reference=bits(run())
            install()
            value=run();candidate=bits(value)
            repeat=candidate==bits(run())
            reports=native.native_reports(value[0][1][1],cfg,[t]*b)
            actual=[r['peak_occupancy'] for r in reports]
            row=dict(cap=cap,tokens=t,actual_peaks=actual,exact=reference==candidate,repeat_exact=repeat,reference=reference,candidate=candidate)
            rows.append(row);a.output.write_text(json.dumps(rows,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('reference','candidate')}),flush=True)
        assert all(r['exact'] and r['repeat_exact'] and r['actual_peaks']==[r['tokens']]*2 for r in rows)
    finally:
        native.native_forward=original

if __name__=='__main__':main()
