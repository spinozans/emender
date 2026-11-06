"""Long nonzero masked document-reset fixtures; expensive, explicit opt-in.

Run separately on a broker lease. This fixture is necessary but does not cover
all operation-rich semantics or replace the trained-model fingerprint battery.
"""
import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from . import c14_b_control_native as native
from . import c14_b_control_graph as graph
from .c14_b_step_composition import bits
from .c14_phase2_staged import install_rowkeys
from .c14_phase5r3_validate import install
from ..x42_pile_scale.velocity_candidate import Policy


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tokens',type=int,choices=(512,1024,2048),required=True)
    p.add_argument('--peak',type=int,default=257)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('3','4','5','6')
    assert not a.output.exists()
    assert 0<a.peak<=a.tokens-7
    torch.set_num_threads(1);torch.manual_seed(8136)
    t=a.tokens;b=1;cap=2*t+1
    policy=Policy(N=2,budget_min=32,budget_ratio=2.,h_min=128.)
    cfg=SimpleNamespace(**(vars(policy.configs()[0])|dict(slots=cap,budget=cap,array_append_only=False,decay_rate=1.)))
    xs=[torch.randn(b,t,d,device='cuda') for d in (64,64)]
    xs += [torch.randn(b,1,128,device='cuda').expand(b,t,128).contiguous(),torch.zeros(b,t,128,device='cuda')]
    xs += [torch.zeros(b,t,device='cuda'),torch.full((b,t),20.,device='cuda'),torch.full((b,t),20.,device='cuda')]
    xs[6][:,a.peak:]=-20.
    xs += [torch.zeros((),device='cuda') for _ in range(3)]
    upstream=torch.randn(b,t,128,device='cuda')
    def run(document):
        # Distinct document identities and native fresh-state allocation per run.
        metadata=([[None]*t],[b'a'*t],[(document,0)],[t-7],[[False]*t],[[97]*t])
        print(f"document={document}: native_forward begin",flush=True)
        out=native.native_forward(xs,cfg,policy.gate_policy(0),metadata)
        print(f"document={document}: native_forward done; replay begin",flush=True)
        replay,grads=graph.backward(xs,upstream,cfg,policy.gate_policy(0),metadata)
        print(f"document={document}: replay done",flush=True)
        torch.testing.assert_close(replay,out[0],atol=2e-5,rtol=2e-5)
        assert torch.count_nonzero(out[0][:,-7:]).item()==0
        reports=native.native_reports(out[1][1],cfg,[t-7])
        assert reports[0]['peak_occupancy']==a.peak
        return bits((out,replay,grads))
    original_forward=native.native_forward;original_kernel=native.kernel
    try:
        install_rowkeys();expected=run(100)
        native.native_forward=original_forward
        install();candidate=run(100);reset=run(101)
        result=dict(tokens=t,cap=cap,peak=a.peak,masked_tail=7,exact=expected==candidate,
                    reset_exact=candidate==reset,reference=expected,candidate=candidate)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        assert result['exact'] and result['reset_exact']
    finally:
        native.native_forward=original_forward;native.kernel=original_kernel

if __name__=='__main__':
    main()
