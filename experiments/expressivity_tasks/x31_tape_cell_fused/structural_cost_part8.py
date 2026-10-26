"""Matched standalone native-initialization and mutation-capture ablations.

These are component measurements, not additive decomposition or universal lower
bounds. Full registered gates remain in cost_part8.json.
"""
import argparse
import json
import os
import torch
from .cell_fused import TapeCell, TapeHeadConfig, tape_scan
from .cost_gates import measure
from .train_gates import require_parity, save
from ..x30_tape_cell.naive_reference import init_state


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parity-receipt',required=True)
    args=p.parse_args();require_parity(args.parity_receipt)
    held=os.environ.get('GPU_LEASE_HELD')
    if held not in tuple(str(i) for i in range(7)) or os.environ.get('CUDA_VISIBLE_DEVICES')!=held:
        raise RuntimeError('broker lease required')
    torch.set_num_threads(1);torch.manual_seed(0)
    result=dict(physical_gpu=held,warmups=5,repetitions=20,tokens=16,measurements=[],
                note='Standalone component timings and matched capture ablations; not additive attribution or a hardware-independent lower bound.')
    for heads in (1,4,16):
        cfg=TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=11776)
        def native():
            state=init_state(64,cfg,'cuda')
            return None,state
        row=dict(component='native_zero_initialization',heads=heads,slots=11776,**measure(native,16))
        result['measurements'].append(row);print(json.dumps(row),flush=True)
    cfg=TapeHeadConfig(key_dim=64,value_dim=128,n_heads=4,budget=11776)
    cell=TapeCell(256,cfg).cuda();x=torch.randn(64,16,256,device='cuda')
    k=cell.W_k(x).reshape(64,16,4,64).detach();q=cell.W_q(x).reshape_as(k)
    v=cell.W_v(x).reshape(64,16,4,128)
    beta=(cell.W_write(x).sigmoid()>cfg.write_enable_threshold).float()
    pop=cell.W_pop(x).sigmoid();logit=cell.W_append(x)
    for capture in (False,True):
        def scan():
            return tape_scan(k,q,v,beta,pop,logit,None,cfg,cell.append_err_coef,record_all=capture)
        row=dict(component='projected_scan',capture=capture,heads=4,slots=11776,**measure(scan,16))
        result['measurements'].append(row);print(json.dumps(row),flush=True)
    rows=result['measurements']
    result['capture_fractional_delta']=rows[-1]['wall_ms_per_token']/rows[-2]['wall_ms_per_token']-1
    save('structural_cost_part8.json',result)


if __name__=='__main__':main()
