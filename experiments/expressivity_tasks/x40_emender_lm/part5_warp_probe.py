"""Scoped reverse-launch occupancy trial; exactness failure restores the launch."""
import argparse
import json
import os
import sys
import hashlib
from pathlib import Path
from unittest.mock import patch
import torch
from . import backward
from .parity import certify


class WarpLaunch:
    def __init__(self,kernel,warps,min_slots=0):
        self.kernel,self.warps,self.min_slots=kernel,warps,min_slots
        self.slot_argument=kernel.arg_names.index('S')
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            if args[self.slot_argument]>=self.min_slots:kwargs['num_warps']=self.warps
            return self.kernel[grid](*args,**kwargs)
        return launch


def main():
    p=argparse.ArgumentParser();p.add_argument('--warps',type=int,choices=[8,16],default=8);p.add_argument('--min-slots',type=int,default=0);p.add_argument('--action',choices=['gate','pilot','curve'],default='gate');a,remaining=p.parse_known_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('GPU_LEASE_HELD'):raise RuntimeError('approved lease required')
    torch.set_num_threads(1)
    root=Path(__file__).resolve().parent/'receipts/part5'
    original=backward.reverse_kernel
    stem=f'warp{a.warps}'+(f'_min{a.min_slots}' if a.min_slots else '')
    if a.action!='gate':
        proof=json.loads((root/f'{stem}_parity.json').read_text())
        assert proof['status']=='passed'
        assert proof['backward_sha256']==hashlib.sha256(Path(backward.__file__).read_bytes()).hexdigest()
        for key,value in proof['flags'].items():assert os.environ.get(key,'0')==value
        os.environ['X40_SCOPED_REVERSE_WARPS']=str(a.warps)
        os.environ['X40_SCOPED_MIN_SLOTS']=str(a.min_slots)
        sys.argv=[sys.argv[0]]+remaining
        with patch.object(backward,'reverse_kernel',WarpLaunch(original,a.warps,a.min_slots)):
            if a.action=='pilot':
                from .part5_pilot import main as run
            else:
                from .train_slice import main as run
            run()
        assert backward.reverse_kernel is original
        return
    result=dict(status='failed',min_slots=a.min_slots,warps=a.warps,scope='reverse launch only; full production B64/T512 gate including pressure and exact gradients',performance='not-run',physical_gpu=os.environ['GPU_LEASE_HELD'])
    result['flags']={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')}
    result['backward_sha256']=hashlib.sha256(Path(backward.__file__).read_bytes()).hexdigest()
    try:
        with patch.object(backward,'reverse_kernel',WarpLaunch(original,a.warps,a.min_slots)):
            result.update(certify())
    except Exception as exc:
        result['error']=repr(exc)
    assert backward.reverse_kernel is original
    result['original_launch_restored']=True
    (root/f'{stem}_parity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
