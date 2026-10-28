"""Part-five gates. Production assertions retain their default scope."""
import argparse
from unittest.mock import patch
import hashlib
import json
import os
from pathlib import Path
import torch
from .parity import certify


def main():
    p=argparse.ArgumentParser();p.add_argument('--batch',type=int,default=64);p.add_argument('--context',type=int,default=512);p.add_argument('--receipt');a=p.parse_args()
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6') or os.environ.get('CUDA_VISIBLE_DEVICES')!=os.environ.get('GPU_LEASE_HELD'):raise RuntimeError('approved lease required')
    torch.set_num_threads(1)
    source=Path(__file__).resolve().parent;root=source/'receipts/part5'
    name='production_parity.json'
    if os.environ.get('X40_BATCHED_READS')=='1':name='reads_parity.json'
    if os.environ.get('X40_FRESH_POOL')=='1' or os.environ.get('X40_STABLE_MAPPING')=='1':name='backward_parity.json'
    if os.environ.get('X40_FUSED_INIT','0')=='1':name='fused_parity.json'
    if a.receipt:name=a.receipt
    metadata=dict(batch=a.batch,context=a.context,flags={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')},
        sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},physical_gpu=os.environ.get('GPU_LEASE_HELD'),python=os.sys.executable)
    try:
        if a.context>512:
            from ..x31_tape_cell_fused import backward as oracle_backward
            from .probe_rebase import TransposedLaunch
            with patch.object(oracle_backward,'rebase',TransposedLaunch()):result=certify(batch=a.batch,context=a.context)
            result['oracle_launch_adapter']='probe-only transposed integer address rebase; unchanged floating reference'
        else:result=certify(batch=a.batch,context=a.context)
    except Exception as exc:
        (root/name).write_text(json.dumps(dict(status='failed',error=repr(exc),**metadata),indent=2)+'\n')
        raise
    result.update(metadata)
    (root/name).write_text(json.dumps(result,indent=2)+'\n')
    print('production parity passed',flush=True)


if __name__=='__main__':main()
