"""Exactness-gated register-cap occupancy trial; preserve the four-warp layout."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch
from . import backward
from .parity import certify

ROOT=Path(__file__).resolve().parent/'receipts/part6'


class RegisterLaunch:
    def __init__(self,kernel,registers):self.kernel,self.registers=kernel,registers
    def __getitem__(self,grid):
        def launch(*args,**kwargs):
            kwargs['maxnreg']=self.registers
            return self.kernel[grid](*args,**kwargs)
        return launch


def main():
    p=argparse.ArgumentParser();p.add_argument('--registers',type=int,choices=[64,96,128,192],default=128)
    p.add_argument('--deferred-pack',action='store_true');p.add_argument('--head-streams',action='store_true');p.add_argument('--action',choices=['gate','pilot','curve','tests'],default='gate')
    p.add_argument('--test-label',default='fixtures');p.add_argument('--batch',type=int,default=128);a,rest=p.parse_known_args()
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1)
    stem=f'registers{a.registers}'+(f'_head_streams_B{a.batch}' if a.head_streams else '')+('_deferred_pack' if a.deferred_pack else '')
    original=backward.reverse_kernel
    from .part6_stream_probe import overlap,gate
    from .part6_deferred_journal import packed_cells
    if a.action=='gate':
        result=dict(physical_gpu=os.environ['GPU_LEASE_HELD'],python=sys.executable,production_batch=64,production_context=512,cell_batch=a.batch if a.head_streams else None,status='failed',registers=a.registers,head_streams=a.head_streams,deferred_pack=a.deferred_pack,batch=a.batch,
                    performance='not-run',scope='Preserve num_warps=4 and original arithmetic; register allocation may spill, exactness remains mandatory.',
                    flags={k:os.environ.get(k,'0') for k in ('X40_BATCHED_READS','X40_FRESH_POOL','X40_FUSED_INIT','X40_STABLE_MAPPING')},
                    backward_sha256=hashlib.sha256(Path(backward.__file__).read_bytes()).hexdigest())
        try:
            with patch.object(backward,'reverse_kernel',RegisterLaunch(original,a.registers)):
                result.update(certify())
            if result['status']=='passed':gate(a.batch,512,suffix=f'_registers{a.registers}'+('_deferred_pack' if a.deferred_pack else ''),deferred_pack=a.deferred_pack,launch_registers=a.registers,stream_candidate=a.head_streams)
        except Exception as exc:result.update(status='failed',error=repr(exc))
        result['original_launch_restored']=backward.reverse_kernel is original
        (ROOT/f'{stem}_parity.json').write_text(json.dumps(result,indent=2)+'\n')
        if result['status']!='passed':raise SystemExit(1)
        return
    proof=json.loads((ROOT/f'{stem}_parity.json').read_text());assert proof['status']=='passed'
    assert proof['backward_sha256']==hashlib.sha256(Path(backward.__file__).read_bytes()).hexdigest()
    for key,value in proof['flags'].items():assert os.environ.get(key,'0')==value
    sys.argv=[sys.argv[0]]+(['--batch',str(a.batch)] if a.action=='pilot' else [])+rest
    with patch.object(backward,'reverse_kernel',RegisterLaunch(original,a.registers)),overlap() if a.head_streams else nullcontext(),packed_cells([]) if a.deferred_pack else nullcontext():
        if a.action=='tests':
            import pytest
            test_status=int(pytest.main(rest))
        elif a.action=='pilot':
            from .part5_pilot import main as run
        else:
            from .train_slice import main as run
        if a.action!='tests':run()
    assert backward.reverse_kernel is original
    if a.action=='tests':
        (ROOT/f'registers{a.registers}_{a.test_label}_tests.json').write_text(json.dumps(dict(status='passed' if test_status==0 else 'failed',exit_code=test_status,registers=a.registers,command=rest,original_launch_restored=True),indent=2)+'\n')
        raise SystemExit(test_status)
    if a.action=='pilot':
        label=rest[rest.index('--label')+1]
        artifact=(ROOT.parent/'part5'/label/'wall.json').resolve()
    else:
        artifact=Path(rest[rest.index('--output')+1])/'summary.json'
    receipt=json.loads(artifact.read_text());receipt.update(maxnreg=a.registers,head_streams=a.head_streams,register_parity_receipt=stem+'_parity.json')
    artifact.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
