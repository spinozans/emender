"""Datatype comparator with fixed tile/dot algorithm and actual recorded views."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from .model import Cell
from .part10_read_probe import timed
from .part10_bf16_read import bf16_read
from .part10_ieee_read import ieee_read
from ..x35_tape_cell_v3 import causal_read

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def main():
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    result=dict(status='failed',physical_gpu=os.environ['GPU_LEASE_HELD'],rows=[],scope='B1 fixed tile/dot algorithm, IEEE FP32 dot versus BF16 MMA; neither full-step attribution nor production FP32 bitwise oracle',sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/'part10_ieee_read.py',ROOT/'part10_bf16_read.py')})
    try:
        for context in (512,2048):
            captured=[];native=causal_read.chunk_read
            def capture(*args):captured.append(tuple(v.detach().clone() for v in args));return native(*args)
            cell=Cell(32768).cuda().train();cell.W_write.weight.data.zero_();cell.W_write.bias.data.fill_(2.);cell.match_coef.data.zero_();cell.error_coef.data.zero_()
            with patch.object(causal_read,'chunk_read',capture):o=cell(torch.randn(1,context,256,device='cuda'),torch.randint(1,32768,(1,context),device='cuda'))
            del o;cell.gate_mean=None
            with torch.no_grad():
                for head,args in enumerate(captured):
                    fp=ieee_read(*args);bf=bf16_read(*args);original=native(*args)
                    native_seconds=timed(lambda:native(*args));ieee_seconds=timed(lambda:ieee_read(*args));bf16_seconds=timed(lambda:bf16_read(*args))
                    row=dict(context=context,head=head,native_seconds=native_seconds,ieee_dot_seconds=ieee_seconds,bf16_dot_seconds=bf16_seconds,fixed_algorithm_speedup=ieee_seconds/bf16_seconds,production_to_bf16_speedup=native_seconds/bf16_seconds,ieee_vs_production_max_abs=float((fp-original).abs().max()),bf16_vs_production_max_abs=float((bf-original).abs().max()))
                    result['rows'].append(row);print(row,flush=True)
            del cell,captured,args;torch.cuda.empty_cache()
        result['status']='passed'
    except Exception as exc:
        import traceback
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:(OUT/'rate_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
