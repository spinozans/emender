"""Read-only production microbatch throughput on recorded B32 views.

The two scales use the same head geometry; repetitions are not a step-time model.
"""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from .model import Cell
from .part10_read_probe import timed
from .part10_bf16_read import bf16_read
from ..x35_tape_cell_v3 import causal_read

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def main():
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    result=dict(status='failed',rows=[],physical_gpu=os.environ['GPU_LEASE_HELD'],scope=__doc__,sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('part10_batch_read_probe.py','part10_bf16_read.py','forward.py','model.py')})
    try:
        for context in (512,2048):
            native=causal_read.chunk_read;captured=[]
            def capture(*args):captured.append(tuple(v.detach().clone() for v in args));return native(*args)
            cell=Cell(32768).cuda().train();cell.W_write.weight.data.zero_();cell.W_write.bias.data.fill_(2.);cell.match_coef.data.zero_();cell.error_coef.data.zero_()
            with patch.object(causal_read,'chunk_read',capture):o=cell(torch.randn(32,context,256,device='cuda'),torch.randint(1,32768,(32,context),device='cuda'))
            del o;cell.gate_mean=None
            with torch.no_grad():
                for head,args in enumerate(captured):
                    fp=native(*args);bf=bf16_read(*args)
                    fp_seconds=timed(lambda:native(*args));bf_seconds=timed(lambda:bf16_read(*args))
                    row=dict(context=context,head=head,batch=32,scales=['22M','1.3B'],key_dim=64,value_dim=128,reachable_slots=args[3].shape[-1],fp32_seconds=fp_seconds,bf16_seconds=bf_seconds,speedup=fp_seconds/bf_seconds,forward_max_abs_difference=float((fp-bf).abs().max()),forward_rms_difference=float((fp-bf).square().mean().sqrt()))
                    result['rows'].append(row);print(row,flush=True)
                    del fp,bf
            del captured,args,cell;torch.cuda.empty_cache()
        result['status']='passed'
    except Exception as exc:
        import traceback
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:(OUT/'batch_read_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
