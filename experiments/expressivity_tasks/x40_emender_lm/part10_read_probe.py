"""Recorded-view read throughput, BF16 operand experiment and cache accounting.

Only B1 measurements: scale heads are identical; full-stack scaling is arithmetic.
"""
import hashlib
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import torch
from .model import Cell, POLICIES
from .part10_bf16_read import bf16_read, reference_read
from ..x35_tape_cell_v3 import causal_read

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def timed(fn,iterations=20):
    for _ in range(3):fn()
    torch.cuda.synchronize();start=time.perf_counter()
    for _ in range(iterations):fn()
    torch.cuda.synchronize();return (time.perf_counter()-start)/iterations

def main():
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    result=dict(status='failed',physical_gpu=os.environ['GPU_LEASE_HELD'],python=os.sys.executable,rows=[],
      scope='B1 recorded views; actual 64-key/128-value head shapes shared at 22M and 1.3B; not a full-model speedup',
      sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('part10_bf16_read.py','part10_read_probe.py','model.py','forward.py','backward.py')})
    try:
        native=causal_read.chunk_read
        for context in (512,2048):
            cell=Cell(32768).cuda().train();cell.W_write.weight.data.zero_();cell.W_write.bias.data.fill_(2.);cell.match_coef.data.zero_();cell.error_coef.data.zero_()
            captured=[]
            def read(*args):
                captured.append(tuple(x.detach().clone() for x in args))
                return native(*args)
            ids=torch.randint(1,32768,(1,context),device='cuda');x=torch.randn(1,context,256,device='cuda')
            with patch.object(causal_read,'chunk_read',read):out=cell(x,ids)
            del out;cell.gate_mean=None
            for head,args in enumerate(captured):
                with torch.no_grad():
                    before=[x.clone() for x in args]
                    fp=native(*args);bf=bf16_read(*args)
                    assert all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(before,args))
                    del before
                    fp_seconds=timed(lambda:native(*args));bf_seconds=timed(lambda:bf16_read(*args))
                    # CPU oracle only a small prefix to keep gathered tensors bounded.
                    small=[v.cpu() for v in args];small[0]=small[0][:,:8].contiguous();small[3]=small[3][:,:8].contiguous();small[4]=small[4][:,:8].contiguous()
                    cpu=reference_read(*small);error=(cpu-bf[:,:8].cpu()).abs()
                    rows=args[3].shape[-1]
                    result['rows'].append(dict(context=context,head=head,slots=POLICIES[head].slots,reachable_slots=rows,batch=1,
                      fp32_seconds=fp_seconds,bf16_seconds=bf_seconds,speedup=fp_seconds/bf_seconds,read_inputs_bytewise_unchanged=True,
                      forward_max_abs_difference=float((fp-bf).abs().max()),forward_rms_difference=float((fp-bf).square().mean().sqrt()),
                      cpu_prefix_max_abs_difference=float(error.max()),cpu_prefix_bitwise_equal=torch.equal(cpu,bf[:,:8].cpu()),
                      output_cache_bytes_per_lane=context*128*4,
                      probability_cache_bytes_per_lane=context*rows*4,
                      cache_note='Outputs alone do not determine probability/query/key/strength adjoints. Probabilities still need store restore and read adjoints; no caching speedup measured.'))
                    print(result['rows'][-1],flush=True)
            del cell,captured,args;torch.cuda.empty_cache()
        result['status']='passed'
    except Exception as exc:
        import traceback
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:(OUT/'read_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
