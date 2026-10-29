"""Bytewise native store/routing check at identical Cell inputs, read variant only."""
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from . import model
from .part10_bf16_read import candidate

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'receipts/part10'

def main():
    assert os.sys.executable=='/home/erikg/emender/.venv/bin/python'
    assert os.environ.get('GPU_LEASE_HELD') in ('1','5','6') and os.environ['CUDA_VISIBLE_DEVICES']==os.environ['GPU_LEASE_HELD']
    torch.set_num_threads(1);torch.manual_seed(3401)
    result=dict(status='failed',rows=[],physical_gpu=os.environ['GPU_LEASE_HELD'],scope='B1 Cell identical inputs, no optimizer; native keys/values/strength/live/age/usage/history plus integral routing; not whole-training store equivalence after changed reads feed later layers',sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('part10_store_probe.py','part10_bf16_read.py','forward.py','backward.py','model.py')})
    try:
        native=model.wired_forward
        for context in (512,2048):
            cell=model.Cell(32768).cuda().train();x=torch.randn(1,context,256,device='cuda');ids=torch.randint(1,32768,(1,context),device='cuda')
            records=[]
            def capture(*args,**kw):
                out,state,trace,pressure=native(*args,**kw)
                fields=[getattr(state.native,name) for name in ('keys','values','strength','live','age','usage')]+[state.history]
                fields+=list(state._replay_trace)
                records.append([v.detach().contiguous().view(torch.uint8).cpu().clone() for v in fields])
                return out,state,trace,pressure
            with patch.object(model,'wired_forward',capture):o=cell(x,ids)
            del o;cell.gate_mean=None;reference=records;records=[]
            with candidate(),patch.object(model,'wired_forward',capture):o=cell(x,ids)
            del o;cell.gate_mean=None
            assert len(reference)==len(records)==4
            for head,(a,b) in enumerate(zip(reference,records)):
                assert len(a)==len(b) and all(torch.equal(v,w) for v,w in zip(a,b)),(context,head)
                result['rows'].append(dict(context=context,head=head,all_native_store_and_route_bytes_equal=True,fields_checked=len(a)))
            del cell,x,ids,reference,records;torch.cuda.empty_cache()
        result['status']='passed'
    except Exception as exc:
        import traceback
        result.update(error=repr(exc),traceback=traceback.format_exc());print(result['traceback'],flush=True)
    finally:(OUT/'store_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'],flush=True)
    if result['status']!='passed':raise SystemExit(1)

if __name__=='__main__':main()
