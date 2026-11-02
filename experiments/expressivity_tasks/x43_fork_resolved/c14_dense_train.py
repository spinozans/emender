"""Dense C14 training, gated on the complete registered raw-byte corpus."""
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from .c11 import Boosted
from .c6 import count
from .c14_documents import sha, evaluation_halves
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'receipts/c14_dense'
DATA=Path('/mnt/nvme2n1/erikg/c14_dense_bytes')

def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

def batches(raw, split):
    """Source order, right padding masked; each row resets model state."""
    rows=[]
    with (DATA/'offsets.jsonl').open() as f:
        for line in f:
            doc=json.loads(line)
            if doc['split'] != split: continue
            for start in range(0,doc['byte_length']-1,128):
                length=min(128,doc['byte_length']-1-start)
                offset=doc['byte_offset']+start
                rows.append((doc['source_index'],start,raw[offset:offset+length+1]))
                if len(rows)==16:
                    yield rows
                    rows=[]
    if rows: yield rows

def tensors(rows):
    x=np.zeros((len(rows),128),dtype=np.int64)
    y=np.zeros_like(x); mask=np.zeros_like(x,dtype=bool)
    for j,(_,_,b) in enumerate(rows):
        length=len(b)-1
        x[j,:length]=b[:-1]; y[j,:length]=b[1:]; mask[j,:length]=True
    return [torch.as_tensor(a,device='cuda') for a in (x,y,mask)]

def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    preflight=json.loads((OUT/'source_preflight.json').read_text())
    assert preflight['status']=='passed', 'full selected corpus required'
    assert sha(DATA/'bytes.bin')==preflight['corpus_sha256']
    assert sha(DATA/'offsets.jsonl')==preflight['offsets_sha256']
    raw=np.memmap(DATA/'bytes.bin',dtype=np.uint8,mode='r')
    valid=[]
    with (DATA/'offsets.jsonl').open() as f:
        for line in f:
            d=json.loads(line)
            if d['split']=='valid': valid.append(d['source_index'])
    halves=evaluation_halves(valid)
    # This immutable construction is shared across heads and future consultation.
    with (OUT/'eval_windows.jsonl').open('w') as f:
        for batch in batches(raw,'valid'):
            for doc,start,b in batch:
                f.write(json.dumps(dict(document=doc,start=start,targets=len(b)-1,half=halves[doc]))+'\n')
    write('eval_construction.json',dict(windows_sha256=sha(OUT/'eval_windows.jsonl'),
        offsets_sha256=preflight['offsets_sha256'],valid_documents=len(valid),
        policy='all real targets, source-order 128-byte independent resets; same positions for consultation',
        support_strata='pending validated causal features; never inferred from absent counts'))
    torch.set_num_threads(1)
    target=2000000; results=[]
    for arm,expected in [('emender',2133700),('transformer',2133748)]:
        torch.manual_seed(3401)
        model=Boosted(arm,target).cuda().train()
        if arm=='emender': target=count(model.head.base)
        assert count(model)==expected
        opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
        zeros=torch.zeros((16,128,260),device='cuda')
        lengths=torch.zeros((16,128),device='cuda')
        curve=[]; steps=targets=epochs=0
        torch.cuda.synchronize(); start=time.monotonic()
        while time.monotonic()-start < 2700:
            for rows in batches(raw,'train'):
                x,y,mask=tensors(rows)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=model(x,zeros[:len(rows)],lengths[:len(rows)])
                    loss=-lp.gather(-1,y[...,None])[...,0][mask].mean()
                assert torch.isfinite(loss)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                opt.step(); steps+=1; targets+=int(mask.sum())
                if steps==1 or steps%100==0:
                    curve.append(dict(step=steps,seconds=time.monotonic()-start,nll=float(loss.detach()),targets=targets))
                    print(arm,curve[-1],flush=True)
                if time.monotonic()-start >= 2700: break
            epochs+=1
        torch.cuda.synchronize(); elapsed=time.monotonic()-start
        model.eval()
        torch.save(model.state_dict(),DATA/f'{arm}_dense.pt')
        sums={}; eval_start=time.monotonic()
        with (OUT/f'{arm}_eval.jsonl').open('w') as f, torch.no_grad():
            for rows in batches(raw,'valid'):
                x,y,mask=tensors(rows)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    lp,_=model(x,zeros[:len(rows)],lengths[:len(rows)])
                nll=(-lp.gather(-1,y[...,None])[...,0]).float().cpu().numpy()
                for j,(doc,at,b) in enumerate(rows):
                    losses=nll[j,:len(b)-1]
                    f.write(json.dumps(dict(document=doc,start=at,nll=losses.tolist()))+'\n')
                    key=halves[doc]
                    total,counted=sums.get(key,(0.,0))
                    sums[key]=(total+float(losses.astype(np.float64).sum()),counted+len(losses))
        row=dict(arm=arm,parameters=expected,steps=steps,targets=targets,epochs_started=epochs,
            active_seconds=elapsed,curve=curve,heldout={k:dict(targets=n,nll=s/n) for k,(s,n) in sums.items()},
            eval_seconds=time.monotonic()-eval_start,checkpoint_sha256=sha(DATA/f'{arm}_dense.pt'),
            eval_sha256=sha(OUT/f'{arm}_eval.jsonl'),cross_document_windows=0,
            boundary_evidence='every row sliced from one hash-verified document; right-pad loss masked; fresh forward per batch',
            support_strata='unavailable pending validated consultation feature bridge')
        results.append(row); write('training.json',dict(models=results,seed=3401,precision='BF16',
            protocol_sha256=sha(ROOT/'PREREGISTRATION_C14_BYTE.md'),dense_amendment_sha256=sha(ROOT/'PREREGISTRATION_C14_DENSE.md'),
            physical_gpu=os.environ['GPU_LEASE_HELD'],code_sha256=sha(__file__)))
        del model; torch.cuda.empty_cache()
if __name__=='__main__': main()
