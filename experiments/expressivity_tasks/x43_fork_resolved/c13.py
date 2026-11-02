"""Exact-step C12 reconstruction followed by frozen coverage-swap inference."""
import json, os, time
from pathlib import Path
import numpy as np
import torch
from .c12 import ROOT, Boosted, count, digest, evaluate
OUT=ROOT/'receipts/c13'
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def predict(model,data,features,lengths,split):
    pp=[];cc=[]
    with torch.inference_mode():
        for lo in range(split,len(data)-1,2048):
            hi=min(lo+2048,len(data)-1);full=(hi-lo)//128*128;ranges=[]
            if full:ranges.append(torch.arange(lo,lo+full,device='cuda').reshape(-1,128))
            if lo+full<hi:ranges.append(torch.arange(lo+full,hi,device='cuda')[None])
            for at in ranges:
                with torch.autocast('cuda',dtype=torch.bfloat16):lp,_=model(data[at],features[at+1],lengths[at+1])
                pp.append(lp.gather(-1,data[at+1,...,None])[...,0].exp().flatten().cpu().numpy())
                cc.append(lp.exp().max(-1).values.flatten().cpu().numpy())
    return dict(p=np.concatenate(pp),confidence=np.concatenate(cc))
def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    torch.set_num_threads(1)
    z=np.load(ROOT/'receipts/c6/index.npz');split=int(.9*len(z['fork']))
    data=torch.tensor(list((ROOT/'input.txt').read_bytes()),device='cuda',dtype=torch.long)
    lengths=torch.tensor(z['context_length'],device='cuda',dtype=torch.float32)
    d=np.load(ROOT/'receipts/c12/disjoint.npz');features=torch.tensor(d['features'][d['ids']],device='cuda')
    self_features=torch.tensor(np.load(ROOT/'receipts/c11/device.npz')['features'],device='cuda')
    meta=dict(protocol_sha256=digest(ROOT/'PREREGISTRATION_C13.md'),source_sha256=digest(Path(__file__)),physical_gpu=os.environ['GPU_LEASE_HELD'],models=[],new_training_arms=0)
    old=json.loads((ROOT/'receipts/c12/training.json').read_text());target=2_000_000
    for arm in ('emender','transformer'):
        row=next(r for r in old['models'] if r['name']==arm+'_disjoint')
        torch.manual_seed(4301);model=Boosted(arm,target).cuda().train()
        if arm=='emender':target=count(model.head.base)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(4301);start=time.monotonic()
        for step in range(row['steps']):
            starts=torch.randint(0,split-128,(16,),device='cuda',generator=rng)
            at=starts[:,None]+torch.arange(128,device='cuda');opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lp,_=model(data[at],features[at+1],lengths[at+1]);loss=-lp.gather(-1,data[at+1,...,None])[...,0].mean()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%2000==0:print(arm,step,float(loss.detach()),flush=True)
        del opt
        model.eval();model.requires_grad_(False)
        reference=np.load(ROOT/'receipts/c12'/(arm+'_disjoint_prediction.npz'))
        parity={}
        for mode in ('with','without'):
            actual=evaluate(model,data,features,lengths,split,mode)
            p=np.exp(-actual);q=np.exp(-reference[mode])
            parity[mode]=dict(passed=bool(np.allclose(p,q,rtol=1e-5,atol=1e-7)),max_absolute_probability_difference=float(np.abs(p-q).max()),n=len(p))
        recovery=dict(arm=arm,steps=row['steps'],seconds=time.monotonic()-start,parity=parity)
        meta['models'].append(recovery);write('recovery.json',meta)
        assert all(x['passed'] for x in parity.values()),'Recovery failed: no swap adopted'
        before={k:v._version for k,v in model.state_dict().items()}
        pred=predict(model,data,self_features,lengths,split)
        assert before=={k:v._version for k,v in model.state_dict().items()}
        np.savez_compressed(OUT/(arm+'_self_prediction.npz'),**pred)
        recovery['prediction_sha256']=digest(OUT/(arm+'_self_prediction.npz'));write('recovery.json',meta)
        print(recovery,flush=True);del model;torch.cuda.empty_cache()
if __name__=='__main__':main()
