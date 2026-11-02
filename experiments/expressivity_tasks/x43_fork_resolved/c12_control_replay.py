"""Recover C11 without-device predictions only after whole-vector replay parity."""
import os,time
import numpy as np
import torch
from .c12 import ROOT, OUT, Boosted, count, digest, write, evaluate

def main():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    protocol=digest(ROOT/'PREREGISTRATION_C12_CONTROL_REPLAY.md');assert protocol in (ROOT/'PROGRAM.md').read_text()
    torch.set_num_threads(1)
    data=torch.tensor(list((ROOT/'input.txt').read_bytes()),device='cuda',dtype=torch.long)
    z=np.load(ROOT/'receipts/c6/index.npz');lengths=torch.tensor(z['context_length'],device='cuda',dtype=torch.float32)
    features=torch.tensor(np.load(ROOT/'receipts/c11/device.npz')['features'],device='cuda')
    split=int(.9*len(data));target=2_000_000;rows=[]
    import json
    oldmeta=json.loads((ROOT/'receipts/c11/training.json').read_text())
    for arm in ('emender','transformer'):
        oldrow=next(row for row in oldmeta['models'] if row['arm']==arm)
        torch.manual_seed(4301);model=Boosted(arm,target).cuda().train()
        if arm=='emender':target=count(model.head.base)
        opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.01)
        rng=torch.Generator(device='cuda').manual_seed(4301)
        torch.cuda.synchronize();start=time.monotonic()
        for step in range(oldrow['steps']):
            starts=torch.randint(0,split-128,(16,),device='cuda',generator=rng)
            at=starts[:,None]+torch.arange(128,device='cuda');opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                lp,_=model(data[at],features[at+1],lengths[at+1]);loss=-lp.gather(-1,data[at+1,...,None])[...,0].mean()
            assert torch.isfinite(loss)
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step()
            if step%1000==0:print(arm,step,float(loss.detach()),flush=True)
        torch.cuda.synchronize();seconds=time.monotonic()-start
        model.eval();withloss=evaluate(model,data,features,lengths,split,'with')
        # C11 evaluated exp(logp) in float32; reconstruct that rounding exactly.
        p=np.exp(-withloss.astype(np.float32)).astype(np.float64)
        original=ROOT/'receipts/c11'/(arm+'_ops_prediction.npz');reference=np.load(original)['p']
        parity=bool(np.allclose(p,reference,rtol=1e-5,atol=1e-7))
        row=dict(arm=arm,steps=oldrow['steps'],seconds=seconds,parity=parity,
            max_absolute_probability_difference=float(np.abs(p-reference).max()),original_sha256=digest(original))
        if parity:
            without=evaluate(model,data,features,lengths,split,'without')
            path=OUT/(arm+'_c11_replay_prediction.npz');np.savez_compressed(path,with_replayed=withloss,without=without)
            row['prediction_sha256']=digest(path)
        rows.append(row)
        write('control_replay.json',dict(protocol_sha256=protocol,source_sha256=digest(__import__('pathlib').Path(__file__)),models=rows))
        print(row,flush=True);del model,opt;torch.cuda.empty_cache()
if __name__=='__main__':main()
