"""C7b: all-position mixture likelihood, jointly differentiated neural gate."""
import json, os, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize
from .c6 import ROOT, Head, count, digest
OUT=ROOT/'receipts/c11'
def write(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')

class Boosted(nn.Module):
    def __init__(self,arm,target,recipe="ops"):
        super().__init__()
        self.head=Head(arm,target)
        self.device_input=nn.Linear(260,256,bias=False)
    def forward(self,ids,features,lengths):
        # Hook is scoped to this forward and removed even if the head raises.
        hook=self.head.base.embed.register_forward_hook(lambda module,args,out: out+self.device_input(features))
        try: lp=self.head(ids,features,False)
        finally: hook.remove()
        return lp,torch.ones_like(lengths)

def prepare_teacher():
    from .direct_sa import classify
    raw=(ROOT/'input.txt').read_bytes(); data=np.frombuffer(raw,dtype=np.uint8)
    pos,lengths,iv=classify(raw); unique,ids=np.unique(iv,axis=0,return_inverse=True)
    z=np.load(ROOT/'receipts/c6/index.npz'); assert np.array_equal(ids,z['ids'])
    probs=np.zeros((len(unique),256),np.float32); split=int(.9*len(raw))
    for k,(lo,hi) in enumerate(unique):
        occ=pos[lo:hi]; c=np.bincount(data[occ[occ<split]],minlength=256)
        if c.sum(): probs[k]=c/c.sum()
    assert np.array_equal(probs[ids,data],z['q'])
    write('teacher.json',dict(exact_target_ratios_match=True,unique_intervals=len(unique),
        sha256=__import__('hashlib').sha256(probs.tobytes()).hexdigest()))
    return probs

def train():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    protocol_hash = digest(ROOT/'PREREGISTRATION_C11.md')
    assert protocol_hash in (ROOT/'PROGRAM.md').read_text()
    torch.set_num_threads(1)
    z = np.load(ROOT/'receipts/c6/index.npz')
    data = torch.tensor(list((ROOT/'input.txt').read_bytes()), device='cuda', dtype=torch.long)
    ids = torch.tensor(z['ids'], device='cuda', dtype=torch.long)
    features = torch.tensor(z['features'], device='cuda', dtype=torch.float32)
    fork = torch.tensor(z['fork'], device='cuda', dtype=torch.bool)
    q = torch.tensor(z['q'], device='cuda', dtype=torch.float32)
    lengths = torch.tensor(z['context_length'], device='cuda', dtype=torch.float32)
    split = int(.9*len(data))
    meta = dict(protocol_sha256=protocol_hash, code_sha256=digest(Path(__file__)),
        index_sha256=digest(ROOT/'receipts/c6/index.npz'),
        physical_gpu=os.environ['GPU_LEASE_HELD'], torch=torch.__version__,
        seed=4301, batch=16, context=128, budget_seconds=180, optimizer='AdamW lr=3e-4 wd=.01 clip=1',
        precision='autocast bfloat16; normalized logits float32',
        scope='corpus-label diagnostic; no causal train-index claim', models=[], failures=[])
    write('training.json',meta)
    probs=prepare_teacher()
    original=features.cpu().numpy()
    per=original[z['ids']].copy()
    per[:,:256]=probs[z['ids']]
    per[:,256]=original[z['ids'],257]
    per[:,257]=original[z['ids'],259]
    per[:,258]=z['fork']
    per[:,259]=np.log1p(z['context_length'])
    np.savez_compressed(OUT/'device.npz',features=per)
    write('device.json',dict(corpus_sha256=digest(ROOT/'input.txt'),device_sha256=digest(OUT/'device.npz'),index_sha256=digest(ROOT/'receipts/c6/index.npz')))
    features=torch.tensor(per,device='cuda')
    target = 2_000_000
    for arm in ('emender','transformer'):
        for recipe in ("ops",):
            aware=False
            name = arm+'_'+recipe
            try:
                torch.manual_seed(4301)
                init = time.monotonic()
                model = Boosted(arm, target,recipe).cuda().train()
                if arm == 'emender':
                    target = count(model.head.base)
                parameters = count(model)
                initialization_seconds = time.monotonic()-init
                opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=3e-4,weight_decay=.01)
                rng = torch.Generator(device='cuda').manual_seed(4301)
                steps=0; curve=[]; fork_tokens=0
                torch.cuda.synchronize(); start=time.monotonic()
                while time.monotonic()-start < 180:
                    starts = torch.randint(0,split-128,(16,),device='cuda',generator=rng)
                    at = starts[:,None]+torch.arange(128,device='cuda')
                    y = data[at+1]
                    opt.zero_grad(set_to_none=True)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        logp,g=model(data[at],features[at+1],lengths[at+1])
                        p=logp.gather(-1,y[...,None])[...,0].exp()
                        loss=-logp.gather(-1,y[...,None])[...,0].mean()
                    assert torch.isfinite(loss), 'nonfinite loss'
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    opt.step(); steps+=1
                    fork_tokens+=int(fork[at+1].sum())
                    if steps==1 or steps%100==0:
                        curve.append(dict(step=steps,seconds=time.monotonic()-start,nll=float(loss.detach())))
                        print(name,curve[-1],flush=True)
                torch.cuda.synchronize()
                training_seconds=time.monotonic()-start
                row=dict(name=name,arm=arm,fork_aware=aware,trainable_parameters=parameters,
                    total_parameters=sum(p.numel() for p in model.parameters()), steps=steps,
                    tokens=steps*16*128,fork_tokens=fork_tokens,training_seconds=training_seconds,
                    initialization_seconds=initialization_seconds,curve=curve)
                # Model is frozen before any held-out inference. No checkpoint files needed.
                model.eval(); eval_start=time.monotonic(); pp=[]; cc=[]; gg=[]
                with torch.no_grad():
                    for lo in range(split,len(data)-1,16*128):
                        # Each row is an independent 128-byte reset, including across batches.
                        hi=min(lo+16*128,len(data)-1)
                        width=128 if hi-lo>=128 else hi-lo
                        full=(hi-lo)//128*128
                        ranges=[]
                        if full: ranges.append(torch.arange(lo,lo+full,device='cuda').reshape(-1,128))
                        if lo+full<hi: ranges.append(torch.arange(lo+full,hi,device='cuda')[None])
                        for at in ranges:
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                lp,g=model(data[at],features[at+1],lengths[at+1])
                            gg.append(g.flatten().cpu().numpy())
                            pp.append(lp.gather(-1,data[at+1,...,None])[...,0].exp().flatten().cpu().numpy())
                            cc.append(lp.max(-1).values.exp().flatten().cpu().numpy())
                p=np.concatenate(pp).astype(np.float64); confidence=np.concatenate(cc)
                assert len(p)==len(data)-split-1 and np.isfinite(p).all() and (p>0).all()
                np.savez_compressed(OUT/(name+'_prediction.npz'),p=p,confidence=confidence,neural_gate=np.concatenate(gg))
                row.update(evaluation_seconds=time.monotonic()-eval_start,
                           prediction_sha256=digest(OUT/(name+'_prediction.npz')))
                meta['models'].append(row)
                write('training.json',meta)
                print(name,'complete',row['steps'],flush=True)
                del opt,model; torch.cuda.empty_cache()
            except Exception:
                error=traceback.format_exc(); print(error,flush=True)
                meta['failures'].append(dict(name=name,error=error)); write('training.json',meta)
                if arm!='gdn2': raise
                break


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    train()
