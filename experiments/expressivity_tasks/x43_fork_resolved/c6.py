"""Hash-bound C6 corpus-label diagnostic; never a causal index claim."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import minimize
from .run import Candidate, Policy, Control, count

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'receipts/c6'
sys.path.insert(0, str(ROOT))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def prepare():
    from .direct_sa import classify
    t = time.monotonic()
    raw = (ROOT/'input.txt').read_bytes()
    data = np.frombuffer(raw, dtype=np.uint8)
    split = int(.9*len(raw))
    pos, lengths, iv = classify(raw)
    labels = np.load(ROOT/'receipts/part1c/labels.npz')
    assert np.array_equal(lengths, labels['context_length'])
    unique, ids = np.unique(iv, axis=0, return_inverse=True)
    features = np.zeros((len(unique), 260), np.float32)
    probs = np.zeros((len(unique), 256), np.float32)
    corpus_support = np.zeros(len(unique), np.int32)
    for k, (lo, hi) in enumerate(unique):
        occ = pos[lo:hi]
        corpus_support[k] = len(np.unique(data[occ]))
        c = np.bincount(data[occ[occ < split]], minlength=256)
        s, total = np.count_nonzero(c), c.sum()
        features[k, :256] = c > 0
        features[k, 256] = s == 0
        features[k, 257:259] = np.log1p([total, s])
        if total:
            p = c/total
            probs[k] = p
            features[k, 259] = -(p[p>0]*np.log(p[p>0])).sum()
    assert np.array_equal(corpus_support[ids]>1, labels['fork'])
    q = probs[ids, data]
    support = features[ids, :256].sum(-1).astype(np.int32)
    np.savez_compressed(OUT/'index.npz', ids=ids, features=features, q=q,
                        support=support, fork=labels['fork'], context_length=lengths)
    write('index.json', dict(seconds=time.monotonic()-t, corpus_sha256=digest(ROOT/'input.txt'),
        n=len(data), split=split, unique_intervals=len(unique),
        scope='corpus-selected intervals and labels; train-prefix counts; transductive diagnostic',
        source_labels_sha256=digest(ROOT/'receipts/part1c/labels.npz'),
        source_direct_sa_sha256=digest(ROOT/'direct_sa.py'), index_sha256=digest(OUT/'index.npz')))


class Head(nn.Module):
    def __init__(self, arm, target):
        super().__init__()
        self.arm = arm
        # Identical extra parameter count for every architecture and control.
        self.condition = nn.Linear(260, 256, bias=False)
        self.ood = nn.Linear(256, 1, bias=False)
        if arm == 'emender':
            self.base = Candidate(Policy(N=2, budget_min=32, budget_ratio=2., h_min=128.),
                                  vocab=256, d=256, layers=2, target=target)
        elif arm == 'transformer':
            self.base = Control(target)
        else:
            import fla
            from fla.layers import GatedDeltaNet
            assert fla.__version__ == '0.5.1'
            self.embed = nn.Embedding(256, 256)
            self.layers = nn.ModuleList([GatedDeltaNet(hidden_size=256, head_dim=64,
                num_heads=4, expand_v=1, layer_idx=i) for i in range(2)])
            self.norms = nn.ModuleList([nn.LayerNorm(256) for _ in range(2)])
            self.norm = nn.LayerNorm(256)
            self.output = nn.Linear(256, 256, bias=False)
            remaining = target + 260*256 + 256 - count(self)
            hidden = max(1, round((remaining-512)/1026))
            self.ff = nn.ModuleList([nn.Sequential(nn.Linear(256,hidden),nn.GELU(),
                                                   nn.Linear(hidden,256)) for _ in range(2)])

    def forward(self, ids, features, aware):
        embed = self.base.embed if self.arm != 'gdn2' else self.embed
        x = embed(ids)
        if aware:
            x = x + self.condition(features)
        if self.arm == 'emender':
            from ..x40_emender_lm.part6_checkpoint import checkpoint_block
            for block in self.base.blocks:
                if self.training and torch.is_grad_enabled():
                    x, _ = checkpoint_block(block, x, ids)
                else:
                    x = block(x, ids)
            x = self.base.norm(x)
            byte = F.linear(x.float(), self.base.head.weight.float())
        elif self.arm == 'transformer':
            x = x + self.base.position(torch.arange(ids.shape[1], device=ids.device))
            mask = torch.ones(ids.shape[1],ids.shape[1],device=ids.device,dtype=torch.bool).triu(1)
            for block in self.base.blocks:
                x = block(x, src_mask=mask, is_causal=True)
            x = self.base.norm(x)
            byte = self.base.head(x).float()
        else:
            for layer, norm, ff in zip(self.layers,self.norms,self.ff):
                x = x + layer(norm(x), use_cache=False)[0]
                x = x + ff(x)
            x = self.norm(x)
            byte = self.output(x).float()
        if not aware:
            return byte.log_softmax(-1)
        return branch_logp(byte, self.ood(x).float(), features[..., :256]>0)


def branch_logp(byte, ood, supported):
    outside = (~supported).sum(-1, keepdim=True)
    # Uniform OOD routing; logsumexp includes a single aggregate OOD channel.
    joined = torch.cat([byte.masked_fill(~supported, -torch.inf),
                        ood.masked_fill(outside == 0, -torch.inf)], -1).log_softmax(-1)
    return torch.where(supported, joined[..., :256],
                       joined[..., 256:] - outside.clamp_min(1).log())


def train():
    assert os.environ.get('GPU_LEASE_HELD') in ('2','4')
    protocol_hash = digest(ROOT/'PREREGISTRATION_C6.md')
    assert protocol_hash in (ROOT/'PROGRAM.md').read_text()
    torch.set_num_threads(1)
    z = np.load(OUT/'index.npz')
    data = torch.tensor(list((ROOT/'input.txt').read_bytes()), device='cuda', dtype=torch.long)
    ids = torch.tensor(z['ids'], device='cuda', dtype=torch.long)
    features = torch.tensor(z['features'], device='cuda', dtype=torch.float32)
    fork = torch.tensor(z['fork'], device='cuda', dtype=torch.bool)
    split = int(.9*len(data))
    meta = dict(protocol_sha256=protocol_hash, code_sha256=digest(Path(__file__)),
        physical_gpu=os.environ['GPU_LEASE_HELD'], torch=torch.__version__,
        seed=4301, batch=16, context=128, budget_seconds=180, optimizer='AdamW lr=3e-4 wd=.01 clip=1',
        precision='autocast bfloat16; normalized logits float32',
        scope='corpus-label diagnostic; no causal train-index claim', models=[], failures=[])
    write('training.json',meta)
    target = 2_000_000
    for arm in ('emender','transformer','gdn2'):
        for aware in (True,False):
            name = arm+('_aware' if aware else '_control')
            try:
                torch.manual_seed(4301)
                init = time.monotonic()
                model = Head(arm, target).cuda().train()
                if arm == 'emender' and aware:
                    target = count(model.base)
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
                        logp=model(data[at],features[ids[at+1]],aware)
                        nll=-logp.gather(-1,y[...,None])[...,0]
                        mask=fork[at+1] if aware else torch.ones_like(y,dtype=torch.bool)
                        loss=nll[mask].mean()
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
                model.eval(); eval_start=time.monotonic(); pp=[]; cc=[]
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
                                lp=model(data[at],features[ids[at+1]],aware)
                            pp.append(lp.gather(-1,data[at+1,...,None])[...,0].exp().flatten().cpu().numpy())
                            cc.append(lp.max(-1).values.exp().flatten().cpu().numpy())
                p=np.concatenate(pp).astype(np.float64); confidence=np.concatenate(cc)
                assert len(p)==len(data)-split-1 and np.isfinite(p).all() and (p>0).all()
                np.savez_compressed(OUT/(name+'_prediction.npz'),p=p,confidence=confidence)
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


def analyze():
    z=np.load(OUT/'index.npz'); split=int(.9*len(z['fork']))
    targets=np.arange(split+1,len(z['fork'])); half=len(targets)//2
    assert half==55769 and len(targets)-half==55770
    fork=z['fork'][targets]; q=z['q'][targets]; support=z['support'][targets]
    lengths=z['context_length'][targets]; calibration=np.arange(len(targets))<half; test=~calibration
    fits={}; predictions={}
    training=json.loads((OUT/'training.json').read_text())
    for row in training['models']:
        name=row['name']; ev=np.load(OUT/(name+'_prediction.npz')); p=ev['p']
        predictions[name]=p
        eligible=calibration&fork&(support>0)
        features=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(lengths),np.log1p(support)])
        center=features[eligible,1:].mean(0); scale=features[eligible,1:].std(0); scale[scale==0]=1
        features[:,1:]=(features[:,1:]-center)/scale
        def objective(w):
            raw_z=features[eligible]@w; zz=np.clip(raw_z,-35,35); g=1/(1+np.exp(-zz))
            prob=(1-g)*p[eligible]+g*q[eligible]
            return -np.log(prob).mean(), features[eligible].T@((p[eligible]-q[eligible])*g*(1-g)/prob*(np.abs(raw_z)<35))/len(prob)
        fit=minimize(objective,np.zeros(4),jac=True,method='L-BFGS-B',bounds=[(-20,20)]*4,
                     options={'maxiter':1000,'ftol':1e-12})
        assert fit.success, fit.message
        fits[name]=dict(weights=fit.x.tolist(),center=center.tolist(),scale=scale.tolist(),
                        calibration_fork_nll=float(fit.fun),calibration_targets=half,
                        prediction_sha256=digest(OUT/(name+'_prediction.npz')))
    # This durable calibration receipt is completed before any final loss is calculated.
    write('frozen_gates.json',fits); gate_hash=digest(OUT/'frozen_gates.json')
    write('calibration_freeze.json',dict(sha256=gate_hash,final_test_scored=False))
    assert digest(OUT/'frozen_gates.json')==gate_hash
    def summary(p):
        nll=-np.log(p)
        masks=dict(overall=np.ones(len(p),bool),fork=fork,forced=~fork,covered=support>0,
                   unsupported=support==0,zero_target=(support>0)&(q==0))
        return {k:dict(n=int((test&m).sum()),nll=float(nll[test&m].mean()) if (test&m).any() else None)
                for k,m in masks.items()}
    report=dict(scope=training['scope'],frozen_gate_sha256=gate_hash,models={},
                c3=dict(emender_standalone=2.222975,transformer_standalone=1.924201,
                        emender_fusion=1.673012,transformer_fusion=1.746996),failures=training['failures'])
    for row in training['models']:
        name=row['name']; p=predictions[name]; f=fits[name]
        ev=np.load(OUT/(name+'_prediction.npz'))
        features=np.column_stack([np.ones(len(p)),ev['confidence'],np.log1p(lengths),np.log1p(support)])
        features[:,1:]=(features[:,1:]-f['center'])/f['scale']
        gate=1/(1+np.exp(-np.clip(features@np.array(f['weights']),-35,35)))
        mixed=np.where(support>0,(1-gate)*p+gate*q,p)
        # C3-comparable corpus forced walk is explicitly transductive.
        standalone=np.where(fork,p,1.) if row['fork_aware'] else p
        fused=np.where(fork,mixed,1.)
        report['models'][name]=dict(neural=summary(p),standalone=summary(standalone),fusion=summary(fused))
    a=report['models']['emender_aware']; b=report['models']['transformer_aware']
    delta=a['standalone']['overall']['nll']-b['standalone']['overall']['nll']
    report.update(decisive_emender_minus_transformer=delta,
        fork_emender_minus_transformer=a['neural']['fork']['nll']-b['neural']['fork']['nll'],
        verdict='survives' if delta<0 else 'refuted',
        limitations=['One seed; no significance claim.',
          'Corpus-wide interval selection and forced labels are transductive; causal training-only selector not executed.',
          'OOD mass is uniformly distributed among unsupported bytes; empty support is uniform.',
          '180-second budget includes first kernel compilation but excludes index construction and initialization.'])
    write('results.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    {'prepare':prepare,'train':train,'analyze':analyze}[sys.argv[1]]()
