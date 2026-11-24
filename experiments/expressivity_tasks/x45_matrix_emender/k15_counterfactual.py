"""K: matched fixed-copy readouts and residual source choice at K15's middle mixer."""
import json
from pathlib import Path
import sys
import time
from itertools import islice

import torch
from torch import nn
from .k15_relation import reads
from .k15_relation_training import window
from .k15_historical_training import j, load_net, ANCHOR_SHA, REACH
from .k15_coupled_training import save
from .k15_coupled_summary import interval


def alternatives(sites):
    # Original address is never replaced within this handle's lifetime.
    original = None
    for t, bs, addresses, stop in sites:
        if t == stop-1:
            original = addresses[-1]
        latest = bs[-1]
        alt = next((i for i in reversed(range(len(bs))) if bs[i] != latest), -1)
        yield original == addresses[-1], bs[alt], ((addresses[-1]-addresses[alt])/stop,
              bs.count(bs[alt])/len(bs), bs.count(latest)/len(bs), (t-stop+1)/32)


class ChoicePort(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.switch = nn.Linear(width+4, 1)
        nn.init.zeros_(self.switch.weight); nn.init.zeros_(self.switch.bias)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, mixer, hidden, ids, session, forced=None, audit=False):
        sites, work = reads(bytes(ids[0].tolist()), session, audit=audit)
        desc = list(alternatives(sites))
        t = [r[0] for r in sites]
        feat = hidden.new_tensor([r[2] for r in desc]).reshape(-1,4)
        score = self.switch(torch.cat((hidden[0,t],feat),dim=1)).flatten().float()
        return hidden.new_zeros(1,ids.shape[1],mixer.value_dim).float(), sites, work, (score,desc)


def assign(stream):
    root = j.B.parents[1]; used = set(); hashes = set()
    for name in ('k15_coupled_I','k15_relation_J'):
        prior = json.loads((root/name/'panels.json').read_text())
        for key in ('train','cal','locked'):
            used.update(d['doc'] for d in prior[key])
    limits = {s:max(d['offset'] for key in ('train','cal','locked') for d in prior[key] if d['split']==s) for s in ('train','calibration')}
    for path in [REACH]+[p for arm in 'DEFGH' for p in (root/f'k15_bookmarks_{arm}').glob('*.jsonl')]:
        with path.open() as f: used.update(json.loads(line)['doc'] for line in f)
    hashes.update(d['sha256'] for d in stream.documents_iter() if d['doc'] in used)
    def choose(split):
        for d in stream.documents(split):
            if d['offset']>limits[split] and d['length']>=1025 and d['doc'] not in used and d['sha256'] not in hashes:
                used.add(d['doc']); hashes.add(d['sha256']); yield d
    tr=list(islice(choose('train'),96)); dev=list(islice(choose('calibration'),112)); assert len(tr)==96 and len(dev)==112
    return dict(pool=tr[:32],train=tr[32:],isolate=dev[:32],curve=dev[32:48],dev=dev[48:80],locked=dev[80:])


def measure(net,stream,d,audit=False):
    x,y=window(stream,d); start=time.perf_counter(); z,sites,work,(score,desc)=net(x,audit=audit)
    if audit: assert torch.equal(z,net(x,doors=False)[0]) and not score.any()
    base=z[0].float().log_softmax(-1).gather(1,y[:,None]).flatten(); t=torch.tensor([r[0] for r in sites],device=y.device,dtype=torch.long)
    def tensor(seq,typ): return torch.tensor(seq,device=y.device,dtype=typ)
    hit=tensor([r[1][-1] for r in sites],torch.long)==y[t]; alt=tensor([r[1] for r in desc],torch.long)==y[t]; valid=tensor([r[0] for r in desc],torch.bool)
    distinct=tensor([len(set(r[1]))>1 for r in sites],torch.bool)
    def local(mass): return -(.95*base[t].exp()+.05*mass).log()
    latest=local(hit); other=local(alt); sw=(score>0)&distinct
    def total(active): return (-base.sum()+(active+base[t]).sum())/len(y)
    losses=dict(latest=total(latest),single=total(torch.where(valid,latest,-base[t])),choice=total(torch.where(sw,other,latest)))
    idx=tensor([i for i,r in enumerate(sites) for _ in r[1]],torch.long)
    ages=tensor([(a-s)/s for _,_,aa,s in sites for a in aa],torch.float32)
    match=tensor([b for _,bb,_,_ in sites for b in bb],torch.long)==y[t][idx]
    for key,beta in [('constant',1.),('fitted',net.port.beta.clamp(-8,8))]:
        w=(ages*beta).exp(); num=w.new_zeros(len(t)).index_add(0,idx,w*match); den=w.new_zeros(len(t)).index_add(0,idx,w)
        losses[key]=total(local(num/den))
    expected=total(latest+score.sigmoid()*(other-latest))
    row=dict(doc=d['doc'],loss={k:v.item() for k,v in losses.items()},active=len(t),fallback=int((~valid).sum()),distinct=int(distinct.sum()),latest_correct=int((hit&distinct).sum()),alt_correct=int((alt&distinct).sum()),switches=int(sw.sum()),refs=len(idx),work=work,seconds=time.perf_counter()-start)
    assert all(torch.isfinite(v) for v in losses.values()); return row,expected,losses['fitted']


def main(out):
    assert j._require_lease() in ('0','1') and j.digest(j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    identity=j.flagship_identity(); assert identity['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'; stream=j.Stream(identity['source'],identity['manifest']); p=assign(stream); (out/'panels.json').write_text(json.dumps(dict(identity=identity,**p)))
    torch.manual_seed(3413); net=load_net(); net.port=ChoicePort(net.backbone.embed.embedding_dim).cuda(); net.eval()
    def evaluate(name,docs):
        with torch.no_grad(), (out/(name+'.jsonl')).open('x') as f:
            rows=[]
            for d in docs:
                row=measure(net,stream,d)[0]; rows.append(row); print(json.dumps(row),file=f,flush=True)
        contrasts={a+'-'+b:interval([r['loss'][a]-r['loss'][b] for r in rows]) for a,b in [('single','latest'),('constant','fitted'),('latest','constant'),('latest','fitted'),('latest','choice')]}; j.emit(panel=name,contrasts=contrasts); return rows,contrasts
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        for phase,lr in [('pool',.05),('train',.003)]:
            opt=torch.optim.Adam(net.port.parameters(),lr=lr)
            for i,d in enumerate(p[phase],1):
                row,choice,pool=measure(net,stream,d,audit=phase=='pool' and i==1); opt.zero_grad(); (pool if phase=='pool' else choice).backward(); opt.step(); j.emit(phase=phase,step=i,**row)
                if phase=='train' and i in (16,32,64):
                    save(out/f'dose{i}.pt',dict(anchor=ANCHOR_SHA,port=net.port.state_dict(),step=i,panels=j.digest(out/'panels.json'))); evaluate(f'curve{i}',p['curve'])
            if phase=='pool':
                j.emit(beta=net.port.beta.item()); rows,_=evaluate('isolate',p['isolate'])
                if sum(r['distinct'] for r in rows)<64 or sum(r['distinct']>0 for r in rows)<8:
                    j.emit(complete=True,stop='insufficient disagreement'); return
        rows,c=evaluate('dev',p['dev']); gate=c['latest-choice'][1]>0 and sum(r['switches']>0 for r in rows)>=8
        if gate: evaluate('confirmation',p['locked'])
        j.emit(complete=True,dev_pass=gate)

if __name__=='__main__': main(Path(sys.argv[1]))


@torch.no_grad()
def copy_logprobs(logits, sites, controller, *, noop=False):
    """Decode one K15 pass without targets. No-op always keeps the latest survivor."""
    score, desc = controller
    lp = logits[0].float().log_softmax(-1)
    if not sites:
        return lp
    pos = [r[0] for r in sites]
    latest = torch.tensor([r[1][-1] for r in sites], device=lp.device)
    alternate = torch.tensor([r[1] for r in desc], device=lp.device)
    source = latest if noop else torch.where(score > 0, alternate, latest)
    probs = .95 * lp[pos].exp()
    probs[torch.arange(len(pos), device=lp.device),source] += .05
    lp[pos] = probs.clamp_min(1e-30).log()
    return lp
