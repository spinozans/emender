"""Qualified 1024-byte K15 study. Checkpoints store port + immutable backbone reference."""
import hashlib
import json
import os
from itertools import islice
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from . import k15_bookmark_joint as j
from .k15_historical_port import HistoricalPort
from .k15_recovery_panel import panel

ANCHOR = j.B.parents[1]/'xsa_anchor_replays_v1/k15/final.pt'
ANCHOR_SHA = 'accb73ce3ad0cbe5b88d439dad8073d17a0ab3ec25a9f3f52b6cfdf7f968ecd0'
REACH = j.B.parents[1]/'k15_historical_reach/reach.jsonl'


def log_distribution(logits, trace):
    z=logits[0].float().log_softmax(-1)
    events=[e for e in trace if 'saved_pos' in e]
    if not events: return z
    pos=[e['stop']-1 for e in events]; byte=[e['calls'][-1][1]['byte'] for e in events]
    marked=z[pos].exp()*.95; marked=marked+.05*F.one_hot(torch.tensor(byte,device=z.device),256)
    mixed=z.clone(); mixed[pos]=marked.log(); return mixed


def inputs(text):
    seq=torch.tensor(list(text),device='cuda'); return seq[None,:-1],seq[1:]


def training_rows(stream):
    docs=list(islice((d for d in stream.documents('train') if d['length']>=2049),256))
    assert len({d['doc'] for d in docs})==256
    with stream.source.open('rb') as f:
        for start in (0,1024):
            for d in docs:
                f.seek(d['offset']); text=f.read(d['length']); assert hashlib.sha256(text).hexdigest()==d['sha256']
                yield d['doc'],start,text[start:start+1025]


def load_net():
    assert j.digest(ANCHOR)==ANCHOR_SHA and j.digest(j.B)=='0b4eeb8208b2a44f366d935a3bbaa702eed13e980258f545973b77a56c2a76f5'
    base,*_=j._build_k10(dict(j.incumbent._tuned_winner()),3401)
    base=j.convert_model_to_composition(base); base.load_state_dict(torch.load(ANCHOR,map_location='cpu',weights_only=True)); base.requires_grad_(False)
    net=j.BookmarkK15(base,j.EXE); net.port=HistoricalPort(net.port.byte.embedding_dim,net.port.action.in_features-net.port.byte.embedding_dim)
    net.port.decision_horizon=1024
    net.port.load_state_dict({n[5:]:v for n,v in torch.load(j.B,map_location='cpu',weights_only=True).items() if n.startswith('port.')})
    return net.cuda()


def load_port_checkpoint(path):
    """Reconstruct deployment model; apply log_distribution to its logits/trace."""
    state=torch.load(path,weights_only=True,map_location='cpu')
    assert state['anchor_sha256']==ANCHOR_SHA and state['arm'] in ('write','index') and state['step']==512
    net=load_net(); net.port.load_state_dict(state['port']); net.eval()
    net.port.intervention='clear' if state['arm']=='index' else 'none'
    return net


def check_prefix(net,x,end,full,events):
    p,t,*_=net(x[:,:end],pad_to=1024,audit=True)
    delta=(log_distribution(p,t)-full[:end]).abs().max().item()
    if events[:len(t)]!=t or delta!=0:
        raise ValueError(f'fixed-bucket prefix failure: end={end} exact_events={events[:len(t)]==t} max_abs_logp={delta}')


@torch.no_grad()
def score(net, stream, arm, out):
    net.eval()
    with (out/(arm+'_eval.jsonl')).open('x') as f, torch.autocast('cuda',dtype=torch.bfloat16):
        for n,(doc,text) in enumerate(panel(stream,'calibration')):
            x,y=inputs(text); keep=torch.arange(len(y),device=y.device); keep=(keep>=32)&(keep%16!=15)
            modes=['off','none','clear','address','echo'] if arm=='write' else ['off','clear']
            for mode in modes:
                net.port.intervention=mode
                z,tr,*_=net(x,doors=mode!='off',audit=not n); lp=log_distribution(z,tr)
                if not n and mode=='none':
                    for end in (16,17,129):
                        check_prefix(net,x,end,lp,tr)
                losses=F.nll_loss(lp,y,reduction='none'); assert losses.isfinite().all()
                f.write(json.dumps(dict(doc=doc,mode=mode,nll=float(losses.mean()),delayed=float(losses[keep].mean()),writes=sum('saved_pos' in e for e in tr),extensions=sum(e['op']==2 for e in tr),miss=sum(e.get('miss',False) for e in tr),trace=tr if not n else None))+'\n')


def replay_g(stream,out):
    from .k15_recovery_g import compare
    gpath=j.B.parents[1]/'k15_bookmarks_G/joint.pt'; dpath=j.B.parents[1]/'k15_bookmarks_D/joint.pt'
    assert j.digest(dpath)==j.D_SHA256
    lines=(gpath.parent/'console').read_text().splitlines(); snapshots=[json.loads(l) for l in lines if l.startswith('{')]
    assert snapshots[-1]=={'complete':True} and j.digest(gpath)==next(r['sha'] for r in snapshots if r.get('snapshot')==4096)
    snap=torch.load(gpath,map_location='cpu',weights_only=True); d=torch.load(dpath,map_location='cpu',weights_only=True)
    assert d['step']==snap['step']==4096 and all(torch.equal(d[k],snap[k]) for k in ('cpu_rng','cuda_rng')); del d
    base,*_=j._build_k10(dict(j.incumbent._tuned_winner()),3401); net=j.BookmarkK15(j.convert_model_to_composition(base),j.EXE); net.port=j.InterventionPort(net.port.byte.embedding_dim,net.port.action.in_features-net.port.byte.embedding_dim); net.load_state_dict(snap['model']); del snap
    compare(net.cuda(),stream,out);j.emit(G_replay=True,final_rng_equal=True)


def main(out):
    assert j._require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'
    assert j.digest(j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    assert j.digest(REACH)=='bb7153fba112d2808afb7d9331b4c3e66cfce4c1f29b3c6cb81b96fc7c41aacb'
    identity=j.flagship_identity(); assert identity['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'; stream=j.Stream(identity['source'],identity['manifest']); j.emit(identity=identity)
    replay_g(stream,out)
    reference=None
    for arm in ('write','index'):
        net=load_net(); net.backbone.eval(); net.port.train(); net.port.intervention='none' if arm=='write' else 'clear'
        opt=torch.optim.AdamW(net.port.parameters(),lr=1e-4,fused=True); torch.manual_seed(3407)
        with (out/(arm+'_train.jsonl')).open('x') as f, torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
            for step,(doc,start,text) in enumerate(training_rows(stream),1):
                x,y=inputs(text)
                if step==1:
                    with torch.no_grad():
                        off=net(x,doors=False)[0].clone(); assert torch.equal(off,net.backbone.head(net.backbone.features(x)))
                        net.port.eval(); z0,t0,*_=net(x,audit=True); lp0=log_distribution(z0,t0)
                        for end in (16,17,129):
                            check_prefix(net,x,end,lp0,t0)
                        net.port.train();j.emit(arm=arm,origin_prefix_exact=True)
                        if reference is None: reference=off
                        else: assert torch.equal(off,reference)
                z,tr,probs,_=net(x,audit=step==1); lp=log_distribution(z,tr); losses=F.nll_loss(lp,y,reduction='none'); loss=losses.mean()
                objective=loss+sum((p+q)*(losses[e['stop']-1:].mean().detach()-2) for p,q,e in zip(probs[::2],probs[1::2],tr)); opt.zero_grad(); objective.backward(); norm=torch.nn.utils.clip_grad_norm_(net.port.parameters(),1.,error_if_nonfinite=True);opt.step()
                f.write(json.dumps(dict(step=step,doc=doc,start=start,sha=hashlib.sha256(text).hexdigest(),loss=float(loss.detach()),grad=float(norm)))+'\n');f.flush()
                if step<=3 or step%32==0:j.emit(arm=arm,step=step,nll=float(loss.detach()))
                if step%128==0:
                    path=out/(arm+'.pt'); tmp=path.with_suffix('.partial'); torch.save(dict(port=net.port.state_dict(),optimizer=opt.state_dict(),step=step,arm=arm,anchor_sha256=ANCHOR_SHA,cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state()),tmp); os.replace(tmp,path)
        del net,opt
        score(load_port_checkpoint(path),stream,arm,out)
    j.emit(complete=True)

if __name__=='__main__': main(Path(sys.argv[1]))
