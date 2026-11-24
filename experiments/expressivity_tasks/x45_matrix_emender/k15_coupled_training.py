"""Stage I: independent, immutable backbone. Invoke via leased stage launcher."""
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
from . import k15_bookmark_joint as j
from .k15_historical_training import inputs, load_net, ANCHOR_SHA
from .k15_coupled_port import CoupledPort


def model(path=None):
    net=load_net(); net.port=CoupledPort(net.port.byte.embedding_dim,net.port.action.in_features-net.port.byte.embedding_dim).cuda(); net.eval()
    if path:
        state=torch.load(path,weights_only=True); assert state['anchor']==ANCHOR_SHA
        net.port.load_state_dict(state['port'])
    return net


def rows(stream):
    def select(split,a,b):
        return list(islice((d for d in stream.documents(split) if d['length']>=1025),a,b))
    train=select('train',0,192); cal=select('calibration',192,224); lock=select('calibration',224,256)
    assert len({d['doc'] for d in train+cal+lock})==256
    for name in ('D/eval','E/original','F/payload','G/panel','H/write_eval'):
        arm,file=name.split('/'); p=j.B.parent.parent/('k15_bookmarks_'+arm)/(file+'.jsonl'); prior={json.loads(l)['doc'] for l in p.open()}; assert not prior.intersection(d['doc'] for d in cal+lock)
    return train,cal,lock


def window(stream,d):
    with stream.source.open('rb') as f: f.seek(d['offset']); data=f.read(d['length'])
    assert hashlib.sha256(data).hexdigest()==d['sha256']; return data[:513]


def losses(z,y): return F.cross_entropy(z[0].float(),y,reduction='none')


def save(path,state):
    tmp=path.with_suffix('.partial'); torch.save(state,tmp); os.replace(tmp,path)


def fixed(z,text,events):
    p=z[0].float().softmax(-1)
    for e in events:
        s=e['stop']; cs=e['candidates']
        if not cs: continue
        _,addr,_=cs[0]
        for k in range(17):
            if addr+k>=s or text[addr:addr+k]!=text[s:s+k]: break
            p[s+k-1]*=.95; p[s+k-1,text[addr+k]]+=.05
    return p.clamp_min(1e-30).log()[None]


@torch.no_grad()
def evaluate(net,stream,cal,out):
    with (out/'cal.tmp').open('w') as f:
        for d in cal:
            text=window(stream,d); x,y=inputs(text); off=net(x,doors=False)[0]; modes={}; choice=None
            for mode in ('learned','always','skip','echo','fixed','off'):
                net.port.mode=mode
                if mode in ('fixed','off'): z=fixed(off,text,events) if mode=='fixed' else off
                else:
                    z,events,*_=net(x,forced=choice if mode=='echo' else None)
                    if mode=='learned':choice={e['stop']:e['write'] for e in events if 'write' in e}
                l=losses(z,y); immediate=[e['stop']-1 for e in events if e['candidates']]; delayed=[t for t0 in immediate for t in range(t0+1,t0+17)]
                modes[mode]=[float(l.mean()),float(l[immediate].mean()) if immediate else None,float(l[delayed].mean()) if delayed else None]; assert l.isfinite().all()
            print(json.dumps(dict(doc=d['doc'],modes=modes,writes=sum(choice.values()),eligible=len(choice))),file=f,flush=True)
    os.replace(out/'cal.tmp',out/'cal.jsonl')


def main(out):
    assert j._require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'; assert j.digest(j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    identity=j.flagship_identity(); assert identity['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'; stream=j.Stream(identity['source'],identity['manifest']); train,cal,lock=rows(stream); (out/'panels.json').write_text(json.dumps(dict(train=train,cal=cal,locked=lock,identity=identity)))
    torch.manual_seed(3411); net=model(); opt=torch.optim.AdamW(net.port.parameters(),lr=.001); path=out/'model.pt'; start=0
    if path.exists():
        state=torch.load(path,weights_only=True); assert state['anchor']==ANCHOR_SHA; net.port.load_state_dict(state['port']); opt.load_state_dict(state['optimizer']); start=state['step']
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        for i,d in enumerate(train[start:],start):
            text=window(stream,d); x,y=inputs(text); stop=64*(1+i%7)
            z,ev,gates,_=net(x,forced={stop:True},audit=i==0); loss=losses(z,y); obj=loss.mean(); adv=None
            if stop in dict(gates):
                z0,ev0,*_=net(x,forced={stop:False}); skip=losses(z0,y); adv=(skip-loss)[stop:stop+16].detach().mean(); obj=(obj+skip.mean())/2+F.binary_cross_entropy_with_logits(dict(gates)[stop].float(),(adv>0).float())
                assert torch.equal(z[:,:stop],z0[:,:stop])
            if i==0 or (adv is not None and not (out/'oracle').exists()):
                with torch.no_grad():
                    replay={e['stop']:e['write'] for e in ev if 'write' in e}; zr,tr,*_=net(x,forced=replay,audit=True); assert tr==ev and torch.equal(z,zr)
                    end=stop+12; zp,tp,*_=net(x[:,:end],forced=replay,audit=True,pad_to=512); assert tp==ev[:len(tp)] and torch.equal(z[:,:end],zp)
                    if adv is not None:
                        zs,ts,*_=net(x[:,:stop+16],forced={stop:False},audit=True,pad_to=512); assert torch.equal(zs,z0[:,:stop+16]); (out/'oracle').write_text(f'forced replay, prefix and write/skip direct suffix exact step {i+1} site {stop}\n')
            opt.zero_grad(); grad=0.
            if obj.requires_grad:
                obj.backward(); grad=float(torch.nn.utils.clip_grad_norm_(net.port.parameters(),1,error_if_nonfinite=True)); opt.step()
            j.emit(step=i+1,doc=d['doc'],nll=float(loss.detach().mean()),adv=float(adv) if adv is not None else None,grad=grad,eligible=len(gates),correct=sum(text[e['selected']]==text[e['stop']] for e in ev if 'selected' in e),echo=sum(text[e['stop']-1]==text[e['stop']] for e in ev if 'selected' in e))
            if (i+1)%16==0:save(path,dict(anchor=ANCHOR_SHA,step=i+1,port=net.port.state_dict(),optimizer=opt.state_dict()))
        assert (out/'oracle').exists(); evaluate(model(path),stream,cal,out); j.emit(complete=True)

if __name__=='__main__':main(Path(sys.argv[1]))
