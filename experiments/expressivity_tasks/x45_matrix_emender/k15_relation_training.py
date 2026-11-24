"""Run the preregistered TRAIN/CAL relation repair inside a project lease."""
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time
import torch
from torch.nn import functional as F
from .k15_coupled_training import save, losses
from .k15_coupled_summary import interval
from .k15_historical_training import load_net, inputs, ANCHOR_SHA, j, REACH
from .k15_relation import RelationPort


def panels(stream):
    root=j.B.parents[1]; prior=json.loads((root/'k15_coupled_I/panels.json').read_text()); used={d['doc'] for p in ('train','cal','locked') for d in prior[p]}
    for path in [REACH]+[p for arm in 'DEFGH' for p in (root/f'k15_bookmarks_{arm}').glob('*.jsonl')]:
        with path.open() as f: used.update(json.loads(line)['doc'] for line in f)
    hashes={d['sha256'] for d in stream.documents_iter() if d['doc'] in used}
    def choose(split):
        for d in stream.documents(split):
            if d['length']>=1025 and d['doc'] not in used and d['sha256'] not in hashes:
                used.add(d['doc']); hashes.add(d['sha256']); yield d
    train=list(islice(choose('train'),256)); other=list(islice(choose('calibration'),96)); assert len(train)==256 and len(other)==96
    return dict(train=train,cal=other[:48],locked=other[48:])


def window(stream,d):
    with stream.source.open('rb') as f: f.seek(d['offset']); raw=f.read(d['length'])
    assert hashlib.sha256(raw).hexdigest()==d['sha256']; return inputs(raw[:1025])


def teacher(off,sites):
    t=[r[0] for r in sites]; b=[r[1][-1] for r in sites]; p=off[0,t].float().softmax(-1)*.95; p[range(len(t)),b]+=.05; return p


def model(path=None):
    net=load_net(); net.port=RelationPort(net.backbone.blocks[len(net.backbone.blocks)//2].mixer.value_dim).cuda(); net.eval()
    if path:
        st=torch.load(path,weights_only=True); assert st['anchor']==ANCHOR_SHA; net.port.load_state_dict(st['port'])
    return net


@torch.no_grad()
def score(net,out,name,docs,stream):
    rows=[]
    with (out/(name+'.jsonl')).open('w') as f:
        for d in docs:
            x,y=window(stream,d); modes={}; ms={}
            for mode in ('off','full','immediate','echo','scramble','fixed'):
                net.port.mode=mode; torch.cuda.synchronize(); start=time.perf_counter()
                if mode=='fixed':
                    z=off.float().log_softmax(-1); z[0,[r[0] for r in full]]=teacher(off,full).clamp_min(1e-30).log()
                else: z,sites,stats,_=net(x,doors=mode!='off')
                torch.cuda.synchronize(); ms[mode]=1000*(time.perf_counter()-start)
                if mode=='off': off=z
                if mode=='full': full=sites; work=stats
                modes[mode]=losses(z,y); assert modes[mode].isfinite().all()
            imm=[t for t,bs,a,s in full if t==s-1]; delayed=[t for t in range(1024) if t%32!=31 or t==1023]
            row=dict(doc=d['doc'],work=work,immediate=len(imm),active=len(full),ms=ms,modes={k:[float(l.mean()),float(l[imm].mean()) if imm else None,float(l[delayed].mean())] for k,l in modes.items()}); rows.append(row); print(json.dumps(row),file=f,flush=True)
    contrasts={f'{ref}-full/{k}':interval([r['modes'][ref][k]-r['modes']['full'][k] for r in rows if r['modes'][ref][k] is not None]) for ref,k in [('off',0),('off',1),('echo',1),('immediate',2),('scramble',2)]}; (out/(name+'_gate.json')).write_text(json.dumps(contrasts)); return all(v[1]>0 for v in contrasts.values())


def main(out):
    assert j._require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'; assert j.digest(j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    identity=j.flagship_identity(); assert identity['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'; stream=j.Stream(identity['source'],identity['manifest']); p=panels(stream); manifest=json.dumps(dict(**p,identity=identity)); (out/'panels.json').write_text(manifest); panelsha=hashlib.sha256(manifest.encode()).hexdigest()
    torch.manual_seed(3412); net=model(); opt=torch.optim.AdamW(net.port.parameters(),lr=.01); start=0; path=out/'model.pt'
    if path.exists():
        st=torch.load(path,weights_only=True); assert st['anchor']==ANCHOR_SHA and st['panels']==panelsha; net.port.load_state_dict(st['port']); opt.load_state_dict(st['optimizer']); start=st['step']
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        for i,d in enumerate(p['train'][start:],start+1):
            x,y=window(stream,d); off=net(x,doors=False)[0]; z,sites,work,_=net(x,audit=i==1); grad=0.; kl=None
            if i==1:
                assert torch.allclose(z,off,atol=.125,rtol=.01); (out/'oracle').write_text('all native hit sets exact first doc; zero-init compatible\n')
            if sites:
                obj=F.kl_div(z[0,[r[0] for r in sites]].float().log_softmax(-1),teacher(off,sites),reduction='batchmean'); opt.zero_grad(); obj.backward(); grad=float(torch.nn.utils.clip_grad_norm_(net.port.parameters(),1,error_if_nonfinite=True)); opt.step(); kl=float(obj.detach())
            j.emit(step=i,doc=d['doc'],work=work,active=len(sites),grad=grad,kl=kl,nll=float(losses(z.detach(),y).mean()))
            if i%32==0:
                save(path,dict(step=i,anchor=ANCHOR_SHA,panels=panelsha,port=net.port.state_dict(),optimizer=opt.state_dict()));
                if i in (64,128,256): os.link(path,out/f'dose{i}.pt')
        net=model(path); passed=score(net,out,'cal',p['cal'],stream); confirm=score(net,out,'confirmation',p['locked'],stream) if passed else None; j.emit(complete=True,passed=passed,confirmed=confirm)

if __name__=='__main__': main(Path(sys.argv[1]))
