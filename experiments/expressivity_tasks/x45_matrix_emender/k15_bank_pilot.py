"""Train-only opportunity and pricing; no assessment payloads opened here."""
from pathlib import Path
from itertools import islice
import json,sys,time
import torch
import numpy as np
from .k15_handle_bank import BankPort
from .k15_historical_training import j,load_net,REACH,ANCHOR_SHA
from .k15_relation_training import window
from .k15_coupled_training import save


def setup():
    assert j._require_lease() in ('0','1') and j.digest(j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    ident=j.flagship_identity(); assert ident['manifest_sha256']=='48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414'
    return j.Stream(ident['source'],ident['manifest']),ident


def panels(stream):
    root=j.B.parents[1]; used=set(); limits={}; hashes=set()
    for path in [REACH]+[p for a in 'DEFGH' for p in (root/f'k15_bookmarks_{a}').glob('*.jsonl')]:
        with path.open() as f: used.update(json.loads(l)['doc'] for l in f)
    for a in ['k15_coupled_I','k15_relation_J','k15_counterfactual_K']:
        for docs in json.loads((root/a/'panels.json').read_text()).values():
            if not isinstance(docs,list): continue
            for d in docs:
                used.add(d['doc']); limits[d['split']]=max(limits.get(d['split'],0),d['offset'])
    hashes.update(d['sha256'] for d in stream.documents_iter() if d['doc'] in used)
    def choose(split):
        for d in stream.documents(split):
            if d['offset']>limits[split] and d['doc'] not in used and d['sha256'] not in hashes and d['length']>=1025:
                used.add(d['doc']); hashes.add(d['sha256']); yield d
    tr=choose('train'); dev=choose('calibration'); result={}
    for name,size,it in [('full',64,tr),('short',64,tr),('train',1024,tr),('curve',32,dev),('dev',32,dev),('locked',32,dev)]:
        result[name]=list(islice(it,size)); assert len(result[name])==size
    return result


@torch.no_grad()
def extract(net,stream,d,audit=False):
    start=time.monotonic(); x,y=window(stream,d)
    with torch.autocast('cuda',dtype=torch.bfloat16): z,(options,meta),(trace,work),hidden=net(x,audit=audit)
    base=-z[0].float().log_softmax(-1).gather(1,y[:,None]).flatten().cpu(); y=y.cpu(); b=base[31:1023].reshape(31,1,32); target=y[31:1023].reshape(31,1,32); opt=torch.from_numpy(options)
    nll=torch.where(opt<0,b,-(.95*(-b).exp()+.05*(opt==target)).log()); gain=(nll[:,:1]-nll); inc=(opt!=opt[:,:1]); distinct=(opt.max(1).values!=opt.min(1).values)
    active=inc.any(-1); g=gain.sum(-1); informative=((g[:,:,None]!=g[:,None,:])&active[:,:,None]&active[:,None,:]).any(-1).any(-1)
    witness=next((r for block in trace for r in block if r['byte']==int(y[r['t']])),None)
    baseline=float(base.sum()+(nll[:,0]-b[:,0]).sum()); row=dict(doc=d['doc'],current_nll=baseline/1024,decisions=int(informative.sum()),eligible=int(inc.any(1).sum()),disagreement=int(distinct.sum()),witness=witness,work=work,seconds=time.monotonic()-start)
    conflict=(opt.max(1).values!=torch.where(opt<0,256,opt).min(1).values) & ((opt>=0).sum(1)>=2)
    return (hidden.cpu(),torch.tensor(meta,dtype=torch.float32),g,baseline,row['eligible'],(gain*conflict[:,None]).sum(-1),int(conflict.sum())),row


def main(out):
    stream,ident=setup(); p=panels(stream); (out/'panels.json').write_text(json.dumps(dict(identity=ident,**p))); net=load_net().eval(); net.port=BankPort()
    for name in ['full','short']:
        net.port.short=name=='short'; rows=[]; deadline=time.monotonic()+300
        with (out/f'{name}.jsonl').open('x') as f:
            for i,d in enumerate(p[name],1):
                assert time.monotonic()<deadline; _,r=extract(net,stream,d,i==1); rows.append(r); print(json.dumps(r),file=f,flush=True); j.emit(panel=name,step=i,seconds=r['seconds'])
                if i in (1,32,64): save(out/f'{name}_progress.pt',dict(step=i,anchor=ANCHOR_SHA,panels=j.digest(out/'panels.json'),bank=net.port.state_dict()))
        passed=sum(r['decisions'] for r in rows)>=64 and sum(r['decisions']>0 for r in rows)>=16 and any(r['witness'] for r in rows)
        j.emit(panel=name,passed=passed,decisions=sum(r['decisions'] for r in rows),docs=sum(r['decisions']>0 for r in rows));
        if passed: break
    j.emit(complete=True)

if __name__=='__main__': main(Path(sys.argv[1]))
