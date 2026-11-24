"""Frozen calibration counterfactuals, holding original decisions and addresses fixed."""
import json
import os
from pathlib import Path
import sys
import torch
from . import k15_coupled_training as t


@torch.no_grad()
def main(out):
    assert t.j._require_lease() in ('0','1') and os.environ['OMP_NUM_THREADS']=='4'
    source=out.parent/'k15_coupled_I'; state=source/'model.pt'; assert t.j.digest(state)=='1f02a84e5c9daf6c1de36c9a2820adbc1dd89f3a1847fca888e34bf5fbe10d8c'
    assert t.j.digest(t.j.EXE)=='87d917c19518e7dd25976420b25b2fbebef5fdcb36a09bc39e0ed76e3c96f40f'
    panel=json.loads((source/'panels.json').read_text()); identity=t.j.flagship_identity(); assert identity==panel['identity']; stream=t.j.Stream(identity['source'],identity['manifest']); cal=panel['cal']; net=t.model(state)
    with torch.autocast('cuda',dtype=torch.bfloat16,cache_enabled=False):
        for i,d in enumerate(cal):
            text=t.window(stream,d); x,y=t.inputs(text); net.port.mode='learned'; z,tr,*_=net(x); l=t.losses(z,y); forced={e['stop']:e['write'] for e in tr if 'write' in e}; stop=64*(1+i%7); adv=None; immediate=[s-1 for s in forced]; delay=[s+j for s in forced for j in range(16)]; changes={}
            for mode in ('no_latent','no_cursor','matched_echo'):
                net.port.mode=mode; zm,tm,*_=net(x,forced=forced); assert tm==tr; delta=t.losses(zm,y)-l
                changes[mode]=[float(delta.mean()),float(delta[immediate].mean()) if immediate else None,float(delta[delay].mean()) if delay else None]
            if stop in forced:
                net.port.mode='learned'; zother,_,*_=net(x,forced={**forced,stop:not forced[stop]}); assert torch.equal(zother[:,:stop],z[:,:stop]); delta=t.losses(zother,y)[stop:stop+16]-l[stop:stop+16]; adv=float(delta.mean())*(1 if forced[stop] else -1)
            t.j.emit(step=i+1,doc=d['doc'],changes=changes,write=forced.get(stop),adv=adv,eligible=len(forced),correct=sum(text[e['selected']]==text[e['stop']] for e in tr if 'selected' in e),echo=sum(text[s-1]==text[s] for s in forced),max_common=[max(c[0] for c in e['candidates']) for e in tr if e['candidates']],distinct=sum(len(set(c[2] for c in e['candidates']))>1 for e in tr))
    t.j.emit(complete=True)

if __name__=='__main__':main(Path(sys.argv[1]))
