"""Read-only P2 checks; never open another assessment panel."""
import argparse
from collections import Counter
import json
import os
import runpy
import subprocess
import tempfile
from pathlib import Path
import time

import numpy as np
import torch
from .k15_prefix_training import (j, OUT, R, ANCHOR_SHA, PrefixK15, PrefixMachine, BookmarkSession, PREFIX, setup, load_net, restore, payload)
from .k15_phrase_qualification import backbone
from .k15_counterfactual import ChoicePort
from .k15_coupled_port import CoupledPort
from .k15_historical_port import HistoricalPort
from .k15_relation import RelationPort
from .k15_handle_bank import BankPort, Recall

AUDIT=OUT.with_name('k15P2-audit')


def history():
    assert os.environ['CUDA_VISIBLE_DEVICES']==''
    root=j.B.parents[1]; base=backbone(); net=j.BookmarkK15(base,j.EXE); width=base.embed.embedding_dim; dim=net.port.action.in_features-width
    for p in sorted([*root.glob('k15*/*.pt'),*Path('output').glob('k15L*/*.pt')]):
        st=torch.load(p,map_location='cpu',weights_only=True,mmap=True)
        parent=p.parent.name
        if parent=='k15_bookmarks_A':
            # Stage A predates native additive bookmark reads; instantiate its pinned implementation.
            ref='bd0c0239:experiments/expressivity_tasks/x45_matrix_emender/k15_bookmarks.py'
            with tempfile.NamedTemporaryFile(suffix='.py') as f:
                f.write(subprocess.check_output(['git','show',ref])); f.flush()
                legacy=runpy.run_path(f.name)['BookmarkK15'](base,j.EXE); legacy.load_state_dict(st,strict=True); del legacy
        elif parent in ('k15_bookmarks_B','k15_bookmarks_D','k15_bookmarks_G'):
            net.port=j.InterventionPort(width,dim); net.load_state_dict(st.get('model',st),strict=True)
        else:
            assert st.get('anchor',st.get('anchor_sha256'))==ANCHOR_SHA
            if 'models' in st:
                for v in st['models'].values(): Recall(width).load_state_dict(v,strict=True)
            else:
                mod={'k15_bookmarks_H':lambda:HistoricalPort(width,dim),'k15_coupled_I':lambda:CoupledPort(width,dim),'k15_relation_J':lambda:RelationPort(dim),'k15_counterfactual_K':lambda:ChoicePort(width),'k15L-pilot':BankPort}[parent]()
                mod.load_state_dict(st.get('port',st.get('bank')),strict=True)
        j.emit(prior=str(p),sha=j.digest(p)); del st
    for s in ('N','O'):
        p=Path('output/k15'+s)/'checkpoint.json'; st=json.loads(p.read_text()); assert st['panel' if 'panel' in st else 'panels']==j.digest(p.with_name('panels.json')) and st['anchor']==ANCHOR_SHA
        j.emit(prior=str(p),sha=j.digest(p),weights=list(st['weights']))
    p=Path('output/k15M/pilot/result.json'); assert json.loads(p.read_text()); j.emit(m_pilot=str(p),sha=j.digest(p))


class TimedSession(BookmarkSession):
    def __init__(self):
        super().__init__(j.EXE); self.work=Counter(); self.seconds=0.
    def rpc(self,cmd):
        t=time.monotonic(); r=super().rpc(cmd); self.seconds+=time.monotonic()-t; self.work[cmd.split()[0]]+=1
        for k in ('tail_starts','sealed_rows'): self.work[k]+=r.get(k,0)
        return r


def gpu():
    assert os.environ['OMP_NUM_THREADS']=='4'; stream,_=setup(); net=load_net().eval(); m=PrefixK15(net.backbone).cuda().eval(); docs=json.loads((OUT/'panels.json').read_text())['train'][:8]; m.requires_grad_(False)
    for name in ('birth','reversed','compiled-8'):
        restore(m,OUT/('birth.pt' if name=='reversed' else name+'.pt')); s=TimedSession()
        try:
            if name=='reversed': m.operand.weight[0,0].neg_()
            state=PrefixMachine(m,s,PREFIX); trace=[]
            for i,y in enumerate(b'ABA'):
                p=state.predict(); e=max(state.pending[2],key=lambda e:float(e.weight)); trace.append(dict(decision=(e.op,*e.cursor),pa=float(p[65]),pb=float(p[66])))
                if name=='compiled-8' and i==0:
                    q=np.load(AUDIT/'cli.npy'); assert np.array_equal(p.cpu().numpy(),q) and abs(q.sum()-1)<1e-5; j.emit(cli_exact=True)
                state.observe(y)
            j.emit(fixture=name,trace=trace)
        finally: s.close()
    for name in ('birth','compiled-8'):
        restore(m,OUT/(name+'.pt'))
        for k,v in m.named_parameters(): v.requires_grad_(not k.startswith('backbone.') or k=='backbone.blocks.0.norm1.weight')
        for step,d in enumerate(docs,1):
            t=time.monotonic(); raw=payload(stream,d)[:136]; s=TimedSession(); loss=[]; delays=Counter(); newborn=Counter(); rows=Counter()
            try:
                state=PrefixMachine(m,s,raw[:128]); m.zero_grad(set_to_none=True)
                for y in raw[128:]:
                    p=state.predict(); n=len(state.rows); _,lit,edges,_,_=state.pending
                    for e in edges:
                        if e.byte in (y,None):
                            w=float((e.weight*(lit[y] if e.byte is None else 1)/p[y]).detach()); delays[n-e.address]+=w; rows[e.op]+=w
                            if e.address>=128: newborn[e.op]+=w
                    loss.append(-p[y].log()); state.observe(y)
                torch.cuda.synchronize(); fwd=time.monotonic()-t; torch.stack(loss).mean().backward(); torch.cuda.synchronize()
                j.emit(audit=name,step=step,doc=d['doc'],per_op_mass=rows,read_delay=delays,newborn_mass=newborn,phase_grad=m.phase.weight.grad.norm(dim=1).tolist(),transition_grad=m.transition.weight.grad.norm(dim=1).tolist(),addr_grad=m.operand.weight.grad.norm().item(),k15_grad=m.backbone.blocks[0].norm1.weight.grad.norm().item(),work=s.work,native_seconds=s.seconds,forward_seconds=fwd,total_seconds=time.monotonic()-t)
            finally: s.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('stage',choices=['history','gpu']); (history if parser.parse_args().stage=='history' else gpu)()
