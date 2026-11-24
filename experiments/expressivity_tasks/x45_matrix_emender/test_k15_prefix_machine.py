import torch
from torch import nn
from experiments.expressivity_tasks.x45_matrix_emender.k15_phrase_qualification import PREFIX,j
from .k15_bookmarks import BookmarkSession
from .k15_prefix_machine import PrefixK15,PrefixMachine


def test_prefix_machine_expanded_paths_and_delayed_credit():
    torch.manual_seed(3411); b=nn.Module(); b.embed=nn.Embedding(256,8); b.norm=nn.LayerNorm(8); b.head=nn.Linear(8,256); b.blocks=nn.ModuleList([nn.Linear(8,8)])
    model=PrefixK15(b); s=BookmarkSession(j.EXE)
    try:
        m=PrefixMachine(model,s,PREFIX); paths=[(0,0,1.)]
        for step,y in enumerate(b'ABABA'):
            p=m.predict(); _,lit,edges,_,_=m.pending
            if step==0: earlier=next(e.weight for e in edges if e.byte==65)
            if step<2: assert int(p.argmax())==y
            # Expand every path from scratch using affine source scores and exact bytes;
            # shared neural output/row payloads condition the reference by definition.
            raw=m.tape.raw; n=len(raw); candidates={n-1,*[i+2 for i in range(n-2) if raw[i:i+2]==raw[-2:]][-2:]}
            matches=[i for k in (2,1) for i in reversed(range(k,n)) if raw[i-k:i]==raw[-k:]]; phrase=matches[0] if matches else -1
            if phrase>=0: candidates.add(phrase)
            aa=sorted(candidates); w=torch.tensor([3*(a==phrase)+(a-n)/n+.1*float(m.rows[a].payload.mean().detach()) for a in aa]).softmax(0); z=float(torch.sigmoid(torch.tensor(3.)))
            ref=torch.zeros_like(p); new=[]
            for d,age,q in paths:
                if d and age<3:
                    ref[raw[n-d]]+=q*z
                    if raw[n-d]==y: new.append((d,age+1,q*z))
                    q*=1-z
                ref+=q*(1-z)*lit.detach(); new.append((0,0,q*(1-z)*float(lit[y].detach())))
                for a,prob in zip(aa,w.tolist()):
                    ref[raw[a]]+=q*z*prob
                    if raw[a]==y: new.append((n-a,1,q*z*prob))
            assert torch.allclose(ref,p,atol=2e-6) and abs(float(p.sum())-1)<1e-5
            paths=[(d,k,q/float(ref[y])) for d,k,q in new]; m.observe(y)
        assert len(paths)>len(m.posterior) and torch.autograd.grad(-p[y].log(),earlier)[0].abs()>0
    finally: s.close()
