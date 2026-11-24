from .k15_handle_bank import Seed,continuations,enumerate_bank
from .k15_relation import reads
import random

class Exact:
    def __init__(self): self.raw=b''
    def rpc(self,line):
        op,h,*_=line.split(); pat=bytes.fromhex(h)
        if op=='append': self.raw+=pat; return dict(visible=len(self.raw))
        return dict(positions=[i for i in range(len(self.raw)-len(pat)) if self.raw[i:i+len(pat)]==pat],tail_starts=0,sealed_rows=0)

def test_current_identical_and_suffix_causal():
    raw=random.Random(42).choices(b'abcd',k=1024); raw=bytes(raw); opts,*_=enumerate_bank(raw,Exact(),audit=True)
    sites,_=reads(raw,Exact()); assert {31+i:v for i,v in enumerate(opts[:,0].flatten()) if v>=0}=={t:bs[-1] for t,bs,_,_ in sites}
    changed=enumerate_bank(raw[:733]+b'x'*291,Exact())[0]; assert (opts[:21]==changed[:21]).all() and (opts[21,:,:30]==changed[21,:,:30]).all()

def test_restart_preserves_original_payload():
    raw=b'abQabRab'+b'x'*25+b'abR'; seed=Seed(8,b'ab',(2,5)); vals,origin=continuations(raw+b'x'*40,32,seed,True,False); assert vals[3]==ord('R') and origin[3]==35 and vals[4]==ord('a')

def test_label_free_decode_and_delayed_gradient():
    import torch
    from .k15_handle_bank import decode_last,Recall
    from experiments.expressivity_tasks.x45_matrix_emender.k15_bank_training import gains
    import numpy as np
    z=torch.zeros(1,33,256); sel=np.full((1,32),-1); assert decode_last(z,sel)[0]==1/256
    sel[0,1]=65; assert abs(float(decode_last(z,sel)[65])-(.05+.95/256))<1e-6
    model=Recall(6); h=torch.randn(3,6); m=torch.rand(3,5,4); m[:,0]=0; m[:,1:,3]=1; g=torch.randn(3,5); item=(h,m,g,0,0,g,0)
    loss=(model(h,m).softmax(1)*g).sum(); loss.backward(); assert model.score[0].weight.grad.norm()>0
    assert set(gains(item,model,4))=={'current','adaptive','fifo','fixed','reassign'}

def test_handle_reassignment_wraps_valid_ranks_only():
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k15_bank_decode import RankControl
    class Scores(torch.nn.Module):
        def forward(self,h,m): return h
    scores=torch.eye(5)[[0,1,2,1]]; m=torch.zeros(4,5,4); m[:3,1:3,3]=1.; m[3,1,3]=1.; assert RankControl(Scores(),'reassign')(scores,m).argmax(1).tolist()==[0,2,1,1]
