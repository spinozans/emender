"""Deploy a retained handle policy and audit causal prefix replays on train only."""
import json,sys,time
from pathlib import Path
import torch
from torch import nn
from .k15_handle_bank import BankPort,DecodePort,Recall,decode_last
from .k15_historical_training import load_net,j,ANCHOR_SHA
from .k15_bank_pilot import setup
from .k15_relation_training import window


class RankControl(nn.Module):
    def __init__(self,model,mode):
        super().__init__(); self.model=model; self.mode=mode
    def forward(self,h,m):
        z=self.model(h,m); a=z.argmax(-1); count=m[:,:,3].sum(-1).long()
        if self.mode=='fifo': pick=(count>0).long()
        elif self.mode=='reassign': pick=torch.where(a>0,1+a%count.clamp_min(1),0)
        else: raise ValueError(self.mode)
        return torch.zeros_like(z).scatter_(1,pick[:,None],1.)


def load_decoder(path,panels,seed=3414):
    state=torch.load(path,weights_only=True,map_location='cpu'); assert state['panels']==j.digest(panels) and state['anchor']==ANCHOR_SHA
    net=load_net().eval(); controller=Recall(net.backbone.embed.embedding_dim).cuda(); controller.load_state_dict(state['models'][str((seed,'fresh'))]); net.port=DecodePort(controller.eval()); return net


@torch.no_grad()
def predict_next(net,ids):
    """One document-local prefix, full label-free 256-byte distribution."""
    if ids.shape[0]!=1 or not 1<=ids.shape[1]<=1024: raise ValueError('one independent 1..1024 byte prefix')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if ids.shape[1]<32:
            z=net(ids,doors=False,pad_to=1024)[0]; return z[0,-1].float().softmax(-1),[],{},[]
        z,selected,work,picks=net(ids,pad_to=1024); return decode_last(z,selected),selected,work,picks


@torch.inference_mode()
def main(out):
    stream,identity=setup(); panels=out.with_name('k15L-pilot')/'panels.json'; panel=json.loads(panels.read_text()); assert panel['identity']==identity
    path=out.with_name('k15L-train')/'dose1024.pt'; net=load_decoder(path,panels); model=net.port.controller
    with (out/'transactions.jsonl').open('x') as f:
        for d in panel['train'][:4]:
            x,y=window(stream,d); net.port=BankPort()
            with torch.autocast('cuda',dtype=torch.bfloat16): z,(opts,m),(trace,_),h=net(x,audit=True)
            ranks=model(h,torch.tensor(m,device=h.device,dtype=torch.float32)).argmax(-1).tolist(); options=opts[range(31),ranks].flatten(); targets=y.tolist(); hit=next((r for i,rows in enumerate(trace) for r in rows if r['rank']==ranks[i] and r['byte']==targets[r['t']]),None); miss=next((i+31 for i,v in enumerate(options) if i+31>=64 and v<0),None)
            for t,event in [(hit['t'] if hit else None,hit),(miss,None)]:
                if t is None: continue
                for mode in ['adaptive','current','fifo','reassign']:
                    net.port=DecodePort(model if mode=='adaptive' else None if mode=='current' else RankControl(model,mode)); torch.cuda.synchronize(); start=time.perf_counter(); dist,sel,work,picks=predict_next(net,x[:,:t+1]); torch.cuda.synchronize(); elapsed=time.perf_counter()-start
                    i=(t-31)//32; lag=(t-31)%32; b=int(sel.flatten()[t-31]); assert b==opts[i,picks[-1],lag] and (mode!='adaptive' or picks[-1]==ranks[i]); ref=z[0,t].float().softmax(-1)
                    if b>=0: ref=ref*.95; ref[b]+=.05
                    delta=float((ref-dist).abs().max()); assert delta<1e-6
                    print(json.dumps(dict(doc=d['doc'],mode=mode,t=t,witness=event,chosen=picks[-1],byte=b,target=targets[t],p=float(dist[targets[t]]),seconds=elapsed,work=work,max_abs_delta=delta)),file=f,flush=True)
            j.emit(doc=d['doc'],hit=bool(hit),miss=miss is not None)
    j.emit(complete=True,checkpoint=j.digest(path))

if __name__=='__main__': main(Path(sys.argv[1]))
