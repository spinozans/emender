"""Per-policy chunk graphs; dense LM and optimizer remain outside capture."""
import torch
from torch import nn
from torch.nn import functional as F


class ScanModule(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg

    def forward(self,k,q,v,mask,pop,append,coef,logits,match,error):
        from .model import scan
        (read,_,_,_),_,_,sig=scan(k,q,v,mask,pop,append,self.cfg,coef,logits,match,error)
        return read,sig


def capture_scan(cfg,inputs):
    sample=tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)
    # make_graphed_callables captures separate forward/backward launch sequences
    # and retains the static buffers and graph pool for their complete lifetime.
    captured=torch.cuda.make_graphed_callables(ScanModule(cfg),sample,allow_unused_input=True)
    signature=tuple((x.shape,x.dtype,x.device,x.requires_grad) for x in sample)
    def checked(*args):
        if tuple((x.shape,x.dtype,x.device,x.requires_grad) for x in args)!=signature:
            raise ValueError('chunk graph input signature changed; disable and rebuild graphs for a new run')
        return captured(*args)
    return checked


class TrainingGraph:
    """Training helper selecting per-chunk graphs, with eager dense arithmetic."""
    def __init__(self,model,x,y,*,gate_cost=0.):
        self.model=model;self.gate_cost=gate_cost;self.shape=x.shape
        for block in model.blocks:block.cell.use_chunk_graphs=True

    def replay(self,x,y):
        if x.shape!=self.shape:raise ValueError('graph shape mismatch')
        self.model.zero_grad(set_to_none=True)
        logits,_=self.model(x)
        self.ce=F.cross_entropy(logits.flatten(0,1),y.flatten())
        loss=self.ce+self.gate_cost*torch.stack([b.cell.gate_mean for b in self.model.blocks]).mean()
        loss.backward()
        return loss


def certify_training_graph(model,x,y,*,gate_cost=0.):
    model.zero_grad(set_to_none=True)
    logits,_=model(x)
    loss=F.cross_entropy(logits.flatten(0,1),y.flatten())
    loss=loss+gate_cost*torch.stack([b.cell.gate_mean for b in model.blocks]).mean()
    loss.backward();reference=loss.detach().clone()
    gradients=[None if p.grad is None else p.grad.clone() for p in model.parameters()]
    del loss,logits
    graph=TrainingGraph(model,x,y,gate_cost=gate_cost)
    result=graph.replay(x,y)
    torch.testing.assert_close(result,reference,rtol=0,atol=0)
    for p,expected in zip(model.parameters(),gradients):
        if expected is None:assert p.grad is None
        else:torch.testing.assert_close(p.grad,expected,rtol=0,atol=0)
    return graph,dict(status='passed',batch=list(x.shape),loss=float(reference),gradients='bitwise eager versus per-chunk graphs')


def disable_chunk_graphs(model):
    for block in model.blocks:
        block.cell.use_chunk_graphs=False
        block.cell.chunk_graphs=[None]*4
        block.cell.gate_mean=None
