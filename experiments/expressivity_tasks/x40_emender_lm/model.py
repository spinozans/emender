"""Six pure memory blocks; no attention or positional attention machinery."""
from dataclasses import asdict
from functools import partial
import math
import struct
from unittest.mock import patch
import torch
from torch import nn
from ..x36_tape_perf import cell_fused as upstream
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig
from ..x37_head_differentiation.model import metrics, parameter_receipt
from ..x39_memory_usage.control_evaluate import profile
from .forward import tape_scan as wired_forward

HORIZONS = (128,512,2048,8192)
BUDGETS = (64,256,736,1472)
POLICIES = tuple(TapeHeadConfig(n_heads=1,budget=b,
    decay_rate=math.exp(math.log(.001)/h),pop_threshold=p)
    for b,h,p in zip(BUDGETS,HORIZONS,(.01,.1,.5,.9)))


def policy_receipt():
    rows=[]
    for cfg,h in zip(POLICIES,HORIZONS):
        rate=struct.unpack('<f',struct.pack('<f',cfg.decay_rate))[0]
        rows.append(dict(config=asdict(cfg),requested_horizon=h,float32_decay=rate,
            float32_t2_horizon=math.log(cfg.free_threshold)/math.log(rate)))
    return rows


def scan(k,q,v,mask,pop,append,cfg,append_coef,logits,match_coef,error_coef):
    evidence=torch.empty(k.shape[0],k.shape[2],k.shape[1],device=k.device)
    errors=torch.empty_like(evidence)
    beta=mask.detach().clone()
    forward=partial(wired_forward,wire_logits=logits.detach(),
        match_coef=match_coef.detach(),error_coef=error_coef.detach(),
        evidence=evidence,error_evidence=errors)
    with patch.object(upstream,'persistent_scan',forward):
        result=upstream.tape_scan(k,q,v,beta,pop,append,None,cfg,append_coef)
    evidence=evidence.transpose(1,2).contiguous().detach()
    errors=errors.transpose(1,2).contiguous().detach()
    sigmoid=(logits+match_coef*evidence+error_coef*errors).sigmoid()
    return result,evidence,errors,sigmoid


class Cell(upstream.TapeCell):
    def __init__(self,vocab):
        super().__init__(256,TapeHeadConfig(n_heads=4,budget=2944))
        del self.W_k,self.W_q
        self.features=nn.ModuleList(); self.geometry=nn.ModuleList()
        for head in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3701+head)
                embedding=nn.Embedding(vocab,64)
                key,query=nn.Linear(64,64),nn.Linear(64,64)
                query.load_state_dict(key.state_dict())
            embedding.requires_grad_(False);key.requires_grad_(False)
            self.features.append(embedding)
            self.geometry.append(nn.ModuleDict(dict(key=key,query=query)))
        self.W_in=nn.Linear(256,64,bias=False)
        with torch.no_grad(): self.W_in.weight.copy_(torch.eye(256)[:64])
        self.W_write.requires_grad_(True)
        # Frozen policy drives; learned content and enable wire only.
        self.W_pop.requires_grad_(False);self.W_append.requires_grad_(False)
        self.append_err_coef.requires_grad_(False)
        self.match_coef=nn.Parameter(torch.full((4,),-1.))
        self.error_coef=nn.Parameter(torch.zeros(4))
        self.configs=POLICIES
        self.telemetry=None
        self.gate_mean=None

    def forward(self,x,ids):
        b,t=ids.shape
        z=self.W_in(x)
        values=self.W_v(x).reshape(b,t,4,128)
        pop,append,logits=self.W_pop(x).sigmoid(),self.W_append(x),self.W_write(x)
        reads=[];telemetry=[];gates=[]
        for h,cfg in enumerate(self.configs):
            features=z+self.features[h](ids)
            k=self.geometry[h]['key'](features).detach().unsqueeze(2)
            q=self.geometry[h]['query'](features).unsqueeze(2)
            sl=slice(h,h+1)
            (r,state,trace,pressure),e,err,sig=scan(k,q,values[:,:,sl].contiguous(),
                (ids!=0).float().unsqueeze(-1),pop[:,:,sl].contiguous(),
                append[:,:,sl].contiguous(),cfg,self.append_err_coef[sl].contiguous(),
                logits[:,:,sl].contiguous(),self.match_coef[sl],self.error_coef[sl])
            reads.append(r);gates.append(sig.mean())
            if not self.training:
                m=profile(state,trace,pressure)
                m.update(metrics(state,trace,pressure))
                m['gate_mean']=float(sig.mean())
                m['match_mean']=float(e.mean());m['error_mean']=float(err.mean())
                telemetry.append(m)
        self.gate_mean=torch.stack(gates).mean()
        self.telemetry=telemetry if not self.training else None
        return torch.cat(reads,2)


class Block(nn.Module):
    def __init__(self,vocab):
        super().__init__()
        self.norm1,self.norm2=nn.LayerNorm(256),nn.LayerNorm(256)
        self.cell=Cell(vocab)
        self.read_out=nn.Linear(512,256)
        self.ff=nn.Sequential(nn.Linear(256,1024),nn.GELU(),nn.Linear(1024,256))

    def forward(self,x,ids):
        x=x+self.read_out(self.cell(self.norm1(x).float(),ids).flatten(2))
        return x+self.ff(self.norm2(x))


class LM(nn.Module):
    def __init__(self,vocab=32768, *, ignition=False):
        super().__init__()
        self.embed=nn.Embedding(vocab,256)
        blocks=[]
        for i in range(6):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3401+i)
                blocks.append(Block(vocab))
        self.blocks=nn.ModuleList(blocks)
        if ignition:
            for block in self.blocks:
                cell=block.cell
                with torch.no_grad():
                    cell.W_write.weight.zero_();cell.W_write.bias.fill_(2.)
                    cell.match_coef.zero_();cell.error_coef.zero_()
                cell.W_write.requires_grad_(False)
                cell.match_coef.requires_grad_(False);cell.error_coef.requires_grad_(False)
        self.norm=nn.LayerNorm(256)
        self.head=nn.Linear(256,vocab,bias=False)

    def features(self,ids):
        x=self.embed(ids)
        for block in self.blocks: x=block(x,ids)
        return self.norm(x)

    def forward(self,ids):
        x=self.features(ids)
        return self.head(x),[b.cell.telemetry for b in self.blocks]


def require_replays(loss):
    seen=set();stack=[loss.grad_fn];count=0
    while stack:
        node=stack.pop()
        if node is None or node in seen: continue
        seen.add(node)
        count+=type(node).__name__=='ReplayBackward'
        stack.extend(n for n,_ in node.next_functions)
    if count!=24: raise AssertionError(f'Expected 24 ReplayBackward nodes, found {count}')
    return count
