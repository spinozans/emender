"""Variable-head pure proxy. Management parameters are immutable policy data.

This candidate is isolated from the certified X40 modules; it requires its own
parity receipt before scientific execution. No attention or projected adapters.
"""
from dataclasses import dataclass
from functools import partial
import math
from unittest.mock import patch
import torch
from torch import nn
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig
from ..x36_tape_perf import cell_fused as upstream
from ..x40_emender_lm.backward import autograd_scan
from ..x40_emender_lm.part6_checkpoint import checkpoint_block
from ..x36_tape_perf.loss import ChunkedLoss
from .velocity_forward import tape_scan


@dataclass(frozen=True)
class GatePolicy:
    match_wire: float
    occ_onset: float
    occ_slope: float


@dataclass(frozen=True)
class Policy:
    h_min: float = 128.
    ratio: float = 4.
    N: int = 4
    budget_min: int = 64
    budget_ratio: float = 2.85
    write_gate_bias: float = 2.
    match_coef: float = -1.
    match_wire: float = 1.
    occ_onset: float = .6
    occ_slope: float = .5
    pop_threshold: float = .1
    w_in_width: int = 64
    gate_cost_lambda: float = .01
    lr: float = 3e-4
    warmup_steps: int = 100

    def __post_init__(self):
        for name,value in vars(self).items():
            if name in ('match_wire','occ_onset','occ_slope'):continue
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
                raise ValueError(name)
        if not isinstance(self.N,int) or not 1 <= self.N <= 8: raise ValueError('N')
        if not isinstance(self.w_in_width,int) or not 1 <= self.w_in_width <= 256: raise ValueError('w_in_width')
        for name,lo,hi in [('match_wire',.25,4.),('occ_onset',.3,.9),('occ_slope',.1,2.)]:
            values=getattr(self,name)
            if isinstance(values,(list,tuple)):
                if len(values) not in (self.N,8):raise ValueError(name+' head count')
                object.__setattr__(self,name,tuple(values))
            else:values=(values,)
            if any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or not lo<=v<=hi for v in values):raise ValueError(name)
        if not 1 < self.h_min <= 1e7 or not 1 <= self.ratio <= 16: raise ValueError('horizons')
        if self.budget_min != int(self.budget_min) or self.budget_min < 1 or self.budget_ratio < 1: raise ValueError('budgets')
        if self.budgets[-1] > 262144: raise ValueError('budget exceeds native limit')
        if not 0 <= self.pop_threshold <= 1 or self.lr <= 0 or self.warmup_steps < 1 or self.gate_cost_lambda < 0: raise ValueError('optimizer/policy')
        for h in self.horizons:
            # The exact-decay spine cannot represent a horizon rounded to rate 1.
            import struct
            r=struct.unpack('<f',struct.pack('<f',math.exp(math.log(.001)/h)))[0]
            if r == 1.: raise ValueError('horizon rounds to infinite fp32 decay')

    def gate_policy(self,head):
        def select(name):
            value=getattr(self,name)
            return value[head] if isinstance(value,tuple) else value
        return GatePolicy(*(select(name) for name in ('match_wire','occ_onset','occ_slope')))

    @property
    def horizons(self): return tuple(self.h_min*self.ratio**i for i in range(self.N))
    @property
    def budgets(self): return tuple(round(self.budget_min*self.budget_ratio**i) for i in range(self.N))
    def configs(self):
        return tuple(TapeHeadConfig(n_heads=1,budget=b,decay_rate=math.exp(math.log(.001)/h),
                                   pop_threshold=self.pop_threshold) for b,h in zip(self.budgets,self.horizons))


def scan(k,q,v,beta,pop,append,cfg,append_coef,logits,match_coef,error_coef,policy):
    shape=(k.shape[0],k.shape[2],k.shape[1])
    evidence=torch.empty(shape,device=k.device);errors=torch.empty_like(evidence)
    occupancy=torch.empty_like(evidence);gates=torch.empty_like(evidence)
    beta=beta.detach().clone()
    forward=partial(tape_scan,wire_logits=logits.detach(),match_coef=match_coef.detach(),
                    error_coef=error_coef.detach(),evidence=evidence,error_evidence=errors,
                    match_wire=policy.match_wire,occ_onset=policy.occ_onset,occ_slope=policy.occ_slope,
                    occupancy=occupancy,gate_values=gates)
    # Same single-threaded injection used by X40. Replay records policy decisions.
    with patch.object(upstream,'persistent_scan',forward):
        if torch.is_grad_enabled() and any(a.requires_grad for a in (k,q,v,beta,pop,append,append_coef)):
            result=autograd_scan((k,q,v,beta,pop,append),None,cfg,append_coef)
        else: result=forward(k,q,v,beta,pop,append,None,cfg,append_coef)
    return result,occupancy.detach(),gates.detach()


class Cell(upstream.TapeCell):
    def __init__(self,vocab,policy):
        super().__init__(256,TapeHeadConfig(n_heads=policy.N,budget=max(policy.budgets)))
        self.policy=policy;self.configs=policy.configs()
        self.gate_policies=tuple(policy.gate_policy(h) for h in range(policy.N))
        del self.W_k,self.W_q
        self.features=nn.ModuleList();self.geometry=nn.ModuleList()
        for head in range(policy.N):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3701+head)
                embedding=nn.Embedding(vocab,64)
                key,query=nn.Linear(64,64),nn.Linear(64,64)
                query.load_state_dict(key.state_dict())
            embedding.requires_grad_(False);key.requires_grad_(False)
            self.features.append(embedding);self.geometry.append(nn.ModuleDict(dict(key=key,query=query)))
        self.W_in=nn.Sequential(nn.Linear(256,policy.w_in_width,bias=False),nn.Linear(policy.w_in_width,64,bias=False))
        for module in (self.W_write,self.W_pop,self.W_append):module.requires_grad_(False)
        self.append_err_coef.requires_grad_(False)
        with torch.no_grad():
            self.W_write.weight.zero_();self.W_write.bias.fill_(policy.write_gate_bias)
        self.register_buffer('match_coef',torch.full((policy.N,),policy.match_coef))
        self.register_buffer('error_coef',torch.zeros(policy.N))
        self.telemetry=None;self.gate_mean=None

    def forward(self,x,ids):
        from ..x39_memory_usage.control_evaluate import profile
        from ..x30_tape_cell.routing_trace import CASE_NAMES
        b,t=ids.shape;n=self.policy.N
        z=self.W_in(x);values=self.W_v(x).reshape(b,t,n,128)
        pop,append,logits=self.W_pop(x).sigmoid(),self.W_append(x),self.W_write(x)
        reads=[];rows=[];gate_means=[]
        for h,cfg in enumerate(self.configs):
            features=z+self.features[h](ids)
            k=self.geometry[h]['key'](features).detach().unsqueeze(2)
            q=self.geometry[h]['query'](features).unsqueeze(2)
            sl=slice(h,h+1)
            # Packed GPT-2 token 0 is content, never a padding sentinel.
            (r,state,trace,pressure),occ,gates=scan(k,q,values[:,:,sl].contiguous(),
                torch.ones(b,t,1,device=x.device),pop[:,:,sl].contiguous(),append[:,:,sl].contiguous(),cfg,
                self.append_err_coef[sl],logits[:,:,sl].contiguous(),self.match_coef[sl],self.error_coef[sl],self.gate_policies[h])
            reads.append(r);gate_means.append(gates.mean())
            if not self.training:
                row=profile(state,trace,pressure)
                touched=torch.zeros(b,cfg.slots,device=x.device,dtype=torch.bool)
                for batch in range(b):
                    targets=torch.cat((trace.alloc_target[batch].flatten(),trace.matched[batch].flatten()))
                    targets=targets[targets>=0];touched[batch,targets]=True
                live=(trace.alloc_target>=0).long().sub((trace.freed!=0).sum(-1)).cumsum(1)
                row.update(budget=cfg.slots,occ_frac=float(live.float().mean()/cfg.slots),
                    gate_mean=float(gates.mean()),prewrite_occ_frac=float(occ.mean()),
                    ever_touched_fraction=float(touched.float().mean()),
                    event_density=float((trace.case!=0).float().mean()))
                rows.append(row)
        self.gate_mean=torch.stack(gate_means).mean();self.telemetry=rows if not self.training else None
        return torch.cat(reads,2)


class Block(nn.Module):
    def __init__(self,vocab,policy,d):
        super().__init__();self.norm1=nn.LayerNorm(d);self.norm2=nn.LayerNorm(d)
        self.input=nn.Linear(d,256);self.cell=Cell(vocab,policy)
        self.read_out=nn.Linear(128*policy.N,d);self.ff=nn.Identity()
    def forward(self,x,ids):
        with torch.autocast('cuda',enabled=False):
            read=self.cell(self.input(self.norm1(x)).float(),ids).flatten(2)
        x=x+self.read_out(read)
        return x+self.ff(self.norm2(x))


class Candidate(nn.Module):
    def __init__(self,policy,vocab=50257,d=768,layers=12,target=150_000_000):
        super().__init__();self.policy=policy
        self.embed=nn.Embedding(vocab,d);self.norm=nn.LayerNorm(d);self.head=nn.Linear(d,vocab,bias=False)
        self.blocks=nn.ModuleList([Block(vocab,policy,d) for _ in range(layers)])
        base=sum(p.numel() for p in self.parameters() if p.requires_grad)
        hidden=round((target-base-layers*d)/(layers*(2*d+1)))
        if hidden<1: raise ValueError('proxy target below compulsory trainable parameters')
        for block in self.blocks:block.ff=nn.Sequential(nn.Linear(d,hidden),nn.GELU(),nn.Linear(hidden,d))
        self.hidden=hidden
    def features(self,ids):
        x=self.embed(ids);gates=[]
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():x,gate=checkpoint_block(block,x,ids)
            else:
                x=block(x,ids);gate=block.cell.gate_mean
            gates.append(gate)
        self.gate_mean=torch.stack(gates).mean()
        return self.norm(x)
    def loss(self,ids,targets):
        with torch.autocast('cuda',dtype=torch.bfloat16):hidden=self.features(ids)
        with torch.autocast('cuda',enabled=False):return ChunkedLoss.apply(hidden.float(),self.head.weight,targets,1024)
    def counts(self):return dict(total=sum(p.numel() for p in self.parameters()),trainable=sum(p.numel() for p in self.parameters() if p.requires_grad))
