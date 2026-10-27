"""X39-local recurrent match wire; X31/X36 sources remain read-only.

Evidence uses certified READ's 2/sqrt(K), before POP/WRITE/DECAY, detached.
The existing append error coefficient remains in append amplitude only.
The added learned per-head match coefficient starts at -1. No new error term.
"""
from functools import partial
from unittest.mock import patch
import torch
from torch import nn
from ..x36_tape_perf import cell_fused as upstream
from .semantic_forward import tape_scan as wired_forward


def scan(k,q,v,mask,pop,append,state,cfg,append_err_coef,write_logits,match_coef):
    # The attended session is single threaded. Patch only the dispatch target
    # during this call, including Replay.forward's nested no-grad invocation.
    evidence=torch.empty(k.shape[0],k.shape[2],k.shape[1],device=k.device)
    beta=mask.detach().clone()
    forward=partial(wired_forward,wire_logits=write_logits.detach(),
                    match_coef=match_coef.detach(),evidence=evidence)
    with patch.object(upstream,'persistent_scan',forward):
        result=upstream.tape_scan(k,q,v,beta,pop,append,state,cfg,append_err_coef)
    evidence=evidence.transpose(1,2).contiguous().detach()
    sigmoid=(write_logits+match_coef*evidence).sigmoid()
    return result,evidence,sigmoid


class SemanticCell(upstream.TapeCell):
    def __init__(self,old,vocab):
        super().__init__(256,old.cfg)
        self.load_state_dict(old.state_dict())
        del self.W_k,self.W_q
        self.features=nn.ModuleList()
        self.geometry=nn.ModuleList()
        for head in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3701+head)
                embedding=nn.Embedding(vocab,64)
                key,query=nn.Linear(64,64),nn.Linear(64,64)
                query.load_state_dict(key.state_dict())
            embedding.requires_grad_(False)
            key.requires_grad_(False);query.requires_grad_(False)
            self.features.append(embedding)
            self.geometry.append(nn.ModuleDict(dict(key=key,query=query)))
        self.W_in=nn.Linear(256,64,bias=False)
        with torch.no_grad():
            self.W_in.weight.copy_(torch.eye(256)[:64])
        self.match_coef=nn.Parameter(torch.full((4,),-1.))
        self.gate_mean=None
        self.evidence=None

    def forward(self,x,token_ids,state=None):
        b,t=token_ids.shape
        shared=self.W_in(x)
        keys=[];queries=[]
        for h in range(4):
            z=self.features[h](token_ids)+shared
            keys.append(self.geometry[h]['key'](z).detach())
            queries.append(self.geometry[h]['query'](z))
        k,q=torch.stack(keys,2),torch.stack(queries,2)
        v=self.W_v(x).reshape(b,t,4,128)
        mask=(token_ids!=0).unsqueeze(-1).expand(b,t,4).float()
        result,self.evidence,sigmoid=scan(k,q,v,mask,self.W_pop(x).sigmoid(),
            self.W_append(x),state,self.cfg,self.append_err_coef,self.W_write(x),self.match_coef)
        self.gate_mean=sigmoid.mean()
        self.sigmoid_per_head=sigmoid.detach().mean((0,1)).tolist() if not self.training else None
        return result
