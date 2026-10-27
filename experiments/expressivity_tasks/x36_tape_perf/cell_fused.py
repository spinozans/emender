"""X31 forward, with a dead-row gradient traffic masked in X36 only."""
import torch
from ..x31_tape_cell_fused.cell_fused import FusedState, TapeHeadConfig
from ..x31_tape_cell_fused.cell_fused import TapeCell as OriginalCell
from ..x31_tape_cell_fused.cell_fused import tape_scan as original_scan


def tape_scan(k,q,v,beta,pop,logit,state,cfg,append_err_coef=None,*,record_payload=False,record_all=False):
    coef=append_err_coef
    native = state.native if isinstance(state,FusedState) else state
    needs_grad = any(x.requires_grad for x in (k,q,v,beta,pop,logit))
    needs_grad |= coef is not None and coef.requires_grad
    if native is not None:
        needs_grad |= any(getattr(native,n).requires_grad for n in ('keys','values','strength'))
    if torch.is_grad_enabled() and needs_grad:
        if record_payload or record_all:
            raise ValueError('training manages its private journal automatically')
        from .backward import autograd_scan
        return autograd_scan((k,q,v,beta,pop,logit),state,cfg,coef)
    return original_scan(k,q,v,beta,pop,logit,state,cfg,coef,
                         record_payload=record_payload,record_all=record_all)


class TapeCell(OriginalCell):
    def forward(self,x,token_ids,state=None):
        b,t=x.shape[:2]
        h,cfg=self.cfg.n_heads,self.cfg
        k=self.W_k(x).reshape(b,t,h,cfg.key_dim).detach()
        q=self.W_q(x).reshape_as(k)
        v=self.W_v(x).reshape(b,t,h,cfg.value_dim)
        beta=((self.W_write(x).sigmoid().detach()>cfg.write_enable_threshold)
              & (token_ids!=0).unsqueeze(-1)).float()
        return tape_scan(k,q,v,beta,self.W_pop(x).sigmoid(),self.W_append(x),
                         state,cfg,self.append_err_coef)
