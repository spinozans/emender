"""Wrapper batching for unchanged registered conditional-control kernels."""
import torch
import numpy as np
import triton
from .c14_b_ablation_native import kernel,eager
from .c14_b_native import native_reports as base_reports


def native_forward(inputs,cfg,policy,metadata):
    k,q,v,av,pop,append,logits,coef,mc,ec=inputs
    owners,raw,windows,valid,keep,predictions=metadata
    b,t,_=k.shape
    cache=getattr(metadata,'encodings',{})
    cache_key=('native',k.device)
    encoded=cache.get(cache_key)
    if encoded is None:
        owner_ids=[];expiry=[]
        for j in range(b):
            canonical={}; ids=[]; ends=[]
            for i in range(t):
                o=owners[j][i]
                ids.append(canonical.setdefault(o,len(canonical)) if o is not None and o.suffix else -1)
                end=valid[j]
                if o is not None:
                    end=next((n for n in range(i+1,valid[j]) if not o.is_current(windows[j],raw[j][:n+1])),valid[j])
                ends.append(end)
            owner_ids.append(ids);expiry.append(ends)
        def ints(x,dtype=torch.int32):return torch.tensor(x,device=k.device,dtype=dtype)
        encoded=(ints(owner_ids),ints(predictions),ints(expiry),ints(keep),ints([list(r) for r in raw]),ints(valid))
        cache[cache_key]=encoded
    output=torch.empty_like(v)
    state=(torch.empty(b,cfg.slots,64,device=k.device),torch.empty(b,cfg.slots,128,device=k.device),
           torch.empty(b,cfg.slots,device=k.device),torch.empty(b,7,cfg.slots,device=k.device,dtype=torch.int32))
    trace=torch.zeros(b,t,18,device=k.device,dtype=torch.int32)
    kernel[(b,)](*(x.contiguous() for x in inputs),*encoded,output,*state,trace,T=t,S=cfg.slots,BS=triton.next_power_of_2(cfg.slots),
        Decay=cfg.decay_rate,Free=cfg.free_threshold,PopThreshold=cfg.pop_threshold,Enable=cfg.write_enable_threshold,
        Eps=cfg.eps_residual,MatchWire=policy.match_wire,OccSlope=policy.occ_slope,OccOnset=policy.occ_onset,AppendOnly=cfg.array_append_only,
        enable_fp_fusion=False,num_warps=8)
    return output,(state,trace)


class NativeReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cfg,policy,metadata,*inputs):
        out,(state,trace)=native_forward(inputs,cfg,policy,metadata)
        ctx.save_for_backward(*inputs,out)
        ctx.cfg,ctx.policy,ctx.store_metadata=cfg,policy,metadata
        ctx.mark_non_differentiable(trace)
        return out,trace

    @staticmethod
    def backward(ctx,grad,grad_trace):
        *inputs,out=ctx.saved_tensors
        # The captured schedule resets every slot on each invocation; no
        # state is written back to the forward store or another window.
        from .c14_b_control_graph import backward as replay_backward
        replay,grads=replay_backward(inputs,grad,ctx.cfg,ctx.policy,ctx.store_metadata)
        torch.testing.assert_close(replay,out,atol=2e-5,rtol=2e-5,msg='native/replay divergence; fail closed')
        return (None,None,None,*grads)


def native_reports(trace,cfg,valid):
    if cfg.slots==triton.next_power_of_2(cfg.slots):return base_reports(trace,cfg,valid)
    # The unchanged control kernel uses S as the empty-read sentinel. For a
    # non-power-of-two cap S is also a padded lane, which can increment the
    # POP trace on an empty store without touching any live slot. Reconstruct
    # exact pre-read cardinality from the existing trace and remove only that
    # phantom counter. Numerical forward/state/backward are unchanged.
    cpu=trace.detach().cpu().clone();rows=cpu.numpy()
    previous=np.zeros(rows.shape[:2],dtype=np.int64)
    previous[:,1:]=rows[:,:-1,14].astype(np.int64)+rows[:,:-1,15]
    pre=previous-rows[:,:,1]+rows[:,:,2]
    assert (pre>=0).all(), 'invalid pre-read occupancy reconstruction'
    rows[:,:,7]=np.where(pre>0,rows[:,:,7],0)
    return base_reports(cpu,cfg,valid)
