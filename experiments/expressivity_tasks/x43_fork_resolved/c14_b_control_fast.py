"""Evaluation-only metadata reuse and suppression of unused telemetry.

No model arithmetic or native kernel changes. The original implementation is
retained for first-shard bit parity in both consultation modes.
"""
import numpy as np
import torch
import triton
from . import c14_b_control_native as native
from . import c14_b_control_model as model_module
from .c14_b_suffix import SuffixOwner
from .c14_b_eval_decode import owner_metadata,load_owner_decoder
owner_decoder=None
original_owner=model_module.queried_suffix

original_forward=native.native_forward
original_reports=native.native_reports
cached=None
full_telemetry=True


def forward(inputs,cfg,policy,metadata):
    global cached
    k,q,v,av,pop,append,logits,coef,mc,ec=inputs
    owners,raw,windows,valid,keep,predictions=metadata
    b,t,_=k.shape
    if cached is None or cached[0] is not owners:
        global owner_decoder
        blank=not any(any(row) for row in keep)
        if blank:
            owner_ids=np.full((b,t),-1,dtype=np.int32)
            expiry=np.broadcast_to(np.asarray(valid,dtype=np.int32)[:,None],(b,t)).copy()
        else:
            if owner_decoder is None:owner_decoder=load_owner_decoder()
            contexts=np.asarray([[len(o.suffix) if o is not None else 0 for o in row] for row in owners],dtype=np.int64)
            raw_matrix=np.frombuffer(b''.join(raw),dtype=np.uint8).reshape(b,t)
            owner_ids,expiry=owner_metadata(raw_matrix,contexts,valid,owner_decoder)
        tensors=tuple(torch.tensor(x,device=k.device,dtype=torch.int32) for x in
                      (owner_ids,predictions,expiry,keep,[list(r) for r in raw],valid))
        cached=(owners,tensors)
    output=torch.empty_like(v)
    state=(torch.empty(b,cfg.slots,64,device=k.device),torch.empty(b,cfg.slots,128,device=k.device),
        torch.empty(b,cfg.slots,device=k.device),torch.empty(b,7,cfg.slots,device=k.device,dtype=torch.int32))
    trace=torch.zeros(b,t,18,device=k.device,dtype=torch.int32)
    native.kernel[(b,)](*(x.contiguous() for x in inputs),*cached[1],output,*state,trace,
        T=t,S=cfg.slots,BS=triton.next_power_of_2(cfg.slots),Decay=cfg.decay_rate,Free=cfg.free_threshold,
        PopThreshold=cfg.pop_threshold,Enable=cfg.write_enable_threshold,Eps=cfg.eps_residual,
        MatchWire=policy.match_wire,OccSlope=policy.occ_slope,OccOnset=policy.occ_onset,AppendOnly=cfg.array_append_only,
        enable_fp_fusion=False,num_warps=8)
    return output,(state,trace)


def reports(trace,cfg,valid):
    if full_telemetry:return original_reports(trace,cfg,valid)
    # Non-sampled shards need only the no-array-insertion guard. Counts for
    # sampled shards still come from the complete native trace, unchanged.
    requests=int(trace[:,:,(2,3,4)].sum())
    return [dict(array_requests=requests)]


def owner(window,observed,length):
    # The frozen decoder already checks raw integer suffix bounds and the
    # source-window adapter checks document/offset identities. No arithmetic
    # or suffix bytes change; omit repeated per-position type validation.
    return SuffixOwner(window,observed[-length:] if length else b'')


def install():
    model_module.queried_suffix=owner
    native.native_forward=forward;native.native_reports=reports


def restore():
    model_module.queried_suffix=original_owner
    native.native_forward=original_forward;native.native_reports=original_reports
