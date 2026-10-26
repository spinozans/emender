"""First-order reverse replay with fixed integral routing and typed undo.

A single program owns each store throughout the reverse chunk. This correctness
implementation reserves a conservative undo capacity and retains a private
O(S) replay copy. It is not a performance-certified training implementation.
"""
import struct
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from .all_field_journal import undo_to
from .lazy_decay import advance_exact


@triton.jit
def rebase(Addresses, Counts, Mapping, Fields:tl.constexpr,
           Capacity, Block:tl.constexpr):
    n=tl.program_id(0).to(tl.int64)
    x=tl.program_id(1)*Block+tl.arange(0,Block)
    active=x<tl.load(Counts+n)
    address=tl.load(Addresses+n*Capacity+x,active,0).to(tl.int64)
    lo=tl.full((Block,),0,tl.int32)
    hi=tl.where(active,Fields,0)
    while tl.sum((lo<hi).to(tl.int32),0)>0:
        searching=lo<hi
        mid=(lo+hi)//2
        base=tl.load(Mapping+3*mid,searching,0)
        right=address>=base
        lo=tl.where(searching & right,mid+1,lo)
        hi=tl.where(searching & ~right,mid,hi)
    slot=lo-1
    valid=active & (slot>=0)
    old=tl.load(Mapping+3*slot,valid,0)
    new=tl.load(Mapping+3*slot+1,valid,0)
    size=tl.load(Mapping+3*slot+2,valid,0)
    matched=valid & (address>=old) & (address<old+size)
    tl.device_assert((~active)|matched,'undo address has no owning field')
    tl.store(Addresses+n*Capacity+x,address-old+new,matched)


def rebase_owned(addresses,counts,old_owners,new_owners,capacity):
    if len(old_owners)!=len(new_owners) or any(
            a.shape!=b.shape or a.dtype!=b.dtype or a.device!=b.device
            or not a.is_contiguous() or not b.is_contiguous()
            for a,b in zip(old_owners,new_owners)):
        raise ValueError('rebase requires matching owned contiguous fields')
    rows=sorted((a.data_ptr(),b.data_ptr(),a.numel()*a.element_size())
                for a,b in zip(old_owners,new_owners))
    mapping=torch.tensor(rows,dtype=torch.int64,pin_memory=True).to(
        addresses.device,non_blocking=True)
    rebase[(len(counts),triton.cdiv(capacity,256))](
        addresses,counts,mapping,len(rows),capacity,256,debug=True)


@triton.jit
def reverse_kernel(K,Q,V,Beta,Pop,Keys,Values,Strength,Live,Touch,
                   Cases,Matched,Alloc,Pops,Freed,Gates,Errors,
                   Addresses,Before,Widths,Counts,Boundaries,
                   GradOut,GKeys,GValues,GStrength,DKout,DQout,DVout,
                   DBeta,DPop,DLogit,DCoef,
                   T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                   BS:tl.constexpr,BV:tl.constexpr,UB:tl.constexpr,Capacity,
                   Fresh:tl.constexpr,Start,Decay:tl.constexpr):
    n=tl.program_id(0)
    s=tl.arange(0,BS)
    d=tl.arange(0,BV)
    valid=s<S
    gs=tl.load(GStrength+n*S+s,valid,0.)
    stop=tl.load(Counts+n)
    coef_grad=tl.full((),0.,tl.float32)
    for reverse in range(T):
        t=T-1-reverse
        inp=n*T+t
        start=tl.load(Boundaries+inp)
        undo_to((Addresses,Before,Widths,Counts,n,Capacity),start,stop,UB)
        stop=start
        tl.debug_barrier()
        live=tl.load(Live+n*S+s,valid,False).to(tl.int1)
        reference=tl.load(Strength+n*S+s,valid,0.)
        if Fresh and t==0:
            strength=reference
        else:
            touch=tl.load(Touch+n*S+s,valid,0)
            strength=advance_exact(reference,tl.where(live,Start+t-touch,0),Decay)
        freed=tl.load(Freed+inp*S+s,valid,0)
        allocation=tl.load(Alloc+inp)
        target=tl.load(Matched+inp)
        case=tl.load(Cases+inp)
        pop_id=tl.load(Pops+inp)
        postpop_live=live & (freed!=1)
        postwrite_live=postpop_live | (s==allocation)
        # Death stops future decay but not the derivative of the multiplication
        # that produced its terminal strength. Allocation replaces its adjoint.
        gs=tl.where(postwrite_live,gs*Decay,gs)
        content=tl.load(V+inp*DV+d,d<DV,0.)
        beta=tl.load(Beta+inp)
        dv=tl.full((BV,),0.,tl.float32)
        dbeta=tl.full((),0.,tl.float32)
        dlogit=tl.full((),0.,tl.float32)
        if (case==4)|(case==5):
            before=tl.load(Values+(n*S+target)*DV+d,d<DV,0.)
            gradient=tl.load(GValues+(n*S+target)*DV+d,d<DV,0.)
            dv=beta*gradient
            dbeta=tl.sum(gradient*(content-before),0)
            tl.store(GValues+(n*S+target)*DV+d,(1.-beta)*gradient,d<DV)
        if allocation>=0:
            gradient=tl.load(GValues+(n*S+allocation)*DV+d,d<DV,0.)
            gate=tl.load(Gates+inp)
            dv=gate*gradient
            dlogit=tl.sum(gradient*content,0)*gate*(1.-gate)
            coef_grad+=dlogit*tl.load(Errors+inp)
            tl.store(GValues+(n*S+allocation)*DV+d,0.,d<DV)
            for j in range(DK):
                gk=tl.load(GKeys+(n*S+allocation)*DK+j)
                tl.store(DKout+inp*DK+j,gk)
                tl.store(GKeys+(n*S+allocation)*DK+j,0.)
            gs=tl.where(s==allocation,0.,gs)
        tl.store(DVout+inp*DV+d,dv,d<DV)
        tl.store(DBeta+inp,dbeta)
        tl.store(DLogit+inp,dlogit)
        pop=tl.load(Pop+inp)
        dpop=-tl.sum(tl.where(s==pop_id,gs*strength,0.),0)
        gs=tl.where(s==pop_id,gs*(1.-pop),gs)
        tl.store(DPop+inp,dpop)
        tl.debug_barrier()
        # Dense read adjoints, evaluated only on the restored pre-read store.
        score=tl.full((BS,),0.,tl.float32)
        scale=2./tl.sqrt(float(DK))
        for j in range(DK):
            key=tl.load(Keys+(n*S+s)*DK+j,valid & live,0.)
            query=tl.load(Q+inp*DK+j)
            score+=key*query
        masked=tl.where(live & valid,score*scale,-1.e30)
        numer=tl.where(valid,tl.exp(masked-tl.max(masked,0)),0.)
        prob=numer/tl.sum(numer,0)
        weight=prob*strength*live
        dw=tl.full((BS,),0.,tl.float32)
        for j in range(DV):
            value=tl.load(Values+(n*S+s)*DV+j,valid & live,0.)
            gout=tl.load(GradOut+inp*DV+j)
            dw+=value*gout
            gv=tl.load(GValues+(n*S+s)*DV+j,valid,0.)
            tl.store(GValues+(n*S+s)*DV+j,gv+weight*gout,valid)
        gs+=prob*live*dw
        dp=dw*strength*live
        ds=prob*(dp-tl.sum(prob*dp,0))*live*scale
        for j in range(DK):
            key=tl.load(Keys+(n*S+s)*DK+j,valid & live,0.)
            query=tl.load(Q+inp*DK+j)
            gk=tl.load(GKeys+(n*S+s)*DK+j,valid,0.)
            tl.store(GKeys+(n*S+s)*DK+j,gk+ds*query,valid)
            tl.store(DQout+inp*DK+j,tl.sum(ds*key,0))
        tl.debug_barrier()
    tl.store(GStrength+n*S+s,gs,valid)
    tl.store(DCoef+n,coef_grad)


@triton.jit
def copy_allocated_payload(Keys, Values, ReplayKeys, ReplayValues, Alloc,
                           T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,
                           DV:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
    n=tl.program_id(0)
    # One program owns a store, so reused allocation IDs cannot race.
    k=tl.arange(0,BK)
    v=tl.arange(0,BV)
    for token in range(T):
        slot=tl.load(Alloc+n*T+token)
        if slot>=0:
            key=tl.load(Keys+(n*S+slot)*DK+k,k<DK,0.)
            value=tl.load(Values+(n*S+slot)*DV+v,v<DV,0.)
            tl.store(ReplayKeys+(n*S+slot)*DK+k,key,k<DK)
            tl.store(ReplayValues+(n*S+slot)*DV+v,value,v<DV)


class Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cfg,start_tokens,history,saved,k,q,v,beta,pop,logit,coef,
                keys,values,strength,live,age,usage):
        from .cell_fused import FusedState,tape_scan
        from ..x30_tape_cell.naive_reference import TapeState
        initial=FusedState(TapeState(keys,values,strength,live,age,usage),
                           history,start_tokens,live_bound=saved["live_bound"],owns_import=saved["owns_native"])
        out,final,trace,pressure=tape_scan(k,q,v,beta,pop,logit,initial,cfg,coef,
                                          record_all=True)
        journal=final.all_field_journal
        # Private copies prevent backward from mutating returned native state.
        # The log's owned addresses are rebased onto these buffers once below.
        if saved["live_bound"] == 0:
            # Empty imports have no readable payload outside allocation events.
            # Keep address space for undo, but copy only rows ever allocated.
            replay=tuple(torch.empty_like(x) if i<2 else x.clone()
                         for i,x in enumerate(journal.owners))
            copy_allocated_payload[(k.shape[0]*k.shape[2],)](
                journal.owners[0],journal.owners[1],replay[0],replay[1],
                trace.alloc_target.transpose(1,2).contiguous(),k.shape[1],
                cfg.slots,cfg.key_dim,cfg.value_dim,
                triton.next_power_of_2(cfg.key_dim),triton.next_power_of_2(cfg.value_dim))
        else:
            replay=tuple(x.clone() for x in journal.owners)
        addresses=journal.addresses
        rebase_owned(addresses,journal.counts,journal.owners,replay,journal.capacity)
        ctx.cfg=cfg
        ctx.start=start_tokens
        ctx.capacity=journal.capacity
        ctx.replay=replay
        ctx.trace=tuple(x.transpose(1,2).clone(memory_format=torch.contiguous_format) for x in
                        (trace.case,trace.matched,trace.alloc_target,trace.pop_target,trace.freed))
        ctx.log=(addresses,journal.before,journal.widths,journal.counts,
                 journal.boundaries,journal.gates,journal.errors)
        ctx.save_for_backward(k,q,v,beta,pop,logit,coef)
        # Native imports are copied once per training chunk. This intentionally
        # favors alias safety over the inference consuming-buffer optimization.
        final.all_field_journal=None
        saved.update(state=final,trace=trace,pressure=pressure)
        return out,final.native.keys,final.native.values,final.native.strength

    @staticmethod
    @once_differentiable
    def backward(ctx,gout,gkeys,gvalues,gstrength):
        k,q,v,beta,pop,logit,coef=ctx.saved_tensors
        cfg=ctx.cfg
        b,t,h,dk=k.shape
        def flat(x): return x.transpose(1,2).contiguous()
        args=list(map(flat,(k,q,v,beta,pop)))
        def gradient(g,like):
            return torch.zeros_like(like) if g is None else g.contiguous().clone()
        replay=tuple(x.clone() for x in ctx.replay)
        keys,values,strength,live,age,history,*meta=replay
        gk=gradient(gkeys,keys);gv=gradient(gvalues,values);gs=gradient(gstrength,strength)
        go=torch.zeros(b,h,t,cfg.value_dim,device=k.device) if gout is None else flat(gout)
        dkout,dqout,dvout,dbeta,dpop,dlogit=[torch.zeros_like(flat(x)) for x in
                                             (k,q,v,beta,pop,logit)]
        dc=torch.empty(b*h,device=k.device)
        original_addresses,before,widths,counts,boundaries,gates,errors=ctx.log
        addresses=original_addresses.clone()
        rebase_owned(addresses,counts,ctx.replay,replay,ctx.capacity)
        reverse_kernel[(b*h,)](*args,keys,values,strength,live,meta[5],
            *ctx.trace,gates,errors,addresses,before,widths,counts,boundaries,
            go,gk,gv,gs,dkout,dqout,dvout,dbeta,dpop,dlogit,dc,
            t,cfg.slots,dk,cfg.value_dim,triton.next_power_of_2(cfg.slots),
            triton.next_power_of_2(cfg.value_dim),min(128,triton.next_power_of_2(max(cfg.slots,cfg.value_dim,dk))),ctx.capacity,True,ctx.start,
            struct.unpack('<f',struct.pack('<f',cfg.decay_rate))[0],
            enable_fp_fusion=False,debug=True,num_warps=16 if cfg.slots>=4096 else 4)
        def unflat(x):return x.transpose(1,2).contiguous()
        return (None,None,None,None,*map(unflat,(dkout,dqout,dvout,dbeta,dpop,dlogit)),
                dc.reshape(b,h).sum(0),gk,gv,gs,None,None,None)


def autograd_scan(inputs,state,cfg,coef):
    from .cell_fused import FusedState
    from ..x30_tape_cell.naive_reference import TapeState,init_state
    k=inputs[0];b,t,h=k.shape[:3]
    if state is None:
        native=init_state(b,cfg,k.device)
        history=torch.zeros(b*h,128,device=k.device,dtype=torch.int32)
        tokens=0
    elif isinstance(state,TapeState):
        native=state
        history=torch.zeros(b*h,128,device=k.device,dtype=torch.int32)
        tokens=0
    else:
        if state.consumed:raise ValueError('FusedState handle was already consumed')
        native=state.native;history=state.history;tokens=state.tokens
    if coef is None:coef=k.new_zeros(h)
    saved={"live_bound": 0 if state is None else (state.live_bound if isinstance(state,FusedState) else None),
           "owns_native": state is None}
    out,keys,values,strength=Replay.apply(cfg,tokens,history,saved,*inputs,coef,
        native.keys,native.values,native.strength,native.live,native.age,native.usage)
    final=saved['state']
    final.native=TapeState(keys,values,strength,final.native.live,final.native.age,native.usage)
    if isinstance(state,FusedState):state.consumed=True
    return out,final,saved['trace'],saved['pressure']
