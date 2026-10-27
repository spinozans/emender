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
from ..x31_tape_cell_fused.all_field_journal import undo_to
from ..x31_tape_cell_fused.lazy_decay import advance_exact


@triton.jit
def rebase(Addresses, Counts, Mapping, Fields:tl.constexpr,
           Capacity, Block:tl.constexpr, PayloadS:tl.constexpr=0, ReversePayload:tl.constexpr=False):
    n=tl.program_id(0).to(tl.int64)
    x=tl.program_id(1)*Block+tl.arange(0,Block)
    active=x<tl.load(Counts+n)
    address=tl.load(Addresses+n*Capacity+x,active,0).to(tl.int64)
    lo=tl.full((Block,),0,tl.int32)
    hi=tl.where(active,Fields,0)
    while tl.sum((lo<hi).to(tl.int32),0)>0:
        searching=lo<hi
        mid=(lo+hi)//2
        base=tl.load(Mapping+4*mid,searching,0)
        right=address>=base
        lo=tl.where(searching & right,mid+1,lo)
        hi=tl.where(searching & ~right,mid,hi)
    slot=lo-1
    valid=active & (slot>=0)
    old=tl.load(Mapping+4*slot,valid,0)
    new=tl.load(Mapping+4*slot+1,valid,0)
    size=tl.load(Mapping+4*slot+2,valid,0)
    matched=valid & (address>=old) & (address<old+size)
    tl.device_assert((~active)|matched,'undo address has no owning field')
    offset=address-old
    dim=tl.load(Mapping+4*slot+3,valid,0)
    if PayloadS>0:
        # Float32 payload offsets are permuted from [store,S,D] to [store,D,S].
        element=offset//4
        safe_dim=tl.maximum(dim,1)
        store=element//(PayloadS*safe_dim)
        if ReversePayload:
            row=element%PayloadS
            col=(element//PayloadS)%safe_dim
            permuted=((store*PayloadS+row)*safe_dim+col)*4
        else:
            row=(element//safe_dim)%PayloadS
            col=element%safe_dim
            permuted=((store*safe_dim+col)*PayloadS+row)*4
        offset=tl.where(dim>0,permuted,offset)
    tl.store(Addresses+n*Capacity+x,offset+new,matched)


def rebase_owned(addresses,counts,old_owners,new_owners,capacity,*,payload_slots=0,reverse_payload=False):
    if len(old_owners)!=len(new_owners) or any(
            (a.shape!=b.shape and not (payload_slots and i<2 and a.transpose(-1,-2).shape==b.shape))
            or a.dtype!=b.dtype or a.device!=b.device
            or not a.is_contiguous() or not b.is_contiguous()
            for i,(a,b) in enumerate(zip(old_owners,new_owners))):
        raise ValueError('rebase requires matching owned contiguous fields')
    rows=sorted((a.data_ptr(),b.data_ptr(),a.numel()*a.element_size(),
                 (a.shape[-2] if reverse_payload else a.shape[-1]) if payload_slots and i<2 else 0)
                for i,(a,b) in enumerate(zip(old_owners,new_owners)))
    mapping=torch.tensor(rows,dtype=torch.int64,pin_memory=True).to(
        addresses.device,non_blocking=True)
    rebase[(len(counts),triton.cdiv(capacity,256))](
        addresses,counts,mapping,len(rows),capacity,256,payload_slots,reverse_payload,debug=True)


@triton.jit
def reverse_kernel(K,Q,V,Beta,Pop,Keys,Values,Strength,Live,Touch,
                   Cases,Matched,Alloc,Pops,Freed,Gates,Errors,
                   Addresses,Before,Widths,Counts,Boundaries,
                   GradOut,GKeys,GValues,GStrength,DKout,DQout,DVout,
                   DBeta,DPop,DLogit,DCoef,
                   T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                   BS:tl.constexpr,BV:tl.constexpr,UB:tl.constexpr,Capacity,SoA:tl.constexpr,
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
        freed=tl.load(Freed+inp.to(tl.int64)*S+s,valid,0)
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
            before=tl.load(Values+((n*DV+d)*S+target if SoA else (n*S+target)*DV+d),d<DV,0.)
            gradient=tl.load(GValues+((n*DV+d)*S+target if SoA else (n*S+target)*DV+d),d<DV,0.)
            dv=beta*gradient
            dbeta=tl.sum(gradient*(content-before),0)
            tl.store(GValues+((n*DV+d)*S+target if SoA else (n*S+target)*DV+d),(1.-beta)*gradient,d<DV)
        if allocation>=0:
            gradient=tl.load(GValues+((n*DV+d)*S+allocation if SoA else (n*S+allocation)*DV+d),d<DV,0.)
            gate=tl.load(Gates+inp)
            dv=gate*gradient
            dlogit=tl.sum(gradient*content,0)*gate*(1.-gate)
            coef_grad+=dlogit*tl.load(Errors+inp)
            tl.store(GValues+((n*DV+d)*S+allocation if SoA else (n*S+allocation)*DV+d),0.,d<DV)
            for j in range(DK):
                gk=tl.load(GKeys+((n*DK+j)*S+allocation if SoA else (n*S+allocation)*DK+j))
                tl.store(DKout+inp*DK+j,gk)
                tl.store(GKeys+((n*DK+j)*S+allocation if SoA else (n*S+allocation)*DK+j),0.)
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
            key=tl.load(Keys+((n*DK+j)*S+s if SoA else (n*S+s)*DK+j),valid & live,0.)
            query=tl.load(Q+inp*DK+j)
            score+=key*query
        masked=tl.where(live & valid,score*scale,-1.e30)
        numer=tl.where(valid,tl.exp(masked-tl.max(masked,0)),0.)
        prob=numer/tl.sum(numer,0)
        weight=prob*strength*live
        dw=tl.full((BS,),0.,tl.float32)
        for j in range(DV):
            value=tl.load(Values+((n*DV+j)*S+s if SoA else (n*S+s)*DV+j),valid & live,0.)
            gout=tl.load(GradOut+inp*DV+j)
            dw+=value*gout
            gv=tl.load(GValues+((n*DV+j)*S+s if SoA else (n*S+s)*DV+j),valid & live,0.)
            tl.store(GValues+((n*DV+j)*S+s if SoA else (n*S+s)*DV+j),gv+weight*gout,valid & live)
        gs+=prob*live*dw
        dp=dw*strength*live
        ds=prob*(dp-tl.sum(prob*dp,0))*live*scale
        for j in range(DK):
            key=tl.load(Keys+((n*DK+j)*S+s if SoA else (n*S+s)*DK+j),valid & live,0.)
            query=tl.load(Q+inp*DK+j)
            gk=tl.load(GKeys+((n*DK+j)*S+s if SoA else (n*S+s)*DK+j),valid & live,0.)
            tl.store(GKeys+((n*DK+j)*S+s if SoA else (n*S+s)*DK+j),gk+ds*query,valid & live)
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


class ReplayStorage:
    """Freeze only when a consuming continuation would overwrite saved state."""
    def __init__(self, owners):
        self.owners=owners
        self.frozen=False
        self.versions=tuple(x._version for x in owners)

    def validate(self):
        if any(x._version != version for x,version in zip(self.owners,self.versions)):
            raise RuntimeError('retained replay state was modified in place')

    def freeze(self):
        self.validate()
        if not self.frozen:
            self.owners=tuple(x.clone() for x in self.owners)
            self.frozen=True
            self.versions=tuple(x._version for x in self.owners)


class Replay(torch.autograd.Function):
    @staticmethod
    def forward(ctx,cfg,start_tokens,history,saved,k,q,v,beta,pop,logit,coef,
                keys,values,strength,live,age,usage):
        ctx.set_materialize_grads(False)
        from .cell_fused import FusedState,tape_scan
        from ..x30_tape_cell.naive_reference import TapeState
        initial=FusedState(TapeState(keys,values,strength,live,age,usage),
                           history,start_tokens,live_bound=saved["live_bound"],owns_import=saved["owns_native"])
        out,final,trace,pressure=tape_scan(k,q,v,beta,pop,logit,initial,cfg,coef,
                                          record_all=True)
        journal=final.all_field_journal
        # Keep the owned final buffers until replay is consumed. A subsequent
        # consuming continuation freezes them before its first mutation.
        ctx.storage=ReplayStorage(journal.owners)
        final.replay_storage=ctx.storage
        ctx.address_owners=journal.owners
        ctx.replay=journal.owners  # address-space accounting; zero forward copies
        ctx.cfg=cfg
        ctx.initial_bound=saved["live_bound"]
        ctx.start=start_tokens
        ctx.capacity=journal.capacity
        ctx.trace=final._replay_trace
        final._replay_trace=None
        ctx.log=(journal.addresses,journal.before,journal.widths,journal.counts,
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
        ctx.storage.validate()
        # Keep the oracle AoS layout at large capacities: the transposed
        # specialization hits a measured replay cliff at S11776.
        payload_soa=cfg.slots<4096
        replay=tuple(x.transpose(-1,-2).clone(memory_format=torch.contiguous_format)
                     if payload_soa and i<2 else x.clone() for i,x in enumerate(ctx.storage.owners))
        keys,values,strength,live,age,history,*meta=replay
        # Replay payloads and adjoints use [store,dim,slot] backing; full-S
        # reduction/accumulation expressions retain the oracle lane shapes.
        gk=gradient(gkeys,ctx.storage.owners[0])
        gv=gradient(gvalues,ctx.storage.owners[1])
        if payload_soa:
            gk=gk.transpose(-1,-2).contiguous()
            gv=gv.transpose(-1,-2).contiguous()
        gs=gradient(gstrength,strength)
        go=torch.zeros(b,h,t,cfg.value_dim,device=k.device) if gout is None else flat(gout)
        dkout,dqout,dvout,dbeta,dpop,dlogit=[torch.zeros_like(flat(x)) for x in
                                             (k,q,v,beta,pop,logit)]
        dc=torch.empty(b*h,device=k.device)
        original_addresses,before,widths,counts,boundaries,gates,errors=ctx.log
        # The log is private to this autograd node. Rebase it in place and
        # restore it on this stream before returning, retaining repeat-backward
        # semantics without a capacity-sized address clone (4.5 GiB at T4096).
        addresses=original_addresses
        rebase_owned(addresses,counts,ctx.address_owners,replay,ctx.capacity,payload_slots=cfg.slots if payload_soa else 0)
        # Eight warps retain exact reduction trees only in the independently
        # tested, empty-origin prefix specialization. Unknown imports and
        # longer continuations keep the oracle's warp count.
        prefix8=(cfg.slots==2944 and cfg.key_dim==64 and cfg.value_dim==128
                 and ctx.initial_bound is not None and ctx.initial_bound+t<=512)
        try:
            reverse_kernel[(b*h,)](*args,keys,values,strength,live,meta[5],
                *ctx.trace,gates,errors,addresses,before,widths,counts,boundaries,
                go,gk,gv,gs,dkout,dqout,dvout,dbeta,dpop,dlogit,dc,
                t,cfg.slots,dk,cfg.value_dim,triton.next_power_of_2(cfg.slots),
                triton.next_power_of_2(cfg.value_dim),min(128,triton.next_power_of_2(max(cfg.slots,cfg.value_dim,dk))),ctx.capacity,payload_soa,True,ctx.start,
                struct.unpack('<f',struct.pack('<f',cfg.decay_rate))[0],
                enable_fp_fusion=False,debug=True,num_warps=16 if cfg.slots>=4096 else (8 if prefix8 else 4))
        finally:
            rebase_owned(addresses,counts,replay,ctx.address_owners,ctx.capacity,
                         payload_slots=cfg.slots if payload_soa else 0,reverse_payload=payload_soa)
        def unflat(x):return x.transpose(1,2).contiguous()
        return (None,None,None,None,*map(unflat,(dkout,dqout,dvout,dbeta,dpop,dlogit)),
                dc.reshape(b,h).sum(0),gk.transpose(-1,-2).contiguous() if payload_soa else gk,
                gv.transpose(-1,-2).contiguous() if payload_soa else gv,gs,None,None,None)


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
        if state.replay_storage is not None:state.replay_storage.freeze()
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
