"""X40-local X36 replay port with exact unit-reference decay lookup.

The X36 implementation is copied to keep upstream sources read-only. Local
changes add unit decay tables and device-authored graph-safe address maps.
First-order exact replay with sparse imported/continued row adjoints.

Short nonempty-entry chunks index the entry-live/allocation union in logical
order. Native owners and cold native-gradient backing retain their addresses;
reductions reconstruct the original logical-slot operand and arithmetic tree.
Fresh prefixes and long chunks keep the independently validated paths.
Recorded write decisions suppress packed value-adjoint traffic when neither
a writing token nor a native-value input gradient can consume it.
"""
import struct
import os
import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from ..x31_tape_cell_fused.all_field_journal import undo_to
from .exact_decay import advance_unit, table


@triton.jit
def rebase(Addresses, Counts, Mapping, Fields:tl.constexpr,
           Capacity, Block:tl.constexpr, PayloadS:tl.constexpr=0, ReversePayload:tl.constexpr=False,
           PoolRows=None,PoolInverse=None,PS:tl.constexpr=0):
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
        if PS>0 and ReversePayload:
            store=element//(PS*safe_dim)
            packed=element%PS
            col=(element//PS)%safe_dim
            row=tl.load(PoolRows+store*PS+packed,matched & (dim>0),-1)
            tl.device_assert((~(matched & (dim>0)))|((row>=0)&(row<PayloadS)),
                             'hot undo address has no logical row')
            permuted=((store*PayloadS+row)*safe_dim+col)*4
        elif PS>0:
            store=element//(PayloadS*safe_dim)
            row=(element//safe_dim)%PayloadS
            col=element%safe_dim
            packed=tl.load(PoolInverse+store*PayloadS+row,matched & (dim>0),-1)
            tl.device_assert((~(matched & (dim>0)))|((packed>=0)&(packed<PS)),
                             'payload undo address outside entry/touched union')
            permuted=((store*safe_dim+col)*PS+packed)*4
        elif ReversePayload:
            store=element//(PayloadS*safe_dim)
            row=element%PayloadS
            col=(element//PayloadS)%safe_dim
            permuted=((store*PayloadS+row)*safe_dim+col)*4
        else:
            store=element//(PayloadS*safe_dim)
            row=(element//safe_dim)%PayloadS
            col=element%safe_dim
            permuted=((store*safe_dim+col)*PayloadS+row)*4
        offset=tl.where(dim>0,permuted,offset)
    tl.store(Addresses+n*Capacity+x,offset+new,matched)


@triton.jit
def mapping_kernel(Old,New,Mapping,Sizes:tl.constexpr,Dims:tl.constexpr):
    # Address tables are device-authored so capture contains no host transfer
    # with a transient pinned-memory lifetime.
    for i in tl.static_range(len(Old)):
        tl.store(Mapping+4*i,Old[i].to(tl.uint64))
        tl.store(Mapping+4*i+1,New[i].to(tl.uint64))
        tl.store(Mapping+4*i+2,Sizes[i])
        tl.store(Mapping+4*i+3,Dims[i])


@triton.jit
def mapping_runtime(Old,New,Mapping,Sizes,Dims):
    # Byte pointers and runtime sizes avoid specializing on allocator address
    # order, while retaining exactly the same sorted integral address table.
    for i in tl.static_range(len(Old)):
        tl.store(Mapping+4*i,Old[i].to(tl.uint64))
        tl.store(Mapping+4*i+1,New[i].to(tl.uint64))
        tl.store(Mapping+4*i+2,Sizes[i])
        tl.store(Mapping+4*i+3,Dims[i])


def rebase_owned(addresses,counts,old_owners,new_owners,capacity,*,payload_slots=0,reverse_payload=False,
                 pool_rows=None,pool_inverse=None):
    ps=0 if pool_rows is None else pool_rows.shape[-1]
    def compatible(i,a,b):
        if ps and i<2:
            native,hot=(b,a) if reverse_payload else (a,b)
            return (native.shape[:-2]==hot.shape[:-2] and native.shape[-2]==payload_slots
                    and native.shape[-1]==hot.shape[-2] and hot.shape[-1]==ps)
        return a.shape==b.shape or (payload_slots and i<2 and a.transpose(-1,-2).shape==b.shape)
    if len(old_owners)!=len(new_owners) or any(
            not compatible(i,a,b)
            or a.dtype!=b.dtype or a.device!=b.device
            or not a.is_contiguous() or not b.is_contiguous()
            for i,(a,b) in enumerate(zip(old_owners,new_owners))):
        raise ValueError('rebase requires matching owned contiguous fields')
    order=sorted(range(len(old_owners)),key=lambda i:old_owners[i].data_ptr())
    old=tuple(old_owners[i] for i in order)
    new=tuple(new_owners[i] for i in order)
    sizes=tuple(x.numel()*x.element_size() for x in old)
    dims=tuple((old_owners[i].shape[-2] if reverse_payload else old_owners[i].shape[-1])
               if payload_slots and i<2 else 0 for i in order)
    mapping=torch.empty((len(order),4),dtype=torch.int64,device=addresses.device)
    if os.environ.get('X40_STABLE_MAPPING','0')=='1':
        mapping_runtime[(1,)](tuple(x.view(torch.uint8) for x in old),
                              tuple(x.view(torch.uint8) for x in new),mapping,sizes,dims)
    else:
        mapping_kernel[(1,)](old,new,mapping,sizes,dims)
    rebase[(len(counts),triton.cdiv(capacity,256))](
        addresses,counts,mapping,len(order),capacity,256,payload_slots,reverse_payload,
        mapping if pool_rows is None else pool_rows,mapping if pool_inverse is None else pool_inverse,ps,debug=True)


@triton.jit
def slot_sum(x, FullLive, BS:tl.constexpr, FullBS:tl.constexpr, Tail=0., Inverse=None, Sparse:tl.constexpr=False):
    if Sparse:
        rank=Inverse
        expanded=tl.gather(x,tl.maximum(rank,0),0)
        return tl.sum(tl.where(rank>=0,expanded,Tail),0)
    # Reconstruct only the reduction operand, not the resident adjoint state.
    # Retain the byte occupancy operand in the reconstructed reduction;
    # unconstrained gather-only padding changes the blocked floating order.
    if BS<FullBS:
        lane=tl.arange(0,FullBS)
        expanded=tl.gather(x,tl.minimum(lane,BS-1),0)
        return tl.sum(tl.where((lane<BS)|(~FullLive),tl.where(lane<BS,expanded,Tail),0.),0)
    return tl.sum(x,0)


@triton.jit
def slot_max(x, FullLive, BS:tl.constexpr, FullBS:tl.constexpr, Inverse=None, Sparse:tl.constexpr=False):
    if Sparse:
        expanded=tl.gather(x,tl.maximum(Inverse,0),0)
        return tl.max(tl.where(Inverse>=0,expanded,-1.e30),0)
    if BS<FullBS:
        lane=tl.arange(0,FullBS)
        expanded=tl.gather(x,tl.minimum(lane,BS-1),0)
        return tl.max(tl.where((lane<BS)|(~FullLive),tl.where(lane<BS,expanded,-1.e30),-1.e30),0)
    return tl.max(x,0)


@triton.jit
def reverse_program(K,Q,V,Beta,Pop,Keys,Values,Strength,Live,Touch,
                   Cases,Matched,Alloc,Pops,Freed,Gates,Errors,
                   Addresses,Before,Widths,Counts,Boundaries,
                   GradOut,GKeys,GValues,GStrength,StrengthScratch,ProbabilityScratch,DKout,DQout,DVout,
                   DBeta,DPop,DLogit,DCoef,
                   T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                   BS:tl.constexpr,BV:tl.constexpr,UB:tl.constexpr,Capacity,SoA:tl.constexpr,
                   Fresh:tl.constexpr,Start,Decay:tl.constexpr,KeyGrad:tl.constexpr,GS:tl.constexpr,RS:tl.constexpr, PoolRows, PoolInverse, PS:tl.constexpr, DecayTable, ColdTail:tl.constexpr=False, NativeValuesGrad:tl.constexpr=True):
    # Wide offsets help the bounded fast path; long/large vectors retain
    # checked int32 offsets to avoid doubling address-register footprints.
    n=tl.program_id(0).to(tl.int64 if S<4096 and T<=512 else tl.int32)
    p=tl.arange(0,BS)
    s=tl.load(PoolRows+n*PS+p,p<PS,-1) if PS else p
    a=p if PS else s
    AS:tl.constexpr=PS if PS else S
    NS:tl.constexpr=PS if PS else S
    d=tl.arange(0,BV)
    valid=(s>=0) & (s<S)
    if T<=512:
        gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.)
    # The recorded decisions are the value-adjoint consumers. With no
    # correction/allocation and no native-value gradient consumer, the read
    # recurrence's value adjoints are dead: no later token can scatter them
    # into a writing token. Keep DVout's exact zero stores, but skip the
    # packed adjoint traffic. This decision stays on device.
    value_adjoint_needed=tl.full((),True,tl.int1)
    if PS>0 and not NativeValuesGrad:
        token=tl.arange(0,triton.next_power_of_2(T))
        cases=tl.load(Cases+n*T+token,token<T,0)
        allocations=tl.load(Alloc+n*T+token,token<T,-1)
        value_adjoint_needed=tl.sum(((cases==4)|(cases==5)|(allocations>=0)).to(tl.int32),0)>0
    stop=tl.load(Counts+n)
    coef_grad=tl.full((),0.,tl.float32)
    for reverse in range(T):
        t=T-1-reverse
        inp=n*T+t
        # Every other input adjoint is unconditionally written below. Keys
        # receive a gradient only on allocation; initialize those lanes here.
        if S<4096 and KeyGrad:
            kd=tl.arange(0,triton.next_power_of_2(DK))
            tl.store(DKout+inp*DK+kd,0.,kd<DK)
        start=tl.load(Boundaries+inp)
        undo_to((Addresses,Before,Widths,Counts,n,Capacity),start,stop,UB)
        stop=start
        tl.debug_barrier()
        full_lane=tl.arange(0,triton.next_power_of_2(S))
        full_live=tl.load(Live+n*S+full_lane,full_lane<S,False).to(tl.int1)
        inverse=tl.load(PoolInverse+n*S+full_lane,full_lane<S,-1) if PS else full_lane
        tl.device_assert((inverse>=0 if PS else full_lane<BS)|(~full_live),'live row outside compact replay')
        if T<=512:
            live=tl.load(Live+n*S+s,valid,False).to(tl.int1)
            reference=tl.load(Strength+n*S+s,valid,0.)
            if Fresh and t==0:
                strength=reference
            else:
                touch=tl.load(Touch+n*S+s,valid,0)
                strength=advance_unit(reference,tl.where(live,Start+t-touch,0),Decay,DecayTable,T)
        else:
            # Exact decay has independent lane arithmetic but many int64
            # temporaries. Evaluate bounded tiles before the full-S float
            # adjoint/reduction vectors, rather than spilling a BS-wide jump.
            for tile in range(tl.cdiv(min(S,BS,RS),512)):
                row=tile*512+tl.arange(0,512)
                active=row<min(S,BS,RS)
                tile_live=tl.load(Live+n*S+row,active,False).to(tl.int1)
                reference=tl.load(Strength+n*S+row,active,0.)
                touch=tl.load(Touch+n*S+row,active,0)
                steps=tl.where(tile_live,Start+t-touch,0)
                if Fresh:steps=tl.where(t==0,0,steps)
                decayed=advance_unit(reference,steps,Decay,DecayTable,T)
                tl.store(StrengthScratch+n*RS+row,decayed,active)
            tl.debug_barrier()
            strength=tl.load(StrengthScratch+n*RS+s,valid & (s<RS),0.)
            live=tl.load(Live+n*S+s,valid,False).to(tl.int1)
            gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.,volatile=True)
        freed=tl.load(Freed+inp.to(tl.int64)*S+s,valid,0)
        allocation=tl.load(Alloc+inp)
        target=tl.load(Matched+inp)
        case=tl.load(Cases+inp)
        pop_id=tl.load(Pops+inp)
        postpop_live=live & (freed!=1)
        postwrite_live=postpop_live | ((s==allocation) & valid)
        # Death stops future decay but not the derivative of the multiplication
        # that produced its terminal strength. Allocation replaces its adjoint.
        gs=tl.where(postwrite_live,gs*Decay,gs)
        content=tl.load(V+inp*DV+d,d<DV,0.)
        beta=tl.load(Beta+inp)
        dv=tl.full((BV,),0.,tl.float32)
        dbeta=tl.full((),0.,tl.float32)
        dlogit=tl.full((),0.,tl.float32)
        atarget=tl.load(PoolInverse+n*S+target,target>=0,0) if PS else target
        if (case==4)|(case==5):
            before=tl.load(Values+((n*DV+d)*NS+atarget if SoA else (n*NS+atarget)*DV+d),d<DV,0.)
            gradient=tl.load(GValues+((n*DV+d)*AS+atarget if SoA else (n*AS+atarget)*DV+d),d<DV,0.)
            dv=beta*gradient
            dbeta=tl.sum(gradient*(content-before),0)
            tl.store(GValues+((n*DV+d)*AS+atarget if SoA else (n*AS+atarget)*DV+d),(1.-beta)*gradient,d<DV)
        aallocation=tl.load(PoolInverse+n*S+allocation,allocation>=0,0) if PS else allocation
        if allocation>=0:
            gradient=tl.load(GValues+((n*DV+d)*AS+aallocation if SoA else (n*AS+aallocation)*DV+d),d<DV,0.)
            gate=tl.load(Gates+inp)
            dv=gate*gradient
            dlogit=tl.sum(gradient*content,0)*gate*(1.-gate)
            coef_grad+=dlogit*tl.load(Errors+inp)
            tl.store(GValues+((n*DV+d)*AS+aallocation if SoA else (n*AS+aallocation)*DV+d),0.,d<DV)
            if KeyGrad:
                for j in range(DK):
                    gk=tl.load(GKeys+((n*DK+j)*AS+aallocation if SoA else (n*AS+aallocation)*DK+j))
                    tl.store(DKout+inp*DK+j,gk)
                    tl.store(GKeys+((n*DK+j)*AS+aallocation if SoA else (n*AS+aallocation)*DK+j),0.)
            gs=tl.where((s==allocation) & valid,0.,gs)
        tl.store(DVout+inp*DV+d,dv,d<DV)
        tl.store(DBeta+inp,dbeta)
        tl.store(DLogit+inp,dlogit)
        pop=tl.load(Pop+inp)
        dpop=-slot_sum(tl.where(s==pop_id,gs*strength,0.),full_live,BS,triton.next_power_of_2(S),Inverse=inverse,Sparse=PS>0)
        gs=tl.where(s==pop_id,gs*(1.-pop),gs)
        tl.store(DPop+inp,dpop)
        if T>512:
            tl.store(GStrength+n*GS+a,gs,valid & (a<GS))
        tl.debug_barrier()
        # Read adjoints cover the complete restored entry/write prefix.
        score=tl.full((BS,),0.,tl.float32)
        scale=2./tl.sqrt(float(DK))
        for j in range(DK):
            key=tl.load(Keys+((n*DK+j)*NS+a if SoA else (n*NS+a)*DK+j),valid & live,0.)
            query=tl.load(Q+inp*DK+j)
            score+=key*query
        masked=tl.where(live & valid,score*scale,-1.e30)
        maximum=slot_max(masked,full_live,BS,triton.next_power_of_2(S),Inverse=inverse,Sparse=PS>0)
        numer=tl.where(valid,tl.exp(masked-maximum),0.)
        # No-live softmax includes every physical slot in the historical
        # expression, even though the eventual read is multiplied by live.
        tail=tl.where(tl.arange(0,triton.next_power_of_2(S))<S,tl.exp(-1.e30-maximum),0.)
        denom=slot_sum(numer,full_live,BS,triton.next_power_of_2(S),tail,Inverse=inverse,Sparse=PS>0)
        prob=numer/denom
        if T>512:
            tl.store(ProbabilityScratch+n*RS+s,prob,valid & (s<RS))
            strength=tl.load(StrengthScratch+n*RS+s,valid & (s<RS),0.,volatile=True)
        weight=prob*strength*live
        dw=tl.full((BS,),0.,tl.float32)
        for j in range(DV):
            value=tl.load(Values+((n*DV+j)*NS+a if SoA else (n*NS+a)*DV+j),valid & live,0.)
            gout=tl.load(GradOut+inp*DV+j)
            dw+=value*gout
            gv=tl.load(GValues+((n*DV+j)*AS+a if SoA else (n*AS+a)*DV+j),valid & live & value_adjoint_needed,0.)
            tl.store(GValues+((n*DV+j)*AS+a if SoA else (n*AS+a)*DV+j),gv+weight*gout,valid & live & value_adjoint_needed)
        if T>512:
            tl.debug_barrier()
            gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.,volatile=True)
            prob=tl.load(ProbabilityScratch+n*RS+s,valid & (s<RS),0.,volatile=True)
            strength=tl.load(StrengthScratch+n*RS+s,valid & (s<RS),0.,volatile=True)
        gs+=prob*live*dw
        if T>512:
            tl.store(GStrength+n*GS+a,gs,valid & (a<GS))
        dp=dw*strength*live
        if ColdTail:
            # Cold native strength may contain -0. Reconstruct the oracle's
            # inactive dp arithmetic, not a uniformly positive-zero operand.
            full_strength=tl.load(Strength+n*S+full_lane,full_lane<S,0.)
            cold_dp=0.*full_strength*0.
            dp_tail=(tail/denom)*cold_dp
        else:
            cold_dp=0.
            dp_tail=0.
        dp_sum=slot_sum(prob*dp,full_live,BS,triton.next_power_of_2(S),dp_tail,Inverse=inverse,Sparse=PS>0)
        ds=prob*(dp-dp_sum)*live*scale
        dq_tail=(tail/denom)*(cold_dp-dp_sum)*0.*scale*0.
        for j in range(DK):
            key=tl.load(Keys+((n*DK+j)*NS+a if SoA else (n*NS+a)*DK+j),valid & live,0.)
            if KeyGrad:
                query=tl.load(Q+inp*DK+j)
                gk=tl.load(GKeys+((n*DK+j)*AS+a if SoA else (n*AS+a)*DK+j),valid & live,0.)
                tl.store(GKeys+((n*DK+j)*AS+a if SoA else (n*AS+a)*DK+j),gk+ds*query,valid & live)
            tl.store(DQout+inp*DK+j,slot_sum(ds*key,full_live,BS,triton.next_power_of_2(S),dq_tail,Inverse=inverse,Sparse=PS>0))
        tl.debug_barrier()
    if T<=512:
        tl.store(GStrength+n*GS+a,gs,valid & (a<GS))
    tl.store(DCoef+n,coef_grad)


@triton.jit
def reverse_kernel(K,Q,V,Beta,Pop,Keys,Values,Strength,Live,Touch,
                   Cases,Matched,Alloc,Pops,Freed,Gates,Errors,
                   Addresses,Before,Widths,Counts,Boundaries,
                   GradOut,GKeys,GValues,GStrength,StrengthScratch,ProbabilityScratch,DKout,DQout,DVout,
                   DBeta,DPop,DLogit,DCoef,
                   T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                   BS:tl.constexpr,BV:tl.constexpr,UB:tl.constexpr,Capacity,SoA:tl.constexpr,
                   Fresh:tl.constexpr,Start,Decay:tl.constexpr,KeyGrad:tl.constexpr,InitialBound:tl.constexpr,GS:tl.constexpr,RS:tl.constexpr, PoolRows, PoolInverse, PS:tl.constexpr, DecayTable, ColdTail:tl.constexpr=False, NativeValuesGrad:tl.constexpr=True):
    # The allocation trace is integral and already resident. Its maximum,
    # together with the entry prefix bound, covers every row read or mutated.
    # Dispatch once per store; no host query or synchronization is introduced.
    if T>512 and InitialBound>=0 and BS>1:
        n=tl.program_id(0).to(tl.int64)
        token=tl.arange(0,triton.next_power_of_2(T))
        allocated=tl.load(Alloc+n*T+token,token<T,-1)
        reach=tl.maximum(InitialBound,tl.max(allocated,0)+1)
        if reach<=BS//2:
            reverse_program(
                K,Q,V,Beta,Pop,Keys,Values,Strength,Live,
                Touch,Cases,Matched,Alloc,Pops,Freed,Gates,Errors,Addresses,
                Before,Widths,Counts,Boundaries,GradOut,GKeys,GValues,GStrength,StrengthScratch,
                ProbabilityScratch,DKout,DQout,DVout,DBeta,DPop,DLogit,DCoef,
                T,S,DK,DV,BS//2,BV,UB,Capacity,SoA,
                Fresh,Start,Decay,KeyGrad,GS,RS,PoolRows,PoolInverse,PS,DecayTable,ColdTail,NativeValuesGrad)
        else:
            reverse_program(
                K,Q,V,Beta,Pop,Keys,Values,Strength,Live,
                Touch,Cases,Matched,Alloc,Pops,Freed,Gates,Errors,Addresses,
                Before,Widths,Counts,Boundaries,GradOut,GKeys,GValues,GStrength,StrengthScratch,
                ProbabilityScratch,DKout,DQout,DVout,DBeta,DPop,DLogit,DCoef,
                T,S,DK,DV,BS,BV,UB,Capacity,SoA,
                Fresh,Start,Decay,KeyGrad,GS,RS,PoolRows,PoolInverse,PS,DecayTable,ColdTail,NativeValuesGrad)
    else:
        reverse_program(
                K,Q,V,Beta,Pop,Keys,Values,Strength,Live,
                Touch,Cases,Matched,Alloc,Pops,Freed,Gates,Errors,Addresses,
                Before,Widths,Counts,Boundaries,GradOut,GKeys,GValues,GStrength,StrengthScratch,
                ProbabilityScratch,DKout,DQout,DVout,DBeta,DPop,DLogit,DCoef,
                T,S,DK,DV,BS,BV,UB,Capacity,SoA,
                Fresh,Start,Decay,KeyGrad,GS,RS,PoolRows,PoolInverse,PS,DecayTable,ColdTail,NativeValuesGrad)


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


def replay_uses_soa(slots,*,pooled_rows=None):
    """Replay backing policy, shared with the measurement receipt writer."""
    return True


class ReplayStorage:
    """Freeze only when a consuming continuation would overwrite saved state."""
    def __init__(self, owners):
        # Preserve addresses and shared version counters without keeping the
        # returned autograd outputs alive through their own replay context.
        self.owners=tuple(x.detach() for x in owners)
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
        from ..x36_tape_perf.cell_fused import FusedState,tape_scan
        from ..x30_tape_cell.naive_reference import TapeState
        initial=FusedState(TapeState(keys,values,strength,live,age,usage),
                           history,start_tokens,live_bound=saved["live_bound"],owns_import=saved["owns_native"])
        # A continued state's reach may be the saturated capacity bound even
        # when its occupancy is small. Reach is not an adjoint cardinality.
        ctx.entry_live=live.detach().reshape(-1,cfg.slots).clone() if saved["live_bound"] != 0 else None
        fresh_pool=saved['live_bound']==0 and k.shape[1]<=512 and os.environ.get('X40_FRESH_POOL','0')=='1'
        if fresh_pool:ctx.entry_live=live.detach().reshape(-1,cfg.slots).clone()
        ctx.exceptional_cold_tail=False
        if fresh_pool:
            ctx.entry_counts=torch.zeros(k.shape[0]*cfg.n_heads,device=k.device,dtype=torch.int64)
            initial._entry_live_max=0
        elif ctx.entry_live is not None:
            ctx.entry_counts=ctx.entry_live.sum(1)
            unusual=(~live) & (torch.signbit(strength) | ~torch.isfinite(strength))
            metadata=torch.stack((ctx.entry_counts.max(),unusual.any().to(torch.int64))).cpu().tolist()
            initial._entry_live_max=int(metadata[0])
            ctx.exceptional_cold_tail=bool(metadata[1])
            del unusual
        out,final,trace,pressure=tape_scan(k,q,v,beta,pop,logit,initial,cfg,coef,
                                          record_all=True)
        journal=final.all_field_journal
        # Keep the owned final buffers until replay is consumed. A subsequent
        # consuming continuation freezes them before its first mutation.
        ctx.storage=ReplayStorage(journal.owners)
        final.replay_storage=ctx.storage
        ctx.address_owners=ctx.storage.owners
        ctx.replay=ctx.storage.owners  # address-space accounting; zero forward copies
        ctx.cfg=cfg
        ctx.initial_bound=saved["live_bound"]
        ctx.entry_live_max=getattr(initial,"_entry_live_max",None)
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
        ctx.storage.validate()
        # Key adjoints feed only key inputs, not query/value/strength adjoints.
        # The LM's detached keys therefore need no key-adjoint matrix.
        key_grad=ctx.needs_input_grad[4] or ctx.needs_input_grad[11]
        replay_rows=(cfg.slots if ctx.initial_bound is None else
                     min(cfg.slots,ctx.initial_bound+t))
        pool_rows=pool_inverse=None
        if ctx.entry_live is not None and t<=512:
            from ..x36_tape_perf.adjoint_pool import build
            pool_rows,pool_inverse,pool_counts=build(ctx.entry_live,ctx.trace[2])
            ctx.pool_stats=dict(union_counts=pool_counts,allocated_rows=pool_rows.shape[-1],
                entry_counts=ctx.entry_counts,intra_chunk_events=t,slots=cfg.slots,
                worst_case_rows=min(cfg.slots,ctx.entry_live_max+t),
                adjoint_bytes=b*h*pool_rows.shape[-1]*(1+cfg.value_dim+(dk if key_grad else 0))*4,
                journal_capacity=ctx.capacity,exceptional_cold_tail=ctx.exceptional_cold_tail)
            replay_rows=pool_rows.shape[-1]
        payload_soa=replay_uses_soa(cfg.slots,pooled_rows=replay_rows if pool_rows is not None else None)
        if pool_rows is not None:
            ctx.pool_stats['payload_layout']='hot_soa'
            ctx.pool_stats['hot_payload_bytes']=b*h*replay_rows*(dk+cfg.value_dim)*4
        from ..x36_tape_perf.replay_setup import prepare
        replay,(gk,gv,gs)=prepare(ctx.storage.owners,(gkeys,gvalues,gstrength),soa=payload_soa,key_grad=key_grad,
            strength_rows=None if ctx.needs_input_grad[13] else replay_rows,pool_rows=pool_rows)
        keys,values,strength,live,age,history,*meta=replay
        # Replay payloads and adjoints use [store,dim,slot] backing; full-S
        # reduction/accumulation expressions retain the oracle lane shapes.
        streamed=t>512
        strength_scratch=torch.empty((b,h,replay_rows),device=k.device) if streamed else strength
        probability_scratch=torch.empty_like(strength_scratch) if streamed else strength
        go=torch.zeros(b,h,t,cfg.value_dim,device=k.device) if gout is None else flat(gout)
        dkout=(torch.zeros_like(args[0]) if cfg.slots>=4096 else torch.empty_like(args[0])) if key_grad else args[0]
        dqout,dvout,dbeta,dpop=[torch.empty_like(x) for x in args[1:]]
        dlogit=torch.empty((b,h,t),device=k.device,dtype=k.dtype)
        dc=torch.empty(b*h,device=k.device)
        original_addresses,before,widths,counts,boundaries,gates,errors=ctx.log
        # The log is private to this autograd node. Rebase it in place and
        # restore it on this stream before returning, retaining repeat-backward
        # semantics without a capacity-sized address clone (4.5 GiB at T4096).
        addresses=original_addresses
        rebase_owned(addresses,counts,ctx.address_owners,replay,ctx.capacity,payload_slots=cfg.slots if payload_soa else 0,
                     pool_rows=pool_rows,pool_inverse=pool_inverse)
        # Eight warps retain exact reduction trees only in the independently
        # tested, empty-origin prefix specialization. Unknown imports and
        # longer continuations keep the oracle's warp count.
        prefix8=(cfg.slots>=2944 and cfg.key_dim==64 and cfg.value_dim==128
                 and pool_rows is None and ctx.initial_bound is not None and ctx.initial_bound+t<=512)
        decay_table,_,_=table(cfg.decay_rate,cfg.free_threshold,t,k.device)
        try:
            reverse_kernel[(b*h,)](*args,keys,values,strength,live,meta[5],
                *ctx.trace,gates,errors,addresses,before,widths,counts,boundaries,
                go,gk,gv,gs,strength_scratch,probability_scratch,dkout,dqout,dvout,dbeta,dpop,dlogit,dc,
                t,cfg.slots,dk,cfg.value_dim,triton.next_power_of_2(max(1,replay_rows)),
                triton.next_power_of_2(cfg.value_dim),min(128,triton.next_power_of_2(max(cfg.slots,cfg.value_dim,dk))),ctx.capacity,payload_soa,True,ctx.start,
                struct.unpack('<f',struct.pack('<f',cfg.decay_rate))[0],key_grad,
                -1 if ctx.initial_bound is None else ctx.initial_bound,gs.shape[-1],replay_rows,
                live if pool_rows is None else pool_rows,live if pool_inverse is None else pool_inverse,0 if pool_rows is None else replay_rows,
                decay_table,
                ColdTail=pool_rows is not None and ctx.exceptional_cold_tail,
                NativeValuesGrad=ctx.needs_input_grad[12],
                enable_fp_fusion=False,debug=True,num_warps=8 if prefix8 else (16 if cfg.slots>=4096 else 4),
                maxnreg=128 if cfg.slots>=4096 else None)
        finally:
            rebase_owned(addresses,counts,replay,ctx.address_owners,ctx.capacity,
                         payload_slots=cfg.slots if payload_soa else 0,reverse_payload=payload_soa,
                         pool_rows=pool_rows,pool_inverse=pool_inverse)
        def unflat(x):return x.transpose(1,2).contiguous()
        input_grads=tuple(unflat(g) if ctx.needs_input_grad[4+i] else None
                          for i,g in enumerate((dkout,dqout,dvout,dbeta,dpop,dlogit)))
        def native_grad(g,index):
            if not ctx.needs_input_grad[index]:return None
            if pool_rows is not None:
                from ..x36_tape_perf.replay_setup import unpack
                return unpack(g,pool_rows,(gkeys,gvalues,gstrength)[index-11],
                              ctx.storage.owners[index-11],soa=payload_soa,strength=index==13)
            return g.transpose(-1,-2).contiguous() if payload_soa else g
        return (None,None,None,None,*input_grads,
                dc.reshape(b,h).sum(0) if ctx.needs_input_grad[10] else None,
                native_grad(gk,11),native_grad(gv,12),native_grad(gs,13) if pool_rows is not None else (gs if ctx.needs_input_grad[13] else None),
                None,None,None)


def autograd_scan(inputs,state,cfg,coef):
    from ..x36_tape_perf.cell_fused import FusedState
    from ..x30_tape_cell.naive_reference import TapeState,init_state
    k=inputs[0];b,t,h=k.shape[:3]
    if state is None:
        if os.environ.get('X40_FUSED_INIT','0')=='1':
            from .fused_init import fresh_state
            native,history=fresh_state(b,cfg,k.device)
        else:
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
