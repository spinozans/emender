"""Generated diagnostic; no production import."""
import triton
import triton.language as tl
from ..x31_tape_cell_fused.all_field_journal import undo_to
from .exact_decay import advance_unit

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


@triton.jit(do_not_specialize=["Token"])
def reverse_kernel(Scalars, Token, Phase:tl.constexpr, K,Q,V,Beta,Pop,Keys,Values,Strength,Live,Touch,
                   Cases,Matched,Alloc,Pops,Freed,Gates,Errors,
                   Addresses,Before,Widths,Counts,Boundaries,
                   GradOut,GKeys,GValues,GStrength,StrengthScratch,ProbabilityScratch,DKout,DQout,DVout,
                   DBeta,DPop,DLogit,DCoef,
                   T:tl.constexpr,S:tl.constexpr,DK:tl.constexpr,DV:tl.constexpr,
                   BS:tl.constexpr,BV:tl.constexpr,UB:tl.constexpr,Capacity,SoA:tl.constexpr,
                   Fresh:tl.constexpr,Start,Decay:tl.constexpr,KeyGrad:tl.constexpr,InitialBound:tl.constexpr,GS:tl.constexpr,RS:tl.constexpr, PoolRows, PoolInverse, PS:tl.constexpr, DecayTable, ColdTail:tl.constexpr=False, NativeValuesGrad:tl.constexpr=True):
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
    t=Token
    inp=n*T+t
    full_lane=tl.arange(0,triton.next_power_of_2(S))
    full_live=tl.load(Live+n*S+full_lane,full_lane<S,False).to(tl.int1)
    inverse=tl.load(PoolInverse+n*S+full_lane,full_lane<S,-1) if PS else full_lane
    tl.device_assert((inverse>=0 if PS else full_lane<BS)|(~full_live),'live row outside compact replay')
    live=tl.load(Live+n*S+s,valid,False).to(tl.int1)
    if Phase==0:
        start=tl.load(Boundaries+inp)
        if t==T-1:
            stop=tl.load(Counts+n)
        else:
            stop=tl.load(Boundaries+inp+1)
        undo_to((Addresses,Before,Widths,Counts,n,Capacity),start,stop,UB)
    if Phase==1:
        gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.)
        coef_grad=tl.load(DCoef+n) if t<T-1 else tl.full((),0.,tl.float32)
        if S<4096 and KeyGrad:
            kd=tl.arange(0,triton.next_power_of_2(DK))
            tl.store(DKout+inp*DK+kd,0.,kd<DK)
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
                logical=tl.load(PoolRows+n*PS+row,active & (row<PS),-1) if PS else row
                active=active & (logical>=0)
                tile_live=tl.load(Live+n*S+logical,active,False).to(tl.int1)
                reference=tl.load(Strength+n*S+logical,active,0.)
                touch=tl.load(Touch+n*S+logical,active,0)
                steps=tl.where(tile_live,Start+t-touch,0)
                if Fresh:steps=tl.where(t==0,0,steps)
                decayed=advance_unit(reference,steps,Decay,DecayTable,T)
                tl.store(StrengthScratch+n*RS+row,decayed,active)
            tl.debug_barrier()
            strength=tl.load(StrengthScratch+n*RS+a,valid & (a<RS),0.)
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
        tl.store(GStrength+n*GS+a,gs,valid & (a<GS))
        tl.debug_barrier()
        tl.store(StrengthScratch+n*RS+a,strength,valid & (a<RS))
        tl.store(DCoef+n,coef_grad)
    if Phase==2:
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
        tl.store(ProbabilityScratch+n*RS+a,prob,valid & (a<RS))
        tl.store(Scalars+n*2,maximum)
        tl.store(Scalars+n*2+1,denom)
    if Phase==3:
        prob=tl.load(ProbabilityScratch+n*RS+a,valid & (a<RS),0.)
        strength=tl.load(StrengthScratch+n*RS+a,valid & (a<RS),0.)
        gs=tl.load(GStrength+n*GS+a,valid & (a<GS),0.)
        maximum=tl.load(Scalars+n*2)
        denom=tl.load(Scalars+n*2+1)
        tail=tl.where(full_lane<S,tl.exp(-1.e30-maximum),0.)
        scale=2./tl.sqrt(float(DK))
        weight=prob*strength*live
        dw=tl.full((BS,),0.,tl.float32)
        for j in range(DV):
            value=tl.load(Values+((n*DV+j)*NS+a if SoA else (n*NS+a)*DV+j),valid & live,0.)
            gout=tl.load(GradOut+inp*DV+j)
            dw+=value*gout
            gv=tl.load(GValues+((n*DV+j)*AS+a if SoA else (n*AS+a)*DV+j),valid & live & value_adjoint_needed,0.)
            tl.store(GValues+((n*DV+j)*AS+a if SoA else (n*AS+a)*DV+j),gv+weight*gout,valid & live & value_adjoint_needed)
        gs+=prob*live*dw
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
