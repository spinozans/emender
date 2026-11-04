"""B-only native shared-store forward. Frozen velocity kernels are untouched.

The backward replays the differentiable eager store from independent empty
window state and checks native/replay output agreement before differentiating.
This does not reuse or claim certification of the standing X40 reverse kernel.
"""
import math
import torch
import triton
import triton.language as tl
from .c14_b_store import SharedStore


@triton.jit
def kernel(K,Q,V,A,Pop,Append,Logit,Coef,Match,Error,Own,Pred,Expiry,Keep,Ids,Valid,
           Out,FinalK,FinalV,FinalS,FinalMeta,Trace,
           T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,
           Decay:tl.constexpr,Free:tl.constexpr,PopThreshold:tl.constexpr,
           Enable:tl.constexpr,Eps:tl.constexpr,MatchWire:tl.constexpr,
           OccSlope:tl.constexpr,OccOnset:tl.constexpr):
    b = tl.program_id(0)
    ss = tl.arange(0,BS)
    kd = tl.arange(0,64)
    vd = tl.arange(0,128)
    keys = tl.full((BS,64),0.,tl.float32)
    values = tl.full((BS,128),0.,tl.float32)
    strength = tl.full((BS,),0.,tl.float32)
    live = tl.full((BS,),False,tl.int1)
    birth = tl.full((BS,),-1,tl.int32)
    order = tl.full((BS,),-1,tl.int32)
    owner = tl.full((BS,),-1,tl.int32)
    prediction = tl.full((BS,),-1,tl.int32)
    pred_at = tl.full((BS,),-1,tl.int32)
    expires = tl.full((BS,),-1,tl.int32)
    valid = tl.load(Valid+b)
    coef = tl.load(Coef)
    mc = tl.load(Match)
    ec = tl.load(Error)
    for t in range(T):
        k = tl.load(K+(b*T+t)*64+kd)
        q = tl.load(Q+(b*T+t)*64+kd)
        v = tl.load(V+(b*T+t)*128+vd)
        av = tl.load(A+(b*T+t)*128+vd)
        append = tl.load(Append+b*T+t)
        logit = tl.load(Logit+b*T+t)
        if t < valid:
            a_alloc=tl.full((),0,tl.int32)
            a_dup=tl.full((),0,tl.int32)
            a_cap=tl.full((),0,tl.int32)
            s_alloc=tl.full((),0,tl.int32)
            s_correct=tl.full((),0,tl.int32)
            s_cap=tl.full((),0,tl.int32)
            byte = tl.load(Ids+b*T+t)
            correct = live & (owner>=0) & (pred_at==t-1) & (prediction!=byte)
            values = tl.where(correct[:,None],values+(v[None,:]-values),values)
            stale = live & (owner>=0) & (expires<=t)
            live = live & ~stale
            own = tl.load(Own+b*T+t)
            keep = tl.load(Keep+b*T+t)
            if (own>=0) & keep:
                dup = tl.min(tl.where(live & (owner==own),ss,S),0)
                free = tl.min(tl.where((ss<S)&~live,ss,S),0)
                has = tl.sum(live.to(tl.int32),0)>0
                best_order = tl.max(tl.where(live,order,-1),0)
                youngest = tl.min(tl.where(live & (order==best_order),ss,S),0)
                adjacent_order = tl.max(tl.where(live & (birth==t-1),order,-1),0)
                adjacent = adjacent_order>=0
                target = tl.where(adjacent,tl.min(tl.where(live & (order==adjacent_order),ss,S),0),youngest)
                prev = tl.sum(tl.where((live & (ss==target) & adjacent)[:,None],values,0.),0)
                diff = av-prev
                err = tl.where(has,tl.sqrt(tl.sum(diff*diff,0)),0.)
                gate = tl.sigmoid(append+coef*err)
                alloc = (dup==S) & (free<S)
                dst = tl.where(dup<S,dup,tl.where(free<S,free,target))
                a_alloc=alloc.to(tl.int32)
                a_dup=(dup<S).to(tl.int32)
                a_cap=((dup==S)&(free==S)).to(tl.int32)
                changed = ss==dst
                values = tl.where(changed[:,None],tl.where(alloc,gate*av[None,:],values+(av[None,:]-values)),values)
                keys = tl.where((changed & alloc)[:,None],k[None,:],keys)
                strength = tl.where(changed & alloc,1.,strength)
                birth = tl.where(changed & alloc,t,birth)
                order = tl.where(changed & alloc,2*t,order)
                owner = tl.where(changed,own,owner)
                prediction = tl.where(changed,tl.load(Pred+b*T+t),prediction)
                pred_at = tl.where(changed,t,pred_at)
                expires = tl.where(changed,tl.load(Expiry+b*T+t),expires)
                live = live | changed
            pre_array=tl.sum((live & (owner>=0)).to(tl.int32),0)
            pre_stream=tl.sum((live & (owner<0)).to(tl.int32),0)
            scores = tl.sum(keys*q[None,:],1)*(2./tl.sqrt(64.))
            masked = tl.where(live,scores,-1.e30)
            maximum = tl.max(masked,0)
            numer = tl.where(live,tl.exp(masked-maximum),0.)
            weights = numer/tl.maximum(tl.sum(numer,0),1.e-30)
            read = tl.sum(weights[:,None]*strength[:,None]*values,0)
            tl.store(Out+(b*T+t)*128+vd,read)
            hits=tl.sum((live & (owner>=0) & (weights*strength>0)).to(tl.int32),0)
            best = tl.min(tl.where(live & (scores==maximum),ss,S),0)
            evidence = tl.where(best<S,maximum,0.)
            pop = tl.load(Pop+b*T+t)
            popped = (ss==best) & (pop>PopThreshold)
            strength = tl.where(popped,strength*(1.-pop),strength)
            live = live & ~(popped & (strength<=Free))
            has = tl.sum(live.to(tl.int32),0)>0
            best_order = tl.max(tl.where(live,order,-1),0)
            youngest = tl.min(tl.where(live & (order==best_order),ss,S),0)
            adjacent_order = tl.max(tl.where(live & (birth==t-1),order,-1),0)
            adjacent = adjacent_order>=0
            target = tl.where(adjacent,tl.min(tl.where(live & (order==adjacent_order),ss,S),0),youngest)
            prev = tl.sum(tl.where((live & (ss==target) & adjacent)[:,None],values,0.),0)
            diff = v-prev
            residual = tl.sum(diff*diff,0)
            err = tl.where(has,tl.sqrt(residual),0.)
            occ = tl.sum(live.to(tl.int32),0).to(tl.float32)/S
            wire = logit+OccSlope*tl.maximum(0.,occ-OccOnset)+mc*(MatchWire*evidence)+ec*err
            if tl.sigmoid(wire)>Enable:
                request = ~has | ~adjacent | (residual<Eps)
                free = tl.min(tl.where((ss<S)&~live,ss,S),0)
                alloc = request & (free<S)
                dst = tl.where(alloc,free,target)
                gate = tl.sigmoid(append+coef*err)
                s_alloc=alloc.to(tl.int32)
                s_correct=(~request).to(tl.int32)
                s_cap=(request & (free==S)).to(tl.int32)
                changed = ss==dst
                values = tl.where(changed[:,None],tl.where(alloc,gate*v[None,:],values+(v[None,:]-values)),values)
                keys = tl.where((changed & alloc)[:,None],k[None,:],keys)
                strength = tl.where(changed & alloc,1.,strength)
                birth = tl.where(changed & alloc,t,birth)
                order = tl.where(changed & alloc,2*t+1,order)
                owner = tl.where(changed,-1,owner)
                prediction = tl.where(changed,-1,prediction)
                pred_at = tl.where(changed,-1,pred_at)
                expires = tl.where(changed,-1,expires)
                live = live | changed
            post_array=tl.sum((live & (owner>=0)).to(tl.int32),0)
            post_stream=tl.sum((live & (owner<0)).to(tl.int32),0)
            decayed=tl.sum(live.to(tl.int32),0)
            strength = tl.where(live,strength*Decay,strength)
            deaths=tl.sum((live & (strength<=Free)).to(tl.int32),0)
            live = live & (strength>Free)
            for field in tl.static_range(18):
                stat=tl.sum(correct.to(tl.int32),0)
                if field==1: stat=tl.sum(stale.to(tl.int32),0)
                if field==2: stat=a_alloc
                if field==3: stat=a_dup
                if field==4: stat=a_cap
                if field==5: stat=pre_array
                if field==6: stat=hits
                if field==7: stat=tl.sum(popped.to(tl.int32),0)
                if field==8: stat=s_alloc
                if field==9: stat=s_correct
                if field==10: stat=s_cap
                if field==11: stat=decayed
                if field==12: stat=deaths
                if field==13: stat=tl.maximum(pre_array,post_array)
                if field==14: stat=tl.sum((live & (owner>=0)).to(tl.int32),0)
                if field==15: stat=tl.sum((live & (owner<0)).to(tl.int32),0)
                if field==16: stat=tl.maximum(pre_array+pre_stream,post_array+post_stream)
                if field==17: stat=tl.maximum(pre_stream,post_stream)
                tl.store(Trace+(b*T+t)*18+field,stat)
        else:
            tl.store(Out+(b*T+t)*128+vd,0.)
    tl.store(FinalK+(b*S+ss[:,None])*64+kd[None,:],keys,ss[:,None]<S)
    tl.store(FinalV+(b*S+ss[:,None])*128+vd[None,:],values,ss[:,None]<S)
    tl.store(FinalS+b*S+ss,strength,ss<S)
    # All bookkeeping needed to compare/restore the physical final state.
    for field in tl.static_range(7):
        value = live.to(tl.int32)
        if field==1: value=birth
        if field==2: value=order
        if field==3: value=owner
        if field==4: value=prediction
        if field==5: value=pred_at
        if field==6: value=expires
        tl.store(FinalMeta+(b*7+field)*S+ss,value,ss<S)


def eager(k,q,v,av,pop,append,logits,coef,mc,ec,cfg,policy,metadata):
    owners,raw,windows,valid,keep,predictions = metadata
    b,t,_ = k.shape
    batch_reads=[]; reports=[]
    for j in range(b):
        store=SharedStore(cfg); tokens=[]
        for i in range(t):
            content=v[j,i]
            if i>=valid[j]:
                tokens.append(torch.zeros_like(content)); continue
            store.observe(windows[j],raw[j][:i+1],raw[j][i],content,i)
            if keep[j][i] and owners[j][i].suffix:
                target,adjacent=store.target(i)
                prev=store.slots[target].value if adjacent else torch.zeros_like(content)
                a=av[j,i]
                err=(a-prev).square().sum().detach().sqrt() if target is not None else a.new_zeros(())
                gate=(append[j,i]+coef*err).sigmoid()
                store.insert_array(k[j,i],a,owners[j][i],predictions[j][i],i,gate)
            read,best,evidence=store.read(q[j,i],torch.zeros_like(content))
            tokens.append(read)
            store.stream(k[j,i],content,pop[j,i],append[j,i],logits[j,i],coef,mc,ec,policy,best,evidence,i)
            store.decay()
        batch_reads.append(torch.stack(tokens)); reports.append(store.report())
    return torch.stack(batch_reads),reports


def native_forward(inputs,cfg,policy,metadata):
    k,q,v,av,pop,append,logits,coef,mc,ec=inputs
    owners,raw,windows,valid,keep,predictions=metadata
    b,t,_=k.shape
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
    output=torch.empty_like(v)
    state=(torch.empty(b,cfg.slots,64,device=k.device),torch.empty(b,cfg.slots,128,device=k.device),
           torch.empty(b,cfg.slots,device=k.device),torch.empty(b,7,cfg.slots,device=k.device,dtype=torch.int32))
    trace=torch.zeros(b,t,18,device=k.device,dtype=torch.int32)
    kernel[(b,)](*(x.contiguous() for x in inputs),ints(owner_ids),ints(predictions),ints(expiry),ints(keep),
        ints([list(r) for r in raw]),ints(valid),output,*state,trace,T=t,S=cfg.slots,BS=triton.next_power_of_2(cfg.slots),
        Decay=cfg.decay_rate,Free=cfg.free_threshold,PopThreshold=cfg.pop_threshold,Enable=cfg.write_enable_threshold,
        Eps=cfg.eps_residual,MatchWire=policy.match_wire,OccSlope=policy.occ_slope,OccOnset=policy.occ_onset,
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
        # Private immutable replay graph, rebuilt from the exact window reset;
        # nothing is written back to the forward store or another window.
        replay_inputs=[x.detach().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
        with torch.enable_grad():
            replay,_=eager(*replay_inputs,ctx.cfg,ctx.policy,ctx.store_metadata)
            torch.testing.assert_close(replay,out,atol=2e-5,rtol=2e-5,msg='native/replay divergence; fail closed')
            active=[x for x in replay_inputs if x.requires_grad]
            grads=torch.autograd.grad(replay,active,grad,allow_unused=True)
        it=iter(grads)
        return (None,None,None,*(next(it) if x.requires_grad else None for x in replay_inputs))


def native_reports(trace,cfg,valid):
    reports=[]
    for j,row in enumerate(trace.cpu().tolist()):
        row=row[:valid[j]]
        def total(n):return sum(r[n] for r in row)
        def peak(n):return max((r[n] for r in row),default=0)
        denom=total(5)
        reports.append(dict(APPEND=total(2)+total(8),POP=total(1)+total(7)+total(12),
            DECAY=total(11),CORRECT=total(0)+total(9),MERGE=total(3)+total(4)+total(10),
            stale_retirements=total(1),natural_retirements=total(12),
            capacity_merges=total(4)+total(10),duplicate_merges=total(3),
            array_requests=total(2)+total(3)+total(4),refused_insertions=0,
            array_read_opportunities=denom,array_slots_found=denom,array_slots_read=total(6),
            array_read_hit_rate=total(6)/denom if denom else None,reads=len(row),
            peak_occupancy=peak(16),peak_array=peak(13),peak_stream=peak(17),
            final_array=row[-1][14] if row else 0,final_stream=row[-1][15] if row else 0,
            occupancy=[(r[15],r[14]) for r in row],budget=cfg.slots,telemetry_source='native_event_trace'))
    return reports
