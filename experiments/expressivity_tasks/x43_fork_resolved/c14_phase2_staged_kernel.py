"""Uncertified experiment: explicit global staging between token iterations."""
import triton
import triton.language as tl

@triton.jit
def kernel(K,Q,V,A,Pop,Append,Logit,Coef,Match,Error,Own,Pred,Expiry,Keep,Ids,Valid,
           Out,FinalK,FinalV,FinalS,FinalMeta,Trace,
           T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,
           Decay:tl.constexpr,Free:tl.constexpr,PopThreshold:tl.constexpr,
           Enable:tl.constexpr,Eps:tl.constexpr,MatchWire:tl.constexpr,
           OccSlope:tl.constexpr,OccOnset:tl.constexpr,AppendOnly:tl.constexpr):
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
    tl.store(FinalK+(b*S+ss[:,None])*64+kd[None,:],keys,ss[:,None]<S)
    tl.store(FinalV+(b*S+ss[:,None])*128+vd[None,:],values,ss[:,None]<S)
    for t in range(T):
        keys = tl.load(FinalK+(b*S+ss[:,None])*64+kd[None,:],ss[:,None]<S,0.)
        values = tl.load(FinalV+(b*S+ss[:,None])*128+vd[None,:],ss[:,None]<S,0.)
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
            if AppendOnly: correct=tl.full((BS,),False,tl.int1)
            values = tl.where(correct[:,None],values+(v[None,:]-values),values)
            stale = live & (owner>=0) & (expires<=t)
            if AppendOnly: stale=tl.full((BS,),False,tl.int1)
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
                target_array=tl.max(tl.where(live & (ss==target),owner,-1),0)>=0
                if ~(AppendOnly & target_array & ~request):
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
            age_mask=live
            if AppendOnly: age_mask=live & (owner<0)
            decayed=tl.sum(age_mask.to(tl.int32),0)
            strength = tl.where(age_mask,strength*Decay,strength)
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
