"""Uncertified split-step/ordered-slot-page candidate. Explicit opt-in only."""
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as tl


@gluon.jit
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))


@gluon.jit
def reduction_input(x, axis: tl.constexpr):
    if len(x.shape) == 1:
        return tl.convert_layout(x, tl.BlockedLayout([1], [32], [8], [0]))
    elif axis == 1:
        return tl.convert_layout(x, tl.BlockedLayout([1, 1], [1, 32], [4, 2], [1, 0]))
    else:
        return tl.convert_layout(x, tl.BlockedLayout([1, 1], [1, 32], [2, 4], [1, 0]))


@gluon.jit
def fixed_sum(x, axis: tl.constexpr):
    result = tl.sum(reduction_input(x, axis), axis)
    if len(x.shape) == 2:
        return tl.convert_layout(result, tl.BlockedLayout([1], [32], [8], [0]))
    else:
        return result


@gluon.jit
def fixed_min(x, axis: tl.constexpr):
    return tl.min(reduction_input(x, axis), axis)


@gluon.jit
def fixed_max(x, axis: tl.constexpr):
    return tl.max(reduction_input(x, axis), axis)


@gluon.jit
def as_rows(x):
    return tl.convert_layout(x, tl.SliceLayout(1, tl.BlockedLayout([1,1], [1,32], [2,4], [1,0])))[:,None]


@gluon.jit
def as_cols(x):
    return tl.convert_layout(x, tl.SliceLayout(0, tl.BlockedLayout([1,1], [1,32], [2,4], [1,0])))[None,:]


@gluon.jit
def previous_value(FinalV, Scratch, b, S:tl.constexpr, BS:tl.constexpr, selected, ss, vd):
    target = fixed_min(tl.where(selected,ss,S),0)
    tl.thread_barrier()
    value = tl.load(FinalV+(b*S+target)*128+vd,target<S,0.)
    # Reference reduction includes positive-zero unselected rows.
    return value + 0.


@gluon.jit
def update_value(FinalV, Scratch, Input, b, S:tl.constexpr, BS:tl.constexpr, ss, changed, alloc, gate):
    tl.store(Scratch+b*S+ss,changed.to(tl.float32),ss<S)
    tl.thread_barrier()
    cd = tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    piece = tl.load(Input+cd)
    for r in range(tl.cdiv(BS,128)):
        rs = r*128+tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
        change = tl.load(Scratch+b*S+rs,rs<S,0.) != 0.
        values = tl.load(FinalV+(b*S+as_rows(rs))*128+as_cols(cd),as_rows(rs)<S,0.)
        values = tl.where(as_rows(change),tl.where(alloc,gate*as_cols(piece),values+(as_cols(piece)-values)),values)
        tl.store(FinalV+(b*S+as_rows(rs))*128+as_cols(cd),values,as_rows(rs)<S)
    tl.thread_barrier()


@gluon.jit
def stream_read(FinalV, Scratch, b, S:tl.constexpr, BS:tl.constexpr, ss, weights, strength, vd):
    tl.store(Scratch+b*7*S+ss,(weights*strength).to(tl.int32,bitcast=True),ss<S)
    tl.thread_barrier()
    rs = tl.arange(0,2,layout=tl.BlockedLayout([1],[32],[8],[0]))
    # First pair initializes the accumulator without an extra zero addition.
    w = tl.load(Scratch+b*7*S+rs,rs<S,0).to(tl.float32,bitcast=True)
    value = tl.load(FinalV+(b*S+as_rows(rs))*128+as_cols(vd),as_rows(rs)<S,0.)
    acc = as_rows(w)*value
    for r in range(1,BS//2):
        rows = 2*r+rs
        w = tl.load(Scratch+b*7*S+rows,rows<S,0).to(tl.float32,bitcast=True)
        value = tl.load(FinalV+(b*S+as_rows(rows))*128+as_cols(vd),as_rows(rows)<S,0.)
        acc = acc + as_rows(w)*value
    return fixed_sum(acc,0)


@gluon.jit
def control(K,Q,V,A,Pop,Append,Logit,Coef,Match,Error,Own,Pred,Expiry,Keep,Ids,Valid,
           Out,FinalK,FinalV,FinalS,FinalMeta,Trace,
           Score,Product,Scalars,t,Phase:tl.constexpr,
           T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,
           Decay:tl.constexpr,Free:tl.constexpr,PopThreshold:tl.constexpr,
           Enable:tl.constexpr,Eps:tl.constexpr,MatchWire:tl.constexpr,
           OccSlope:tl.constexpr,OccOnset:tl.constexpr,AppendOnly:tl.constexpr):
    tl.static_assert(7*S >= 128, "small contexts must use the reference kernel")
    b = tl.program_id(0)
    ss = tl.arange(0,BS,layout=tl.BlockedLayout([1],[32],[8],[0]))
    kd = tl.arange(0,64,layout=tl.BlockedLayout([1],[32],[8],[0]))
    vd = tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    strength = tl.load(FinalS+b*S+ss,ss<S,0.)
    live = tl.load(FinalMeta+(b*7)*S+ss,ss<S,0)!=0
    birth = tl.load(FinalMeta+(b*7+1)*S+ss,ss<S,-1)
    order = tl.load(FinalMeta+(b*7+2)*S+ss,ss<S,-1)
    owner = tl.load(FinalMeta+(b*7+3)*S+ss,ss<S,-1)
    prediction = tl.load(FinalMeta+(b*7+4)*S+ss,ss<S,-1)
    pred_at = tl.load(FinalMeta+(b*7+5)*S+ss,ss<S,-1)
    expires = tl.load(FinalMeta+(b*7+6)*S+ss,ss<S,-1)
    valid=tl.load(Valid+b)
    coef=tl.load(Coef);mc=tl.load(Match);ec=tl.load(Error)
    if t < valid:
        k=tl.load(K+(b*T+t)*64+kd)
        q=tl.load(Q+(b*T+t)*64+kd)
        v=tl.load(V+(b*T+t)*128+vd)
        av=tl.load(A+(b*T+t)*128+vd)
        append=tl.load(Append+b*T+t)
        logit=tl.load(Logit+b*T+t)
        if Phase == 0:
            a_alloc=0
            a_dup=0
            a_cap=0
            s_alloc=0
            s_correct=0
            s_cap=0
            byte = tl.load(Ids+b*T+t)
            correct = live & (owner>=0) & (pred_at==t-1) & (prediction!=byte)
            if AppendOnly: correct=tl.full((BS,),False,tl.int1,layout=tl.BlockedLayout([1],[32],[8],[0]))
            update_value(FinalV,FinalS,V+(b*T+t)*128,b,S,BS,ss,correct,False,1.)
            stale = live & (owner>=0) & (expires<=t)
            if AppendOnly: stale=tl.full((BS,),False,tl.int1,layout=tl.BlockedLayout([1],[32],[8],[0]))
            live = live & ~stale
            own = tl.load(Own+b*T+t)
            keep = tl.load(Keep+b*T+t)
            if (own>=0) & keep:
                dup = fixed_min(tl.where(live & (owner==own),ss,S),0)
                free = fixed_min(tl.where((ss<S)&~live,ss,S),0)
                has = fixed_sum(live.to(tl.int32),0)>0
                best_order = fixed_max(tl.where(live,order,-1),0)
                youngest = fixed_min(tl.where(live & (order==best_order),ss,S),0)
                adjacent_order = fixed_max(tl.where(live & (birth==t-1),order,-1),0)
                adjacent = adjacent_order>=0
                target = tl.where(adjacent,fixed_min(tl.where(live & (order==adjacent_order),ss,S),0),youngest)
                prev = previous_value(FinalV,FinalMeta,b,S,BS,live & (ss==target) & adjacent,ss,vd)
                diff = av-prev
                err = tl.where(has,tl.sqrt(fixed_sum(diff*diff,0)),0.)
                gate = sigmoid(append+coef*err)
                alloc = (dup==S) & (free<S)
                dst = tl.where(dup<S,dup,tl.where(free<S,free,target))
                a_alloc=alloc.to(tl.int32)
                a_dup=(dup<S).to(tl.int32)
                a_cap=((dup==S)&(free==S)).to(tl.int32)
                changed = ss==dst
                update_value(FinalV,FinalS,A+(b*T+t)*128,b,S,BS,ss,changed,alloc,gate)
                tl.store(FinalK+(b*S+dst)*64+kd,k,alloc & (dst<S) & (kd<64))
                strength = tl.where(changed & alloc,1.,strength)
                birth = tl.where(changed & alloc,t,birth)
                order = tl.where(changed & alloc,2*t,order)
                owner = tl.where(changed,own,owner)
                prediction = tl.where(changed,tl.load(Pred+b*T+t),prediction)
                pred_at = tl.where(changed,t,pred_at)
                expires = tl.where(changed,tl.load(Expiry+b*T+t),expires)
                live = live | changed
            pre_array=fixed_sum((live & (owner>=0)).to(tl.int32),0)
            pre_stream=fixed_sum((live & (owner<0)).to(tl.int32),0)
            for field in tl.static_range(6):
                stat=fixed_sum(correct.to(tl.int32),0)
                if field==1: stat=fixed_sum(stale.to(tl.int32),0)
                if field==2: stat=a_alloc
                if field==3: stat=a_dup
                if field==4: stat=a_cap
                if field==5: stat=pre_array
                tl.store(Trace+(b*T+t)*18+field,stat)
            tl.store(Scalars+b*8,pre_array)
            tl.store(Scalars+b*8+1,pre_stream)
        else:
            s_alloc=0;s_correct=0;s_cap=0
            pre_array=tl.load(Scalars+b*8)
            pre_stream=tl.load(Scalars+b*8+1)
            scores=tl.load(Score+b*BS+ss)
            maximum=tl.load(Scalars+b*8+2).to(tl.float32,bitcast=True)
            product=tl.load(Product+b*BS+ss)
            hits=fixed_sum((live & (owner>=0) & (product>0)).to(tl.int32),0)
            best = fixed_min(tl.where(live & (scores==maximum),ss,S),0)
            evidence = tl.where(best<S,maximum,0.)
            pop = tl.load(Pop+b*T+t)
            popped = (ss==best) & (pop>PopThreshold)
            strength = tl.where(popped,strength*(1.-pop),strength)
            live = live & ~(popped & (strength<=Free))
            has = fixed_sum(live.to(tl.int32),0)>0
            best_order = fixed_max(tl.where(live,order,-1),0)
            youngest = fixed_min(tl.where(live & (order==best_order),ss,S),0)
            adjacent_order = fixed_max(tl.where(live & (birth==t-1),order,-1),0)
            adjacent = adjacent_order>=0
            target = tl.where(adjacent,fixed_min(tl.where(live & (order==adjacent_order),ss,S),0),youngest)
            prev = previous_value(FinalV,FinalMeta,b,S,BS,live & (ss==target) & adjacent,ss,vd)
            diff = v-prev
            residual = fixed_sum(diff*diff,0)
            err = tl.where(has,tl.sqrt(residual),0.)
            occ = fixed_sum(live.to(tl.int32),0).to(tl.float32)/S
            wire = logit+OccSlope*tl.maximum(0.,occ-OccOnset)+mc*(MatchWire*evidence)+ec*err
            if sigmoid(wire)>Enable:
                request = ~has | ~adjacent | (residual<Eps)
                target_array=fixed_max(tl.where(live & (ss==target),owner,-1),0)>=0
                if ~(AppendOnly & target_array & ~request):
                    free = fixed_min(tl.where((ss<S)&~live,ss,S),0)
                    alloc = request & (free<S)
                    dst = tl.where(alloc,free,target)
                    gate = sigmoid(append+coef*err)
                    s_alloc=alloc.to(tl.int32)
                    s_correct=(~request).to(tl.int32)
                    s_cap=(request & (free==S)).to(tl.int32)
                    changed = ss==dst
                    update_value(FinalV,FinalS,V+(b*T+t)*128,b,S,BS,ss,changed,alloc,gate)
                    tl.store(FinalK+(b*S+dst)*64+kd,k,alloc & (dst<S) & (kd<64))
                    strength = tl.where(changed & alloc,1.,strength)
                    birth = tl.where(changed & alloc,t,birth)
                    order = tl.where(changed & alloc,2*t+1,order)
                    owner = tl.where(changed,-1,owner)
                    prediction = tl.where(changed,-1,prediction)
                    pred_at = tl.where(changed,-1,pred_at)
                    expires = tl.where(changed,-1,expires)
                    live = live | changed
            post_array=fixed_sum((live & (owner>=0)).to(tl.int32),0)
            post_stream=fixed_sum((live & (owner<0)).to(tl.int32),0)
            age_mask=live
            if AppendOnly: age_mask=live & (owner<0)
            decayed=fixed_sum(age_mask.to(tl.int32),0)
            strength = tl.where(age_mask,strength*Decay,strength)
            deaths=fixed_sum((live & (strength<=Free)).to(tl.int32),0)
            live = live & (strength>Free)
            for field in tl.static_range(6,18):
                stat=0
                if field==6: stat=hits
                if field==7: stat=fixed_sum(popped.to(tl.int32),0)
                if field==8: stat=s_alloc
                if field==9: stat=s_correct
                if field==10: stat=s_cap
                if field==11: stat=decayed
                if field==12: stat=deaths
                if field==13: stat=tl.maximum(pre_array,post_array)
                if field==14: stat=fixed_sum((live & (owner>=0)).to(tl.int32),0)
                if field==15: stat=fixed_sum((live & (owner<0)).to(tl.int32),0)
                if field==16: stat=tl.maximum(pre_array+pre_stream,post_array+post_stream)
                if field==17: stat=tl.maximum(pre_stream,post_stream)
                tl.store(Trace+(b*T+t)*18+field,stat)
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


@gluon.jit
def initialize(FinalK,FinalV,FinalS,FinalMeta,S:tl.constexpr):
    b=tl.program_id(0);page=tl.program_id(1)
    rs=page*128+tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    kd=tl.arange(0,64,layout=tl.BlockedLayout([1],[32],[8],[0]))
    vd=tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    tl.store(FinalK+(b*S+as_rows(rs))*64+as_cols(kd),0.,as_rows(rs)<S)
    tl.store(FinalV+(b*S+as_rows(rs))*128+as_cols(vd),0.,as_rows(rs)<S)
    tl.store(FinalS+b*S+rs,0.,rs<S)
    for f in tl.static_range(7):
        tl.store(FinalMeta+(b*7+f)*S+rs,0 if f==0 else -1,rs<S)


@gluon.jit
def score_pages(Q,FinalK,Score,Valid,t,T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr):
    b=tl.program_id(0);page=tl.program_id(1)
    if t < tl.load(Valid+b):
        rs=page*128+tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
        kd=tl.arange(0,64,layout=tl.BlockedLayout([1],[32],[8],[0]))
        q=tl.load(Q+(b*T+t)*64+kd)
        keys=tl.load(FinalK+(b*S+as_rows(rs))*64+as_cols(kd),as_rows(rs)<S,0.)
        value=fixed_sum(keys*as_cols(q),1)*(2./tl.sqrt(64.))
        tl.store(Score+b*BS+rs,value,rs<BS)


@gluon.jit
def normalize(Score,FinalS,FinalMeta,Product,Scalars,Valid,t,S:tl.constexpr,BS:tl.constexpr):
    b=tl.program_id(0)
    if t < tl.load(Valid+b):
        ss=tl.arange(0,BS,layout=tl.BlockedLayout([1],[32],[8],[0]))
        live=tl.load(FinalMeta+b*7*S+ss,ss<S,0)!=0
        strength=tl.load(FinalS+b*S+ss,ss<S,0.)
        scores=tl.load(Score+b*BS+ss)
        masked=tl.where(live,scores,-1.e30)
        maximum=fixed_max(masked,0)
        numer=tl.where(live,tl.exp(masked-maximum),0.)
        weights=numer/tl.maximum(fixed_sum(numer,0),1.e-30)
        tl.store(Product+b*BS+ss,weights*strength)
        tl.store(Scalars+b*8+2,maximum.to(tl.int32,bitcast=True))


@gluon.jit
def read_page(FinalV,Product,Carry,Out,Valid,t,page,T:tl.constexpr,S:tl.constexpr,BS:tl.constexpr,PAGE:tl.constexpr):
    b=tl.program_id(0)
    vd=tl.arange(0,128,layout=tl.BlockedLayout([1],[32],[8],[0]))
    rs=tl.arange(0,2,layout=tl.BlockedLayout([1],[32],[8],[0]))
    if t < tl.load(Valid+b):
        if page==0:
            w=tl.load(Product+b*BS+rs)
            value=tl.load(FinalV+(b*S+as_rows(rs))*128+as_cols(vd),as_rows(rs)<S,0.)
            acc=as_rows(w)*value
            start=1
        else:
            acc=tl.load(Carry+(b*2+as_rows(rs))*128+as_cols(vd))
            start=0
        for r in range(start,PAGE//2):
            rows=page*PAGE+2*r+rs
            if page*PAGE+2*r < BS:
                w=tl.load(Product+b*BS+rows,rows<BS,0.)
                value=tl.load(FinalV+(b*S+as_rows(rows))*128+as_cols(vd),as_rows(rows)<S,0.)
                acc=acc+as_rows(w)*value
        tl.store(Carry+(b*2+as_rows(rs))*128+as_cols(vd),acc)
        if (page+1)*PAGE>=BS:
            tl.store(Out+(b*T+t)*128+vd,fixed_sum(acc,0))
    else:
        tl.store(Out+(b*T+t)*128+vd,0.)


class SplitKernel:
    """Diagnostic only. Dedicated scratch per call; no production installer."""
    compiled=None
    def __getitem__(self,grid):
        def launch(*a,**kw):
            import torch
            b=grid[0];s=kw['S'];bs=kw['BS'];tmax=kw['T']
            score=torch.empty((b,bs),device=a[0].device)
            product=torch.empty_like(score)
            scalars=torch.empty((b,8),device=a[0].device,dtype=torch.int32)
            carry=torch.empty((b,2,128),device=a[0].device)
            opts=dict(num_warps=8,enable_fp_fusion=False)
            initialize[(b,triton.cdiv(s,128))](*a[17:21],S=s,**opts)
            for t in range(tmax):
                self.compiled=control[grid](*a,score,product,scalars,t,Phase=0,**kw)
                score_pages[(b,triton.cdiv(bs,128))](a[1],a[17],score,a[15],t,T=tmax,S=s,BS=bs,**opts)
                normalize[grid](score,a[19],a[20],product,scalars,a[15],t,S=s,BS=bs,**opts)
                for page in range(triton.cdiv(bs,256)):
                    read_page[grid](a[18],product,carry,a[16],a[15],t,page,T=tmax,S=s,BS=bs,PAGE=256,**opts)
                self.compiled=control[grid](*a,score,product,scalars,t,Phase=1,**kw)
            return self.compiled
        return launch

kernel=SplitKernel()
