"""Batch the registered conditional-control replay orchestration; the production scans are untouched.

The scalar SharedStore remains the independent oracle. Routing decisions stay
outside autograd, as in that oracle. Slot tensors retain immutable versions for
PyTorch autograd, with one operation across all rows/slots instead of a Python
operation and device synchronization per slot. No reverse kernel is introduced.
"""
import torch


def encode(metadata, device):
    cache=getattr(metadata,"encodings",{})
    key=("replay",device)
    if key in cache:return cache[key]
    native=cache.get(("native",device))
    if native is not None:
        result=tuple(native[i].long() for i in (0,2,3,1,4,5))
        cache[key]=result
        return result
    owners, raw, windows, valid, keep, predictions = metadata
    ids=[]; ends=[]
    for j, row in enumerate(owners):
        canonical={}; ii=[]; ee=[]
        for i,o in enumerate(row):
            ii.append(canonical.setdefault(o,len(canonical)) if o is not None and o.suffix else -1)
            ee.append(next((n for n in range(i+1,valid[j]) if not o.is_current(windows[j],raw[j][:n+1])),valid[j]) if o is not None else valid[j])
        ids.append(ii);ends.append(ee)
    result=tuple(torch.tensor(x,device=device,dtype=torch.int64) for x in
                 (ids,ends,keep,predictions,[list(r) for r in raw],valid))
    cache[key]=result
    return result


def replay(k,q,v,av,pop,append,logits,coef,mc,ec,cfg,policy,metadata,*,encoded=None):
    b,t,_=k.shape;s=cfg.slots;device=k.device
    own_at,expiry_at,keep,predictions,raw,valid=encode(metadata,device) if encoded is None else encoded
    slots=torch.arange(s,device=device)[None,:].expand(b,s)
    batch=torch.arange(b,device=device)
    keys=k.new_zeros(b,s,64);values=v.new_zeros(b,s,128);strength=v.new_zeros(b,s)
    live=torch.zeros(b,s,device=device,dtype=torch.bool)
    birth=slots.new_full((b,s),-1);order=birth.clone();owner=birth.clone()
    prediction=birth.clone();pred_at=birth.clone();expires=birth.clone()
    outputs=[]
    value_deps=slots.new_zeros((b,s));strength_deps=value_deps.clone()
    used=slots.new_zeros(())
    state=(keys,values,strength,live,birth,order,owner,prediction,pred_at,expires,value_deps,strength_deps,used)
    clocks=torch.arange(t,device=device)
    step=compiled_transition if k.is_cuda else transition
    # The gate battery intentionally exercises multiple budgets and batch
    # shapes; scope the compiler cache allowance to this B-only adapter.
    with torch._dynamo.config.patch(recompile_limit=64):
        for i in range(t):
            token_inputs=tuple(x[:,i] for x in (k,q,v,av,pop,append,logits))+(coef,mc,ec)
            token_metadata=tuple(x[:,i] for x in (own_at,expiry_at,keep,predictions,raw))+(valid,)
            state,read=step(state,token_inputs,token_metadata,slots,batch,clocks[i],cfg,policy)
            outputs.append(read)
    return torch.stack(outputs,1),state[-1]


def transition(state,inputs,metadata,slots,batch,i,cfg,policy):
    k,q,v,av,pop,append,logits,coef,mc,ec=inputs
    own_at,expiry_at,keep,predictions,raw,valid=metadata
    b,s=slots.shape
    keys,values,strength,live,birth,order,owner,prediction,pred_at,expires,value_deps,strength_deps,used=state
    def target():
        adjacent=live & (birth==i-1)
        ao=torch.where(adjacent,order,-1).max(-1).values
        newest=torch.where(live,order,-1).max(-1).values
        chosen=torch.where(ao>=0,ao,newest)
        index=torch.where(live & (order==chosen[:,None]),slots,s).min(-1).values.clamp_max(s-1)
        return index,ao>=0,live.any(-1)
    active=i<valid
    with torch.no_grad():
        correct=live & (owner>=0) & (pred_at==i-1) & (prediction!=raw[:,None]) & active[:,None]
        if cfg.array_append_only:correct=torch.zeros_like(correct)
    content=v[:,None,:]
    values=torch.where(correct[...,None],values+(content-values),values)
    with torch.no_grad():
        value_deps=torch.where(correct,value_deps | (1<<2),value_deps)
        if not cfg.array_append_only:
            live=live & ~(active[:,None] & (owner>=0) & (expires<=i))
        dst,adjacent,has=target()
        previous=torch.where(adjacent[:,None],values.detach()[batch,dst],0.)
        error=torch.where(has,(av.detach()-previous).square().sum(-1).sqrt(),0.)
        duplicate=torch.where(live & (owner==own_at[:,None]),slots,s).min(-1).values
        free=torch.where(~live,slots,s).min(-1).values
        insert=active & (own_at>=0) & keep.bool()
        allocation=insert & (duplicate==s) & (free<s)
        dst=torch.where(duplicate<s,duplicate,torch.where(free<s,free,dst))
        changed=insert[:,None] & (slots==dst[:,None])
    gate=(append+coef*error).sigmoid()
    new=torch.where(allocation[:,None,None],gate[:,None,None]*av[:,None,:],values+(av[:,None,:]-values))
    values=torch.where(changed[...,None],new,values)
    keys=torch.where((changed & allocation[:,None])[...,None],k[:,None,:],keys)
    strength=torch.where(changed & allocation[:,None],1.,strength)
    with torch.no_grad():
        value_deps=torch.where(changed,torch.where(allocation[:,None],(1<<3)|(1<<5)|(1<<7),value_deps | (1<<3)),value_deps)
        strength_deps=torch.where(changed & allocation[:,None],0,strength_deps)
        birth=torch.where(changed & allocation[:,None],i,birth)
        order=torch.where(changed & allocation[:,None],2*i,order)
        owner=torch.where(changed,own_at[:,None],owner)
        prediction=torch.where(changed,predictions[:,None],prediction)
        pred_at=torch.where(changed,i,pred_at)
        expires=torch.where(changed,expiry_at[:,None],expires)
        live=live | changed
    scores=(keys*q[:,None,:]).sum(-1)*.25
    masked=torch.where(live,scores,-1.e30)
    weights=masked.softmax(-1)*live
    read=(weights[...,None]*strength[...,None]*values).sum(1)
    read=torch.where(active[:,None],read,0.)
    with torch.no_grad():
        read_deps=torch.where(live & active[:,None],value_deps | strength_deps | (1<<1),0)
        # Bits represent structural graph dependencies, including the
        # cancelling old value in old+(new-old). Preserve None versus zero
        # gradients so AdamW does not decay an originally unused parameter.
        for bit in (1,2,3,4,5,7):
            used=used | torch.where(((read_deps & (1<<bit))!=0).any(),1<<bit,0)
        best=masked.detach().argmax(-1)
        evidence=torch.where(live.any(-1),scores.detach()[batch,best],0.)
        popped=live & (slots==best[:,None]) & (pop[:,None].detach()>cfg.pop_threshold) & active[:,None]
    strength=torch.where(popped,strength*(1-pop[:,None]),strength)
    with torch.no_grad():
        strength_deps=torch.where(popped,strength_deps | (1<<4),strength_deps)
        live=live & ~(popped & (strength.detach()<=cfg.free_threshold))
        dst,adjacent,has=target()
        previous=torch.where(adjacent[:,None],values.detach()[batch,dst],0.)
        residual=(v.detach()-previous).square().sum(-1)
        error=torch.where(has,residual.sqrt(),0.)
        occ=live.sum(-1)/s
        wire=logits.detach()+policy.occ_slope*(occ-policy.occ_onset).clamp_min(0.)+mc.detach()*(policy.match_wire*evidence)+ec.detach()*error
        enabled=active & (wire.sigmoid()>cfg.write_enable_threshold)
        if cfg.array_append_only:
            enabled=enabled & ~(has & adjacent & (residual>=cfg.eps_residual) & (owner[batch,dst]>=0))
        request=~has | ~adjacent | (residual<cfg.eps_residual)
        free=torch.where(~live,slots,s).min(-1).values
        allocation=enabled & request & (free<s)
        dst=torch.where(allocation,free,dst)
        changed=enabled[:,None] & (slots==dst[:,None])
    gate=(append+coef*error).sigmoid()
    new=torch.where(allocation[:,None,None],gate[:,None,None]*content,values+(content-values))
    values=torch.where(changed[...,None],new,values)
    keys=torch.where((changed & allocation[:,None])[...,None],k[:,None,:],keys)
    strength=torch.where(changed & allocation[:,None],1.,strength)
    with torch.no_grad():
        value_deps=torch.where(changed,torch.where(allocation[:,None],(1<<2)|(1<<5)|(1<<7),value_deps | (1<<2)),value_deps)
        strength_deps=torch.where(changed & allocation[:,None],0,strength_deps)
        birth=torch.where(changed & allocation[:,None],i,birth)
        order=torch.where(changed & allocation[:,None],2*i+1,order)
        owner=torch.where(changed,-1,owner)
        prediction=torch.where(changed,-1,prediction)
        pred_at=torch.where(changed,-1,pred_at)
        expires=torch.where(changed,-1,expires)
        live=live | changed
    age=live & active[:,None]
    if cfg.array_append_only:age=age & (owner<0)
    strength=torch.where(age,strength*cfg.decay_rate,strength)
    with torch.no_grad():live=live & ~(age & (strength.detach()<=cfg.free_threshold))
    return (keys,values,strength,live,birth,order,owner,prediction,pred_at,expires,value_deps,strength_deps,used),read


# Inductor compiles this Python orchestration into fused ordinary tensor ops.
# No production store source or reverse kernel is changed. A single clock
# tensor avoids specializing a new graph for every position.
compiled_transition=torch.compile(transition,fullgraph=True,dynamic=False,
    options={'triton.cudagraphs':False,'compile_threads':1})
