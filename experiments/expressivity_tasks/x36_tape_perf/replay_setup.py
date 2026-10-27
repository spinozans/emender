"""Copy hot replay payloads and initialize packed adjoints in one activity."""
import torch
import triton
import triton.language as tl


@triton.jit
def unpack_kernel(Packed, Logical, Dense, S:tl.constexpr, PS:tl.constexpr,
                  D:tl.constexpr, SoA:tl.constexpr, Block:tl.constexpr):
    store=tl.program_id(0)
    x=tl.program_id(1)*Block+tl.arange(0,Block)
    row=x//D
    col=x%D
    logical=tl.load(Logical+store*PS+row,row<PS,-1)
    active=(row<PS) & (logical>=0)
    value=tl.load(Packed+((store*D+col)*PS+row if SoA else (store*PS+row)*D+col),active,0.)
    tl.store(Dense+(store*S+logical)*D+col,value,active)


def unpack(packed, logical, original, owner, *, soa, strength=False):
    # Untouched native rows retain their output adjoints. The historical
    # strength recurrence adds positive zero even to inactive cold rows.
    dense=torch.zeros_like(owner) if original is None else original.contiguous().clone()
    if strength:
        dense=dense+0.
    d=1 if strength else owner.shape[-1]
    s=owner.shape[-1] if strength else owner.shape[-2]
    ps=logical.shape[-1]
    unpack_kernel[(logical.shape[0],triton.cdiv(ps*d,256))](
        packed,logical,dense,s,ps,d,soa and not strength,256)
    return dense


@triton.jit
def setup_kernel(Sources, Destinations, Gradients, GradDest,
                 Starts:tl.constexpr, Numel:tl.constexpr, GStarts:tl.constexpr, GNumel:tl.constexpr, Strides:tl.constexpr,
                 Missing:tl.constexpr, H:tl.constexpr, S:tl.constexpr,
                 Dims:tl.constexpr, SoA:tl.constexpr, GS:tl.constexpr, PoolRows, PS:tl.constexpr):
    pid=tl.program_id(0)
    for i in tl.static_range(len(Sources)):
        if (pid>=Starts[i]) & (pid<Starts[i+1]):
            block=pid-Starts[i]
            if i<2 and SoA:
                rows:tl.constexpr=PS if PS else S
                row=(block%tl.cdiv(rows,32))*32+tl.arange(0,32)
                col=((block//tl.cdiv(rows,32))%tl.cdiv(Dims[i],32))*32+tl.arange(0,32)
                store=block//(tl.cdiv(rows,32)*tl.cdiv(Dims[i],32))
                logical=tl.load(PoolRows+store*PS+row,row<PS,-1) if PS else row
                valid=(row[:,None]<rows) & (logical[:,None]>=0) & (col[None,:]<Dims[i])
                value=tl.load(Sources[i]+(store*S+logical[:,None])*Dims[i]+col[None,:],valid,0)
                tl.store(Destinations[i]+(store*Dims[i]+col[None,:])*rows+row[:,None],value,
                         (row[:,None]<rows) & (col[None,:]<Dims[i]))
            else:
                x=block*256+tl.arange(0,256)
                value=tl.load(Sources[i]+x,x<Numel[i],0)
                tl.store(Destinations[i]+x,value,x<Numel[i])
    for i in tl.static_range(3):
        if (pid>=GStarts[i]) & (pid<GStarts[i+1]):
            block=pid-GStarts[i]
            if i<2 and SoA:
                row=(block%tl.cdiv(PS if PS else S,32))*32+tl.arange(0,32)
                col=((block//tl.cdiv(PS if PS else S,32))%tl.cdiv(Dims[i],32))*32+tl.arange(0,32)
                store=block//(tl.cdiv(PS if PS else S,32)*tl.cdiv(Dims[i],32))
                valid=(row[:,None]<(PS if PS else S)) & (col[None,:]<Dims[i])
                logical=tl.load(PoolRows+store*PS+row,row<PS,-1) if PS else row
                valid=valid & (logical[:,None]>=0)
                if Missing[i]:
                    value=tl.full((32,32),0.,tl.float32)
                else:
                    address=(store//H)*Strides[i][0]+(store%H)*Strides[i][1]+logical[:,None]*Strides[i][2]+col[None,:]*Strides[i][3]
                    value=tl.load(Gradients[i]+address,valid,0.)
                tl.store(GradDest[i]+(store*Dims[i]+col[None,:])*(PS if PS else S)+row[:,None],value,(row[:,None]<(PS if PS else S)) & (col[None,:]<Dims[i]))
            else:
                x=block*256+tl.arange(0,256)
                valid=x<GNumel[i]
                if Missing[i]:
                    value=tl.full((256,),0.,tl.float32)
                elif i<2:
                    rows:tl.constexpr=PS if PS else S
                    store=x//(rows*Dims[i])
                    row=(x//Dims[i])%rows
                    logical=tl.load(PoolRows+store*PS+row,valid,-1) if PS else row
                    valid=valid & (logical>=0)
                    address=(store//H)*Strides[i][0]+(store%H)*Strides[i][1]+logical*Strides[i][2]+(x%Dims[i])*Strides[i][3]
                    value=tl.load(Gradients[i]+address,valid,0.)
                else:
                    store=x//GS
                    row=x%GS
                    logical=tl.load(PoolRows+store*PS+row,valid,-1) if PS else row
                    address=(store//H)*Strides[i][0]+(store%H)*Strides[i][1]+logical*Strides[i][2]
                    valid=valid & (logical>=0)
                    value=tl.load(Gradients[i]+address,valid,0.)
                tl.store(GradDest[i]+x,value,valid)


def prepare(owners,gradients,*,soa,key_grad=True,strength_rows=None,pool_rows=None):
    """Return independent contiguous owners and adjoints in replay layout.

    Native-output adjoints may be missing, expanded, or strided. Payload
    transpose tiles preserve bits; no arithmetic or reduction is changed.
    With key_grad=False the unused key-adjoint pointer aliases owned keys;
    its setup interval is empty and the caller disables every adjoint write.
    pool_rows selects exact-union payload/adjoint backing. Untouched native
    rows remain in immutable owner backing; unpack restores required native
    gradients. Without a pool, strength_rows bounds private strength adjoints.
    """
    if any(not x.is_contiguous() for x in owners):
        raise ValueError('replay requires contiguous owned fields')
    b,h,s,dk=owners[0].shape
    dv=owners[1].shape[-1]
    ps=0 if pool_rows is None else pool_rows.shape[-1]
    if ps and not soa:
        raise ValueError('hot replay payloads require SoA backing')
    replay=tuple(torch.empty((b,h,x.shape[-1],ps),device=x.device,dtype=x.dtype)
                 if ps and i<2 else torch.empty_like(x.transpose(-1,-2) if soa and i<2 else x,
                                 memory_format=torch.contiguous_format)
                 for i,x in enumerate(owners))
    adjoints=tuple(
        torch.empty((b,h,ps) if i==2 else ((b,h,x.shape[-2],ps) if soa else (b,h,ps,x.shape[-1])),device=x.device,dtype=x.dtype)
        if ps and (i or key_grad) else
        torch.empty((b,h,strength_rows),device=x.device,dtype=x.dtype)
        if i==2 and strength_rows is not None and strength_rows<s
        else torch.empty_like(x) if i or key_grad else x
        for i,x in enumerate(replay[:3]))
    missing=tuple(g is None for g in gradients)
    sources=tuple(owners[i] if g is None else g for i,g in enumerate(gradients))
    numel=tuple(x.numel() for x in (*owners,*adjoints))
    starts=[0]
    for i,n in enumerate(numel):
        payload=i in (0,1,len(owners),len(owners)+1)
        dim=dk if i in (0,len(owners)) else dv
        blocks=(b*h*triton.cdiv(ps if ps else s,32)*triton.cdiv(dim,32) if soa and payload
                else triton.cdiv(n,256))
        if i==len(owners) and not key_grad:blocks=0
        starts.append(starts[-1]+blocks)
    setup_kernel[(starts[-1],)](owners,replay,sources,adjoints,tuple(starts[:len(owners)+1]),numel[:len(owners)],
                               tuple(starts[len(owners):]),numel[len(owners):],
                               tuple(x.stride() for x in sources),missing,h,s,(dk,dv),soa,adjoints[2].shape[-1],owners[3] if pool_rows is None else pool_rows,ps,
                               num_warps=4,enable_fp_fusion=False)
    return replay,adjoints
