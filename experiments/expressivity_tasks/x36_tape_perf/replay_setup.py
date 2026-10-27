"""Copy owned replay fields and initialize native adjoints in one activity."""
import torch
import triton
import triton.language as tl


@triton.jit
def setup_kernel(Sources, Destinations, Gradients, GradDest,
                 Starts:tl.constexpr, Numel:tl.constexpr, GStarts:tl.constexpr, GNumel:tl.constexpr, Strides:tl.constexpr,
                 Missing:tl.constexpr, H:tl.constexpr, S:tl.constexpr,
                 Dims:tl.constexpr, SoA:tl.constexpr):
    pid=tl.program_id(0)
    for i in tl.static_range(len(Sources)):
        if (pid>=Starts[i]) & (pid<Starts[i+1]):
            block=pid-Starts[i]
            if i<2 and SoA:
                row=(block%tl.cdiv(S,32))*32+tl.arange(0,32)
                col=((block//tl.cdiv(S,32))%tl.cdiv(Dims[i],32))*32+tl.arange(0,32)
                store=block//(tl.cdiv(S,32)*tl.cdiv(Dims[i],32))
                valid=(row[:,None]<S) & (col[None,:]<Dims[i])
                value=tl.load(Sources[i]+(store*S+row[:,None])*Dims[i]+col[None,:],valid,0)
                tl.store(Destinations[i]+(store*Dims[i]+col[None,:])*S+row[:,None],value,valid)
            else:
                x=block*256+tl.arange(0,256)
                value=tl.load(Sources[i]+x,x<Numel[i],0)
                tl.store(Destinations[i]+x,value,x<Numel[i])
    for i in tl.static_range(3):
        if (pid>=GStarts[i]) & (pid<GStarts[i+1]):
            block=pid-GStarts[i]
            if i<2 and SoA:
                row=(block%tl.cdiv(S,32))*32+tl.arange(0,32)
                col=((block//tl.cdiv(S,32))%tl.cdiv(Dims[i],32))*32+tl.arange(0,32)
                store=block//(tl.cdiv(S,32)*tl.cdiv(Dims[i],32))
                valid=(row[:,None]<S) & (col[None,:]<Dims[i])
                if Missing[i]:
                    value=tl.full((32,32),0.,tl.float32)
                else:
                    address=(store//H)*Strides[i][0]+(store%H)*Strides[i][1]+row[:,None]*Strides[i][2]+col[None,:]*Strides[i][3]
                    value=tl.load(Gradients[i]+address,valid,0.)
                tl.store(GradDest[i]+(store*Dims[i]+col[None,:])*S+row[:,None],value,valid)
            else:
                x=block*256+tl.arange(0,256)
                valid=x<GNumel[i]
                if Missing[i]:
                    value=tl.full((256,),0.,tl.float32)
                elif i<2:
                    store=x//(S*Dims[i])
                    address=(store//H)*Strides[i][0]+(store%H)*Strides[i][1]+((x//Dims[i])%S)*Strides[i][2]+(x%Dims[i])*Strides[i][3]
                    value=tl.load(Gradients[i]+address,valid,0.)
                else:
                    store=x//S
                    address=(store//H)*Strides[i][0]+(store%H)*Strides[i][1]+(x%S)*Strides[i][2]
                    value=tl.load(Gradients[i]+address,valid,0.)
                tl.store(GradDest[i]+x,value,valid)


def prepare(owners,gradients,*,soa,key_grad=True):
    """Return independent contiguous owners and adjoints in replay layout.

    Native-output adjoints may be missing, expanded, or strided. Payload
    transpose tiles preserve bits; no arithmetic or reduction is changed.
    With key_grad=False the unused key-adjoint pointer aliases owned keys;
    its setup interval is empty and the caller disables every adjoint write.
    """
    if any(not x.is_contiguous() for x in owners):
        raise ValueError('replay requires contiguous owned fields')
    b,h,s,dk=owners[0].shape
    dv=owners[1].shape[-1]
    replay=tuple(torch.empty_like(x.transpose(-1,-2) if soa and i<2 else x,
                                 memory_format=torch.contiguous_format)
                 for i,x in enumerate(owners))
    adjoints=tuple(torch.empty_like(x) if i or key_grad else x for i,x in enumerate(replay[:3]))
    missing=tuple(g is None for g in gradients)
    sources=tuple(owners[i] if g is None else g for i,g in enumerate(gradients))
    numel=tuple(x.numel() for x in (*owners,*owners[:3]))
    starts=[0]
    for i,n in enumerate(numel):
        payload=i in (0,1,len(owners),len(owners)+1)
        dim=dk if i in (0,len(owners)) else dv
        blocks=(b*h*triton.cdiv(s,32)*triton.cdiv(dim,32) if soa and payload
                else triton.cdiv(n,256))
        if i==len(owners) and not key_grad:blocks=0
        starts.append(starts[-1]+blocks)
    setup_kernel[(starts[-1],)](owners,replay,sources,adjoints,tuple(starts[:len(owners)+1]),numel[:len(owners)],
                               tuple(starts[len(owners):]),numel[len(owners):],
                               tuple(x.stride() for x in sources),missing,h,s,(dk,dv),soa,
                               num_warps=4,enable_fp_fusion=False)
    return replay,adjoints
