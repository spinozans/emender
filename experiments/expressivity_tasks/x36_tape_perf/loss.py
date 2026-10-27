"""Memory-bounded vocabulary projection for the matched long-context gate."""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable


class ChunkedLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, chunk):
        ctx.save_for_backward(hidden,weight,targets)
        ctx.chunk=chunk
        flat=hidden.reshape(-1,hidden.shape[-1])
        labels=targets.reshape(-1)
        total=hidden.new_zeros(())
        for begin in range(0,len(flat),chunk):
            total+=F.cross_entropy(F.linear(flat[begin:begin+chunk],weight),
                                   labels[begin:begin+chunk],reduction='sum')
        return total/len(flat)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        hidden,weight,targets=ctx.saved_tensors
        flat=hidden.reshape(-1,hidden.shape[-1])
        labels=targets.reshape(-1)
        dh=torch.empty_like(flat);dw=torch.zeros_like(weight)
        scale=upstream/len(flat)
        for begin in range(0,len(flat),ctx.chunk):
            with torch.enable_grad():
                h=flat[begin:begin+ctx.chunk].detach().requires_grad_()
                w=weight.detach().requires_grad_()
                loss=F.cross_entropy(F.linear(h,w),labels[begin:begin+ctx.chunk],reduction='sum')
                gh,gw=torch.autograd.grad(loss,(h,w),grad_outputs=scale)
            dh[begin:begin+ctx.chunk].copy_(gh);dw.add_(gw)
        return dh.reshape_as(hidden),dw,None,None


def bounded_loss(model,ids,targets,chunk=1024):
    hidden=model.embed(ids)
    for block in model.blocks:
        hidden,_=block(hidden,ids)
    return ChunkedLoss.apply(model.norm(hidden),model.head.weight,targets,chunk)
