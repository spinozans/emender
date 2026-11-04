"""Capture the batched PyTorch replay launch schedule, never a new store kernel.

All ownership, routing, values and upstream gradients are dynamic graph inputs.
The captured graph includes ordinary PyTorch autograd, with immutable slot
versions. Each call copies its gradients out before buffers can be reused.
The scalar eager oracle remains independent and the native/replay check remains
mandatory on every backward. CUDA graphs only remove Python launch overhead.
"""
from collections import OrderedDict
import torch
from .c14_b_control_replay import replay,encode

CACHE=OrderedDict()

class ReplayGraph:
    def __init__(self,inputs,gradient,cfg,policy,metadata):
        self.inputs=[x.detach().clone().requires_grad_(i in (1,2,3,4,5,6,7)) for i,x in enumerate(inputs)]
        self.encoded=encode(metadata,inputs[0].device)
        self.gradient=gradient.contiguous().clone()
        active=[x for x in self.inputs if x.requires_grad]
        def run():
            with torch.enable_grad():
                output,dependencies=replay(*self.inputs,cfg,policy,None,encoded=self.encoded)
                grads=torch.autograd.grad(output,active,self.gradient,allow_unused=True)
            return output,dependencies,grads
        # Allocate and warm ordinary PyTorch kernels on a side stream before
        # capture. No optimizer, RNG, telemetry or trained state is touched.
        stream=torch.cuda.Stream(device=inputs[0].device)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):run()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.output,self.dependencies,self.grads=run()

    def run(self,inputs,gradient,metadata):
        with torch.no_grad():
            for static,value in zip(self.inputs,inputs):static.copy_(value)
            for static,value in zip(self.encoded,encode(metadata,inputs[0].device)):static.copy_(value)
            self.gradient.copy_(gradient)
        self.graph.replay()
        dependencies=int(self.dependencies)
        result=[];it=iter(self.grads)
        for i,x in enumerate(self.inputs):
            g=next(it) if x.requires_grad else None
            result.append(g.clone() if g is not None and dependencies & (1<<i) else None)
        return self.output,result


def backward(inputs,gradient,cfg,policy,metadata):
    key=(repr(cfg),repr(policy),tuple((tuple(x.shape),x.dtype,x.device) for x in inputs))
    if key not in CACHE:
        if len(CACHE)>=4:CACHE.popitem(last=False)
        CACHE[key]=ReplayGraph(inputs,gradient,cfg,policy,metadata)
    CACHE.move_to_end(key)
    return CACHE[key].run(inputs,gradient,metadata)
