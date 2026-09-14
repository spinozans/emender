"""Read-only capture of the first shared recurrence call, after adapter decisions."""
import importlib
import inspect
from scripts.e97_tensor_capsule import pack,unpack
from scripts.e97_first_divergence import digest

TENSORS=('S0','k','v','q','decay','g','erase_gate','value_write_gate','reset_before','valid_mask')


def describe(capsule):
    values=unpack(capsule,'cpu')
    return dict(values={k:digest(v) for k,v in values.items()},
        layout={k:None if d is None else {n:v for n,v in d.items() if n!='original_address'} for k,d in capsule['tensors'].items()})


class Capture:
    def __init__(self):
        self.owner=importlib.import_module('ndm.triton.e88_triton_optimized')
        self.original=self.owner.e88_triton;self.signature=inspect.signature(self.original)
        self.calls=0;self.inputs=None;self.outputs=None;self.meta=None;self.unchanged=None
        def observe(*args,**kwargs):
            self.calls+=1
            if self.calls!=1:return self.original(*args,**kwargs)
            bound=self.signature.bind(*args,**kwargs);bound.apply_defaults();arguments=bound.arguments
            tensors={k:v for k,v in arguments.items() if k in TENSORS}
            self.meta={k:v for k,v in arguments.items() if k not in TENSORS}
            self.inputs=pack(tensors)
            result=self.original(*args,**kwargs)
            if not isinstance(result,tuple) or len(result)!=2:raise ValueError('recurrence result contract')
            self.outputs=pack(dict(out=result[0],S_final=result[1]))
            self.unchanged=describe(self.inputs)==describe(pack(tensors))
            return result  # same original output/state objects
        self.observer=observe;self.owner.e88_triton=observe

    def close(self):
        if self.owner.e88_triton is not self.observer:raise ValueError('capture ownership changed')
        self.owner.e88_triton=self.original

    def finish(self):
        if self.calls<1 or self.inputs is None or self.outputs is None or not self.unchanged:raise ValueError('missing/mutated recurrence capture')
        return dict(meta=self.meta,inputs=describe(self.inputs),outputs=describe(self.outputs),total_calls=self.calls,inputs_unchanged=True)
