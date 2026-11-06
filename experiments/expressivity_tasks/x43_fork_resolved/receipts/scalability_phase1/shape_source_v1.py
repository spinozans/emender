"""Experimental reach-bounded launch of the unchanged control kernel.

Explicit installation is diagnostic only until full fingerprints pass.
"""
import inspect
import triton
import torch
from . import c14_b_control_native as native


def install():
    original = native.native_forward
    kernel = native.kernel
    class Bounded:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                # Two allocations per token from the kernel's empty state.
                kwargs['BS'] = triton.next_power_of_2(min(kwargs['S'], max(1, 2*kwargs['T'])))
                return kernel[grid](*args, **kwargs)
            return launch
    namespace = dict(vars(native), kernel=Bounded())
    exec(compile(inspect.getsource(original), '<reach-bounded native wrapper>', 'exec'), namespace)
    bounded = namespace['native_forward']
    def forward(inputs, cfg, policy, metadata):
        result = bounded(inputs, cfg, policy, metadata)
        tile = triton.next_power_of_2(min(cfg.slots, max(1, 2*inputs[0].shape[1])))
        if tile >= triton.next_power_of_2(cfg.slots):
            return result
        out, (state, trace) = result
        keys, values, strength, meta = state
        keys[:,tile:].zero_()
        values[:,tile:].zero_()
        strength[:,tile:].zero_()
        meta[:,:,tile:].fill_(-1)
        meta[:,0,tile:].zero_()
        # Reference padded lane S contributes to POP on empty reads only.
        if cfg.slots < triton.next_power_of_2(cfg.slots):
            previous = torch.zeros_like(trace[:,:,14])
            previous[:,1:] = trace[:,:-1,14]+trace[:,:-1,15]
            pre_read = previous-trace[:,:,1]+trace[:,:,2]
            valid = torch.arange(trace.shape[1],device=trace.device)[None,:] < torch.as_tensor(metadata[3],device=trace.device)[:,None]
            trace[:,:,7].add_(((pre_read == 0) & (inputs[4] > cfg.pop_threshold) & valid).to(trace.dtype))
        return out, (state, trace)
    native.native_forward = forward
    return original


def main():
    install()
    from .c14_launch_validation import main as validate
    validate()

if __name__ == '__main__':
    main()
