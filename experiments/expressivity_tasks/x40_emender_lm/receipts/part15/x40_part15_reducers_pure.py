import triton
@triton.jit
def combine_live(a,b):
    return a+b
@triton.jit
def combine_requests(a,b):
    return a+b
@triton.jit
def combine_caps(a,b):
    return a+b
@triton.jit
def combine_remaining(a,b):
    return a+b
@triton.jit
def combine_active(a,b):
    return a+b
