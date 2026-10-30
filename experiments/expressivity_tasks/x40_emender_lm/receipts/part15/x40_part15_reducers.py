import triton
import triton.language as tl
@triton.jit
def combine_live(a,b):
    wide=a.to(tl.int64)+b.to(tl.int64)
    tl.device_assert((wide>=-2147483648)&(wide<=2147483647),'PART15_REDUCE_LIVE')
    return a+b
@triton.jit
def combine_requests(a,b):
    wide=a.to(tl.int64)+b.to(tl.int64)
    tl.device_assert((wide>=-2147483648)&(wide<=2147483647),'PART15_REDUCE_REQUESTS')
    return a+b
@triton.jit
def combine_caps(a,b):
    wide=a.to(tl.int64)+b.to(tl.int64)
    tl.device_assert((wide>=-2147483648)&(wide<=2147483647),'PART15_REDUCE_CAPS')
    return a+b
@triton.jit
def combine_remaining(a,b):
    wide=a.to(tl.int64)+b.to(tl.int64)
    tl.device_assert((wide>=-2147483648)&(wide<=2147483647),'PART15_REDUCE_REMAINING')
    return a+b
@triton.jit
def combine_active(a,b):
    wide=a.to(tl.int64)+b.to(tl.int64)
    tl.device_assert((wide>=-2147483648)&(wide<=2147483647),'PART15_REDUCE_ACTIVE')
    return a+b
