"""Exact unit-reference decay tables, never pow or approximate decay."""
from functools import lru_cache
import struct
import numpy as np
import torch
import triton
import triton.language as tl
from ..x31_tape_cell_fused.lazy_decay import advance_exact


@lru_cache(maxsize=128)
def host_table(decay,threshold,length):
    rate=np.float32(decay);free=np.float32(threshold)
    values=np.empty(length+1,dtype=np.float32);values[0]=1.
    for i in range(length):values[i+1]=np.float32(values[i]*rate)
    value=np.float32(1.);horizon=0
    if rate==1.:return values,-1,1.
    while value>free:
        updated=np.float32(value*rate)
        if updated==value:return values,-1,float(value)
        value=updated;horizon+=1
    return values,horizon,float(value)


@lru_cache(maxsize=128)
def table(decay,threshold,length,device):
    rate=struct.unpack('<f',struct.pack('<f',decay))[0]
    values,horizon,terminal=host_table(rate,threshold,length)
    return torch.from_numpy(values.copy()).to(device),horizon,terminal


@triton.jit
def advance_unit(s,steps,decay:tl.constexpr,Table,Limit:tl.constexpr):
    unit=(s==1.) & (steps>=0) & (steps<=Limit)
    cached=tl.load(Table+steps,unit,0.)
    fallback=advance_exact(s,tl.where(unit,0,steps),decay)
    return tl.where(unit,cached,fallback)
