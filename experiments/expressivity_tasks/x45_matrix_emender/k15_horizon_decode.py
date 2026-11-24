"""Chosen-policy next-byte service. The native member persists; neural prefixes replay."""
import time
import numpy as np
import torch
from .k15_horizon import Resident


def interpolate(base,evidence,weight):
    n=evidence.total()
    if not n: return base
    a=n/(1+n) if weight=='count' else weight
    result=(1-a)*base
    for k,v in evidence.items(): result[k]+=a*v/n
    return result


@torch.inference_mode()
def transactions(net,exe,mode,w,observed,h):
    assert len(observed)>=h>=128 and h%32==0
    tic=time.perf_counter(); state=None if mode=='base' else Resident(exe,mode); distributions=[]; times=[]
    try:
        if state:
            for s in range(0,h,32): state.append(observed[s:s+32])
        prefill=time.perf_counter()-tic
        with torch.autocast('cuda',dtype=torch.bfloat16):
            for s in range(h,len(observed)+1):
                start=time.perf_counter()
                if state and s>h: state.append(observed[s-1:s])
                ev=state.read() if state else {}; ids=torch.tensor(list(observed[s-128:s]),device='cuda')[None]; base=net(ids,doors=False)[0][0,-1].float().softmax(-1).cpu().numpy()
                distributions.append(interpolate(base,ev,w) if ev else base); times.append(time.perf_counter()-start)
    finally:
        if state: state.session.close()
    return np.stack(distributions),dict(prefill=prefill,wall=time.perf_counter()-tic,seconds=times)
