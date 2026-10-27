"""Separate store counts, native payload, and metadata; never promote views."""
import json
import os
from pathlib import Path
import torch
from .backing import FIELDS
from .cell_fused import TapeHeadConfig, tape_scan

NAMES = ('heap', 'position', 'due', 'terminal', 'birth', 'touch',
         'bitmap_l0', 'bitmap_l1', 'bitmap_root', 'birth_next', 'birth_prev',
         'birth_head', 'heap_size', 'references', 'legacy_scores')


def inventory(state):
    native = state.native.compact
    stores = native.live.shape[0] * native.live.shape[1]
    def describe(x):
        size = x.numel() * x.element_size()
        return dict(shape=list(x.shape), bytes=size, bytes_per_store=size/stores)
    return dict(stores=stores, physical_slots=native.live.shape[2],
                live_rows=int(native.live.sum()),
                native={name:describe(getattr(native,name)) for name in FIELDS},
                metadata={name:describe(x) for name,x in zip(NAMES,state.metadata)},
                history=describe(state.history))


@torch.no_grad()
def main():
    assert os.environ.get('GPU_LEASE_HELD') in list('0123456')
    rows=[]
    for heads in (1,16):
        cfg=TapeHeadConfig(key_dim=64,value_dim=128,n_heads=heads,budget=11776)
        k=torch.zeros(64,16,heads,64,device='cuda')
        v=torch.zeros(64,16,heads,128,device='cuda')
        off=torch.zeros(64,16,heads,device='cuda')
        state=None
        for chunk in range(1,5):
            _,state,_,_=tape_scan(k,k,v,off,off,off,state,cfg)
            rows.append(dict(heads=heads,chunk=chunk,**inventory(state)))
    result=dict(physical_gpu=os.environ['GPU_LEASE_HELD'],measurements=rows,
                interpretation='Metadata follows the touched physical prefix, not architectural S. '
                'It is not live proportional. The historical 14 MB is aggregate base state '
                'across 1024 stores, including payload; it is not 14 MB of metadata per store.')
    Path(__file__).with_name('metadata_part6.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
