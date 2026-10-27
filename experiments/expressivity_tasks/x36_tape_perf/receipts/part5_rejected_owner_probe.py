"""Rejected diagnostic: passes original code, so does not establish OOM cause.

Run the broker wrapper's validate command with this repository-relative input:
experiments/expressivity_tasks/x36_tape_perf/receipts/part5_rejected_owner_probe.py
and the filter -k discarded_chunk_releases.
This probe was removed from the ordinary suite and is archived as evidence.
"""
import gc
import weakref
import torch
from experiments.expressivity_tasks.x36_tape_perf.cell_fused import tape_scan, TapeHeadConfig


def test_discarded_chunk_releases_replay_storage():
    cfg=TapeHeadConfig(n_heads=2,budget=37,key_dim=4,value_dim=5)
    inputs=[torch.randn(shape,device='cuda',requires_grad=True) for shape in
            [(2,9,2,4)]*2+[(2,9,2,5)]+[(2,9,2)]*3]
    out,final,trace,pressure=tape_scan(*inputs,None,cfg)
    storage=weakref.ref(out.grad_fn.storage)
    out.sum().backward()
    del out,final,trace,pressure
    gc.collect()
    assert storage() is None, 'discarded output edges retained replay payload storage'
