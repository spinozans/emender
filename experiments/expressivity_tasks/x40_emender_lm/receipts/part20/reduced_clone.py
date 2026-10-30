"""Bounded diagnostic of the failed host-API clone gate, after source rollback.

The same candidate is installed only in this process. No file is modified.
This B1/T8 case cannot stand in for any required certified-shape gate.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch
from experiments.expressivity_tasks.x31_tape_cell_fused import kernels as x31
from experiments.expressivity_tasks.x40_emender_lm import kernels as x40
from experiments.expressivity_tasks.x40_emender_lm.parity import certify
assert sys.executable == '/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3', '4')
torch.set_num_threads(1)
r=Path(__file__).resolve().parent
for module in (x31,x40):
    fn=module.materialize_state
    old='born=tl.load(Birth+n*S+slots,valid,Clock)'
    assert fn.src.count(old)==1
    fn._unsafe_update_src(fn.src.replace(old,'born=tl.load(Birth+n*S+slots,valid & live,Clock)'))
    fn.hash=None
fn=x40.scan_kernel
old='cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live,0.)'
assert fn.src.count(old)==1
fn._unsafe_update_src(fn.src.replace(old,'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live & (r_touch<Start+t-1),0.)'))
fn.hash=None
print(json.dumps(dict(scope='B1/T8 diagnostic only, process-local candidate after rollback',
    sources={str(Path(m.__file__)):hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest() for m in (x31,x40)})),flush=True)
result=certify(batch=1,context=8)
(r/'reduced_clone.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
