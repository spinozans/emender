"""Temporary source-only ordering experiment. Never installs a production fix."""
import hashlib
import json
import runpy
import sys
from experiments.expressivity_tasks.x40_emender_lm import kernels

source = kernels.scan_kernel.src
before = hashlib.sha256(source.encode()).hexdigest()
# Bound the experiment to the two read-reduction regions and history reload.
# Keep every arithmetic expression and its reduction tree unchanged.
markers = [
    '                r_scores=tl.sum(r_keys*r_query[None,:],1)*(2.0/tl.sqrt(float(DK)))',
    '                r_maximum=tl.max(r_masked,0)',
    '                    r_partial=tl.sum(r_weighted[:,None]*r_values,0)',
    '                r_best=tl.min(tl.where(r_live & (r_masked==r_maximum),r_columns,S),0)',
    '                tl.store(ReadDenom+r_header,tl.sum(r_numer,0))',
    '                tl.store(ReadBest+r_header,S)',
    '                rr_best=tl.min(tl.where(rr_maximum==tl.max(rr_maximum,0),rr_candidates,S),0)',
    '                rr_scale=tl.exp(rr_maximum-tl.max(rr_maximum,0))',
    '                rr_total=tl.sum(rr_denom*rr_scale,0)',
    '                rr_read=tl.sum(rr_payload*rr_scale[:,None],0)/tl.maximum(rr_total,1.0e-30)',
    '        journal_store(History + n * 128 + ring, case, True, J, RecordAll)',
]
for marker in markers:
    assert source.count(marker) == 1, marker
    indent = marker[:len(marker)-len(marker.lstrip())]
    source = source.replace(marker, marker+'\n'+indent+'tl.debug_barrier()')
# Avoid the unselected undefined cache operand without changing chosen values.
old = 'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live,0.)'
new = 'cached=tl.load(ReadStrengthCache+n*S+r_columns,r_live & (r_touch<Start+t-1),0.)'
assert source.count(old) == 1
source = source.replace(old, new)
kernels.scan_kernel._unsafe_update_src(source)
kernels.scan_kernel.hash = None
print(json.dumps(dict(experiment='diagnostic only; no certification', before=before,
    after=hashlib.sha256(source.encode()).hexdigest(), barriers=len(markers), cache_mask=True)), flush=True)
sys.argv = ['isolated.py', 'original']
runpy.run_path('experiments/expressivity_tasks/x40_emender_lm/receipts/part15/isolated.py', run_name='__main__')
