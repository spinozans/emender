"""Pressure self-consistency diagnostic; never enables a performance gate.

A temporary local X31 kernel fork adds a store/load barrier solely to test the
race hypothesis. The upstream files and production certification stay intact.
"""
import difflib
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch
from ..x31_tape_cell_fused import cell_fused as oracle
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig

ROOT=Path(__file__).resolve().parent


def main():
    if os.environ.get('GPU_LEASE_HELD') not in ('1','5','6'):
        raise RuntimeError('approved lease required')
    torch.set_num_threads(1)
    torch.manual_seed(4001)
    b,t,h=1,512,1
    cfg=TapeHeadConfig(n_heads=1,budget=64,decay_rate=math.exp(-1/1000),pop_threshold=.1)
    k=torch.randn(b,t,h,64,device='cuda')*.3
    q=torch.randn_like(k)*.3
    v=torch.randn(b,t,h,128,device='cuda')
    pop=torch.rand(b,t,h,device='cuda')
    append=torch.randn_like(pop)
    beta=torch.ones_like(pop); beta[:,::17]=0
    path=ROOT.parent/'x31_tape_cell_fused/kernels.py'
    original=path.read_text()
    needle='        journal_store(History + n * 128 + ring, case, True, J, RecordAll)\n'
    assert original.count(needle)==1
    modified=original.replace(needle,needle+'        tl.debug_barrier()\n')
    outdir=ROOT/'receipts/part4'
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/'pressure_barrier_diagnostic.patch').write_text(''.join(difflib.unified_diff(
        original.splitlines(True),modified.splitlines(True),fromfile='X31 original (unchanged)',tofile='temporary diagnostic fork')))
    # Absolute imports let the local fork use the unmodified X31 dependencies.
    source=modified.replace('from .','from experiments.expressivity_tasks.x31_tape_cell_fused.')
    rows=[]; paired=[]
    with tempfile.TemporaryDirectory(prefix='x40_pressure_') as directory:
        candidate_path=Path(directory)/'barrier_kernel.py';candidate_path.write_text(source)
        spec=importlib.util.spec_from_file_location('x40_pressure_barrier_diagnostic',candidate_path)
        candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
        for name,kernel in (('unchanged_x31',oracle.scan_kernel),('diagnostic_barrier',candidate.scan_kernel)):
            for grad in (False,True):
                previous_result=None
                with patch.object(oracle,'scan_kernel',kernel):
                    for repeat in range(3):
                        args=[x.detach().clone().requires_grad_(grad) for x in (q,v,pop,append)]
                        out,st,tr,p=oracle.tape_scan(k,args[0],args[1],beta,args[2],args[3],None,cfg)
                        c=tr.case.cpu();actual=p.detach().cpu(); expected=torch.empty_like(actual)
                        for i in range(t):
                            window=c[:,max(0,i-127):i+1]
                            expected[:,i]=(window==5).sum(1).float()/((window>0)&(window!=4)).sum(1).clamp_min(1).float()
                        # tl.div and torch division can differ by one ULP. Retain
                        # exact mismatches, but separately count semantic errors.
                        error=(actual-expected).abs();bad=error>1e-6
                        fields=[out.detach().cpu(),st.native.keys.detach().cpu(),st.native.values.detach().cpu(),
                            st.native.strength.detach().cpu(),st.native.live.cpu(),st.native.age.cpu(),c,
                            tr.matched.cpu(),tr.alloc_target.cpu(),tr.pop_target.cpu(),tr.freed.cpu()]
                        row=dict(kernel=name,grad=grad,repeat=repeat,bitwise_mismatches=int((actual!=expected).sum()),
                            semantic_mismatches=int(bad.sum()),maximum_error=float(error.max()),indices=bad.nonzero()[:8].tolist(),
                            actual=actual[bad][:8].tolist(),expected=expected[bad][:8].tolist())
                        if previous_result is not None:
                            old_pressure,old_fields=previous_result
                            row.update(repeat_pressure_mismatches=int((actual!=old_pressure).sum()),
                                repeat_pressure_max_error=float((actual-old_pressure).abs().max()),
                                other_fields_bitwise_equal=all(torch.equal(x,y) for x,y in zip(fields,old_fields)))
                        previous_result=(actual,fields)
                        rows.append(row)
                        del out,st,tr,p,args
                paired.append((name,grad,previous_result))
    controls=[]
    for grad in (False,True):
        a=next(r for n,g,r in paired if n=='unchanged_x31' and g==grad)
        z=next(r for n,g,r in paired if n=='diagnostic_barrier' and g==grad)
        controls.append(dict(grad=grad,other_fields_bitwise_equal=all(torch.equal(x,y) for x,y in zip(a[1],z[1]))))
    assert path.read_text()==original, 'upstream source must remain unchanged'
    assert all(r['semantic_mismatches']==0 for r in rows if r['kernel']=='diagnostic_barrier')
    assert all(r.get('repeat_pressure_mismatches',0)==0 for r in rows if r['kernel']=='diagnostic_barrier')
    assert all(r.get('other_fields_bitwise_equal',True) for r in rows)
    assert all(r['other_fields_bitwise_equal'] for r in controls)
    result=dict(scope='unchanged X31 self-consistency; no X40 wire or policy translation; diagnostic fork is not a certification oracle',
        physical_gpu=os.environ['GPU_LEASE_HELD'],device=torch.cuda.get_device_name(),python=os.sys.executable,
        oracle_source_sha256=hashlib.sha256(original.encode()).hexdigest(),
        diagnostic_source_sha256=hashlib.sha256(source.encode()).hexdigest(),rows=rows,controls=controls)
    (outdir/'pressure_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
