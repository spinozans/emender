"""Candidate checks, each process isolated and each failure propagated."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"part21"))
from experiments.expressivity_tasks.x31_tape_cell_fused.counted_copy import counted_clone

root=Path(__file__).resolve().parent
assert sys.executable=='/home/erikg/emender/.venv/bin/python'
torch.set_num_threads(1)
print(json.dumps({'mode':'production','source_sha256':{str(p):__import__('hashlib').sha256(p.read_bytes()).hexdigest() for name in ('x31_tape_cell_fused','x36_tape_perf','x40_emender_lm') for p in (Path('experiments/expressivity_tasks')/name).glob('*.py')}}),flush=True)
mode=sys.argv[1]
if mode=='lifecycle':
    spec=importlib.util.spec_from_file_location('part20_lifecycle',root.parent/'part20/lifecycle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result={target:module.validate(target) for target in ('x40','x31')}
    print(json.dumps({'status':'passed','fixtures':sum(map(len,result.values())),'details':result}),flush=True)
elif mode in ('oracles','snapshot'):
    import pytest
    target='experiments/expressivity_tasks/x31_tape_cell_fused/tests'
    if mode=='snapshot':
        target+='/test_all_field_journal.py::test_all_field_snapshot_parent_and_pressure_expiry'
    raise SystemExit(pytest.main(['-q',target]))
elif mode=='counted':
    # Undefined tails must never be read; destinations are fully defined zero tails.
    for dtype in (torch.uint64,torch.int32):
        source=torch.empty((3,777),device='cuda',dtype=dtype)
        counts=torch.tensor([0,1,777],device='cuda',dtype=torch.int64)
        source[1,:1]=13;source[2,:]=29
        result=counted_clone(source,counts)
        assert torch.equal(result[0],torch.zeros_like(result[0]))
        assert result[1,0].item()==13
        assert torch.equal(result[1,1:],torch.zeros_like(result[1,1:]))
        assert torch.equal(result[2],source[2])
    source=torch.empty((1,65536*256+1),device='cuda',dtype=torch.uint64)
    result=counted_clone(source,torch.zeros(1,device='cuda',dtype=torch.int64))
    assert (result.view(torch.int64)==0).all()
    print(json.dumps({'status':'passed','counted_clone_cases':7,'large_log_blocks':65537}),flush=True)
elif mode=='certify':
    from experiments.expressivity_tasks.x40_emender_lm.parity import certify
    if int(sys.argv[3])>512:
        from experiments.expressivity_tasks.x40_emender_lm.probe_rebase import TransposedLaunch,validate
        print(json.dumps({'transposed_rebase':validate()}),flush=True)
        from experiments.expressivity_tasks.x31_tape_cell_fused import backward
        backward.rebase=TransposedLaunch()
    result=certify(batch=int(sys.argv[2]),context=int(sys.argv[3]))
    result.update(cuda_peak_allocated=torch.cuda.max_memory_allocated(),cuda_peak_reserved=torch.cuda.max_memory_reserved())
    print(json.dumps(result),flush=True)
elif mode=='isolated':
    import hashlib
    import runpy
    fixture=Path('/tmp/x40_part15_scan10.pt')
    print(json.dumps({'fixture_sha256':hashlib.sha256(fixture.read_bytes()).hexdigest()}),flush=True)
    sys.argv=['isolated.py','original']
    runpy.run_path(str(root.parent/'part15/isolated.py'),run_name='__main__')
else:
    raise ValueError(mode)
