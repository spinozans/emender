"""Candidate checks, each process isolated and each failure propagated."""
import importlib.util
import json
from pathlib import Path
import sys
import torch
from candidate import install, counted_clone

root=Path(__file__).resolve().parent
assert sys.executable=='/home/erikg/emender/.venv/bin/python'
torch.set_num_threads(1)
print(json.dumps({'candidate_sources':install()}),flush=True)
mode=sys.argv[1]
if mode=='lifecycle':
    spec=importlib.util.spec_from_file_location('part20_lifecycle',root.parent/'part20/lifecycle.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    result={target:module.validate(target) for target in ('x40','x31')}
    print(json.dumps({'status':'passed','fixtures':sum(map(len,result.values())),'details':result}),flush=True)
elif mode=='oracles':
    import pytest
    raise SystemExit(pytest.main(['-q','experiments/expressivity_tasks/x31_tape_cell_fused/tests']))
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
    print(json.dumps({'status':'passed','counted_clone_cases':6}),flush=True)
elif mode=='certify':
    from experiments.expressivity_tasks.x40_emender_lm.parity import certify
    print(json.dumps(certify(batch=int(sys.argv[2]),context=int(sys.argv[3]))),flush=True)
else:
    raise ValueError(mode)
