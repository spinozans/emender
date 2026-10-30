"""Fail-closed first-stage promotion gates; no acceptance from partial passes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

assert sys.executable == '/home/erikg/emender/.venv/bin/python'
assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3', '4')
OUT = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument('mode', choices=('lifecycle', 'certify', 'shape', 'isolated', 'oracles', 'curve'))
p.add_argument('--batch', type=int, default=64)
p.add_argument('--context', type=int, default=512)
a = p.parse_args()
result = dict(status='failed', mode=a.mode, batch=a.batch, context=a.context,
              physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
result['sources'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (OUT.parents[1]/'kernels.py', OUT.parents[2]/'x31_tape_cell_fused/kernels.py')}
try:
    if a.mode == 'lifecycle':
        from lifecycle import validate
        result['x40'] = validate('x40')
        result['x31'] = validate('x31')
    elif a.mode == 'isolated':
        import runpy
        fixture = Path('/tmp/x40_part15_scan10.pt')
        result['fixture_sha256'] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        sys.argv = ['isolated.py', 'original']
        runpy.run_path(str(OUT.parent/'part15/isolated.py'), run_name='__main__')
    elif a.mode == 'oracles':
        import pytest
        result['pytest_exit_code'] = pytest.main(['-q',
            'experiments/expressivity_tasks/x31_tape_cell_fused/tests',
            '--junitxml='+str(OUT/'oracles.xml')])
        assert result['pytest_exit_code'] == 0
    elif a.mode == 'curve':
        from experiments.expressivity_tasks.x40_emender_lm import train_slice
        source = Path(train_slice.__file__).read_text()
        # Apply this session's broker allowlist only; leave the baseline and math intact.
        source = source.replace("not in ('1','5','6')", "not in ('2','3','4')")
        sys.argv = ['train_slice.py', '--output', str(OUT/'curve'), '--steps', '64', '--graph']
        result['harness_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        exec(compile(source, train_slice.__file__, 'exec'),
             dict(__name__='__main__', __file__=train_slice.__file__, __package__=train_slice.__package__))
        rows = [json.loads(line) for line in (OUT/'curve/curve.jsonl').read_text().splitlines()]
        result['curve_steps'] = len(rows)
        result['maximum_loss_difference'] = max(row['absolute_difference'] for row in rows)
        assert len(rows) == 64
        assert result['maximum_loss_difference'] == 0, 'requested bit-matched curve gate failed'
    elif a.mode == 'certify':
        from experiments.expressivity_tasks.x40_emender_lm.parity import certify
        if a.context > 512:
            from unittest.mock import patch
            from experiments.expressivity_tasks.x31_tape_cell_fused import backward
            from experiments.expressivity_tasks.x40_emender_lm.probe_rebase import TransposedLaunch, validate
            result['existing_part5_adapter'] = validate()
            with patch.object(backward, 'rebase', TransposedLaunch()):
                result['certification'] = certify(batch=a.batch, context=a.context)
        else:
            result['certification'] = certify(batch=a.batch, context=a.context)
    else:
        import torch
        from experiments.expressivity_tasks.x40_emender_lm.model import POLICIES, scan
        torch.set_num_threads(1)
        result['policies'] = []
        for policy, cfg in enumerate(POLICIES):
            torch.manual_seed(4001)
            k = torch.randn(a.batch, a.context, 1, 64, device='cuda')*.3
            q = torch.randn_like(k)*.3
            v = torch.randn(a.batch, a.context, 1, 128, device='cuda')
            pop = torch.rand(a.batch, a.context, 1, device='cuda')
            append = torch.randn_like(pop)
            logits = torch.randn_like(pop)+.5
            mask = torch.ones_like(pop)
            mask[:,::17] = 0
            with torch.no_grad():
                scan(k,q,v,mask,pop,append,cfg,torch.full((1,),.03,device='cuda'),
                     logits,torch.full((1,),-1.,device='cuda'),torch.full((1,),.07,device='cuda'))
            torch.cuda.synchronize()
            result['policies'].append(dict(policy=policy,slots=cfg.slots))
    result['status'] = 'passed'
except BaseException as exc:
    result.update(error=repr(exc), traceback=traceback.format_exc())
    print(result['traceback'], flush=True)
finally:
    (OUT / (os.environ['X40_PART20_RUN_NAME']+'.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result), flush=True)
if result['status'] != 'passed':
    raise SystemExit(1)
