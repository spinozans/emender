"""Isolated mask test, including strict unchanged-X31 certification.

Use only through run.sh. Results describe a process-local candidate.
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback

from candidate import install

OUT = Path(__file__).resolve().parent


def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    assert os.environ['CUDA_VISIBLE_DEVICES'] in ('2', '3', '4')
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=('isolated', 'certify', 'certify-adapted', 'lifecycle', 'shape', 'pressure-control', 'minimal'))
    p.add_argument('--batch', type=int, default=64)
    p.add_argument('--context', type=int, default=512)
    p.add_argument('--scratch-barriers', action='store_true')
    p.add_argument('--store-layout', action='store_true')
    p.add_argument('--capture', action='store_true')
    p.add_argument('--unique-writers', action='store_true')
    p.add_argument('--inline-store', action='store_true')
    a = p.parse_args()
    assert not a.unique_writers or a.scratch_barriers
    result = dict(status='failed', candidate=install(store_layout=a.store_layout), mode=a.mode,
                  physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'])
    print(json.dumps(result), flush=True)
    if a.scratch_barriers:
        from scratch_ordering import install as order_scratch
        order_scratch(unique_writers=a.unique_writers)
        result['scratch_barriers'] = 'diagnostic PTX only; control-flow legality unproved'
    if a.capture:
        assert not a.scratch_barriers
        from capture import install as capture
        capture()
    try:
        if a.mode == 'isolated':
            sys.argv = ['isolated.py', 'original']
            runpy.run_path(str(OUT.parent / 'part15/isolated.py'), run_name='__main__')
        elif a.mode == 'certify':
            from experiments.expressivity_tasks.x40_emender_lm.parity import certify
            result['certification'] = certify(batch=a.batch, context=a.context)
        elif a.mode == 'certify-adapted':
            from unittest.mock import patch
            from experiments.expressivity_tasks.x31_tape_cell_fused import backward
            from experiments.expressivity_tasks.x40_emender_lm.probe_rebase import TransposedLaunch, validate
            from experiments.expressivity_tasks.x40_emender_lm.parity import certify
            result['oracle_launch_adapter'] = validate()
            result['adapter_scope'] = 'existing probe_rebase.TransposedLaunch, journal address mapping only; strict pressure retained'
            with patch.object(backward, 'rebase', TransposedLaunch()):
                result['certification'] = certify(batch=a.batch, context=a.context)
        elif a.mode == 'minimal':
            sys.argv = ['reduce_probe.py'] + (['--inline'] if a.inline_store else [])
            runpy.run_path(str(OUT / 'reduce_probe.py'), run_name='__main__')
        elif a.mode == 'pressure-control':
            import hashlib
            from experiments.expressivity_tasks.x40_emender_lm import pressure_probe
            path = Path(pressure_probe.__file__)
            source = path.read_text()
            changed = source.replace("not in ('1','5','6')", "not in ('2','3','4')")
            changed = changed.replace("outdir=ROOT/'receipts/part4'", "outdir=ROOT/'receipts/part18/pressure_control'")
            assert changed != source
            result['control_only'] = 'unchanged X31 control before production repair; not a post-fix connection test'
            result['probe_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            exec(compile(changed, str(path), 'exec'), dict(__name__='__main__',
                 __file__=str(path), __package__=pressure_probe.__package__))
        elif a.mode == 'lifecycle':
            from lifecycle import validate
            result['lifecycle'] = validate()
        else:
            import torch
            from experiments.expressivity_tasks.x40_emender_lm.model import POLICIES, scan
            torch.set_num_threads(1)
            result['shapes'] = []
            for policy, cfg in enumerate(POLICIES):
                torch.manual_seed(4001)
                k = torch.randn(a.batch, a.context, 1, 64, device='cuda') * .3
                q = torch.randn_like(k) * .3
                v = torch.randn(a.batch, a.context, 1, 128, device='cuda')
                pop = torch.rand(a.batch, a.context, 1, device='cuda')
                append = torch.randn_like(pop)
                logits = torch.randn_like(pop) + .5
                mask = torch.ones_like(pop)
                mask[:, ::17] = 0
                with torch.no_grad():
                    scan(k, q, v, mask, pop, append, cfg,
                         torch.full((1,), .03, device='cuda'), logits,
                         torch.full((1,), -1., device='cuda'),
                         torch.full((1,), .07, device='cuda'))
                torch.cuda.synchronize()
                result['shapes'].append(dict(policy=policy, batch=a.batch,
                                             context=a.context, slots=cfg.slots))
        result['status'] = 'passed'
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    finally:
        name = os.environ.get('X40_PART18_RUN_NAME', f'{a.mode}_B{a.batch}_T{a.context}')
        (OUT / f'{name}.json').write_text(json.dumps(result, indent=2) + '\n')
    if result['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
