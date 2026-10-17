"""Operational depth-throughput calibration, never curriculum/accuracy evidence."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import time

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage5b_growth_8m as growth
from experiments.expressivity_tasks.run_invertible_stage3_8m import frozen_pools


def run(out):
    import schedulefree
    import torch
    import torch.nn.functional as F

    out.mkdir(exist_ok=False)
    rows = []
    for arm in ('e97', 'gdn2'):
        torch.manual_seed(0)
        model = growth.build_model(arm, 61).cuda()
        opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=.001, betas=(.9, .95), weight_decay=.01)
        rng = np.random.default_rng(0)
        pools = {d: frozen_pools(0, d, 61)['val'] for d in growth.DEPTHS}
        task = growth.InvertibleQuadraticTask(61)
        for depth in growth.DEPTHS:
            elapsed = []
            for index in range(40):
                torch.cuda.synchronize()
                start = time.perf_counter()
                opt.train()
                model.train()
                x, y, mask = [torch.from_numpy(a).cuda() for a in task.generate_batch(32, depth + 2, rng)]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = growth.padded_forward(model, x)
                loss = F.cross_entropy(logits[mask].float(), y[mask])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                torch.cuda.synchronize()
                if index >= 10:
                    elapsed.append(time.perf_counter() - start)
                del logits, loss, x, y, mask
            opt.eval()
            start = time.perf_counter()
            for lower in growth.DEPTHS:
                if lower <= depth:
                    growth.evaluate(model, pools[lower], 61)
            torch.cuda.synchronize()
            evaluation = time.perf_counter() - start
            start = time.perf_counter()
            cp = growth.checkpoint(out, model, opt, rng, depth, depth, 40, {'purpose': 'throughput only'})
            checkpoint = time.perf_counter() - start
            Path(cp['path']).unlink()
            row = {'arm': arm, 'depth': depth, 'warmup_updates': 10, 'timed_updates': 30,
                   'seconds_per_update': float(np.mean(elapsed)),
                   'steps_per_second': 1 / float(np.mean(elapsed)),
                   'all_lower_retention_eval_seconds': evaluation,
                   'checkpoint_save_and_hash_seconds': checkpoint,
                   'projected_seconds_per_update': float(np.mean(elapsed)) + (evaluation + checkpoint) / growth.EVAL_EVERY}
            rows.append(row)
            growth.write_json(out / 'throughput.json', {
                'purpose': 'synthetic operational timing only; not growth or learning evidence',
                'gpu_name': torch.cuda.get_device_name(), 'rows': rows,
                'checkpoint_overhead': 'conservative every evaluation (500 steps), including best saves',
                'projection_warning': 'short isolated timing is an estimate, not a walltime guarantee; six-way contention and later training can differ'})
            print(row, flush=True)
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    # Kept separate from curriculum trajectories; no validation scores are retained.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    run(parser.parse_args().out)
