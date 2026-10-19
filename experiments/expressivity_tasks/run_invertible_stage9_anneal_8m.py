"""X9: scratch T16 E97; AdamW warmup/cosine regime versus X7 constant ScheduleFree."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage7_deepfirst_8m import (
    build_model as x7_model, digest, frozen_pools, forward_batch,
    evaluate, write_json, PRECISION as X7_PRECISION,
)
from experiments.expressivity_tasks.run_invertible_stage6_truecur_8m import prune
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage9_anneal_8m')
DEPTH = 16
CEILING = 500000
EVAL_EVERY = 500
NAMES = ('e97_bf16_seed0',)
WARMUP = 5000
PEAK_LR = 1e-3
MIN_LR = 1e-5
PRECISION = {
    'e97_bf16_seed0': X7_PRECISION['e97_bf16_seed0'],
}


def build_model(arm, precision):
    if (arm, precision) != ('e97', 'bf16'):
        raise ValueError('unregistered arm/precision')
    return x7_model(arm, precision)


def learning_rate(step):
    """Actual LR for update 1..500000; step zero is an untrained baseline."""
    if not 0 <= step <= CEILING:
        raise ValueError('update outside registered schedule')
    if step <= WARMUP:
        return PEAK_LR * step / WARMUP
    progress = (step - WARMUP) / (CEILING - WARMUP)
    return MIN_LR + .5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def make_optimizer(parameters):
    import torch
    return torch.optim.AdamW(parameters, lr=learning_rate(0), betas=(.9, .999),
                             eps=1e-8, weight_decay=.01)


def checkpoint(out, model, opt, rng, step, depth, stage_step, config):
    """Atomic X9 checkpoint with ordinary (not ScheduleFree averaged) weights."""
    import torch
    path = out / f'step_{step:06d}_T{depth}_s{stage_step:05d}.pt'
    tmp = path.with_suffix('.pt.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'depth': depth,
                'stage_step': stage_step, 'config': config, 'weights': 'ordinary AdamW parameters'}, tmp)
    tmp.replace(path)
    return {'path': str(path), 'sha256': digest(path), 'step': step,
            'depth': depth, 'stage_step': stage_step, 'optimizer_state_saved': True}


def run(args):
    import torch
    import torch.nn.functional as F
    name = f'{args.arm}_{args.precision}_seed{args.seed}'
    if name not in NAMES:
        raise ValueError('unregistered run')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    sources = [__file__, 'experiments/expressivity_tasks/run_invertible_stage7_deepfirst_8m.py',
               'experiments/expressivity_tasks/run_invertible_stage2.py', 'ndm/models/hybrid_ladder.py', 'ndm/recurrent_precision.py', 'ndm/models/e88_fla_hybrid.py',
               'ndm/models/fla_gated_delta.py', 'ndm/triton/e88_triton_forward.py',
               'ndm/triton/e88_triton_backward.py', 'ndm/triton/e88_triton_optimized.py',
               'experiments/expressivity_tasks/run_invertible_stage3_8m.py',
               'experiments/expressivity_tasks/run_invertible_stage5_curriculum_8m.py',
               'experiments/expressivity_tasks/run_invertible_stage6_truecur_8m.py',
               'experiments/expressivity_tasks/eval_invertible_stage6_positions.py',
               'experiments/expressivity_tasks/tasks/invertible_quadratic.py']
    import fla
    fla_root = Path(fla.__file__).parent
    fla_paths = ['layers/gated_deltanet.py', 'modules/l2norm.py', 'ops/gated_delta_rule/chunk.py',
                 'ops/gated_delta_rule/chunk_fwd.py', 'ops/gated_delta_rule/gate.py',
                 'ops/gated_delta_rule/fused_recurrent.py', 'ops/gated_delta_rule/wy_fast.py',
                 'ops/common/chunk_delta_h.py', 'ops/common/chunk_o.py']
    config = {**vars(args), 'name': name, 'drivers': DEPTH, 'p': 61, 'vocab': 3721,
              'depths': [DEPTH], 'initialization': 'from scratch; no checkpoint loaded',
              'ladder': False, 'carry_supervision': False, 'stagnation_kill': False,
              'safety_ceiling': CEILING, 'stop_policy': '500K updates safety ceiling only; no threshold stop',
              'eval_every': EVAL_EVERY, 'eval_examples': 512, 'batch': 32, 'lr': .001,
              'optimizer': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
              'clip': 1., 'warmup': WARMUP, 'min_lr': MIN_LR,
              'schedule': 'linear step/5000 to 1e-3; cosine (step-5000)/495000 to 1e-5 at500000',
              'intervention': 'regime bundle: ordinary AdamW + standard betas + warmup/cosine; not schedule-only',
              'weight_decay_comparison': 'nominal .01 retained; SF decay acts at y in normalized update, AdamW decouples decay; no exact trajectory equivalence', 'precision_scope': PRECISION[name],
              'training': 'fresh procedural batches; frozen validation never sampled for training',
              'retention': 'X6D fix: immutable entry + best + latest/exit below400GiB; hash manifests',
              'loss_weighting': 'uniform mean cross entropy over all driver targets; validation unweighted',
              'padding': 'X3 right pad to16; T16 has 18 tokens, padded to32',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'host': os.uname().nodename, 'started_at': time.time(), 'torch_version': torch.__version__,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {p: digest(p) for p in sources},
              'fla_source_sha256': {str(fla_root / p): digest(fla_root / p) for p in fla_paths},
              'eval_backend': 'X7 sequential Triton E97, legacy recurrent precision; no parameter averaging'}
    pool = frozen_pools(args.seed, DEPTH, 61)['val']
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T16_{i}': a for i, a in enumerate(pool)})
    write_json(out / 'data_receipt.json', {'sha256': digest(out / 'frozen_validation.npz'),
        'array_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in pool],
        'examples': 512, 'targets': int(pool[2].sum()), 'selection': 'exact X3/X6 T16 validation only'})
    torch.manual_seed(args.seed)
    model = build_model(args.arm, args.precision).cuda()
    config['trainable_parameters'] = sum(p.numel() for p in model.parameters())
    assert config['trainable_parameters'] == 9900800
    write_json(out / 'config.json', config)
    opt = make_optimizer(model.parameters())
    assert not opt.state
    rng, task = np.random.default_rng(args.seed), InvertibleQuadraticTask(61)
    history, checkpoints, best, entry = [], {}, None, None
    started = time.monotonic()
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for step in range(CEILING + 1):
            if step:
                lr = learning_rate(step)
                for group in opt.param_groups:
                    group['lr'] = lr
                model.train()
                opt.zero_grad(set_to_none=True)
                logits, y, mask = forward_batch(model, task.generate_batch(32, DEPTH + 2, rng))
                loss = F.cross_entropy(logits[mask].float(), y[mask])
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite training loss')
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                opt.step()
                log.write(json.dumps({'step': step, 'depth': DEPTH, 'loss': loss.item(),
                    'grad_norm': norm.item(), 'lr': opt.param_groups[0]['lr'], 'elapsed_s': time.monotonic() - started}) + '\n')
                del logits, loss, y, mask
            if step % EVAL_EVERY and step != CEILING:
                continue
            metrics = evaluate(model, pool)
            current = metrics['aggregate']
            is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
            cp = None
            if is_best or step % 1000 == 0 or step == CEILING:
                cp = checkpoint(out, model, opt, rng, step, DEPTH, step, config)
                checkpoints[cp['path']] = cp
                latest = cp['path']
            if is_best:
                best = {**current, 'step': step, 'checkpoint': cp}
            if step == 0:
                entry = cp['path']
            row = {'step': step, 'depth': DEPTH, **metrics, 'checkpoint': cp,
                   'best_accuracy': best['accuracy'], 'elapsed_s': time.monotonic() - started,
                   'decision': 'safety_ceiling' if step == CEILING else 'continue'}
            history.append(row)
            write_json(out / 'validation.json', history)
            write_json(out / 'checkpoints.json', checkpoints)
            print(json.dumps({'name': name, **row}), flush=True)
            prune(out, checkpoints, {entry, best['checkpoint']['path'], latest})
    write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': CEILING,
        'stop_reason': 'safety_ceiling', 'best': best, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97'], required=True)
    parser.add_argument('--precision', choices=['bf16'], required=True)
    parser.add_argument('--seed', type=int, choices=[0], default=0)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
