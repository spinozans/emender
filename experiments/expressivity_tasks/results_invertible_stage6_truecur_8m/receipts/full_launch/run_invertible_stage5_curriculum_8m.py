"""X5: from-scratch X3 arms, optimizer-continuous doubling-depth curriculum."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage3_8m import (
    build_model, digest, evaluate, frozen_pools, padded_forward, write_json,
)
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage5_curriculum_8m')
DEPTHS = (4, 8, 16, 32, 64, 128)
THRESHOLD = .90
STAGE_BUDGET = 10000
EVAL_EVERY = 500
FLOOR = 1 / 3721
MARGINAL = math.log(3721)
SUFFIX_THEOREM = ('For this invertible map, a predictor restricted to a proper driver suffix '
                  'with at least two omitted drivers has full-state accuracy 1/3721 '
                  '(0.026874%) under uniform independent inputs; this is a suffix-only '
                  'information bound, not a capacity bound on either recurrent model.')


def matrix():
    return [{'name': f'{arm}_seed{seed}', 'arm': arm, 'seed': seed}
            for seed in range(3) for arm in ('e97', 'gdn2')]


def stage_decision(accuracy, stage_step):
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or not 0 <= stage_step <= STAGE_BUDGET:
        raise ValueError('invalid curriculum observation')
    if accuracy >= THRESHOLD:
        return 'threshold'
    return 'budget_exhausted' if stage_step == STAGE_BUDGET else 'continue'


def validation_pools(seed):
    # The exact X2/X3 algorithm and split RNG: T=128 is the archived primary val.
    return {depth: frozen_pools(seed, depth, 61)['val'] for depth in DEPTHS}


def archive_pools(out, seed):
    pools = validation_pools(seed)
    path = out / 'frozen_validation.npz'
    np.savez_compressed(path, **{f'T{d}_{i}': a for d, pool in pools.items() for i, a in enumerate(pool)})
    receipts = {}
    for depth, pool in pools.items():
        aligned = []
        for stage in ('results_invertible_stage2', 'results_invertible_stage3_8m'):
            for prior in (ROOT.parent / stage).glob('*/frozen_pools.npz'):
                config = prior.parent / 'config.json'
                if not config.exists():
                    continue
                cfg = json.loads(config.read_text())
                if cfg.get('seed') != seed or cfg.get('drivers') != depth or cfg.get('p', 61) != 61:
                    continue
                with np.load(prior) as data:
                    if not all(np.array_equal(data[f'val_{i}'], a) for i, a in enumerate(pool)):
                        raise ValueError(f'X-series validation mismatch: {prior}')
                aligned.append({'path': str(prior), 'sha256': digest(prior)})
        receipts[str(depth)] = {'shape': list(pool[0].shape), 'targets': int(pool[2].sum()),
                               'array_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in pool],
                               'aligned_x_series_pools': aligned}
    write_json(out / 'data_receipt.json', {'path': str(path), 'sha256': digest(path),
                                         'depths': receipts, 'selection': 'validation only; no test selection'})
    return pools


def checkpoint(out, model, opt, rng, step, depth, stage_step, config):
    import torch
    path = out / f'step_{step:06d}_T{depth}_s{stage_step:05d}.pt'
    tmp = path.with_suffix('.pt.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'depth': depth,
                'stage_step': stage_step, 'config': config, 'weights': 'ScheduleFree eval x'}, tmp)
    tmp.replace(path)
    return {'path': str(path), 'sha256': digest(path), 'step': step,
            'depth': depth, 'stage_step': stage_step, 'optimizer_state_saved': True}


def prune(out, checkpoints, required):
    """Below the X-series 400GiB floor retain boundaries, stage best, current and 10K."""
    if shutil.disk_usage(out).free >= 400 * 1024**3:
        return
    removed = []
    for path, cp in checkpoints.items():
        if path not in required and cp.get('stage_step') != 0 and cp['step'] % 10000 and Path(path).exists():
            Path(path).unlink()
            removed.append(path)
    with (out / 'pruning.jsonl').open('a') as f:
        f.write(json.dumps({'time': time.time(), 'pruned': removed,
                            'retained_hashes': {p: c['sha256'] for p, c in checkpoints.items() if Path(p).exists()},
                            'free_bytes': shutil.disk_usage(out).free}) + '\n')


def run(args):
    import torch
    import torch.nn.functional as F
    import schedulefree
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'depths': DEPTHS, 'threshold': THRESHOLD, 'stage_budget': STAGE_BUDGET,
              'hard_cap': 60000, 'eval_every': EVAL_EVERY, 'batch': 32, 'p': 61, 'vocab': 3721,
              'lr': .001, 'optimizer': 'AdamWScheduleFree', 'betas': [.9, .95], 'weight_decay': .01,
              'clip': 1., 'warmup': 0, 'dtype': 'bf16 autocast, fp32 parameters',
              'initialization': 'from scratch; no checkpoint loaded',
              'continuity': 'same model, optimizer and training RNG throughout all stages',
              'training': 'fresh procedural batch at current depth only; full-state loss at every driver',
              'geometry': {'dim': 256, 'depth': 4, 'n_heads': 32, 'n_state': 32, 'mlp_ratio': 2.,
                           'expansion': 1. if args.arm == 'e97' else 8.625},
              'padding': 'X3 causal right-pad to16, sliced before metrics/loss',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'host': os.uname().nodename, 'started_at': time.time(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {p: digest(p) for p in [__file__,
                  'experiments/expressivity_tasks/run_invertible_stage3_8m.py',
                  'experiments/expressivity_tasks/run_invertible_stage2.py',
                  'experiments/expressivity_tasks/tasks/invertible_quadratic.py',
                  'ndm/models/hybrid_ladder.py']},
              'torch_version': torch.__version__, 'suffix_floor_percent': FLOOR * 100,
              'marginal_reference_nats': MARGINAL, 'suffix_theorem': SUFFIX_THEOREM}
    write_json(out / 'config.json', config)
    pools = archive_pools(out, args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.arm, 61).cuda()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == {'e97': 9900800, 'gdn2': 9887056}[args.arm]
    if args.arm == 'gdn2':
        assert all(type(layer).__name__ == 'FLAGatedDeltaNetLayer' and layer.allow_neg_eigval
                   and layer.gdn.allow_neg_eigval for layer in model.layers)
    else:
        assert all(layer.use_split_edit and layer.state_activation == 'tanh' and not layer.raw_write
                   and not layer.linear_state for layer in model.layers)
    config['trainable_parameters'] = count
    write_json(out / 'config.json', config)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=.001, betas=(.9, .95), weight_decay=.01)
    assert not opt.state
    rng, task = np.random.default_rng(args.seed), InvertibleQuadraticTask(61)
    history, stages, checkpoints, required = [], [], {}, set()
    total_step, tokens, train_seconds = 0, 0, 0.
    started = time.time()
    with (out / 'training.jsonl').open('a', buffering=1) as log, (out / 'transitions.jsonl').open('a', buffering=1) as transitions:
        for depth in DEPTHS:
            losses, best, previous_best_path = [], None, None
            entry_step = total_step
            for stage_step in range(STAGE_BUDGET + 1):
                if stage_step:
                    t = time.time()
                    opt.train()
                    model.train()
                    x, y, mask = [torch.from_numpy(a).cuda() for a in task.generate_batch(32, depth + 2, rng)]
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        logits = padded_forward(model, x)
                    loss = F.cross_entropy(logits[mask].float(), y[mask])
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite training loss')
                    loss_value = loss.item()
                    losses.append(loss_value)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    torch.cuda.synchronize()
                    train_seconds += time.time() - t
                    tokens += x.numel()
                    total_step += 1
                    log.write(json.dumps({'step': total_step, 'depth': depth, 'stage_step': stage_step,
                                          'loss': loss_value, 'grad_norm': norm.item(),
                                          'elapsed_s': time.time() - started}) + '\n')
                    del logits, loss, x, y, mask
                if stage_step % EVAL_EVERY:
                    continue
                opt.eval()  # Evaluate/save ScheduleFree x, restore y on the next training step.
                metrics = {str(d): evaluate(model, pools[d], 61) for d in DEPTHS if d <= depth}
                current = metrics[str(depth)]
                decision = stage_decision(current['accuracy'], stage_step)
                is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                cp = None
                if is_best or stage_step % 1000 == 0 or decision != 'continue':
                    cp = checkpoint(out, model, opt, rng, total_step, depth, stage_step, config)
                    checkpoints[cp['path']] = cp
                if is_best:
                    if previous_best_path:
                        required.discard(previous_best_path)
                    previous_best_path = cp['path']
                    required.add(cp['path'])
                    best = {**current, 'stage_step': stage_step, 'step': total_step, 'checkpoint': cp}
                if stage_step == 0 or decision != 'continue':
                    required.add(cp['path'])
                row = {'step': total_step, 'depth': depth, 'stage_step': stage_step, 'decision': decision,
                       'validation_by_depth': metrics, 'best_accuracy': best['accuracy'],
                       'train_loss_first500': float(np.mean(losses[:500])) if losses else None,
                       'train_loss_final500': float(np.mean(losses[-500:])) if losses else None,
                       'marginal_reference_nats': MARGINAL, 'suffix_floor_percent': FLOOR * 100}
                history.append(row)
                write_json(out / 'validation.json', history)
                write_json(out / 'checkpoints.json', checkpoints)
                print(json.dumps({'arm': args.arm, 'seed': args.seed, **row}), flush=True)
                prune(out, checkpoints, required | ({cp['path']} if cp else set()))
                if stage_step == 0:
                    transitions.write(json.dumps({'event': 'stage_entry', 'depth': depth, 'step': total_step,
                                                  'checkpoint': cp, 'optimizer_reset': False,
                                                  'previous_stage_exit': stages[-1]['checkpoint'] if stages else None}) + '\n')
                if decision != 'continue':
                    stage = {'depth': depth, 'entry_step': entry_step, 'steps': stage_step,
                             'exit_step': total_step, 'decision': decision,
                             'steps_to_threshold': stage_step if decision == 'threshold' else None,
                             'final_accuracy': current['accuracy'], 'final_loss': current['loss'],
                             'best': best, 'retention': metrics, 'checkpoint': cp,
                             'train_loss_first500': row['train_loss_first500'],
                             'train_loss_final500': row['train_loss_final500']}
                    stages.append(stage)
                    transitions.write(json.dumps({'event': 'stage_exit', **stage}) + '\n')
                    write_json(out / 'stages.json', stages)
                    break
            if decision == 'budget_exhausted':
                break
    achieved = [s['depth'] for s in stages if s['decision'] == 'threshold']
    result = {'arm': args.arm, 'seed': args.seed, 'status': 'complete',
              'frontier': max(achieved, default=0), 'stopping_depth': depth,
              'stopping_accuracy': stages[-1]['final_accuracy'],
              'stop_reason': 'completed_curriculum' if depth == 128 and decision == 'threshold' else decision,
              'total_steps': total_step, 'stages': stages, 'input_tokens': tokens,
              'train_seconds': train_seconds, 'wall_seconds': time.time() - started,
              'suffix_floor_percent': FLOOR * 100, 'marginal_reference_nats': MARGINAL,
              'suffix_theorem': SUFFIX_THEOREM, 'finished_at': time.time()}
    assert total_step <= 60000
    write_json(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97', 'gdn2'], required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
