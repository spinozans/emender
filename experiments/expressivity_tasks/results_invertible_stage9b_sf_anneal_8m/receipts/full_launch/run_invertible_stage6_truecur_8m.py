"""X6: true bottom-up growth, with and without training-only carry supervision."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage3_8m import (
    build_model as x3_build_model, digest, frozen_pools, padded_forward, write_json,
    finish_metrics,
)
# Reuse the unchanged X5 data, checkpoint and capacity-retention implementations.
from experiments.expressivity_tasks.run_invertible_stage5_curriculum_8m import (
    checkpoint,
)


from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage6_truecur_8m')
DEPTHS = (1, 2, 4, 8, 16, 32, 64, 128)
THRESHOLD = .90
STAGE_CEILING = 500000
GLOBAL_CAP = 4000000
EVAL_EVERY = 500
FLOOR = 1 / 3721
MARGINAL = math.log(3721)
SUFFIX_THEOREM = ('For this invertible map, a predictor restricted to a proper driver suffix '
                  'with at least two omitted drivers has full-state accuracy 1/3721 '
                  '(0.026874%) under uniform independent inputs; this is a suffix-only '
                  'information bound, not a capacity bound on either recurrent model.')


NEUTRAL = 3721  # Input-only reserved token; output remains exactly 3721 states.


def matrix():
    return [{'name': f'{curriculum}_{arm}_seed{seed}', 'curriculum': curriculum,
             'arm': arm, 'seed': seed}
            for curriculum in ('A', 'B') for seed in range(3) for arm in ('e97', 'gdn2')]


def carry_probability(stage_step):
    if not 0 <= stage_step <= STAGE_CEILING:
        raise ValueError('invalid carry annealing step')
    return .5 * (1 - stage_step / STAGE_CEILING)


def carry_inputs(x, y, probability, rng=None):
    """Insert previous-state slots, never current/future targets; p=0 is target-independent."""
    if not 0 <= probability <= 1:
        raise ValueError('invalid carry probability')
    depth = x.shape[1] - 2
    positions = np.r_[0, 1, 2 + 2 * np.arange(depth)]
    expanded = np.full((len(x), 2 * depth + 1), NEUTRAL, dtype=x.dtype)
    expanded[:, positions] = x
    if probability and depth > 1:
        selected = rng.random((len(x), depth - 1)) < probability
        expanded[:, 3::2] = np.where(selected, y[:, 2:-1], NEUTRAL)
    return expanded, positions


def build_model(arm, curriculum):
    import torch
    model = x3_build_model(arm, 61)
    if curriculum == 'B':
        class CarryEmbedding(torch.nn.Module):
            def __init__(self, embedding):
                super().__init__()
                self.weight = embedding.weight
                self.neutral = torch.nn.Parameter(torch.empty(1, embedding.embedding_dim))
                torch.nn.init.normal_(self.neutral)
            def forward(self, x):
                return torch.nn.functional.embedding(x, torch.cat((self.weight, self.neutral)))
        model.embed = CarryEmbedding(model.embed)
    return model


def archive_pools(out, seed):
    import hashlib
    pools = {d: frozen_pools(seed, d, 61)['val'] for d in DEPTHS}
    path = out / 'frozen_validation.npz'
    np.savez_compressed(path, **{f'T{d}_{i}': a for d, pool in pools.items() for i, a in enumerate(pool)})
    write_json(out / 'data_receipt.json', {'path': str(path), 'sha256': digest(path),
        'selection': 'exact X3 frozen validation pools; no test selection',
        'depths': {str(d): {'shape': list(pool[0].shape), 'targets': int(pool[2].sum()),
            'array_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]}
            for d, pool in pools.items()}})
    return pools


def forward_batch(model, batch, curriculum, probability=0., rng=None):
    import torch
    x, y, mask = batch
    positions = None
    if curriculum == 'B':
        x, positions = carry_inputs(x, y, probability, rng)
    tokens = x.size
    x, y, mask = [torch.from_numpy(a).cuda() for a in (x, y, mask)]
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = padded_forward(model, x)
    if positions is not None:
        logits = logits[:, positions]
    return logits, y, mask, tokens


def evaluate(model, pool, curriculum):
    import torch
    import torch.nn.functional as F
    from experiments.expressivity_tasks.run_invertible_stage2 import metric_counts
    model.eval()
    totals, loss_sum = {}, 0.
    with torch.no_grad():
        for start in range(0, len(pool[0]), 32):
            # Evaluation has no true-state feedback, irrespective of training p.
            logits, y, mask, _ = forward_batch(model, tuple(a[start:start+32] for a in pool), curriculum)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite evaluation logits')
            loss_sum += F.cross_entropy(logits[mask].float(), y[mask], reduction='sum').item()
            counts = metric_counts(logits.argmax(-1).cpu().numpy(), y.cpu().numpy(), mask.cpu().numpy(), 61)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    return finish_metrics(totals, loss_sum, 61)


def stage_decision(accuracy, stage_step, total_step):
    if (not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or
            not 0 <= stage_step <= STAGE_CEILING or not 0 <= total_step <= GLOBAL_CAP):
        raise ValueError('invalid growth observation')
    if accuracy >= THRESHOLD:
        return 'threshold'
    if total_step == GLOBAL_CAP:
        return 'global_safety_cap'
    if stage_step == STAGE_CEILING:
        return 'ceiling_non_threshold'
    return 'continue'


def prune(out, checkpoints, required):
    """X6 bounded retention: boundaries, stage-best and latest, below 400GiB free."""
    import shutil
    if shutil.disk_usage(out).free >= 400 * 1024**3:
        return
    removed = []
    for path in checkpoints:
        if path not in required and Path(path).exists():
            Path(path).unlink()
            removed.append(path)
    if removed:
        with (out / 'pruning.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': time.time(), 'pruned': removed,
                'retained_hashes': {p: c['sha256'] for p, c in checkpoints.items() if Path(p).exists()},
                'free_bytes': shutil.disk_usage(out).free}) + '\n')


def restore_best(best, model, opt):
    """Restore the selected weights AND optimizer without rewinding consumed data/compute."""
    import torch
    cp = best['checkpoint']
    if digest(cp['path']) != cp['sha256']:
        raise ValueError('best checkpoint hash mismatch')
    payload = torch.load(cp['path'], weights_only=False)
    model.load_state_dict(payload['model'])
    opt.load_state_dict(payload['optimizer'])
    return cp


def run(args):
    import torch
    import torch.nn.functional as F
    import schedulefree
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'depths': DEPTHS, 'threshold': THRESHOLD, 'stage_safety_ceiling': STAGE_CEILING,
              'hard_cap': GLOBAL_CAP, 'stagnation_kill': False,
              'retention': 'below400GiB retain only stage-entry/exit, stage-best and latest; audited prune manifests',
              'ceiling_action': 'continue next stage with best-val model AND optimizer; no training RNG rewind',
              'carry_schedule': 'B only: linear .5 to 0 across 500K stage updates; resets each stage',
              'carry_slot': 'B only: previous joint state before driver>=2, neutral id3721 otherwise; eval always neutral',
              'input_vocab': 3722 if args.curriculum == 'B' else 3721,
              'output_vocab': 3721, 'eval_every': EVAL_EVERY, 'batch': 32, 'p': 61, 'vocab': 3721,
              'lr': .001, 'optimizer': 'AdamWScheduleFree', 'betas': [.9, .95], 'weight_decay': .01,
              'clip': 1., 'warmup': 0, 'dtype': 'bf16 autocast, fp32 parameters',
              'initialization': 'from scratch; no checkpoint loaded',
              'continuity': 'weights AND optimizer carry at threshold; best-val weights AND optimizer at ceiling; training RNG never rewound',
              'training': 'fresh procedural batch at current depth only; full-state loss at every driver',
              'geometry': {'dim': 256, 'depth': 4, 'n_heads': 32, 'n_state': 32, 'mlp_ratio': 2.,
                           'expansion': 1. if args.arm == 'e97' else 8.625},
              'padding': 'X3 causal right-pad to16, sliced before metrics/loss',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'host': os.uname().nodename, 'started_at': time.time(),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'source_sha256': {p: digest(p) for p in [__file__,
                  'experiments/expressivity_tasks/run_invertible_stage5_curriculum_8m.py',
                  'experiments/expressivity_tasks/run_invertible_stage3_8m.py',
                  'experiments/expressivity_tasks/run_invertible_stage2.py',
                  'experiments/expressivity_tasks/tasks/invertible_quadratic.py',
                  'ndm/models/hybrid_ladder.py']},
              'torch_version': torch.__version__, 'suffix_floor_percent': FLOOR * 100,
              'marginal_reference_nats': MARGINAL, 'suffix_theorem': SUFFIX_THEOREM}
    write_json(out / 'config.json', config)
    pools = archive_pools(out, args.seed)
    torch.manual_seed(args.seed)
    model = build_model(args.arm, args.curriculum).cuda()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert count == {'e97': 9900800, 'gdn2': 9887056}[args.arm] + (256 if args.curriculum == 'B' else 0)
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
    boundary_checkpoints = set()
    total_step, tokens, train_seconds = 0, 0, 0.
    latest_path = None
    started = time.time()
    with (out / 'training.jsonl').open('a', buffering=1) as log, (out / 'transitions.jsonl').open('a', buffering=1) as transitions:
        for depth in DEPTHS:
            losses, best, previous_best_path = [], None, None
            entry_step = total_step
            remaining = min(STAGE_CEILING, GLOBAL_CAP - total_step)
            for stage_step in range(remaining + 1):
                if stage_step:
                    t = time.time()
                    opt.train()
                    model.train()
                    batch = task.generate_batch(32, depth + 2, rng)
                    opt.zero_grad(set_to_none=True)
                    probability = carry_probability(stage_step) if args.curriculum == 'B' else 0.
                    logits, y, mask, batch_tokens = forward_batch(model, batch, args.curriculum, probability, rng)
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
                    tokens += batch_tokens
                    total_step += 1
                    log.write(json.dumps({'step': total_step, 'depth': depth, 'stage_step': stage_step,
                                          'loss': loss_value, 'grad_norm': norm.item(), 'carry_probability': probability, 'input_tokens': batch_tokens,
                                          'elapsed_s': time.time() - started}, separators=(',', ':')) + '\n')
                    del logits, loss, y, mask
                if stage_step % EVAL_EVERY and stage_step != remaining:
                    continue
                opt.eval()  # Evaluate/save ScheduleFree x, restore y on the next training step.
                metrics = {str(d): evaluate(model, pools[d], args.curriculum) for d in DEPTHS if d <= depth}
                current = metrics[str(depth)]
                is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                best_accuracy = max(current['accuracy'], best['accuracy'] if best else 0.)
                decision = stage_decision(current['accuracy'], stage_step, total_step)
                cp = None
                if is_best or stage_step % 1000 == 0 or decision != 'continue':
                    cp = checkpoint(out, model, opt, rng, total_step, depth, stage_step, config)
                    checkpoints[cp['path']] = cp
                    latest_path = cp['path']
                if is_best:
                    # An entry can stop being best without ceasing to be a boundary.
                    if previous_best_path and previous_best_path not in boundary_checkpoints:
                        required.discard(previous_best_path)
                    previous_best_path = cp['path']
                    required.add(cp['path'])
                    best = {**current, 'stage_step': stage_step, 'step': total_step, 'checkpoint': cp}
                if stage_step == 0 or decision != 'continue':
                    boundary_checkpoints.add(cp['path'])
                    required.add(cp['path'])
                row = {'step': total_step, 'depth': depth, 'stage_step': stage_step, 'decision': decision,
                       'validation_by_depth': metrics, 'best_accuracy': best['accuracy'],
                       'carry_probability': carry_probability(stage_step) if args.curriculum == 'B' else 0.,
                       'train_loss_delta_marginal': float(np.mean(losses[-500:])) - MARGINAL if losses else None,
                       'wall_elapsed_s': time.time() - started,
                       'training_steps_per_second': total_step / train_seconds if train_seconds else None,
                       'train_loss_first500': float(np.mean(losses[:500])) if losses else None,
                       'train_loss_final500': float(np.mean(losses[-500:])) if losses else None,
                       'marginal_reference_nats': MARGINAL, 'suffix_floor_percent': FLOOR * 100}
                history.append(row)
                write_json(out / 'validation.json', history)
                write_json(out / 'checkpoints.json', checkpoints)
                print(json.dumps({'arm': args.arm, 'curriculum': args.curriculum, 'seed': args.seed, **row}), flush=True)
                prune(out, checkpoints, required | ({latest_path} if latest_path else set()))
                if stage_step == 0:
                    transitions.write(json.dumps({'event': 'stage_entry', 'depth': depth, 'step': total_step,
                                                  'checkpoint': cp, 'optimizer_reset': False,
                                                  'previous_stage_exit': stages[-1]['checkpoint'] if stages else None,
                                                  'carried_checkpoint': stages[-1]['carried_checkpoint'] if stages else None}) + '\n')
                if decision != 'continue':
                    stage = {'depth': depth, 'entry_step': entry_step, 'steps': stage_step,
                             'exit_step': total_step, 'decision': decision,
                             'steps_to_threshold': stage_step if decision == 'threshold' else None,
                             'final_accuracy': current['accuracy'], 'final_loss': current['loss'],
                             'best': best, 'retention': metrics, 'checkpoint': cp,
                             'carried_checkpoint': best['checkpoint'] if decision == 'ceiling_non_threshold' else cp,
                             'ceiling_without_threshold': decision in ('ceiling_non_threshold', 'global_safety_cap'),
                             'train_loss_first500': row['train_loss_first500'],
                             'train_loss_final500': row['train_loss_final500']}
                    stages.append(stage)
                    transitions.write(json.dumps({'event': 'stage_exit', **stage}) + '\n')
                    write_json(out / 'stages.json', stages)
                    write_json(out / 'growth_curve.json', {
                        'arm': args.arm, 'curriculum': args.curriculum, 'seed': args.seed, 'stages': stages,
                        'deepest_threshold_crossed': max((s['depth'] for s in stages if s['decision'] == 'threshold'), default=0),
                        'total_steps': total_step, 'suffix_floor_percent': FLOOR * 100,
                        'marginal_reference_nats': MARGINAL})
                    break
            if total_step >= GLOBAL_CAP:
                break
            if decision == 'ceiling_non_threshold' and depth != DEPTHS[-1]:
                restored = restore_best(best, model, opt)
                transitions.write(json.dumps({'event': 'restore_best_after_ceiling', 'depth': depth,
                    'step': total_step, 'checkpoint': restored, 'optimizer_restored': True,
                    'training_rng_rewound': False}) + '\n')
    achieved = [s['depth'] for s in stages if s['decision'] == 'threshold']
    result = {'arm': args.arm, 'curriculum': args.curriculum, 'seed': args.seed, 'status': 'complete',
              'deepest_threshold_crossed': max(achieved, default=0), 'stopping_depth': depth,
              'stopping_accuracy': stages[-1]['final_accuracy'],
              'stop_reason': 'global_safety_cap' if total_step >= GLOBAL_CAP else 'completed_stage_ladder',
              'total_steps': total_step, 'stages': stages, 'input_tokens': tokens,
              'train_seconds': train_seconds, 'wall_seconds': time.time() - started,
              'suffix_floor_percent': FLOOR * 100, 'marginal_reference_nats': MARGINAL,
              'suffix_theorem': SUFFIX_THEOREM, 'finished_at': time.time()}
    assert total_step <= GLOBAL_CAP
    write_json(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curriculum', choices=['A', 'B'], required=True)
    parser.add_argument('--arm', choices=['e97', 'gdn2'], required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
