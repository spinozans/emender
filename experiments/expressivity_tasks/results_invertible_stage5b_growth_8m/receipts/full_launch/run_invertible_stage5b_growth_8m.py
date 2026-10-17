"""X5b: open-horizon, threshold-driven growth with calibrated stagnation."""
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
    build_model, digest, evaluate, padded_forward, write_json,
)
# Reuse the unchanged X5 data, checkpoint and capacity-retention implementations.
from experiments.expressivity_tasks.run_invertible_stage5_curriculum_8m import (
    archive_pools, checkpoint, matrix, prune,
)


from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage5b_growth_8m')
DEPTHS = (4, 8, 16, 32, 64, 128)
THRESHOLD = .90
STAGE_CEILING = 500000
GLOBAL_CAP = 2000000
WINDOW = 2000
VAL_GAIN = .001
LOSS_GAIN = .001
EVAL_EVERY = 500
FLOOR = 1 / 3721
MARGINAL = math.log(3721)
SUFFIX_THEOREM = ('For this invertible map, a predictor restricted to a proper driver suffix '
                  'with at least two omitted drivers has full-state accuracy 1/3721 '
                  '(0.026874%) under uniform independent inputs; this is a suffix-only '
                  'information bound, not a capacity bound on either recurrent model.')


class Stagnation:
    """Two consecutive joint plateaus between disjoint 2K-update windows."""

    def __init__(self):
        self.previous = None
        self.consecutive = 0

    def observe(self, stage_step, losses, best_accuracy):
        if not stage_step or stage_step % WINDOW:
            return None
        if len(losses) != stage_step or not all(math.isfinite(x) for x in losses[-WINDOW:]):
            raise ValueError('invalid stagnation loss window')
        mean_loss = float(np.mean(losses[-WINDOW:]))
        if mean_loss <= 0 or not math.isfinite(best_accuracy) or not 0 <= best_accuracy <= 1:
            raise ValueError('invalid stagnation observation')
        row = {'stage_step': stage_step, 'window_start': stage_step - WINDOW + 1,
               'window_end': stage_step, 'mean_train_loss': mean_loss,
               'best_accuracy': best_accuracy, 'best_val_gain': None,
               'relative_train_loss_decrease': None, 'joint_plateau': False}
        if self.previous is not None:
            row['best_val_gain'] = best_accuracy - self.previous['best_accuracy']
            row['relative_train_loss_decrease'] = (self.previous['mean_train_loss'] - mean_loss) / self.previous['mean_train_loss']
            row['joint_plateau'] = (row['best_val_gain'] < VAL_GAIN and
                                    row['relative_train_loss_decrease'] < LOSS_GAIN)
            self.consecutive = self.consecutive + 1 if row['joint_plateau'] else 0
        row['consecutive_joint_plateaus'] = self.consecutive
        row['stagnated'] = self.consecutive >= 2
        self.previous = row
        return row


def stage_decision(accuracy, stage_step, total_step, stagnated=False):
    if (not math.isfinite(accuracy) or not 0 <= accuracy <= 1 or
            not 0 <= stage_step <= STAGE_CEILING or not 0 <= total_step <= GLOBAL_CAP):
        raise ValueError('invalid growth observation')
    if accuracy >= THRESHOLD:
        return 'threshold'
    if stagnated:
        return 'stagnation'
    if total_step == GLOBAL_CAP:
        return 'global_safety_cap'
    if stage_step == STAGE_CEILING:
        return 'stage_safety_ceiling'
    return 'continue'


def run(args):
    import torch
    import torch.nn.functional as F
    import schedulefree
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    config = {**vars(args), 'depths': DEPTHS, 'threshold': THRESHOLD, 'stage_safety_ceiling': STAGE_CEILING,
              'hard_cap': GLOBAL_CAP, 'stagnation_window': WINDOW,
              'stagnation_val_gain': VAL_GAIN, 'stagnation_loss_gain': LOSS_GAIN,
              'stagnation_comparisons': 2,
              'stagnation_definition': 'disjoint windows; cumulative best-val gain and relative mean-loss decrease; two consecutive joint plateaus; earliest 6000; negative gains count plateau', 'eval_every': EVAL_EVERY, 'batch': 32, 'p': 61, 'vocab': 3721,
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
    latest_path = None
    started = time.time()
    with (out / 'training.jsonl').open('a', buffering=1) as log, (out / 'transitions.jsonl').open('a', buffering=1) as transitions:
        for depth in DEPTHS:
            losses, best, previous_best_path = [], None, None
            entry_step = total_step
            stagnation = Stagnation()
            remaining = min(STAGE_CEILING, GLOBAL_CAP - total_step)
            for stage_step in range(remaining + 1):
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
                if stage_step % EVAL_EVERY and stage_step != remaining:
                    continue
                opt.eval()  # Evaluate/save ScheduleFree x, restore y on the next training step.
                metrics = {str(d): evaluate(model, pools[d], 61) for d in DEPTHS if d <= depth}
                current = metrics[str(depth)]
                is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
                best_accuracy = max(current['accuracy'], best['accuracy'] if best else 0.)
                plateau = stagnation.observe(stage_step, losses, best_accuracy)
                decision = stage_decision(current['accuracy'], stage_step, total_step,
                                          plateau is not None and plateau['stagnated'])
                cp = None
                if is_best or stage_step % 1000 == 0 or decision != 'continue':
                    cp = checkpoint(out, model, opt, rng, total_step, depth, stage_step, config)
                    checkpoints[cp['path']] = cp
                    latest_path = cp['path']
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
                       'stagnation_window': plateau,
                       'train_loss_delta_marginal': float(np.mean(losses[-500:])) - MARGINAL if losses else None,
                       'wall_elapsed_s': time.time() - started,
                       'training_steps_per_second': total_step / train_seconds if train_seconds else None,
                       'train_loss_first500': float(np.mean(losses[:500])) if losses else None,
                       'train_loss_final500': float(np.mean(losses[-500:])) if losses else None,
                       'marginal_reference_nats': MARGINAL, 'suffix_floor_percent': FLOOR * 100}
                history.append(row)
                write_json(out / 'validation.json', history)
                write_json(out / 'checkpoints.json', checkpoints)
                print(json.dumps({'arm': args.arm, 'seed': args.seed, **row}), flush=True)
                prune(out, checkpoints, required | ({latest_path} if latest_path else set()))
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
                             'stagnation_window': plateau,
                             'safety_stop': decision in ('stage_safety_ceiling', 'global_safety_cap'),
                             'train_loss_first500': row['train_loss_first500'],
                             'train_loss_final500': row['train_loss_final500']}
                    stages.append(stage)
                    transitions.write(json.dumps({'event': 'stage_exit', **stage}) + '\n')
                    write_json(out / 'stages.json', stages)
                    write_json(out / 'growth_curve.json', {
                        'arm': args.arm, 'seed': args.seed, 'stages': stages,
                        'deepest_threshold_crossed': max((s['depth'] for s in stages if s['decision'] == 'threshold'), default=0),
                        'total_steps': total_step, 'suffix_floor_percent': FLOOR * 100,
                        'marginal_reference_nats': MARGINAL})
                    break
            if decision != 'threshold' or total_step >= GLOBAL_CAP:
                break
    achieved = [s['depth'] for s in stages if s['decision'] == 'threshold']
    result = {'arm': args.arm, 'seed': args.seed, 'status': 'complete',
              'frontier': max(achieved, default=0), 'stopping_depth': depth,
              'stopping_accuracy': stages[-1]['final_accuracy'],
              'stop_reason': 'completed_curriculum' if depth == 128 and decision == 'threshold' else ('global_safety_cap' if total_step >= GLOBAL_CAP and decision == 'threshold' else decision),
              'total_steps': total_step, 'stages': stages, 'input_tokens': tokens,
              'train_seconds': train_seconds, 'wall_seconds': time.time() - started,
              'suffix_floor_percent': FLOOR * 100, 'marginal_reference_nats': MARGINAL,
              'suffix_theorem': SUFFIX_THEOREM, 'finished_at': time.time()}
    assert total_step <= GLOBAL_CAP
    write_json(out / 'result.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97', 'gdn2'], required=True)
    parser.add_argument('--seed', type=int, choices=range(3), required=True)
    parser.add_argument('--out', required=True)
    run(parser.parse_args())
