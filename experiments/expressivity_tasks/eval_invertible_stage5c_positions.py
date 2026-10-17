"""X5cP: one frozen-T4 pass per already-selected best checkpoint, no training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage3_8m import (
    build_model, digest, finish_metrics, padded_forward, write_json,
)
from experiments.expressivity_tasks.run_invertible_stage2 import metric_counts

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage5c_growth_8m')
COUNT_KEYS = ('correct', 'count', 'x_correct', 'y_correct', 'out_of_range')


def position_counts(pred, target, mask, losses, p=61):
    """Driver1 is token index2; retain joint-argmax coordinate semantics."""
    if pred.shape != (512, 6) or target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError('expected frozen T4 shape (512, 6)')
    if mask[:, :2].any() or not mask[:, 2:].all() or losses.shape != (512, 4):
        raise ValueError('expected exactly four driver targets per sequence')
    rows = []
    for index in range(2, 6):
        m = np.zeros_like(mask)
        m[:, index] = True
        counts = metric_counts(pred, target, m, p)
        rows.append({'driver_position': index - 1, 'token_index': index,
                     **{k: counts[k] for k in COUNT_KEYS},
                     'loss': float(losses[:, index - 2].astype(np.float64).mean())})
    return rows


def reconcile(aggregate, positions, best, final):
    sums = {k: sum(row[k] for row in positions) for k in COUNT_KEYS}
    if sums != {k: aggregate[k] for k in COUNT_KEYS}:
        raise ValueError('position totals do not reconcile')
    keys = (*COUNT_KEYS, 'terminal_correct', 'terminal_count',
            'post_suffix7_correct', 'post_suffix7_count')
    differences = {k: aggregate[k] - best[k] for k in keys if aggregate[k] != best[k]}
    loss_delta = aggregate['loss'] - best['loss']
    return {'position_sums': sums, 'best_count_differences': differences,
            'best_counts_match': not differences, 'best_loss_delta': loss_delta,
            'best_loss_matches_at_1e-6': abs(loss_delta) <= 1e-6,
            'recorded_best': best, 'recorded_final': final,
            'best_minus_final_counts': {k: aggregate[k] - final[k] for k in keys},
            'best_minus_final_loss': aggregate['loss'] - final['loss'],
            'position_mean_loss_delta': sum(row['loss'] for row in positions) / 4 - aggregate['loss']}


def read_json(path):
    return json.loads(path.read_text())


def evaluate_run(root, name):
    import torch
    import torch.nn.functional as F

    run = root / name
    cfg, result = read_json(run / 'config.json'), read_json(run / 'result.json')
    stage = next(s for s in result['stages'] if s['depth'] == 4)
    best, final = stage['best'], stage['retention']['4']
    cp = best['checkpoint']
    if digest(cp['path']) != cp['sha256']:
        raise ValueError(f'{name}: best checkpoint hash mismatch')
    if digest(stage['checkpoint']['path']) != stage['checkpoint']['sha256']:
        raise ValueError(f'{name}: final checkpoint hash mismatch')
    data_receipt = read_json(run / 'data_receipt.json')
    data_path = run / 'frozen_validation.npz'
    if digest(data_path) != data_receipt['sha256']:
        raise ValueError(f'{name}: validation NPZ hash mismatch')
    with np.load(data_path) as data:
        pool = tuple(data[f'T4_{i}'] for i in range(3))
    hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
    if hashes != data_receipt['depths']['4']['array_sha256']:
        raise ValueError(f'{name}: T4 array hash mismatch')
    history = read_json(run / 'validation.json')
    best_row = next(row for row in history if row['step'] == best['step'] and row['depth'] == 4)
    if any(best_row['validation_by_depth']['4'][k] != best[k] for k in best_row['validation_by_depth']['4']):
        raise ValueError(f'{name}: recorded best/validation mismatch')
    if history[-1]['validation_by_depth']['4'] != final:
        raise ValueError(f'{name}: recorded final/validation mismatch')
    torch.manual_seed(cfg['seed'])
    model = build_model(cfg['arm'], 61).cuda()
    raw = torch.load(cp['path'], map_location='cpu', weights_only=False)
    if (raw['step'], raw['depth'], raw['stage_step'], raw['weights']) != (
            cp['step'], 4, cp['stage_step'], 'ScheduleFree eval x'):
        raise ValueError(f'{name}: checkpoint identity mismatch')
    model.load_state_dict(raw['model'], strict=True)
    del raw
    model.eval()
    predictions, losses = [], []
    totals, loss_sum = {}, 0.
    with torch.no_grad():
        for start in range(0, 512, 32):
            x, y, mask = [torch.from_numpy(a[start:start + 32]).cuda() for a in pool]
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = padded_forward(model, x)
            if not torch.isfinite(logits).all():
                raise FloatingPointError('nonfinite evaluation logits')
            pred = logits.argmax(-1).cpu().numpy()
            predictions.append(pred)
            loss_sum += F.cross_entropy(logits[mask].float(), y[mask], reduction='sum').item()
            losses.append(F.cross_entropy(logits[mask].float(), y[mask], reduction='none').reshape(32, 4).cpu().numpy())
            for key, value in metric_counts(pred, pool[1][start:start + 32], pool[2][start:start + 32]).items():
                totals[key] = totals.get(key, 0) + value
    positions = position_counts(np.concatenate(predictions), pool[1], pool[2], np.concatenate(losses))
    aggregate = finish_metrics(totals, loss_sum, 61)
    comparison = reconcile(aggregate, positions, best, final)
    receipt = {
        'task': 'X5cP', 'name': name, 'depth': 4, 'examples_per_position': 512,
        'selection': 'existing stage.best; accuracy descending, loss ascending; no re-selection',
        'weights': 'ScheduleFree eval x; model.eval(); no optimizer transform or training',
        'checkpoint': cp, 'recorded_final_checkpoint': stage['checkpoint'],
        'data': {'path': str(data_path), 'sha256': data_receipt['sha256'], 'T4_array_sha256': hashes},
        'sources': {str(p): digest(p) for p in [run / 'result.json', run / 'validation.json',
                    run / 'config.json', run / 'data_receipt.json', Path(__file__),
                    Path('experiments/expressivity_tasks/run_invertible_stage3_8m.py'),
                    Path('experiments/expressivity_tasks/run_invertible_stage2.py'),
                    Path('ndm/models/hybrid_ladder.py')]},
        'runtime': {'host': platform.node(), 'python': platform.python_version(),
                    'torch': torch.__version__, 'cuda': torch.version.cuda,
                    'gpu': torch.cuda.get_device_name(), 'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
                    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                    'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                    'batch': 32, 'forward_batches': 16, 'precision': 'bf16 autocast, fp32 parameters',
                    'padding': 'causal right-pad to16, slice before metrics'},
        'coordinate_definition': 'decode the same joint argmax as x=token//61, y=token%61; exclude invalid tokens',
        'positions': positions, 'aggregate': aggregate, 'reconciliation': comparison,
    }
    dest = root / 'receipts' / name / 'positional_receipts.json'
    write_json(dest, receipt)
    print(json.dumps({'name': name, 'joint_by_position': [r['correct'] for r in positions],
                      'best_counts_match': comparison['best_counts_match'],
                      'best_loss_delta': comparison['best_loss_delta']}), flush=True)
    if not comparison['best_counts_match'] or not comparison['best_loss_matches_at_1e-6']:
        raise RuntimeError(f'{name}: discrepancy retained in {dest}; stopping, no patch/rerun')


def main():
    import torch
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    for arm in ('e97', 'gdn2'):
        for seed in range(3):
            evaluate_run(args.root, f'{arm}_seed{seed}')


if __name__ == '__main__':
    main()
