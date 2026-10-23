"""X26 partial replication under reconciled sources; X23E generators unchanged."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.expressivity_tasks import run_merrill26_separators as base
from experiments.expressivity_tasks.merrill26_replication_models import build_model
from experiments.expressivity_tasks.tasks.merrill26_separators import matrix_product

ROOT = Path('experiments/expressivity_tasks/results_stage26_replication')
COMPARISON_ROOT = Path('experiments/expressivity_tasks/results_stage23e_separators')
CARDS = {'rnn_graph_seed0': 2, 'delta_graph_seed0': 3,
         'rnn_integer_seed0': 4, 'delta_integer_seed0': 5}
BATCH, LR, WEIGHT_DECAY = 128, 1e-4, 1e-4


def configure():
    base.ROOT = ROOT


def link_datasets():
    target = COMPARISON_ROOT.resolve() / 'datasets'
    if not target.is_dir():
        raise ValueError('X23E shared datasets unavailable')
    link = ROOT / 'datasets'
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target, target_is_directory=True)
    if link.resolve() != target:
        raise ValueError('X26 must use the actual X23E dataset directory')


class CurriculumSampler:
    """Repo's T=5->100/150-epoch curriculum on the unchanged frozen pool.

    Epoch = one shuffled pass over eligible examples. Carry a short epoch tail
    into the next one to retain exactly 128 examples per optimizer update.
    """
    def __init__(self, dataset, rng, curriculum):
        self.dataset, self.rng, self.curriculum = dataset, rng, curriculum
        self.epoch, self.position, self.order = -1, 0, np.array([], dtype=np.int64)
        self.max_t = 100

    def draw(self, count):
        result = []
        while len(result) < count:
            if self.position == len(self.order):
                self.epoch += 1
                self.max_t = max(5, round(5 + 95 * min(self.epoch / 150, 1))) if self.curriculum else 100
                self.order = self.rng.permutation(np.flatnonzero(self.dataset.lengths <= self.max_t))
                self.position = 0
                if not len(self.order):
                    raise ValueError('empty curriculum epoch')
            take = min(count - len(result), len(self.order) - self.position)
            result.extend(self.order[self.position:self.position + take])
            self.position += take
        return result


def auxiliary_targets(tokens):
    """Exact >0 prefix targets only at completed matrices; no model-input leak."""
    matrices = (np.asarray(tokens, dtype=np.int64) - 1).reshape(-1, 3, 3)
    product = np.eye(3, dtype=object)
    targets = []
    for matrix in matrices[:-1]:
        product = product @ matrix
        targets.append(int(product[0, 0] > 0))
    return targets


def forward_rows(model, rows, task, auxiliary=False, auxiliary_count=1):
    device = next(model.parameters()).device
    lengths = torch.tensor([len(x) for x, _ in rows], device=device)
    padded = (int(lengths.max()) + 15) // 16 * 16
    x = np.full((len(rows), padded), 0 if task == 'graph' else 3, dtype=np.int64)
    for i, (tokens, _) in enumerate(rows):
        x[i, :len(tokens)] = tokens
    targets = torch.tensor([y for _, y in rows], device=device, dtype=torch.long)
    context = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()
    with context:
        logits = model(torch.from_numpy(x).to(device), lengths).float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite logits; upstream nan_to_num is deliberately not used')
    margin = logits[:, 1] - logits[:, 0] if task == 'graph' else logits[
        torch.arange(len(rows), device=device), lengths - 1]
    loss = F.binary_cross_entropy_with_logits(margin, targets.float(), reduction='sum')
    # The main term is summed over examples; caller divides by the effective batch.
    if auxiliary:
        prefix_logits, prefix_targets = [], []
        for i, (tokens, _) in enumerate(rows):
            labels = auxiliary_targets(tokens)
            if labels:
                prefix_logits.append(logits[i, 8:len(tokens) - 9:9])
                prefix_targets.extend(labels)
        if prefix_logits:
            aux = F.binary_cross_entropy_with_logits(torch.cat(prefix_logits),
                torch.tensor(prefix_targets, device=device, dtype=torch.float32), reduction='sum')
            loss = loss + .3 * BATCH * aux / max(auxiliary_count, 1)
    return loss, (margin >= 0).long(), targets


def evaluate(model, dataset, budget):
    # Reuse the reviewed length-stratified evaluator, not its HybridLadder forward.
    old = base.forward_rows
    base.forward_rows = forward_rows
    try:
        return base.evaluate(model, dataset, budget)
    finally:
        base.forward_rows = old


def smoke(arm, task, out):
    torch.manual_seed(239999)
    model = build_model(arm, task).cuda()
    torch.cuda.reset_peak_memory_stats()
    payload, y = (np.zeros(301, dtype=np.int8), 1) if task == 'graph' else base.sample(
        task, 300, np.random.default_rng(239999), 1)
    tokens = base.encode(task, payload)
    started = time.monotonic()
    model.train()
    loss, _, _ = forward_rows(model, [(tokens, y)], task, auxiliary=task == 'integer',
                               auxiliary_count=299)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        forward_rows(model, [(tokens, y)], task)
    torch.cuda.synchronize()
    row = {'arm': arm, 'task': task, 'n': 300, 'tokens': len(tokens), 'batch': 1,
           'optimizer_updates': 0, 'status': 'passed', 'grad_norm': float(grad),
           'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
           'seconds': time.monotonic() - started,
           'parameters': sum(p.numel() for p in model.parameters())}
    base.write_json(out, row)
    print(json.dumps(row), flush=True)


def run(arm, task):
    configure()
    link_datasets()
    reg = base.verify_registration()
    name = f'{arm}_{task}_seed0'
    if name not in CARDS or name not in reg['runs']:
        raise ValueError('unauthorized rung')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    step, best_step, tests, history = 0, None, None, []
    try:
        gate = json.loads((ROOT / 'receipts' / f'smoke_{arm}_{task}.json').read_text())
        if gate['status'] != 'passed' or gate['optimizer_updates'] != 0:
            raise ValueError('zero-update max-bin qualification required')
        torch.manual_seed(0)
        model = build_model(arm, task).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        rng = np.random.default_rng(23000)
        train, validation = base.data(task, 'train'), base.data(task, 'validation')
        sampler = CurriculumSampler(train, rng, task == 'integer')
        config = {'name': name, 'claim': 'partial replication under reconciled sources',
            'seed': 0, 'batch': BATCH, 'clip': 1., 'maximum_steps': base.CEILING,
            'optimizer': {'name': 'AdamW', 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                          'betas': [.9, .999], 'eps': 1e-8},
            'auxiliary_weight': .3 if task == 'integer' else 0.,
            'curriculum': 'T5->100 over 150 eligible-pool epochs' if task == 'integer' else None,
            'registration_sha256': base.digest(ROOT / 'registration.json'),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'host': os.uname().nodename,
            'datasets': {s: json.loads((ROOT / 'datasets' / task / f'{s}.json').read_text())
                         for s in ('train', 'validation')}}
        base.write_json(out / 'config.json', config)
        best, streak, reason = None, 0, None
        budget = 8192 if task == 'graph' else 16384
        with (out / 'training.jsonl').open('a', buffering=1) as log:
            for step in range(base.CEILING + 1):
                base.safety(out, started)
                if step:
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    indices = sampler.draw(BATCH)
                    aux_count = sum(int(train.lengths[i]) - 1 for i in indices)
                    total_loss = 0.
                    for rows in base.batches(train, indices, budget):
                        loss, _, _ = forward_rows(model, rows, task, auxiliary=task == 'integer',
                                                  auxiliary_count=aux_count)
                        (loss / BATCH).backward()
                        total_loss += loss.item()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    log.write(json.dumps({'step': step, 'loss': total_loss / BATCH, 'lr': LR,
                        'grad_norm': grad.item(), 'epoch': sampler.epoch, 'curriculum_max_t': sampler.max_t,
                        'elapsed_s': time.monotonic() - started,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated()}) + '\n')
                if step % base.EVAL_EVERY and step != base.CEILING:
                    continue
                current = evaluate(model, validation, budget)
                key = (-current['accuracy'], current['loss'])
                if best is None or key < best:
                    best, best_step = key, step
                    path = out / 'best.pt'
                    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(),
                        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
                        'cuda_rng': torch.cuda.get_rng_state_all(), 'step': step, 'config': config},
                        path.with_suffix('.pt.tmp'))
                    path.with_suffix('.pt.tmp').replace(path)
                    base.write_json(out / 'checkpoint.json', {'path': str(path), 'sha256': base.digest(path), 'step': step})
                streak = streak + 1 if step and current['correct'] == current['count'] else 0
                row = {'step': step, 'validation': current, 'best_step': best_step,
                       'perfect_streak': streak, 'elapsed_s': time.monotonic() - started}
                history.append(row)
                base.write_json(out / 'validation.json', history)
                print(json.dumps({'name': name, **row}), flush=True)
                if streak >= 3:
                    reason = 'three_perfect_validation_evaluations'
                    break
        reason = reason or '60000_update_cap'
        checkpoint = json.loads((out / 'checkpoint.json').read_text())
        if base.digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('checkpoint bytes changed')
        model.load_state_dict(torch.load(checkpoint['path'], map_location='cuda', weights_only=False)['model'])
        tests = {}
        for split in ('test_id', 'test_ood1', 'test_ood2'):
            base.safety(out, started)
            tests[split] = evaluate(model, base.data(task, split), budget)
            base.write_json(out / 'tests.json', tests)
        status = 'complete'
    except (torch.OutOfMemoryError, OSError, TimeoutError) as exc:
        status, reason = 'censored', f'infrastructure: {type(exc).__name__}: {exc}'
    except Exception as exc:
        base.write_json(out / 'result.json', {'name': name, 'status': 'error', 'step': step,
            'stop_reason': repr(exc), 'scientific_verdict': 'not adjudicated'})
        raise
    base.write_json(out / 'result.json', {'name': name, 'status': status, 'step': step,
        'stop_reason': reason, 'best_step': best_step, 'tests': tests,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time(),
        'scientific_verdict': 'pending analysis' if status == 'complete' else 'not adjudicated; right-censored'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['rnn', 'delta', 'rwkv'], required=True)
    parser.add_argument('--task', choices=['graph', 'integer'], required=True)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    if args.smoke:
        smoke(args.arm, args.task, args.smoke)
    else:
        run(args.arm, args.task)
