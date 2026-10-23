"""X23E registered separator rungs; no optimizer update is allowed by --smoke."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from experiments.expressivity_tasks.tasks.merrill26_separators import (
    TASKS, VOCABS, CLASSES, SPLITS, Dataset, make_dataset, sample, encode,
)

ROOT = Path('experiments/expressivity_tasks/results_stage23e_separators')
CEILING, EVAL_EVERY, BATCH = 60000, 500, 64
WEIGHT_DECAY = .01
FREE_BYTES, RUN_BYTES, WALL_SECONDS = 400 * 1024**3, 40 * 1024**3, 7 * 86400


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(row, indent=2) + '\n')
    tmp.replace(path)


def verify_registration():
    path = ROOT / 'registration.json'
    reg = json.loads(path.read_text())
    sources = {str(path): digest(path), **reg['source_sha256'], **reg['dependency_sha256']}
    for name, sha in sources.items():
        if digest(name) != sha:
            raise ValueError('registered bytes changed: ' + name)
        if not Path(name).is_absolute():
            blob = subprocess.check_output(['git', 'show', 'HEAD:' + name])
            if hashlib.sha256(blob).hexdigest() != sha:
                raise ValueError('not committed: ' + name)
    return reg


def build_model(arm, task):
    from ndm.models.hybrid_ladder import HybridLadderLM
    model = HybridLadderLM(vocab_size=VOCABS[task], dim=256, depth=4,
        layer_pattern=['E97' if arm == 'e97' else 'fla-gdn'],
        layer_kwargs=[{'state_activation': 'tanh', 'raw_write': False} if arm == 'e97'
                      else {'allow_neg_eigval': True}], n_heads=32, n_state=32,
        expansion=1.0 if arm == 'e97' else 8.625, use_triton_e88=arm == 'e97', mlp_ratio=2.)
    if arm == 'e97':
        for layer in model.layers:
            layer.recurrent_state_precision = 'legacy'
            layer.fused_inference = True
    return model


def learning_rate(step):
    if not 0 <= step <= CEILING:
        raise ValueError('step outside registration')
    if step <= 5000:
        return .001 * step / 5000
    return .00001 + .5 * (.001 - .00001) * (1 + math.cos(math.pi * (step - 5000) / (CEILING - 5000)))


def batches(dataset, indices, budget):
    """Length-aware microbatches; padding never changes supervised causal prefixes."""
    rows, maximum = [], 0
    for index in indices:
        row = dataset.example(int(index))
        length = len(row[0])
        if rows and max(maximum, length) * (len(rows) + 1) > budget:
            yield rows
            rows, maximum = [], 0
        rows.append(row)
        maximum = max(maximum, length)
    if rows:
        yield rows


def forward_rows(model, rows, task):
    import torch
    import torch.nn.functional as F
    lengths = [len(x) for x, _ in rows]
    padded = (max(lengths) + 15) // 16 * 16
    pad = 0 if task == 'graph' else 3
    x = np.full((len(rows), padded), pad, dtype=np.int64)
    for i, (tokens, _) in enumerate(rows):
        x[i, :len(tokens)] = tokens
    target = torch.tensor([y for _, y in rows], device='cuda', dtype=torch.long)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(torch.from_numpy(x).cuda())[
            torch.arange(len(rows), device='cuda'), torch.tensor(lengths, device='cuda') - 1].float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError('nonfinite logits')
    if CLASSES[task] == 2:
        margin = logits[:, 1] - logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(margin, target.float(), reduction='sum')
        prediction = (margin > 0).long()
    else:
        loss = F.cross_entropy(logits, target, reduction='sum')
        prediction = logits.argmax(-1)
    return loss, prediction, target


def evaluate(model, dataset, budget):
    import torch
    model.eval()
    correct = loss_sum = count = 0
    by_length = {}
    with torch.no_grad():
        for rows in batches(dataset, range(len(dataset)), budget):
            loss, pred, y = forward_rows(model, rows, dataset.task)
            hits = (pred == y).cpu().tolist()
            correct += sum(hits)
            loss_sum += loss.item()
            count += len(rows)
            for (tokens, _), hit in zip(rows, hits):
                key = str(len(tokens))
                entry = by_length.setdefault(key, {'correct': 0, 'count': 0})
                entry['correct'] += int(hit)
                entry['count'] += 1
    return {'correct': correct, 'count': count, 'accuracy': correct / count,
            'loss': loss_sum / count, 'by_token_length': by_length}


def safety(out, started):
    if shutil.disk_usage(out).free < FREE_BYTES:
        raise OSError('disk guard: less than 400 GiB free')
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) > RUN_BYTES:
        raise OSError('disk guard: rung exceeds 40 GiB')
    if time.monotonic() - started > WALL_SECONDS:
        raise TimeoutError('seven-day wall guard')


def data(task, split):
    """Shared immutable compact pools, never materialize unary corpora on disk."""
    folder = ROOT / 'datasets' / task
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f'{split}.npz'
    with (folder / '.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        receipt = path.with_suffix('.json')
        if not path.exists():
            if receipt.exists():
                raise ValueError('receipted data disappeared; refuse regeneration')
            pool = make_dataset(task, *SPLITS[split])
            tmp = path.with_suffix('.npz.tmp')
            pool.save(tmp)
            tmp.replace(path)
            write_json(receipt, {'task': task, 'split': split, 'sha256': digest(path),
                'spec': SPLITS[split], 'labels': np.bincount(pool.labels, minlength=CLASSES[task]).tolist(),
                'lengths_sha256': hashlib.sha256(pool.lengths.tobytes()).hexdigest(),
                'created_at': time.time()})
        if digest(path) != json.loads(receipt.read_text())['sha256']:
            raise ValueError('dataset identity changed')
    return Dataset.load(task, path)


def smoke(arm, task, out):
    """Untrained max-bin forward/backward, then untrained eval; zero updates."""
    import torch
    torch.manual_seed(239999)
    model = build_model(arm, task).cuda()
    torch.cuda.reset_peak_memory_stats()
    if task == 'graph':
        # Worst unary sequence: one path 0->1->...->300, not a random shorter graph.
        payload = np.zeros(301, dtype=np.int8)
        y = 1
    else:
        payload, y = sample(task, 300, np.random.default_rng(239999), 1 if task != 'mod97' else None)
    tokens = encode(task, payload)
    started = time.monotonic()
    model.train()
    loss, _, _ = forward_rows(model, [(tokens, y)], task)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        forward_rows(model, [(tokens, y)], task)
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    row = {'arm': arm, 'task': task, 'n': 300, 'tokens': len(tokens), 'batch': 1,
        'optimizer_updates': 0, 'peak_allocated_bytes': peak, 'seconds': time.monotonic() - started,
        'parameters': sum(p.numel() for p in model.parameters()), 'status': 'passed'}
    write_json(out, row)
    print(json.dumps(row), flush=True)


def run(arm, task):
    import torch
    name = f'{arm}_{task}_seed0'
    reg = verify_registration()
    if name not in reg['runs']:
        raise ValueError('unauthorized rung')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    step, reason, best_step, history, tests = 0, None, None, [], None
    try:
        smoke_row = json.loads((ROOT / 'receipts' / f'smoke_{arm}_{task}.json').read_text())
        if smoke_row['status'] != 'passed' or smoke_row['optimizer_updates'] != 0:
            raise RuntimeError('max-bin smoke did not pass')
        torch.manual_seed(0)
        model = build_model(arm, task).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=0., betas=(.9, .999), eps=1e-8, weight_decay=WEIGHT_DECAY)
        torch.cuda.reset_peak_memory_stats()
        rng = np.random.default_rng(23000)
        train, validation = data(task, 'train'), data(task, 'validation')
        config = {'name': name, 'architecture': arm, 'task': task, 'seed': 0,
            'optimizer': {'name': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8,
                          'weight_decay': WEIGHT_DECAY, 'lr_update1': learning_rate(1),
                          'lr_update5000': learning_rate(5000), 'lr_update60000': learning_rate(60000)},
            'registration_sha256': digest(ROOT / 'registration.json'),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'parameters': sum(p.numel() for p in model.parameters()),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'host': os.uname().nodename,
            'datasets': {s: json.loads((ROOT / 'datasets' / task / f'{s}.json').read_text())
                         for s in ('train', 'validation')}}
        write_json(out / 'config.json', config)
        best, streak = None, 0
        # Bound padded activations while keeping exactly 64 examples per optimizer update.
        budget = 8192 if task == 'graph' else 16384
        with (out / 'training.jsonl').open('a', buffering=1) as log:
            for step in range(CEILING + 1):
                safety(out, started)
                if step:
                    model.train()
                    opt.zero_grad(set_to_none=True)
                    indices = rng.integers(0, len(train), BATCH)
                    total_loss = 0.
                    for rows in batches(train, indices, budget):
                        loss, _, _ = forward_rows(model, rows, task)
                        (loss / BATCH).backward()
                        total_loss += loss.item()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    for group in opt.param_groups:
                        group['lr'] = learning_rate(step)
                    opt.step()
                    log.write(json.dumps({'step': step, 'loss': total_loss / BATCH,
                        'lr': learning_rate(step), 'grad_norm': grad.item(),
                        'elapsed_s': time.monotonic() - started,
                        'peak_allocated_bytes': torch.cuda.max_memory_allocated()}) + '\n')
                if step % EVAL_EVERY and step != CEILING:
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
                    write_json(out / 'checkpoint.json', {'path': str(path), 'sha256': digest(path), 'step': step})
                streak = streak + 1 if step and current['correct'] == current['count'] else 0
                row = {'step': step, 'validation': current, 'best_step': best_step,
                       'perfect_streak': streak, 'elapsed_s': time.monotonic() - started}
                history.append(row)
                write_json(out / 'validation.json', history)
                print(json.dumps({'name': name, **row}), flush=True)
                if streak >= 3:
                    reason = 'three_perfect_validation_evaluations'
                    break
            reason = reason or '60000_update_cap'
        # OOD has no role in stopping, LR, or checkpoint selection.
        cp = json.loads((out / 'checkpoint.json').read_text())
        if digest(cp['path']) != cp['sha256']:
            raise ValueError('checkpoint bytes changed')
        model.load_state_dict(torch.load(cp['path'], map_location='cuda', weights_only=False)['model'])
        tests = {}
        for split in ('test_id', 'test_ood1', 'test_ood2'):
            safety(out, started)
            tests[split] = evaluate(model, data(task, split), budget)
            write_json(out / 'tests.json', tests)
        status = 'complete'
    except (torch.OutOfMemoryError, OSError, TimeoutError) as exc:
        status, reason = 'censored', f'infrastructure: {type(exc).__name__}: {exc}'
    except Exception as exc:
        # A software failure is recorded, not misreported as an expressivity outcome.
        write_json(out / 'result.json', {'name': name, 'status': 'error', 'step': step,
            'stop_reason': repr(exc), 'scientific_verdict': 'not adjudicated'})
        raise
    write_json(out / 'result.json', {'name': name, 'status': status, 'step': step,
        'stop_reason': reason, 'best_step': best_step, 'tests': tests,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time(),
        'scientific_verdict': 'pending analysis' if status == 'complete' else 'not adjudicated; right-censored'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['e97', 'gdn2'], required=True)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    if args.smoke:
        smoke(args.arm, args.task, args.smoke)
    else:
        run(args.arm, args.task)
