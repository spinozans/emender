"""X26b native IMM task-code control; upstream collate/models/loss stay unmodified."""
from __future__ import annotations

import argparse
from dataclasses import fields
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

from experiments.expressivity_tasks import run_merrill26_separators as base
from experiments.expressivity_tasks.merrill26_replication_models import upstream, SOURCES
from experiments.expressivity_tasks.run_merrill26_replication import CurriculumSampler, BATCH, LR, WEIGHT_DECAY

ROOT = Path('experiments/expressivity_tasks/results_stage26b_native_control')
CARDS = {'rnn_integer_seed0': 4, 'delta_integer_seed0': 5}
SPLITS = ('train', 'val_bin0', 'test_bin0', 'test_bin1', 'test_bin2')


def configure():
    base.ROOT = ROOT


def source(arm):
    return upstream('imm_unbound', 'rnn' if arm == 'rnn' else 'deltanet')


def build_model(arm):
    module = source(arm)
    if arm == 'rnn':
        return module.ModelRNNFinalBinary(301, 256, 256, 2, .3, 'gru')
    return module.ModelDeltaNetFinalBinary(301, 256, 1, 4, .2,
                                          deltanet_mode='chunk', allow_neg_eigval=False)


def prepare():
    """Run the shipped generator with supervisor-approved flags, without balancing."""
    target = ROOT / 'datasets'
    if target.exists():
        raise ValueError('native dataset already exists; refuse regeneration')
    command = [os.sys.executable, str(SOURCES / 'imm_unbound/gen.py'), '--out_dir', str(target),
               '--seed', '0', '--T_train', '100', '--T_gap', '100', '--n_train_per_len', '700',
               '--n_val_per_len', '200', '--n_test_per_len', '100']
    subprocess.run(command, check=True)
    receipt = {'command': command, 'balanced': False, 'generator_sha256': base.digest(command[1]),
               'files': {str(p): base.digest(p) for p in sorted(target.glob('*.txt'))}, 'splits': {}}
    for split in SPLITS:
        y = np.loadtxt(target / f'{split}_tgt.txt', dtype=np.int64)
        receipt['splits'][split] = {'count': len(y), 'labels': np.bincount(y, minlength=2).tolist(),
                                    'majority_accuracy': float(max(np.bincount(y)) / len(y))}
    base.write_json(ROOT / 'receipts/data.json', receipt)


def dataset(arm, split):
    return source(arm).PreloadedNoModBinaryDataset(str(ROOT / 'datasets' / f'{split}_src.txt'),
        str(ROOT / 'datasets' / f'{split}_tgt.txt'), 'pm1', quiet=True)


def forward(model, arm, items, auxiliary=False):
    module = source(arm)
    batch = module.collate_batch(items, teacher_force_state=False)
    device = next(model.parameters()).device
    for field in fields(batch):
        setattr(batch, field.name, getattr(batch, field.name).to(device))
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(batch.tok_type, batch.tok_val_phi,
                       batch.lengths if arm == 'rnn' else batch.attn01)
        loss = module.loss_final_bce(logits, batch.y, batch.y_mask)
        if auxiliary:
            loss = loss + .3 * module.loss_aux_bce(logits, batch.aux_label, batch.aux_mask)
    if not torch.isfinite(logits).all() or not torch.isfinite(loss):
        raise FloatingPointError('nonfinite native output/loss')
    picked = logits[batch.y_mask].float()
    return loss, picked, batch.y


@torch.no_grad()
def evaluate(model, arm, ds):
    model.eval()
    total, correct, count = 0., 0, 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    by_length = {}
    order = np.argsort(ds.Ts, kind='stable')
    for start in range(0, len(order), BATCH):
        items = [ds[int(i)] for i in order[start:start + BATCH]]
        loss, logits, labels = forward(model, arm, items)
        pred = (logits >= 0).long()
        total += loss.item() * len(items)
        correct += int((pred == labels).sum())
        count += len(items)
        for item, y, p in zip(items, labels.cpu().tolist(), pred.cpu().tolist()):
            confusion[y, p] += 1
            row = by_length.setdefault(str(item['T']), {'count': 0, 'correct': 0})
            row['count'] += 1
            row['correct'] += int(y == p)
    recalls = [float(confusion[i, i] / confusion[i].sum()) if confusion[i].sum() else None
               for i in range(2)]
    return {'count': count, 'correct': correct, 'accuracy': correct / count, 'loss': total / count,
            'confusion_true_by_pred': confusion.tolist(), 'class_recall': recalls,
            'balanced_accuracy': sum(recalls) / 2 if None not in recalls else None,
            'by_matrix_length': by_length}


def smoke(arm, out):
    """Real max-bin batch128 backward and eval, with zero optimizer updates."""
    torch.manual_seed(239999)
    module = source(arm)
    spec = importlib.util.spec_from_file_location('x26b_generator', SOURCES / 'imm_unbound/gen.py')
    generator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(generator)
    import random
    rng = random.Random(239999)
    items = []
    for i in range(BATCH):
        n = 300 - i % 3
        mats = np.stack([generator.sample_pm1_matrix(rng) for _ in range(n)])
        items.append({'T': n, 'mats_pm1': torch.from_numpy(mats.reshape(n, 9)).short(),
            'state_prev_phi': torch.from_numpy(module.signed_log1p_np(
                module.compute_states_prev_nomod(mats.reshape(n, 9)))),
            'y': generator.label_for_seq(list(mats))})
    model = build_model(arm).cuda().train()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    loss, _, _ = forward(model, arm, items, True)
    loss.backward()
    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    model.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad():
        forward(model, arm, items)
    torch.cuda.synchronize()
    row = {'status': 'passed', 'arm': arm, 'task': 'integer', 'n': 300, 'tokens': 301,
           'batch': BATCH, 'optimizer_updates': 0, 'grad_norm': float(grad),
           'seconds': time.monotonic() - started, 'parameters': sum(p.numel() for p in model.parameters()),
           'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    base.write_json(out, row)
    print(json.dumps(row), flush=True)


def run(arm):
    configure()
    reg = base.verify_registration()
    name = f'{arm}_integer_seed0'
    if name not in reg['runs'] or reg['cards'] != CARDS:
        raise ValueError('unauthorized native control')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    started, step, best_step = time.monotonic(), 0, None
    try:
        gate = json.loads((ROOT / 'receipts' / f'smoke_{arm}_integer.json').read_text())
        if gate['status'] != 'passed' or gate['optimizer_updates'] != 0:
            raise ValueError('unqualified smoke')
        torch.manual_seed(0)
        model = build_model(arm).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        train, validation = dataset(arm, 'train'), dataset(arm, 'val_bin0')
        train.lengths = np.asarray(train.Ts)
        sampler = CurriculumSampler(train, np.random.default_rng(23000), True)
        config = {'name': name, 'seed': 0, 'batch': BATCH, 'clip': 1., 'maximum_steps': base.CEILING,
            'optimizer': {'name': 'AdamW', 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                          'betas': [.9, .999], 'eps': 1e-8},
            'curriculum': 'T5->100/150 eligible-pool epochs; epoch tails carried to128',
            'auxiliary_weight': .3, 'teacher_force_state': False, 'max_len': 301,
            'parameters': sum(p.numel() for p in model.parameters()),
            'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            'registration_sha256': base.digest(ROOT / 'registration.json'),
            'data_receipt_sha256': base.digest(ROOT / 'receipts/data.json'),
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'host': os.uname().nodename}
        base.write_json(out / 'config.json', config)
        history, best, streak = [], None, 0
        with (out / 'training.jsonl').open('a', buffering=1) as log:
            for step in range(base.CEILING + 1):
                base.safety(out, started)
                if step:
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    items = [train[int(i)] for i in sampler.draw(BATCH)]
                    loss, _, _ = forward(model, arm, items, True)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    optimizer.step()
                    log.write(json.dumps({'step': step, 'loss': loss.item(), 'grad_norm': float(grad),
                        'lr': LR, 'epoch': sampler.epoch, 'curriculum_max_t': sampler.max_t,
                        'elapsed_s': time.monotonic() - started}) + '\n')
                if step % base.EVAL_EVERY and step != base.CEILING:
                    continue
                current = evaluate(model, arm, validation)
                key = (-current['accuracy'], current['loss'])
                if best is None or key < best:
                    best, best_step = key, step
                    path = out / 'best.pt'
                    torch.save({'model': model.state_dict(), 'step': step, 'config': config},
                               path.with_suffix('.pt.tmp'))
                    path.with_suffix('.pt.tmp').replace(path)
                    base.write_json(out / 'checkpoint.json', {'path': str(path), 'step': step,
                                                             'sha256': base.digest(path)})
                streak = streak + 1 if step and current['correct'] == current['count'] else 0
                row = {'step': step, 'validation': current, 'best_step': best_step,
                       'perfect_streak': streak, 'elapsed_s': time.monotonic() - started}
                history.append(row)
                base.write_json(out / 'validation.json', history)
                print(json.dumps({'name': name, **row}), flush=True)
                if streak >= 3:
                    break
        reason = 'three_perfect_validation_evaluations' if streak >= 3 else '60000_update_cap'
        checkpoint = json.loads((out / 'checkpoint.json').read_text())
        if base.digest(checkpoint['path']) != checkpoint['sha256']:
            raise ValueError('checkpoint changed')
        model.load_state_dict(torch.load(checkpoint['path'], weights_only=False)['model'])
        tests = {}
        for split in SPLITS[2:]:
            base.safety(out, started)
            tests[split] = evaluate(model, arm, dataset(arm, split))
            base.write_json(out / 'tests.json', tests)
        status = 'complete'
    except (torch.OutOfMemoryError, OSError, TimeoutError) as exc:
        status, reason, tests = 'censored', repr(exc), None
    except Exception as exc:
        base.write_json(out / 'result.json', {'name': name, 'status': 'error', 'step': step,
            'stop_reason': repr(exc), 'scientific_verdict': 'not adjudicated'})
        raise
    base.write_json(out / 'result.json', {'name': name, 'status': status, 'step': step,
        'stop_reason': reason, 'best_step': best_step, 'tests': tests,
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time(),
        'scientific_verdict': 'pending contract comparison' if status == 'complete' else 'right-censored'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['rnn', 'delta'])
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--smoke', type=Path)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.arm is None:
        parser.error('--arm required for smoke/training')
    elif args.smoke:
        smoke(args.arm, args.smoke)
    else:
        run(args.arm)
