"""X26c external reporting only: shipped IMM launchers and trainers stay byte-identical."""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from experiments.expressivity_tasks import run_merrill26_separators as base
from experiments.expressivity_tasks import run_merrill26_native_control as native

ROOT = Path('experiments/expressivity_tasks/results_stage26c_raw_defaults')
CARDS = {'rnn_integer_seed0': 4, 'delta_integer_seed0': 5}
DATA = native.ROOT / 'datasets'


def configure():
    base.ROOT = ROOT


def environment(path):
    from importlib.metadata import version
    base.write_json(path, {'python': sys.version, 'executable': sys.executable,
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'numpy': np.__version__, 'flash-linear-attention': version('flash-linear-attention'),
        'triton': version('triton'), 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'pip_freeze': subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)})


def confusion_metrics(confusion):
    c = np.asarray(confusion, dtype=np.int64)
    recalls = [float(c[i, i] / c[i].sum()) if c[i].sum() else None for i in range(2)]
    return {'count': int(c.sum()), 'accuracy': float(c.trace() / c.sum()),
        'confusion_true_by_pred': c.tolist(), 'class_recall': recalls,
        'balanced_accuracy': sum(recalls) / 2 if None not in recalls else None,
        'class_counts': c.sum(axis=1).tolist(),
        'majority_accuracy': float(c.sum(axis=1).max() / c.sum())}


def checkpoint_model(arm, checkpoint):
    m, a = native.source(arm), checkpoint['args']
    if a['teacher_force_state'] or a['batch_size'] != 256 or a['max_steps'] != 30000:
        raise ValueError('checkpoint is not registered raw defaults')
    if arm == 'rnn':
        model = m.ModelRNNFinalBinary(a['max_len'], a['d_model'], a['hidden'],
                                      a['layers'], a['dropout'], a['rnn_type'])
    else:
        model = m.ModelDeltaNetFinalBinary(a['max_len'], a['d_model'], a['layers'],
            a['heads'], a['dropout'], a['deltanet_mode'], a['allow_neg_eigval'], a['chunk_size'])
    model.load_state_dict(checkpoint['model'], strict=True)
    return model.cuda().eval()


@torch.no_grad()
def evaluate_split(model, arm, split, args):
    m = native.source(arm)
    ds = m.PreloadedNoModBinaryDataset(str(DATA / f'{split}_src.txt'),
                                      str(DATA / f'{split}_tgt.txt'), 'pm1', quiet=True)
    # Match the shipped evaluation bucket ordering/batch shapes, not the reconciled loop.
    loader = m.make_loader(ds, args['batch_size'], False, 0, 4, False,
                          args['bucket_size'], args['seed'] + 999,
                          teacher_force_state=False)
    confusion = np.zeros((2, 2), dtype=np.int64)
    by_length = {}
    for batch in loader:
        for f in fields(batch):
            setattr(batch, f.name, getattr(batch, f.name).cuda())
        with torch.amp.autocast('cuda', enabled=args['amp'],
                               dtype=m.pick_autocast_dtype(torch.device('cuda'), args['amp_dtype'])):
            logits = model(batch.tok_type, batch.tok_val_phi,
                           batch.lengths if arm == 'rnn' else batch.attn01)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('nonfinite frozen-checkpoint evaluation')
        predicted = (torch.sigmoid(logits[batch.y_mask]) >= .5).long().cpu().tolist()
        labels = batch.y.cpu().tolist()
        lengths = (batch.lengths - 1).cpu().tolist()
        for y, p, length in zip(labels, predicted, lengths):
            confusion[y, p] += 1
            by_length.setdefault(str(length), np.zeros((2, 2), dtype=np.int64))[y, p] += 1
    return {**confusion_metrics(confusion),
            'by_matrix_length': {k: confusion_metrics(v) for k, v in by_length.items()}}


def evaluate(arm):
    configure()
    reg = base.verify_registration()
    name = f'{arm}_integer_seed0'
    out = ROOT / name
    path = out / 'checkpoint.pt'
    sha = base.digest(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    model = checkpoint_model(arm, checkpoint)
    metrics = {split: evaluate_split(model, arm, split, checkpoint['args'])
               for split in ('val_bin0', 'test_bin0', 'test_bin1', 'test_bin2')}
    if base.digest(path) != sha:
        raise ValueError('checkpoint changed during evaluation')
    base.write_json(out / 'result.json', {'name': name, 'status': 'complete',
        'stop_reason': 'shipped pipeline exited successfully; frozen best-ID checkpoint evaluated',
        'checkpoint_sha256': sha, 'checkpoint_args': checkpoint['args'],
        'registration_sha256': base.digest(ROOT / 'registration.json'),
        'data_receipt_sha256': reg['data_receipt_sha256'], 'metrics': metrics,
        'scientific_verdict': 'pending registered three-outcome adjudication; raw accuracy is not proof',
        'finished_at': time.time()})


def smoke(arm, out):
    # Only smoke batch size changes; this function never runs in shipped training.
    native.BATCH = 256
    native.smoke(arm, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('rnn', 'delta'))
    parser.add_argument('--smoke')
    parser.add_argument('--environment')
    args = parser.parse_args()
    if args.environment:
        environment(args.environment)
    elif args.smoke:
        smoke(args.arm, args.smoke)
    else:
        evaluate(args.arm)
