#!/usr/bin/env python3
"""X13 STEP 0: hash-pinned champion, historical runner paths, no training."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from unittest.mock import patch

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as x10
from experiments.expressivity_tasks import run_invertible_stage12_separation_8m as x12

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage13_zerocheck_8m')
CHAMPION = Path('experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m/adamw_seed0/step_468000.pt')
CHAMPION_SHA256 = '0501a81ff50e34017a973f400eec5975faaf5bd9d84df64d6fa84b51ed6c9fa7'


def array_hashes(pool):
    return [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]


def cpu_path(module, batch):
    """Execute the actual loader/padder; replace ONLY device transfer/autocast."""
    import torch
    captured = {}

    class Capture(torch.nn.Module):
        def forward(self, tokens):
            captured['model_input'] = tokens.clone().numpy()
            return torch.zeros((*tokens.shape, 1))

    with patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self), \
            patch.object(torch.amp, 'autocast', lambda *a, **kw: nullcontext()):
        logits, targets, mask = module.x7.forward_batch(Capture(), batch)
    return captured['model_input'], targets.numpy(), mask.numpy(), tuple(logits.shape)


def scalar_decode(inputs, targets, mask):
    """Independent integer arithmetic oracle, not the task's trajectory helper."""
    tokens = []
    x, y = int(inputs[0]) - 61, int(inputs[1]) - 122
    assert 0 <= x < 61 and 0 <= y < 61
    for j, token in enumerate(inputs):
        if j < 2:
            assert not mask[j] and targets[j] == 0
            tokens.append({'token_index': j, 'role': 'initial_x' if j == 0 else 'initial_y',
                           'input': int(token), 'mask': False, 'target': int(targets[j])})
        else:
            assert 0 <= token < 61 and mask[j]
            x, y = y, (x + y * y + int(token)) % 61
            assert int(targets[j]) == 61 * x + y
            tokens.append({'token_index': j, 'driver_position': j - 1, 'role': 'driver',
                           'input': int(token), 'mask': True, 'state': [x, y],
                           'target': int(targets[j])})
    return tokens


def equivalence(pool):
    indices = [0, 1, 2, 7, 63, 127, 255, 511]
    left_pool = x12.x7.frozen_pools(0, 64, 61)['val']
    assert array_hashes(left_pool) == array_hashes(pool)
    left_batch = tuple(a[indices] for a in left_pool)
    right_batch = tuple(a[indices] for a in pool)
    left, right = cpu_path(x12, left_batch), cpu_path(x10, right_batch)
    differences = []
    for name, a, b in zip(('padded_input', 'targets', 'mask'), left[:3], right[:3]):
        if not np.array_equal(a, b):
            differences.append({'field': name, 'stage12': a.tolist(), 'stage10': b.tolist()})
    if left[3] != right[3]:
        differences.append({'field': 'logit_shape', 'stage12': left[3], 'stage10': right[3]})
    examples = []
    for row, index in enumerate(indices):
        examples.append({'pool_index': index, 'tokens': scalar_decode(*[a[row] for a in left_batch]),
                         'stage12_padded_input': left[0][row].tolist(),
                         'stage10_padded_input': right[0][row].tolist(),
                         'stage12_targets': left[1][row].tolist(), 'stage10_targets': right[1][row].tolist(),
                         'stage12_mask': left[2][row].tolist(), 'stage10_mask': right[2][row].tolist()})
    assert left[0].shape == (8, 80) and np.all(left[0][:, 66:] == 0)
    assert left[3] == (8, 66, 1)
    return {'clean': not differences, 'differences': differences, 'examples': examples,
            'pool_array_sha256': array_hashes(pool),
            'same_forward_callable': x12.x7.forward_batch is x10.x7.forward_batch,
            'same_pool_callable': x12.x7.frozen_pools is x10.x7.frozen_pools,
            'same_task_class': x12.x10.InvertibleQuadraticTask is x10.InvertibleQuadraticTask,
            'same_padder_callable': x12.x7.padded_forward is x10.x7.padded_forward,
            'cpu_substitutions': 'cuda transfer -> identity; CUDA autocast -> nullcontext; capturing model only; actual loader/padder unchanged'}


def audit():
    import torch
    out = ROOT / 'audit'
    out.mkdir(parents=True, exist_ok=False)
    champion_hash = x10.x7.digest(CHAMPION)
    assert champion_hash == CHAMPION_SHA256
    result_path = CHAMPION.parent / 'result.json'
    original = json.loads(result_path.read_text())
    assert original['best']['checkpoint']['sha256'] == CHAMPION_SHA256
    pool16 = x12.x7.frozen_pools(0, 16, 61)['val']
    pool64 = x10.x7.frozen_pools(0, 64, 61)['val']
    archive = np.load(CHAMPION.parent / 'frozen_validation.npz')
    assert array_hashes(pool16) == array_hashes(tuple(archive[f'T16_{j}'] for j in range(3)))
    archive64 = np.load(x12.ROOT / 'e97_bf16_seed0/frozen_validation.npz')
    assert array_hashes(pool64) == array_hashes(tuple(archive64[f'T64_{j}'] for j in range(3)))
    historical = {}
    for label, root in [('stage10', x10.ROOTS['adamw']), ('stage12', x12.ROOT)]:
        reg = json.loads((root / 'registration.json').read_text())
        loaded = {name: sha for name, sha in reg['source_sha256'].items()
                  if name.startswith(('ndm/', 'experiments/expressivity_tasks/')) and name.endswith('.py')}
        for name, sha in loaded.items():
            if x10.x7.digest(name) != sha:
                raise ValueError('historical source drift: ' + name)
        historical[label] = {'registration_sha256': x10.x7.digest(root / 'registration.json'), 'source_sha256': loaded}
    registration = {'task': 'X13 STEP 0', 'created_at': time.time(), 'champion': str(CHAMPION),
        'champion_sha256': champion_hash, 'champion_step': 468000,
        'champion_result_sha256': x10.x7.digest(result_path), 'historical_sources': historical,
        'audit_source_sha256': x10.x7.digest(__file__),
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'host': socket.gethostname(), 'torch': torch.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
        't16_pool_array_sha256': array_hashes(pool16), 't64_pool_array_sha256': array_hashes(pool64),
        'order': ['a champion T16 via stage12', 'b CPU T64 path equivalence', 'c if clean champion T64 via stage10'],
        'registered_broken_threshold': 'P1 < 400/512 (far below champion 512); any P1-P4 <400 also fails closed',
        'scope': 'NO mechanism claims; X12 E97 interpretation suspended pending audit; no training in STEP 0'}
    x10.x7.write_json(out / 'registration.json', registration)
    # T16 needs only the positional evaluator's expected-length assertion/reshape
    # adjusted from64 to16. No loader, model, padder, targets or mask are changed.
    model = x12.build_model('e97').cuda()
    checkpoint = torch.load(CHAMPION, map_location='cpu', weights_only=False)
    assert checkpoint['step'] == 468000
    model.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    with patch.object(x12, 'DEPTH', 16):
        a = x12.evaluate(model, pool16)
    counts = [p['correct'] for p in a['positions']]
    expected_rows = json.loads((CHAMPION.parent / 'validation.json').read_text())
    expected = next(r for r in expected_rows if r['step'] == 468000)
    a.update(champion_sha256=champion_hash, original_counts=[p['correct'] for p in expected['positions']],
             evaluator_adjustment='stage12.DEPTH=16 ONLY for evaluator loss reshape/position assertion',
             broken=any(n < 400 for n in counts[:4]), clean=counts[:4] == [512] * 4)
    x10.x7.write_json(out / 'a_stage12_champion_t16.json', a)
    print('STEP0(a)', counts, flush=True)
    if not a['clean']:
        verdict = 'BROKEN' if a['broken'] else 'NOT_CLEAN_ATTENDED_REVIEW'
        x10.x7.write_json(out / 'verdict.json', {'verdict': verdict, 'seed_replication_authorized': False,
            'reason': 'champion T16 stage12 path does not retain four perfect positions', 'finished_at': time.time()})
        return 1
    b = equivalence(pool64)
    x10.x7.write_json(out / 'b_cpu_t64_equivalence.json', b)
    print('STEP0(b) clean=', b['clean'], flush=True)
    if not b['clean']:
        x10.x7.write_json(out / 'verdict.json', {'verdict': 'BROKEN', 'seed_replication_authorized': False,
            'reason': 'CPU T64 builder divergence', 'finished_at': time.time()})
        return 1
    # Trusted stage10 positional evaluator uses the same loader; only its T16
    # loss reshape depth is changed. This exercises T64 without a new data path.
    with patch.object(x10.x7, 'DEPTH', 64):
        c = x10.x7.evaluate(model, pool64)
    x10.x7.write_json(out / 'c_stage10_champion_zeroshot_t64.json', {**c,
        'evaluator_adjustment': 'stage10.x7.DEPTH=64 ONLY for evaluator loss reshape/prefix fallback',
        'weights_unchanged': True, 'training_updates': 0, 'champion_sha256': champion_hash})
    print('STEP0(c)', [p['correct'] for p in c['positions']], flush=True)
    verdict = {'verdict': 'CLEAN', 'seed_replication_authorized': True,
               'a_p1_p4': counts[:4], 'c_p1_p4': [p['correct'] for p in c['positions'][:4]],
               't16_same_checkpoint_all_position_counts_match': counts == a['original_counts'],
               'b_token_equivalence': True, 'finished_at': time.time(),
               'interpretation': 'T16 data path exonerated; CPU T64 construction equivalent. Zero-shot counts are a runtime probe, not a mechanism claim.',
               'receipt_sha256': {p.name: x10.x7.digest(p) for p in sorted(out.glob('*.json'))}}
    x10.x7.write_json(out / 'verdict.json', verdict)
    return 0


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    raise SystemExit(audit())
