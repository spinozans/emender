"""X13 STEP 1: audited X12 recipe, scratch E97 T64 seeds1/2, one million updates."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage12_separation_8m as x12
from scripts.audit_invertible_stage13_zerocheck import ROOT, array_hashes

x10, x7 = x12.x10, x12.x7
SEEDS = (1, 2)
NAMES = tuple(f'e97_bf16_seed{seed}' for seed in SEEDS)
DEPTH, CEILING, EVAL_EVERY = x12.DEPTH, x12.CEILING, x12.EVAL_EVERY
# Reuse the exonerated instrument and update verbatim, not a repaired/new path.
build_model, train_update, evaluate = x12.build_model, x12.train_update, x12.evaluate
make_optimizer, learning_rate = x12.make_optimizer, x12.learning_rate


def require_clean_audit(root=ROOT):
    verdict = json.loads((root / 'audit/verdict.json').read_text())
    if (verdict['verdict'] != 'CLEAN' or not verdict['seed_replication_authorized']
            or verdict['a_p1_p4'] != [512] * 4 or not verdict['b_token_equivalence']):
        raise ValueError('STEP 0 not clean: no seed replication')
    for name, sha in verdict['receipt_sha256'].items():
        if x7.digest(root / 'audit' / name) != sha:
            raise ValueError('changed audit receipt: ' + name)
    return verdict


def run(seed):
    import torch
    require_clean_audit(ROOT)
    registration = x10.verify_registration(ROOT)
    name = f'e97_bf16_seed{seed}'
    if seed not in SEEDS or name not in registration['authorized_runs']:
        raise ValueError('unauthorized seed')
    admission = json.loads((ROOT / 'receipts' / name / 'admission.json').read_text())
    if (admission['registration_sha256'] != x7.digest(ROOT / 'registration.json')
            or admission['audit_verdict_sha256'] != x7.digest(ROOT / 'audit/verdict.json')
            or admission['gpu'] != int(os.environ['CUDA_VISIBLE_DEVICES'])):
        raise ValueError('invalid audited admission')
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=False)
    pool = x7.frozen_pools(seed, DEPTH, 61)['val']
    hashes = array_hashes(pool)
    if hashes != registration['pool_array_sha256'][str(seed)]:
        raise ValueError('pool identity changed')
    np.savez_compressed(out / 'frozen_validation.npz', **{f'T64_{i}': a for i, a in enumerate(pool)})
    x7.write_json(out / 'data_receipt.json', {'array_sha256': hashes,
                  'archive_sha256': x7.digest(out / 'frozen_validation.npz')})
    torch.manual_seed(seed)
    model = build_model('e97').cuda()
    assert len(model.layers) == 4 and sum(p.numel() for p in model.parameters()) == x12.PARAMETERS['e97']
    opt = make_optimizer(model)
    assert not opt.state
    rng, task = np.random.default_rng(seed), x10.InvertibleQuadraticTask(61)
    config = {'arm': 'adamw', 'architecture': 'e97', 'seed': seed, 'name': name,
              'weights': x10.point('adamw'), 'registration': registration,
              'registration_sha256': x7.digest(ROOT / 'registration.json'), 'admission': admission,
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'host': os.uname().nodename,
              'initialization': 'scratch; no champion weights loaded; empty ordinary AdamW state'}
    x7.write_json(out / 'config.json', config)
    history, checkpoints, best, entry = [], {}, None, None
    started = time.monotonic()
    with (out / 'training.jsonl').open('a', buffering=1) as log:
        for step in range(CEILING + 1):
            if step:
                metrics = train_update(model, opt, task.generate_batch(32, DEPTH + 2, rng), step)
                log.write(json.dumps({'step': step, **metrics, 'elapsed_s': time.monotonic() - started}) + '\n')
            if step % EVAL_EVERY and step != CEILING:
                continue
            metrics = evaluate(model, pool)
            current = metrics['aggregate']
            is_best = best is None or (-current['accuracy'], current['loss']) < (-best['accuracy'], best['loss'])
            cp = None
            if is_best or step % 1000 == 0 or step == CEILING:
                cp = x10.checkpoint(out, model, opt, rng, step, config)
                checkpoints[cp['path']] = cp
                latest = cp['path']
            if is_best:
                best = {**current, 'step': step, 'checkpoint': cp}
            if step == 0:
                entry = cp['path']
            row = {'step': step, **metrics, 'checkpoint': cp, 'elapsed_s': time.monotonic() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                   'decision': 'safety_ceiling' if step == CEILING else 'continue'}
            history.append(row)
            x7.write_json(out / 'validation.json', history)
            x7.write_json(out / 'checkpoints.json', checkpoints)
            print(json.dumps({'name': name, **row}), flush=True)
            x10.prune(out, checkpoints, {entry, best['checkpoint']['path'], latest})
    x7.write_json(out / 'result.json', {'name': name, 'status': 'complete', 'total_steps': CEILING,
        'stop_reason': 'safety_ceiling', 'best': best, 'final': history[-1],
        'wall_seconds': time.monotonic() - started, 'finished_at': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    run(parser.parse_args().seed)
