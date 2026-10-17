"""X5D diagnostic: independent scratch depth probes through the unchanged X5 loop.

This is not a replacement curriculum or authorization to relaunch the six X5 runs.
Only diagnostic depth/budget/threshold/evaluation constants and pool selection
are overridden; model, optimizer, masking, padding and training are X5's run().
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage5_curriculum_8m as x5
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask

RESULTS = Path('experiments/expressivity_tasks/results_invertible_stage5_instrument')


def decode_batch(batch, depth, examples=3):
    """Check every target using scalar integer arithmetic, not trajectory/forward."""
    inputs, targets, mask = batch
    assert inputs.ndim == 2 and inputs.shape[1] == depth + 2
    assert inputs.shape == targets.shape == mask.shape
    assert not mask[:, :2].any() and mask[:, 2:].all()
    assert not targets[:, :2].any()
    decoded = []
    for row, (tokens, labels) in enumerate(zip(inputs, targets)):
        x, y = int(tokens[0]) - 61, int(tokens[1]) - 122
        assert 0 <= x < 61 and 0 <= y < 61
        initial, states = [x, y], []
        for driver, label in zip(tokens[2:], labels[2:]):
            driver = int(driver)
            assert 0 <= driver < 61
            x, y = y, (x + y*y + driver) % 61
            assert int(label) == 61*x + y
            states.append({'state': [x, y], 'joint_class': int(label)})
        if row < examples:
            decoded.append({'tokens': tokens.tolist(), 'initial': initial,
                            'drivers': tokens[2:].tolist(), 'composed': states,
                            'targets': labels.tolist(), 'mask': mask[row].tolist()})
    return {'shape': list(inputs.shape), 'driver_count': depth,
            'supervised_positions': int(mask.sum()), 'decoded': decoded,
            'array_sha256': [hashlib.sha256(a.tobytes()).hexdigest() for a in batch]}


def verify_data(out):
    """Receipt the executed source, archived pools and historical length evidence."""
    out.mkdir(parents=True, exist_ok=True)
    old = x5.ROOT
    launch = old / 'receipts/full_launch'
    paths = {'x5': Path(x5.__file__),
             'x3': Path('experiments/expressivity_tasks/run_invertible_stage3_8m.py'),
             'x2': Path('experiments/expressivity_tasks/run_invertible_stage2.py'),
             'task': Path('experiments/expressivity_tasks/tasks/invertible_quadratic.py')}
    source = {k: {'path': str(p), 'sha256': x5.digest(p)} for k, p in paths.items()}
    for key, snapshot in [('x3', 'x3_runner_at_launch.py'), ('x2', 'x2_runner_at_launch.py')]:
        assert x5.digest(launch / snapshot) == source[key]['sha256']
    records = {}
    for run in sorted(old.glob('*_seed*')):
        cfg = json.loads((run / 'config.json').read_text())
        for key in ('x3', 'x2', 'task'):
            assert cfg['source_sha256'][str(paths[key])] == source[key]['sha256']
        assert cfg['source_sha256'][str(paths['x5'])] == x5.digest(launch / 'runner_at_launch.py')
        with np.load(run / 'frozen_validation.npz') as data:
            pool = tuple(data[f'T4_{i}'] for i in range(3))
        validation = decode_batch(pool, 4)
        fresh = x5.frozen_pools(cfg['seed'], 4, 61)['val']
        assert all(np.array_equal(a, b) for a, b in zip(pool, fresh))
        # Same class, seed, batch and first call as the historical X5 training loop.
        train = InvertibleQuadraticTask(61).generate_batch(32, 4+2, np.random.default_rng(cfg['seed']))
        result = json.loads((run / 'result.json').read_text())
        logs = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
        assert len(logs) == result['total_steps'] == 10000
        assert all(row['depth'] == 4 and row['stage_step'] == row['step'] for row in logs)
        assert result['input_tokens'] == 32 * (4+2) * len(logs)
        receipts = json.loads((run / 'data_receipt.json').read_text())
        assert receipts['depths']['4']['array_sha256'] == validation['array_sha256']
        records[run.name] = {'training_first_batch_reconstructed': decode_batch(train, 4),
                             'frozen_validation': validation,
                             'training_jsonl_sha256': x5.digest(run / 'training.jsonl'),
                             'frozen_validation_sha256': x5.digest(run / 'frozen_validation.npz'),
                             'steps': len(logs), 'reported_input_tokens': result['input_tokens'],
                             'inferred_unpadded_length': result['input_tokens'] // (32*len(logs)),
                             'training_log_keys': sorted(logs[0]),
                             'historical_per_batch_lengths_recorded': False}
    receipt = {'source': source, 'historical_runs': records,
               'historical_x5_sha256': x5.digest(launch / 'runner_at_launch.py'),
               'historical_limit': 'No per-batch shapes, token dumps or hashes; first training batch is reconstructed, not an archived sample. Aggregate input_tokens and immutable executed source corroborate length 6.'}
    x5.write_json(out / 'data_verification.json', receipt)
    return receipt


def probe(arm, depth, out):
    budget = (60000 if arm == 'e97' else 18500) if depth == 1 else 10000
    threshold = .97 if depth == 1 else .9
    x5.DEPTHS = (depth,)
    x5.STAGE_BUDGET = budget
    x5.THRESHOLD = threshold
    # T1 matches X3's observation grid; T2/T4 retain the original X5 grid.
    x5.EVAL_EVERY = 250 if depth == 1 else 500
    original_archive = x5.archive_pools
    original_task = x5.InvertibleQuadraticTask
    stream = hashlib.sha256()
    receipt = {'arm': arm, 'depth': depth, 'budget': budget, 'threshold': threshold,
               'eval_every': x5.EVAL_EVERY, 'initialization': 'independent scratch, seed 0',
               'batches': 0, 'input_tokens': 0, 'supervised_positions': 0,
               'unpadded_shape': [32, depth+2], 'padded_length': 16,
               'overrides': ['DEPTHS', 'STAGE_BUDGET', 'THRESHOLD', 'EVAL_EVERY',
                             'archive_pools (same X3 pool function, selected diagnostic depth)',
                             'InvertibleQuadraticTask (instrumented subclass, unchanged generation)']}

    def archive(out, seed):
        pools = original_archive(out, seed)
        receipt['validation'] = decode_batch(pools[depth], depth)
        return pools

    class InstrumentedTask(original_task):
        def generate_batch(self, batch, length, rng):
            arrays = super().generate_batch(batch, length, rng)
            assert length == depth+2
            assert arrays[0].shape == (32, depth+2)
            assert not arrays[2][:, :2].any() and arrays[2][:, 2:].all()
            if receipt['batches'] == 0:
                receipt['first_training_batch'] = decode_batch(arrays, depth)
            for array in arrays:
                stream.update(array.tobytes())
            receipt['batches'] += 1
            receipt['input_tokens'] += arrays[0].size
            receipt['supervised_positions'] += int(arrays[2].sum())
            return arrays

    x5.archive_pools = archive
    x5.InvertibleQuadraticTask = InstrumentedTask
    try:
        x5.run(argparse.Namespace(arm=arm, seed=0, out=str(out)))
    finally:
        x5.archive_pools = original_archive
        x5.InvertibleQuadraticTask = original_task
    receipt['training_stream_sha256'] = stream.hexdigest()
    result = json.loads((out / 'result.json').read_text())
    assert receipt['batches'] == result['total_steps']
    assert receipt['input_tokens'] == result['input_tokens']
    x5.write_json(out / 'instrument_receipt.json', receipt)
    # Score each position separately using the final eval-x checkpoint, with no
    # selection on these diagnostic panels. Check real-model causal visibility.
    import torch
    model = x5.build_model(arm, 61).cuda()
    state = torch.load(result['stages'][-1]['checkpoint']['path'], map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    model.eval()
    pool = x5.frozen_pools(0, depth, 61)['val']
    positions = {}
    for position in range(2, depth+2):
        mask = np.zeros_like(pool[2])
        mask[:, position] = True
        positions[str(position-1)] = x5.evaluate(model, (pool[0], pool[1], mask), 61)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        tokens = torch.from_numpy(pool[0][:32]).cuda()
        full = x5.padded_forward(model, tokens)
        prefix = x5.padded_forward(model, tokens[:, :3])
        delta = (full[:, :3].float()-prefix.float()).abs().max().item()
    assert delta == 0, f'future drivers changed first-position logits: {delta}'
    x5.write_json(out / 'position_metrics.json', {'per_driver': positions,
                  'first_driver_logits_max_difference_full_vs_truncated': delta,
                  'checkpoint': result['stages'][-1]['checkpoint']})
    del model, full, prefix, tokens
    if depth == 1:
        prior = x5.ROOT.parent / 'results_invertible_stage3_8m' / f'one_step_p61_{arm}_lr0.001_seed0'
        x3_train = [json.loads(line) for line in (prior / 'training.jsonl').read_text().splitlines()]
        x5_train = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
        loss_deltas = [abs(a['loss']-b['loss']) for a, b in zip(x3_train, x5_train)]
        old_val = {r['step']: r for r in json.loads((prior / 'validation.json').read_text())}
        pairs = [{'step': r['step'], 'x3_accuracy': old_val[r['step']]['accuracy'],
                  'x5_accuracy': r['validation_by_depth']['1']['accuracy'],
                  'x3_loss': old_val[r['step']]['loss'],
                  'x5_loss': r['validation_by_depth']['1']['loss']}
                 for r in json.loads((out / 'validation.json').read_text())]
        x5.write_json(out / 'x3_equivalence.json', {
            'compared_training_updates': len(loss_deltas),
            'max_training_loss_absolute_difference': max(loss_deltas),
            'all_training_losses_identical': all(d == 0 for d in loss_deltas),
            'validation_pairs': pairs})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--arm', choices=['e97', 'gdn2'])
    parser.add_argument('--depth', type=int, choices=[1, 2, 4])
    parser.add_argument('--out', type=Path, default=RESULTS)
    args = parser.parse_args()
    if args.data_only:
        verify_data(args.out)
    elif args.arm is None or args.depth is None:
        parser.error('--arm and --depth required for a probe')
    else:
        probe(args.arm, args.depth, args.out)


if __name__ == '__main__':
    main()
