"""CPU-only final X4 artifact audit; no training and no model forward."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import numpy as np

from experiments.expressivity_tasks.invertible_stage4_13b import ARMS, ROOT
from experiments.expressivity_tasks.run_invertible_stage2 import (
    FLOOR, MARGINAL_REFERENCE, MATCHED_STEPS, digest, plateau, selected, write_json,
)

root = ROOT
launch = json.loads((root / 'receipts/launch_commit.json').read_text())
commit = launch['commit']
assert launch['remote_commit_before_launch'] == commit
assert commit.startswith('c849812c')
assert commit == subprocess.check_output(['git', 'rev-parse', commit], text=True).strip()
source = subprocess.check_output(['git', 'show', f'{commit}:experiments/expressivity_tasks/run_invertible_stage2.py'])
source_hash = hashlib.sha256(source).hexdigest()
assert source_hash == digest(root / 'receipts/runner_at_launch.py') == launch['runner_sha256']
assert digest(root / 'amendment5.json') == launch['amendment_sha256']
amendment = json.loads((root / 'amendment5.json').read_text())
summary = json.loads((root / 'results_summary.json').read_text())
assert summary['status'] == 'complete' and set(summary['runs']) == set(ARMS)
assert summary['amendment5'] == amendment
assert summary['marginal_optimal_reference'] == MARGINAL_REFERENCE
assert amendment['base_commit'] == '8ea0c761f234e0a4d535bf463d7ef55ddcbda024'
assert datetime.fromisoformat(amendment['registered_at']) < datetime.fromisoformat(launch['recorded_at'])
assert len(summary['paired_e97_minus_gdn2']) == 3
prior = root.with_name('results_invertible_stage2b')
x2 = root.with_name('results_invertible_stage2')
for filename, expected in amendment['basis'].items():
    if filename != 'task_sha256':
        assert digest(filename.removesuffix('_sha256')) == expected

for arm in ('e97', 'gdn2'):
    cfg = summary['runs'][f'headline_{arm}_seed0']['config']
    from scripts.expressivity_1p3b import CHECKPOINTS
    assert digest(cfg['source_checkpoint']) == cfg['source_checkpoint_sha256']
    assert digest(CHECKPOINTS[arm]['args']) == cfg['source_args_sha256']
checked = {}
leases = []
for name in ARMS:
    run = root / name
    result = summary['runs'][name]
    cfg = result['config']
    assert cfg['arm'] in ('e97', 'gdn2') and cfg['seed'] in (0, 1, 2)
    assert cfg['protocol_amendment'] == 5 and cfg['lr'] == .001 and not cfg['pilot']
    assert cfg['hard_cap'] == 60000 and cfg['min_relative_train_loss_improvement'] == .001
    assert cfg['plateau_eligible_step'] == 10000 and cfg['min_improvement'] == .001
    assert cfg['source_sha256'] == source_hash
    assert cfg['marginal_optimal_reference'] == MARGINAL_REFERENCE
    assert cfg['train_sequences'] == 'fresh procedural per batch'
    assert cfg['optimizer_state'] == 'fresh; no released or previous fine-tune optimizer state loaded'
    assert cfg['checkpoint_dir'] == f'/mnt/nvme1n1/erikg/invertible_stage4_13b/{name}'
    old_cfg = json.loads((prior / name / 'config.json').read_text())
    for key in ('source_checkpoint_sha256', 'source_args_sha256', 'trainable_parameters',
                'recurrent_states', 'padding', 'emission', 'batch', 'betas', 'weight_decay',
                'grad_clip', 'warmup', 'dtype', 'optimizer', 'drivers'):
        assert cfg[key] == old_cfg[key], (name, key)
    data = result['data_receipt']
    assert data == amendment['frozen_data_receipts'][str(cfg['seed'])]
    assert data == json.loads((x2 / name / 'data_receipt.json').read_text())
    assert data == json.loads((prior / name / 'data_receipt.json').read_text())
    x3 = root.with_name('results_invertible_stage3_8m')
    for lr in ('0.001', '0.0003', '0.0001'):
        other = x3 / f"primary_p61_{cfg['arm']}_lr{lr}_seed{cfg['seed']}"
        assert data == json.loads((other / 'data_receipt.json').read_text())
        assert digest(other / 'frozen_pools.npz') == data['sha256']
    if cfg['seed'] == 0:
        for tag in ('2e-4', '1e-3'):
            other = root.with_name('results_invertible_stage2c') / f"pilot_{tag}_{cfg['arm']}_seed0"
            assert data == json.loads((other / 'data_receipt.json').read_text())
            assert digest(other / 'frozen_pools.npz') == data['sha256']
    assert digest(run / 'frozen_pools.npz') == data['sha256']
    with np.load(run / 'frozen_pools.npz') as pools:
        for pool, receipt in data['pools'].items():
            assert hashlib.sha256(pools[f'{pool}_0'].tobytes()).hexdigest() == receipt['input_sha256']
    history = json.loads((run / 'validation.json').read_text())
    lines = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
    losses = [r['loss'] for r in lines]
    assert [r['step'] for r in lines] == list(range(1, result['stop_step'] + 1))
    assert all(r['marginal_optimal_reference'] == MARGINAL_REFERENCE and r['suffix_floor'] == FLOOR
               and math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in lines)
    assert result['total_training_tokens'] == 32 * 130 * result['stop_step']
    assert [r['step'] for r in history] == list(range(0, result['stop_step'] + 1, 250))
    best, chain = None, set()
    for i, row in enumerate(history):
        if selected(row, best):
            best = row
            chain.add(row['step'])
        assert row['best_accuracy'] == best['accuracy']
        if row['step'] >= 2000:
            s = row['step']
            assert math.isclose(statistics.mean(losses[s-2000:s-1750]), row['train_loss_window_first250'], rel_tol=1e-12)
            assert math.isclose(statistics.mean(losses[s-250:s]), row['train_loss_window_final250'], rel_tol=1e-12)
        if i < len(history) - 1:
            assert not plateau(history[:i+1], .001), (name, row['step'])
    assert best['step'] == result['step']
    assert 10000 <= result['stop_step'] <= 60000
    assert (result['stop_reason'] == 'plateau' and plateau(history, .001)) or (result['stop_reason'] == 'hard_cap' and result['stop_step'] == 60000 and not plateau(history, .001))
    assert (root / 'receipts' / name / 'exit.txt').read_text().splitlines()[0] == 'exit_code=0'
    assert (root / 'receipts' / f'matched_{name}' / 'exit.txt').read_text().splitlines()[0] == 'exit_code=0'
    assert (run / 'selection_frozen.json').stat().st_mtime_ns < (run / 'best_test.json').stat().st_mtime_ns
    lease = dict(line.split('=', 1) for line in (root / 'receipts' / name / 'lease.txt').read_text().splitlines())
    assert datetime.fromisoformat(lease['start']) > datetime.fromisoformat(launch['recorded_at'])
    leases.append(lease['gpu'])
    cps = json.loads((run / 'checkpoints.json').read_text())
    required = (chain | {s for s in MATCHED_STEPS if s <= result['stop_step']} | {result['stop_step']})
    assert all(Path(cps[str(s)]['path']).exists() for s in required)
    required_hashes = {}
    for s in sorted(required):
        cp = cps[str(s)]
        assert Path(cp['path']).parent == Path(cfg['checkpoint_dir']) and Path(cp['path']).is_absolute()
        assert cp['bytes'] == Path(cp['path']).stat().st_size
        assert digest(cp['path']) == cp['sha256']
        required_hashes[str(s)] = cp['sha256']
    assert {str(s) for s in MATCHED_STEPS if s <= result['stop_step']} <= set(result['matched_test'])
    assert str(result['stop_step']) in result['matched_test']
    if str(result['step']) in result['matched_test']:
        assert result['test'] == result['matched_test'][str(result['step'])]
    all_panels = [result['test'], *result['matched_test'].values()]
    stop_union = {str(summary['runs'][n]['stop_step']) for n in ARMS}
    assert set(result['cross_plateau_test']) == stop_union
    for s, cross in result['cross_plateau_test'].items():
        if cross['status'] == 'evaluated':
            cp = cross['checkpoint']
            assert digest(cp['path']) == cp['sha256']
            all_panels.append(cross['panels'])
            if s in result['matched_test']:
                assert cross['panels'] == result['matched_test'][s]
        else:
            assert cross['reason'] in ('arm already stopped; no extra training', 'checkpoint pruned before stop step became required')
            if cross['reason'].startswith('arm already'):
                assert int(s) > result['stop_step']
            else:
                assert not Path(cps[s]['path']).exists()
    for panels in all_panels:
        assert set(panels) == {'test_1x', 'test_4x', 'test_8x'}
        for length, m in panels.items():
            scale = int(length.split('_')[1][:-1])
            assert m['count'] == 512 * 128 * scale
            assert m['post_suffix7_count'] == 512 * (128 * scale - 8)
            assert m['terminal_count'] == 512
            for key, count in [('accuracy', 'correct'), ('x_accuracy', 'x_correct'), ('y_accuracy', 'y_correct')]:
                assert m[key] == m[count] / m['count']
            assert m['post_suffix7_accuracy'] == m['post_suffix7_correct'] / m['post_suffix7_count']
            assert m['terminal_accuracy'] == m['terminal_correct'] / 512
            assert m['suffix_floor'] == FLOOR and m['marginal_optimal_reference'] == MARGINAL_REFERENCE
            assert math.isfinite(m['loss'])
    events = []
    if (run / 'pruning.jsonl').exists():
        for line in (run / 'pruning.jsonl').read_text().splitlines():
            event = json.loads(line)
            required_at_event = {s for s in chain if s <= event['at_step']} | MATCHED_STEPS | set(event['known_stop_steps']) | {event['at_step']}
            assert not set(event['pruned_steps']) & required_at_event
            assert all(cps[s]['sha256'] == v for s, v in event['retained_hashes'].items())
            events.append(event)
    last = history[-1]
    checked[name] = {'status': 'passed', 'required_checkpoint_hashes': required_hashes,
        'data_sha256': data['sha256'], 'pruning_events': len(events),
        'training_updates': len(lines), 'selected_step': result['step'], 'stop_step': result['stop_step'],
        'stop_window': {'first250_mean': last['train_loss_window_first250'],
                        'final250_mean': last['train_loss_window_final250'],
                        'relative_train_gain': (last['train_loss_window_first250'] - last['train_loss_window_final250']) / last['train_loss_window_first250'],
                        'best_val_gain': last['best_accuracy'] - history[-9]['best_accuracy']},
        'marginal_optimal_reference': MARGINAL_REFERENCE, 'suffix_floor': FLOOR,
        'all_counts_replayed': True, 'selection_before_test': True, 'all_required_weights_rehashed': True}
assert len(set(leases)) == 6
assert (root / 'receipts/train_wave_exit.txt').read_text().splitlines()[1] == 'exit_code=0'
assert (root / 'receipts/eval_wave_exit.txt').read_text().splitlines()[1] == 'exit_code=0'
for arm in ('e97', 'gdn2'):
    for length in ('test_1x', 'test_4x', 'test_8x'):
        for key in ('accuracy', 'loss', 'post_suffix7_accuracy'):
            a = summary['aggregate'][arm][f'{length}_{key}']
            values = [summary['runs'][f'headline_{arm}_seed{s}']['test'][length][key] for s in range(3)]
            assert a['n_seeds'] == 3 and a['mean'] == statistics.mean(values) and a['sample_sd'] == statistics.stdev(values)
write_json(root / 'receipts/final_audit.json', {'status': 'passed', 'audited_at': datetime.now(timezone.utc).isoformat(),
    'execution_commit': commit, 'runner_sha256': source_hash, 'arms': checked,
    'six_exclusive_training_leases': leases, 'all_six_train_and_cross_eval_exits_zero': True,
    'frozen_pools_byte_identical_to_X2_X2b_X2c_X3': True, 'released_source_weights_and_args_rehashed': True,
    'stopping_and_selection_replayed': True,
    'no_extra_training_for_panels': True, 'named_exact_marginal_reference_verified': True})
print('PASS: all six matched1e-3 runs; exact frozen pools; fresh recipe; actual updates; strict calibrated first plateau; best-val selection before test; all required binary hashes/counts; matched/x4/x8/cross-stop; named exact reference; exit0 and six exclusive leases.')
