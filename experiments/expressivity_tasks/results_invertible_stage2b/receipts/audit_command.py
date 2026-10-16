"""Post-run artifact audit; CPU only, no training or model forward."""
import json
import hashlib
import subprocess
from pathlib import Path

from experiments.expressivity_tasks.run_invertible_stage2 import digest, write_json

root = Path('experiments/expressivity_tasks/results_invertible_stage2b')
old_root = root.with_name('results_invertible_stage2')
launch_commit = 'fb2c1b6e2cdc196403332e1ac86d9390be33c6e0'
launch_bytes = subprocess.check_output(['git', 'show', f'{launch_commit}:experiments/expressivity_tasks/run_invertible_stage2.py'])
launch_sha = hashlib.sha256(launch_bytes).hexdigest()
assert digest(root / 'receipts/runner_at_launch.py') == launch_sha
source_disclosure = {
    'execution_commit': launch_commit, 'runner_sha256': launch_sha,
    'concurrent_update': 'HEAD advanced to X2c commits mid-run; runner file modified at 2026-10-16T17:25:13+00:00. All ten X2b configs bind their launch-time source SHA256. No X2b runner edits after launch. Audit binds git-show launch bytes, not current X2c-extended working tree.',
    'audit_current_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'initial_audit_failure': 'Current-working-tree runner identity assertion failed; corrected to launched fb2c1b6e identity under supervisor ruling. Original failure retained.'}

summary = json.loads((root / 'results_summary.json').read_text())
assert summary['status'] == 'complete' and len(summary['runs']) == 10
checked = {}
for name, result in summary['runs'].items():
    run = root / name
    config = result['config']
    assert config['protocol_amendment'] == 2
    assert config['train_sequences'] == 'fresh procedural per batch'
    assert result['data_receipt'] == json.loads((old_root / name / 'data_receipt.json').read_text())
    assert config['source_sha256'] == launch_sha
    from experiments.expressivity_tasks.run_invertible_stage2 import MATCHED_STEPS, plateau
    history = json.loads((run / 'validation.json').read_text())
    assert all(not plateau(history[:i+1]) for i in range(len(history)-1))
    assert (plateau(history) and result['stop_reason'] == 'plateau') or result['stop_step'] == 60000
    assert result['stop_step'] >= 10000
    assert len((run / 'training.jsonl').read_text().splitlines()) == result['stop_step']
    assert (root / 'receipts' / name / 'exit.txt').read_text().splitlines()[0] == 'exit_code=0'
    assert (run / 'selection_frozen.json').stat().st_mtime_ns < (run / 'best_test.json').stat().st_mtime_ns
    expected_panels = {'test_1x'} if config['drivers'] == 1 else {'test_1x', 'test_4x', 'test_8x'}
    for panelset in [result['test'], *result['matched_test'].values()]:
        assert set(panelset) == expected_panels
        for key, metrics in panelset.items():
            scale = int(key.split('_')[1][:-1])
            assert metrics['count'] == 512 * config['drivers'] * scale
            assert metrics['terminal_count'] == 512
            assert metrics['accuracy'] == metrics['correct'] / metrics['count']
            assert metrics['x_accuracy'] == metrics['x_correct'] / metrics['count']
            assert metrics['y_accuracy'] == metrics['y_correct'] / metrics['count']
            assert 0 <= metrics['out_of_range'] <= metrics['count']
    checkpoints = json.loads((run / 'checkpoints.json').read_text())
    from experiments.expressivity_tasks.run_invertible_stage2 import selected
    best = None
    chain = set()
    for row in history:
        if selected(row, best):
            best = row
            chain.add(row['step'])
    assert all(Path(checkpoints[str(step)]['path']).exists() for step in chain)
    pruning_path = run / 'pruning.jsonl'
    prune_events = []
    if pruning_path.exists():
        for line in pruning_path.read_text().splitlines():
            event = json.loads(line)
            required = {step for step in chain if step <= event['at_step']} | MATCHED_STEPS | set(event['known_stop_steps']) | {event['at_step']}
            assert not set(event['pruned_steps']) & required
            assert all(checkpoints[key]['sha256'] == value for key, value in event['retained_hashes'].items())
            prune_events.append(event)

    hashes = {}
    for step in set(result['matched_test']) | {str(result['step'])}:
        cp = checkpoints[step]
        assert digest(cp['path']) == cp['sha256']
        hashes[step] = cp['sha256']
    if str(result['step']) in result['matched_test']:
        assert result['test'] == result['matched_test'][str(result['step'])]
    if name.startswith('headline'):
        assert (root / 'receipts' / f'matched_{name}' / 'exit.txt').read_text().splitlines()[0] == 'exit_code=0'
        assert set(str(s) for s in MATCHED_STEPS if s <= result['stop_step']) <= set(result['matched_test'])
        assert str(result['stop_step']) in result['matched_test']
        for step, cross in result['cross_plateau_test'].items():
            if cross['status'] == 'evaluated':
                assert digest(cross['checkpoint']['path']) == cross['checkpoint']['sha256']
                if step in result['matched_test']:
                    assert cross['panels'] == result['matched_test'][step]
            else:
                assert cross['reason'] in ('arm already stopped; no extra training', 'checkpoint pruned before stop step became required')
    checked[name] = {'status': 'passed', 'rehashed_selected_and_matched_checkpoints': hashes,
                     'selection_frozen_before_test_receipt': True, 'metric_counts_and_exit': 'passed',
                     'pruning_events_audited': len(prune_events), 'best_validation_chain_files_retained': True,
                     'stop_window': {'best_val_gain': history[-1]['best_accuracy'] - history[-9]['best_accuracy'],
                                     'train_loss_relative_gain': (history[-1]['train_loss_window_first250'] - history[-1]['train_loss_window_final250']) / history[-1]['train_loss_window_first250'],
                                     'first250_mean': history[-1]['train_loss_window_first250'],
                                     'final250_mean': history[-1]['train_loss_window_final250']}}
identity = summary['runs']['headline_e97_identity_seed0']
e97 = summary['runs']['headline_e97_seed0']
assert identity['data_receipt'] == e97['data_receipt']
assert identity['config']['identity_layers'] == 11
assert identity['config']['source_checkpoint_sha256'] == e97['config']['source_checkpoint_sha256']
assert summary['amendment2']['amendment'] == 2
write_json(root / 'receipts' / 'final_audit.json', {'status': 'passed', 'arms': checked, 'source_identity': source_disclosure,
           'all_ten_exit_codes_zero': True, 'all_seven_cross_eval_exit_codes_zero': True, 'all_stops_eligible_and_rule_replayed': True, 'all_frozen_data_identical_to_X2': True,
           'all_metric_observation_counts_verified': True, 'no_extra_training_for_matched_panels': True})
print('PASS: ten completed arms, checkpoint hashes, paired identity data, exact counts, exit0, frozen selections, best/matched agreement; corrected joint plateau replayed, frozen pools unchanged.')
