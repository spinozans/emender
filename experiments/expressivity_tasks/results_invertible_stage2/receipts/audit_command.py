"""Post-run artifact audit; CPU only, no training or model forward."""
import json
from pathlib import Path

from experiments.expressivity_tasks.run_invertible_stage2 import digest, write_json

root = Path('experiments/expressivity_tasks/results_invertible_stage2')
summary = json.loads((root / 'results_summary.json').read_text())
assert summary['status'] == 'complete' and len(summary['runs']) == 10
checked = {}
for name, result in summary['runs'].items():
    run = root / name
    config = result['config']
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
    hashes = {}
    for step in set(result['matched_test']) | {str(result['step'])}:
        cp = checkpoints[step]
        assert digest(cp['path']) == cp['sha256']
        hashes[step] = cp['sha256']
    if str(result['step']) in result['matched_test']:
        assert result['test'] == result['matched_test'][str(result['step'])]
    if name.startswith('headline'):
        assert result['stop_step'] == 2000 and result['stop_reason'] == 'plateau'
        assert {'1000', '2000'} == set(result['matched_test'])
        assert {'2000'} == set(result['cross_plateau_test'])
        assert result['cross_plateau_test']['2000']['panels'] == result['matched_test']['2000']
    checked[name] = {'status': 'passed', 'rehashed_selected_and_matched_checkpoints': hashes,
                     'selection_frozen_before_test_receipt': True, 'metric_counts_and_exit': 'passed'}
identity = summary['runs']['headline_e97_identity_seed0']
e97 = summary['runs']['headline_e97_seed0']
assert identity['data_receipt'] == e97['data_receipt']
assert identity['config']['identity_layers'] == 11
assert identity['config']['source_checkpoint_sha256'] == e97['config']['source_checkpoint_sha256']
assert summary['contingency']['triggered'] is True
write_json(root / 'receipts' / 'final_audit.json', {'status': 'passed', 'arms': checked,
           'all_ten_exit_codes_zero': True, 'all_headline_stops_2000': True,
           'all_metric_observation_counts_verified': True, 'no_extra_training_for_matched_panels': True})
print('PASS: ten completed arms, checkpoint hashes, paired identity data, exact counts, exit0, frozen selections, best/matched agreement; all headline plateaus2000.')
