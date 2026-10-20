"""Read-only, CPU-only reconciliation of X6's completed/infra-partial receipts.

Run from the repository root; stdout is the deterministic compact evidence JSON.
This checks recorded metrics, not checkpoint predictions or training trajectories.
"""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def metric_check(metrics, depth):
    assert metrics['count'] == 512 * depth
    assert metrics['terminal_count'] == 512
    assert metrics['post_suffix7_count'] == 512 * max(depth - 8, 0)
    assert metrics['accuracy'] == metrics['correct'] / metrics['count']
    assert 0 <= metrics['correct'] <= min(metrics['x_correct'], metrics['y_correct'])
    assert metrics['out_of_range'] == 0
    assert math.isfinite(metrics['loss'])


def audit():
    registration = load(ROOT / 'registration.json')
    output = {'registration_sha256': sha256(ROOT / 'registration.json'), 'runs': {}}
    pools = {}
    totals = {'full_ladders': 0, 'infra_partials': 0, 'stage_exits': 0,
              'thresholds': 0, 'ceilings': 0, 'validation_rows': 0}
    for cell in registration['matrix']:
        name = cell['name']
        run = ROOT / name
        config = load(run / 'config.json')
        data = load(run / 'data_receipt.json')
        assert sha256(run / 'frozen_validation.npz') == data['sha256']
        pool_key = str(cell['seed'])
        if pool_key in pools:
            assert pools[pool_key] == data['depths']
        pools[pool_key] = data['depths']
        rows = load(run / 'validation.json')
        transitions = [json.loads(line) for line in (run / 'transitions.jsonl').read_text().splitlines()]
        stages = [{k: v for k, v in row.items() if k != 'event'}
                  for row in transitions if row['event'] == 'stage_exit']
        assert stages == load(run / 'stages.json')
        assert [s['depth'] for s in stages] == registration['depths'][:len(stages)]
        for row in rows:
            for depth, metrics in row['validation_by_depth'].items():
                metric_check(metrics, int(depth))
        compact_stages = []
        for stage in stages:
            depth = stage['depth']
            observations = [r for r in rows if r['depth'] == depth]
            # Entry, every 500 stage updates, then exit; exit may repeat the last grid point.
            assert set(r['stage_step'] for r in observations) == set(range(0, stage['steps'] + 1, 500))
            best = stage['best']
            assert any(r['step'] == best['step'] and all(
                r['validation_by_depth'][str(depth)][k] == v
                for k, v in best.items() if k not in ('stage_step', 'step', 'checkpoint'))
                for r in observations)
            observed_best = max((r['validation_by_depth'][str(depth)] for r in observations),
                                key=lambda m: (m['correct'], -m['loss']))
            assert (best['correct'], best['loss']) == (observed_best['correct'], observed_best['loss'])
            last = observations[-1]
            assert last['step'] == stage['exit_step']
            assert last['validation_by_depth'] == stage['retention']
            exit_metrics = stage['retention'][str(depth)]
            assert exit_metrics['accuracy'] == stage['final_accuracy']
            assert exit_metrics['loss'] == stage['final_loss']
            assert stage['exit_step'] == stage['entry_step'] + stage['steps']
            if stage['decision'] == 'threshold':
                assert stage['steps_to_threshold'] == stage['steps']
                assert stage['final_accuracy'] >= registration['threshold']
                totals['thresholds'] += 1
            else:
                assert stage['decision'] == 'ceiling_non_threshold'
                assert stage['steps'] == registration['stage_ceiling']
                assert best['accuracy'] < registration['threshold']
                totals['ceilings'] += 1
            compact_stages.append({k: stage[k] for k in (
                'depth', 'steps', 'entry_step', 'exit_step', 'decision',
                'steps_to_threshold', 'best', 'final_accuracy', 'final_loss')}
                | {'exit_metrics': exit_metrics,
                   'exit_retention_T1': stage['retention']['1']})
        terminal = load(ROOT / 'receipts' / name / 'terminal.json')
        files = ['config.json', 'data_receipt.json', 'transitions.jsonl', 'stages.json', 'validation.json']
        if name != 'B_e97_seed2':
            marker = load(ROOT / 'markers' / f'{name}.DONE')
            result = load(run / 'result.json')
            assert marker == terminal
            assert terminal['exit_code'] == 0
            assert sha256(run / 'result.json') == marker['result_sha256']
            assert result['stages'] == stages
            assert result['stop_reason'] == 'completed_stage_ladder'
            assert len(stages) == 8
            totals['full_ladders'] += 1
            files.append('result.json')
        else:
            marker = load(ROOT / 'markers' / f'{name}.INFRA-FAILURE-DISK.json')
            assert marker['not_a_training_verdict'] is True
            assert terminal['exit_code'] == 1 and len(stages) == 6
            assert load(ROOT / 'markers' / f'{name}.FAIL.reclassified-infra.json') == terminal
            assert not (run / 'result.json').exists()
            assert rows[-1]['depth'] == 64 and rows[-1]['stage_step'] == 272500
            assert '[Errno 28] No space left on device' in (ROOT / 'receipts' / name / 'console.txt').read_text()
            totals['infra_partials'] += 1
        output['runs'][name] = {
            'source_commit': config['source_commit'], 'source_sha256': config['source_sha256'],
            'parameters': config['trainable_parameters'], 'terminal': terminal, 'marker': marker,
            'file_sha256': {f: sha256(run / f) for f in files},
            'validation_rows': len(rows), 'stages': compact_stages,
            'last_validation': rows[-1]}
        totals['stage_exits'] += len(stages)
        totals['validation_rows'] += len(rows)
    assert totals['full_ladders'] == 11 and totals['infra_partials'] == 1
    original = ROOT / 'markers/B_e97_seed2.INFRA-FAILURE-DISK.json'
    correction_path = ROOT / 'markers/B_e97_seed2.INFRA-FAILURE-DISK.correction.json'
    correction = load(correction_path)
    assert correction['original_record_text_verbatim'] == original.read_text()
    assert correction['original_sha256'] == sha256(original)
    partial_stages = output['runs']['B_e97_seed2']['stages']
    for stage in partial_stages[:2]:
        corrected = correction['corrected_banked_science'][f"T{stage['depth']}"]
        assert corrected['stage_steps_to_threshold'] == stage['steps_to_threshold']
        assert corrected['global_step'] == stage['exit_step']
    output['infra_correction_sha256'] = sha256(correction_path)
    output['infra_console_sha256'] = sha256(ROOT / 'receipts/B_e97_seed2/console.txt')
    output['totals'] = totals
    output['historical_controls'] = {}
    for experiment, name in [('stage10_optimizers', 'adamw_seed0'),
                             ('stage10_optimizers', 'sf_seed0'),
                             ('stage7b_stock', 'gdn2_uniform_seed0')]:
        root = ROOT.parent / f'results_invertible_{experiment}_8m'
        result_path = root / name / 'result.json'
        result = load(result_path)
        terminal = load(root / 'receipts' / name / 'terminal.json')
        marker = load(root / 'markers' / f'{name}.DONE')
        assert marker == terminal and terminal['exit_code'] == 0
        assert terminal['result_sha256'] == sha256(result_path)
        assert result['total_steps'] == 500000
        final = result['final']
        output['historical_controls'][f'{experiment}/{name}'] = {
            'result_sha256': sha256(result_path), 'terminal': terminal,
            'final_counts': [p['correct'] for p in final['positions']],
            'final_loss': final['aggregate']['loss'],
            'perfect_prefix_depth': final['perfect_prefix_depth']}
    return output


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, sort_keys=True))
