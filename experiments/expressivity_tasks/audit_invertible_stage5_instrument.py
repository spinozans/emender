"""Offline X5D receipt audit; no training, resubmission or capability claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.expressivity_tasks import run_invertible_stage3_8m as x3
from experiments.expressivity_tasks import run_invertible_stage5_curriculum_8m as x5
from experiments.expressivity_tasks.tasks.invertible_quadratic import InvertibleQuadraticTask
from experiments.expressivity_tasks.verify_invertible_stage5_instrument import RESULTS


def replay(seed, depth, steps):
    task, rng = InvertibleQuadraticTask(61), np.random.default_rng(seed)
    digest = hashlib.sha256()
    for _ in range(steps):
        for array in task.generate_batch(32, depth+2, rng):
            digest.update(array.tobytes())
    return rng.bit_generator.state, digest.hexdigest()


def historical_rng_audit(out):
    import torch
    rows, replays = {}, {}
    for run in sorted(x5.ROOT.glob('*_seed*')):
        cfg = json.loads((run / 'config.json').read_text())
        result = json.loads((run / 'result.json').read_text())
        cp = result['stages'][-1]['checkpoint']
        assert x5.digest(cp['path']) == cp['sha256']
        raw = torch.load(cp['path'], map_location='cpu', weights_only=False)
        seed = cfg['seed']
        if seed not in replays:
            replays[seed] = replay(seed, 4, 10000)
        assert raw['numpy_rng'] == replays[seed][0]
        assert raw['depth'] == 4 and raw['step'] == raw['stage_step'] == 10000
        rows[run.name] = {'checkpoint': cp, 'depth': raw['depth'], 'step': raw['step'],
                          'replayed_depth4_rng_equals_checkpoint': True,
                          'replayed_training_stream_sha256': replays[seed][1],
                          'numpy_rng': raw['numpy_rng']}
        del raw
    x5.write_json(out / 'historical_rng_audit.json', rows)
    return rows


def audit_probes(out):
    import torch
    assert x5.build_model is x3.build_model
    assert x5.evaluate is x3.evaluate
    assert x5.padded_forward is x3.padded_forward
    for snapshot, source in [('instrument_at_launch.py', 'verify_invertible_stage5_instrument.py'),
                             ('launcher_at_launch.sh', 'launch_invertible_stage5_instrument.sh')]:
        assert x5.digest(out / 'receipts/probes' / snapshot) == x5.digest(Path(__file__).with_name(source))
    summary = {}
    for depth in (1, 2, 4):
        for arm in ('e97', 'gdn2'):
            run = out / f'{arm}_T{depth}'
            result = json.loads((run / 'result.json').read_text())
            config = json.loads((run / 'config.json').read_text())
            instrument = json.loads((run / 'instrument_receipt.json').read_text())
            positions = json.loads((run / 'position_metrics.json').read_text())
            steps = result['total_steps']
            expected_budget = (60000 if arm == 'e97' else 18500) if depth == 1 else 10000
            assert config['depths'] == [depth] and config['stage_budget'] == expected_budget
            assert config['threshold'] == (.97 if depth == 1 else .9)
            assert config['seed'] == 0 and config['lr'] == .001 and config['batch'] == 32
            for path, expected_hash in config['source_sha256'].items():
                assert x5.digest(path) == expected_hash
            assert config['trainable_parameters'] == {'e97': 9900800, 'gdn2': 9887056}[arm]
            assert result['status'] == 'complete' and result['stopping_depth'] == depth
            assert steps <= expected_budget
            if depth == 1:
                assert result['stopping_accuracy'] >= .97
            else:
                assert steps == 10000 and result['stop_reason'] == 'budget_exhausted'
            assert instrument['batches'] == steps
            assert instrument['supervised_positions'] == 32*depth*steps
            assert instrument['input_tokens'] == result['input_tokens'] == 32*(depth+2)*steps
            rng, stream = replay(0, depth, steps)
            assert stream == instrument['training_stream_sha256']
            checkpoint = result['stages'][-1]['checkpoint']
            assert x5.digest(checkpoint['path']) == checkpoint['sha256']
            raw = torch.load(checkpoint['path'], map_location='cpu', weights_only=False)
            assert rng == raw['numpy_rng']
            del raw
            rows = [json.loads(line) for line in (run / 'training.jsonl').read_text().splitlines()]
            assert [r['step'] for r in rows] == list(range(1, steps+1))
            assert all(r['depth'] == depth and r['stage_step'] == r['step'] for r in rows)
            per_position = [v['accuracy'] for v in positions['per_driver'].values()]
            assert abs(np.mean(per_position)-result['stopping_accuracy']) < 1e-12
            assert positions['first_driver_logits_max_difference_full_vs_truncated'] == 0
            val = json.loads((run / 'validation.json').read_text())
            record = {'steps': steps, 'budget': expected_budget,
                      'final_accuracy': result['stopping_accuracy'],
                      'best_accuracy': result['stages'][-1]['best']['accuracy'],
                      'first_90_percent_step': next((r['step'] for r in val if r['validation_by_depth'][str(depth)]['accuracy'] >= .9), None),
                      'at_10000': next((r['validation_by_depth'][str(depth)] for r in val if r['step'] == 10000), None),
                      'final_per_driver_accuracy': per_position,
                      'first_driver_causal_logits_max_delta': 0,
                      'replayed_stream_equals_observed_stream': True,
                      'replayed_rng_equals_final_checkpoint': True,
                      'checkpoint_hash_verified': True}
            if depth == 1:
                eq = json.loads((run / 'x3_equivalence.json').read_text())
                record['x3_max_training_loss_absolute_difference'] = eq['max_training_loss_absolute_difference']
                prior = x3.ROOT / f'one_step_p61_{arm}_lr0.001_seed0'
                old = json.loads((prior / 'validation.json').read_text())
                record['x3_first_90_percent_step'] = next(r['step'] for r in old if r['accuracy'] >= .9)
                record['x3_first_97_percent_step'] = next(r['step'] for r in old if r['accuracy'] >= .97)
                record['x3_at_10000'] = next(r for r in old if r['step'] == 10000)
            if depth == 4:
                prior = x5.ROOT / f'{arm}_seed0'
                original = json.loads((prior / 'result.json').read_text())
                original_rows = [json.loads(line) for line in (prior / 'training.jsonl').read_text().splitlines()]
                record['original_x5_stopping_accuracy'] = original['stopping_accuracy']
                record['original_x5_max_training_loss_absolute_difference'] = max(
                    abs(a['loss']-b['loss']) for a, b in zip(original_rows, rows))
            summary[run.name] = record
    assert (out / 'receipts/probes/exit.txt').read_text().startswith('exit_code=0')
    x5.write_json(out / 'verification_summary.json', summary)
    # Hash all portable evidence, never checkpoint payloads or this manifest itself.
    files = {str(p.relative_to(out)): x5.digest(p) for p in sorted(out.rglob('*'))
             if p.is_file() and p.suffix != '.pt' and p.name != 'artifacts_sha256.json'}
    x5.write_json(out / 'artifacts_sha256.json', files)
    return summary


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--out', type=Path, default=RESULTS)
    parser.add_argument('--historical-only', action='store_true')
    args = parser.parse_args()
    historical_rng_audit(args.out)
    if not args.historical_only:
        print(json.dumps(audit_probes(args.out), indent=2))


if __name__ == '__main__':
    main()
