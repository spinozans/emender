"""Audit X5 receipts and publish the depth-frontier table (no test selection)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from experiments.expressivity_tasks.run_invertible_stage5_curriculum_8m import (
    DEPTHS, EVAL_EVERY, FLOOR, MARGINAL, ROOT, STAGE_BUDGET, SUFFIX_THEOREM,
    THRESHOLD, digest, matrix, stage_decision, write_json,
)


def load(path):
    return json.loads(Path(path).read_text())


def same_state(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def audit_checkpoint_payloads(out):
    """Check actual optimizer clocks and tensor-identical model/optimizer/RNG transitions."""
    import torch
    transitions = [json.loads(s) for s in (out / 'transitions.jsonl').read_text().splitlines()]
    previous_exit = None
    for entry, exit_ in zip(transitions[::2], transitions[1::2]):
        first = torch.load(entry['checkpoint']['path'], map_location='cpu', weights_only=False)
        last = torch.load(exit_['checkpoint']['path'], map_location='cpu', weights_only=False)
        for raw, receipt in ((first, entry['checkpoint']), (last, exit_['checkpoint'])):
            assert raw['step'] == receipt['step']
            assert raw['depth'] == receipt['depth'] and raw['stage_step'] == receipt['stage_step']
            assert raw['weights'] == 'ScheduleFree eval x'
            assert all(g['k'] == raw['step'] and not g['train_mode'] for g in raw['optimizer']['param_groups'])
            assert bool(raw['optimizer']['state']) == (raw['step'] > 0)
            assert torch.equal(raw['model']['embed.weight'], raw['model']['out_proj.weight'])
            assert all(torch.isfinite(t).all() for t in raw['model'].values())
            assert len(raw['cuda_rng']) == 1 and raw['torch_rng'].numel() > 0
        if previous_exit is not None:
            for key in ('model', 'optimizer', 'numpy_rng', 'torch_rng', 'cuda_rng'):
                assert same_state(first[key], previous_exit[key]), f'{out}: discontinuous {key}'
        previous_exit = last
    return True


def audit_run(root, row):
    out = root / row['name']
    result, config = load(out / 'result.json'), load(out / 'config.json')
    assert result['status'] == 'complete'
    assert result['arm'] == config['arm'] == row['arm']
    assert result['seed'] == config['seed'] == row['seed']
    assert config['trainable_parameters'] == {'e97': 9900800, 'gdn2': 9887056}[row['arm']]
    assert config['depths'] == list(DEPTHS) and config['stage_budget'] == STAGE_BUDGET
    assert config['threshold'] == THRESHOLD and config['eval_every'] == EVAL_EVERY
    assert config['lr'] == .001 and config['batch'] == 32
    assert config['betas'] == [.9, .95] and config['weight_decay'] == .01
    assert config['optimizer'] == 'AdamWScheduleFree' and config['warmup'] == 0 and config['clip'] == 1.
    assert config['initialization'] == 'from scratch; no checkpoint loaded'
    assert 'exit_code=0' in (root / 'receipts' / row['name'] / 'exit.txt').read_text()
    data = load(out / 'data_receipt.json')
    assert digest(data['path']) == data['sha256']
    with np.load(data['path']) as pools:
        for d in DEPTHS:
            assert pools[f'T{d}_0'].shape == (512, d + 2)
            assert pools[f'T{d}_2'].sum() == 512 * d
    checkpoints = load(out / 'checkpoints.json')
    retained = 0
    for p, cp in checkpoints.items():
        if Path(p).exists():
            assert digest(p) == cp['sha256'], p
            retained += 1
        else:
            assert (out / 'pruning.jsonl').exists(), p
            pruned = [json.loads(s) for s in (out / 'pruning.jsonl').read_text().splitlines()]
            assert any(p in item['pruned'] for item in pruned), p
    history = load(out / 'validation.json')
    training = [json.loads(s) for s in (out / 'training.jsonl').read_text().splitlines()]
    assert [s['step'] for s in training] == list(range(1, result['total_steps'] + 1))
    stages = load(out / 'stages.json')
    assert stages == result['stages'] and stages
    assert [s['depth'] for s in stages] == list(DEPTHS[:len(stages)])
    transitions = [json.loads(s) for s in (out / 'transitions.jsonl').read_text().splitlines()]
    assert len(transitions) == 2 * len(stages)
    previous_step = 0
    for i, stage in enumerate(stages):
        d, steps = stage['depth'], stage['steps']
        assert stage['entry_step'] == previous_step
        assert stage['exit_step'] == previous_step + steps
        h = [s for s in history if s['depth'] == d]
        assert [s['stage_step'] for s in h] == list(range(0, steps + 1, EVAL_EVERY))
        for entry in h:
            assert set(entry['validation_by_depth']) == {str(depth) for depth in DEPTHS if depth <= d}
            metric = entry['validation_by_depth'][str(d)]
            assert entry['decision'] == stage_decision(metric['accuracy'], entry['stage_step'])
            assert metric['count'] == 512 * d
            assert metric['accuracy'] == metric['correct'] / metric['count']
            assert entry['marginal_reference_nats'] == MARGINAL
            assert entry['suffix_floor_percent'] == FLOOR * 100
        assert all(s['decision'] == 'continue' for s in h[:-1])
        assert h[-1]['decision'] == stage['decision']
        assert h[-1]['validation_by_depth'] == stage['retention']
        assert stage['final_accuracy'] == h[-1]['validation_by_depth'][str(d)]['accuracy']
        local_training = [s for s in training if s['depth'] == d]
        assert [s['stage_step'] for s in local_training] == list(range(1, steps + 1))
        if steps:
            assert stage['train_loss_first500'] == float(np.mean([s['loss'] for s in local_training[:500]]))
            assert stage['train_loss_final500'] == float(np.mean([s['loss'] for s in local_training[-500:]]))
        entry, exit_ = transitions[2*i:2*i+2]
        assert entry['event'] == 'stage_entry' and not entry['optimizer_reset']
        assert exit_['event'] == 'stage_exit' and exit_['checkpoint'] == stage['checkpoint']
        assert entry['previous_stage_exit'] == (stages[i-1]['checkpoint'] if i else None)
        for cp in (entry['checkpoint'], exit_['checkpoint'], stage['best']['checkpoint']):
            assert Path(cp['path']).exists() and digest(cp['path']) == cp['sha256']
        if stage['decision'] == 'threshold':
            assert stage['final_accuracy'] >= THRESHOLD and stage['steps_to_threshold'] == steps
        else:
            assert i == len(stages) - 1 and steps == STAGE_BUDGET
            assert stage['final_accuracy'] < THRESHOLD and stage['steps_to_threshold'] is None
        previous_step += steps
    achieved = [s['depth'] for s in stages if s['decision'] == 'threshold']
    assert result['frontier'] == max(achieved, default=0)
    assert result['total_steps'] == previous_step <= 60000
    assert result['stopping_depth'] == stages[-1]['depth']
    assert result['stopping_accuracy'] == stages[-1]['final_accuracy']
    expected_stop = ('completed_curriculum' if stages[-1]['depth'] == 128
                     and stages[-1]['decision'] == 'threshold' else stages[-1]['decision'])
    assert result['stop_reason'] == expected_stop
    loss_trends = {}
    for stage in stages:
        window = np.array([s['loss'] for s in training if s['depth'] == stage['depth']][-2000:])
        assert np.isfinite(window).all()
        slope = float(np.polyfit(np.arange(len(window)), window, 1)[0]) if len(window) > 1 else None
        loss_trends[str(stage['depth'])] = {'trailing_updates': len(window), 'ols_nats_per_update': slope}
    return {**result, 'name': row['name'], 'loss_trends': loss_trends, 'retained_checkpoints_verified': retained,
            'config_sha256': digest(out / 'config.json'), 'result_sha256': digest(out / 'result.json')}


def report(summary):
    runs = summary['runs']
    lines = ['# X5 — curriculum depth frontier (8M scale)', '',
             '**Headline: deepest stage crossing 90% full-state validation accuracy.** '
             '`none (<4)` means no registered stage crossed; it is not a tested T=0 result.', '',
             '| Architecture | Seed | Frontier T | Steps to threshold (T:steps) | Stopping T | Final accuracy | Total updates |',
             '|---|---:|---:|---|---:|---:|---:|']
    for r in runs:
        crossed = ', '.join(f"{s['depth']}:{s['steps_to_threshold']}" for s in r['stages'] if s['decision'] == 'threshold') or 'none'
        frontier = str(r['frontier']) if r['frontier'] else 'none (<4)'
        lines.append(f"| {r['arm']} | {r['seed']} | {frontier} | {crossed} | {r['stopping_depth']} | {100*r['stopping_accuracy']:.4f}% | {r['total_steps']} |")
    lines += ['', '## Reference beside the frontier', '', SUFFIX_THEOREM,
              f'Uniform marginal reference: ln(3721) = **{MARGINAL:.6f} nats**; suffix floor **{100*FLOOR:.6f}%**.', '',
              '## Stage loss levels and retention', '',
              'Loss columns are observed means over the first/final 500 training updates, not a fitted '
              'mechanism or an additional early-stop rule. Δ is final training loss minus ln(3721).', '',
              '| Arm/seed | T | Steps | First-500 CE | Final-500 CE | Δ marginal | Final val CE | Best val accuracy | Retention at exit (T:accuracy) |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in runs:
        for s in r['stages']:
            first, last = s['train_loss_first500'], s['train_loss_final500']
            retention = ', '.join(f"{d}:{100*m['accuracy']:.4f}%" for d, m in s['retention'].items())
            fmt = lambda v: f'{v:.6f}' if v is not None else 'n/a (0 updates)'
            lines.append(f"| {r['arm']}/{r['seed']} | {s['depth']} | {s['steps']} | {fmt(first)} | {fmt(last)} | {fmt(last-MARGINAL if last is not None else None)} | {s['final_loss']:.6f} | {100*s['best']['accuracy']:.4f}% | {retention} |")
    lines += ['', 'Trailing training-loss OLS slopes (up to 2,000 updates; descriptive, not a stop rule):']
    for r in runs:
        for d, trend in r.get('loss_trends', {}).items():
            slope = trend['ols_nats_per_update']
            lines.append(f"- {r['arm']}/{r['seed']}, T={d}: {slope:.8f} nats/update over {trend['trailing_updates']} updates." if slope is not None
                         else f"- {r['arm']}/{r['seed']}, T={d}: insufficient updates for a slope.")
    lines += ['Budget exhaustion, not a pre-registered convergence/plateau test, ends a non-crossing stage; '
              'the observed loss levels and slopes do not establish an asymptotic plateau.', '',
              '## What this changes about the X-series reading', '']
    if all(r['frontier'] == 0 for r in runs):
        lines += ['Both architectures stop at the initial T=4 stage in all three seeds: none reaches '
                  '90% within 10,000 updates. No doubling transition is therefore authorized by the '
                  'registered rule, and T=8–128 are **unattempted**, not measured failures. There is no '
                  'lower completed curriculum stage whose retention could be lost; T=4 is evaluated '
                  'at entry and every 500 updates through the stopping boundary. The experiment '
                  'moves the observed failure boundary from flat T=128 to the much shallower initial '
                  'T=4 under this recipe, but does not establish an intrinsic architecture capability '
                  'limit or test whether a curriculum starting below T=4 could bootstrap composition. '
                  'These six frontiers support no architectural superiority claim.']
    else:
        for arm in ('e97', 'gdn2'):
            rows = [r for r in runs if r['arm'] == arm]
            lines.append(f"{arm}: achieved depths {[r['frontier'] for r in rows]}, stopping depths "
                         f"{[r['stopping_depth'] for r in rows]}. These are recipe/budget-specific "
                         'observations, not intrinsic capacity bounds. Completed-stage retention is '
                         'reported above and in every 500-step validation receipt.')
        lines.append('The curriculum measures staged learnability rather than interpreting the flat-T=128 '
                     'failure as a mechanism. Unattempted depths remain unmeasured; completion at 128 '
                     'would be a lower bound on the observed frontier, not an ultimate capacity ceiling.')
    lines += ['', '## Recipe, receipts and validation', '',
              'From scratch, seeds 0/1/2; exact X3 E97-MLP (9,900,800 parameters) and real fla-gdn '
              'GDN2 (9,887,056; allow_neg_eigval=True). Task vocab 3721; batch 32; matched lr 1e-3; '
              'ScheduleFree AdamW β=(.9,.95), wd=.01, clip=1, bf16 autocast, no warmup. '
              'Fresh procedural full-state supervision only at the current depth. Same optimizer, '
              'weights and training RNG carried across stages; ScheduleFree eval x weights determine '
              'threshold crossings. Observed crossings are resolved on the registered 500-step evaluation grid.', '',
              'T=4→8→16→32→64→128; ≥90% advances; 10K updates without crossing stops; total cap 60K. '
              'At entry, every 500 updates and every exit, score current and all lower registered '
              'depths on 512 frozen sequences each. Frozen pools use the exact X2/X3 algorithm; '
              'aligned archived validation arrays are checked for equality and their file hashes '
              'recorded, including T=128 even when unattempted. No test-pool selection.', '',
              f"Results root: `{ROOT}`. `registration.json` predates launch; `receipts/` retains commands, "
              'exclusive six-GPU leases, source snapshots, source hashes, package versions, GPU/disk '
              'snapshots and exits. Each run retains config, frozen pool arrays/hashes, per-update '
              'training.jsonl, validation.json, transitions.jsonl, stages.json, checkpoints.json and '
              'result.json. Atomic local checkpoints include model, optimizer and both RNG states; '
              'git excludes checkpoint payloads, retaining their paths and SHA-256 receipts. '
              'Below 400GiB free the established retention policy preserves stage boundaries, '
              'stage best, current and 10K checkpoints; any pruning is audited.', '',
              'Independent aggregation verifies all six exits, model/recipe counts, exact update '
              'sequences, the earliest observed threshold crossing or 10K failure, all lower-depth '
              'evaluation panels, stage transitions, loss means, frozen pool hashes and every '
              'retained checkpoint hash. Boundary checkpoint payloads additionally verify cumulative '
              'optimizer clocks and exact model/optimizer/RNG continuity when a stage advances. '
              'CPU tests exercise inclusive thresholds, stop priority, '
              'optimizer continuity, retention panels, pool identity and capacity pruning. '
              'These are local single-GPU task probes, not resilient DiLoCo training: no R/NDP '
              'conformance or Slurm Partition/QOS claim is made (R16 evidence discipline only).', '',
              'A first launcher preflight stopped before GPU acquisition because the venv lacks '
              'pip; package enumeration was switched to stdlib importlib.metadata. That failed '
              'preflight and its source snapshot are retained; no training was started or recipe changed.', '',
              'After launch, a one-line capacity-retention correction explicitly protects stage-entry '
              'checkpoints even when a later best supersedes them. No run triggered pruning (free '
              'space remained above 400GiB), so this correction does not alter executed training or '
              'any frontier. Exact executed source remains in `receipts/full_launch/runner_at_launch.py`; '
              'the registration and each run config bind that snapshot, not the later correction. '
              'A regression test covers the corrected non-10K stage-entry case.', '']
    return '\n'.join(lines)


def aggregate(root=ROOT, report_path=Path('X5-report.md')):
    registration = load(root / 'registration.json')
    assert registration['matrix'] == matrix()
    assert registration['runner_sha256'] == digest(root / 'receipts/full_launch/runner_at_launch.py')
    assert registration['launcher_sha256'] == digest(root / 'receipts/full_launch/launcher_at_launch.sh')
    assert 'exit_code=0' in (root / 'receipts/full_launch/exit.txt').read_text()
    runs = [audit_run(root, row) for row in matrix()]
    for seed in range(3):
        e = load(root / f'e97_seed{seed}' / 'data_receipt.json')
        g = load(root / f'gdn2_seed{seed}' / 'data_receipt.json')
        assert e['sha256'] == g['sha256'], f'seed {seed}: unmatched frozen pools'
    for run in runs:
        run['checkpoint_payloads_verified'] = audit_checkpoint_payloads(root / run['name'])
        cfg = load(root / run['name'] / 'config.json')
        source = next(v for k, v in cfg['source_sha256'].items() if k.endswith('run_invertible_stage5_curriculum_8m.py'))
        assert source == registration['runner_sha256']
    summary = {'status': 'complete_audited', 'generated_at': time.time(), 'registration': registration,
               'runs': runs, 'frontiers': {arm: [r['frontier'] for r in runs if r['arm'] == arm]
                                          for arm in ('e97', 'gdn2')},
               'suffix_theorem': SUFFIX_THEOREM, 'suffix_floor_percent': FLOOR * 100,
               'marginal_reference_nats': MARGINAL}
    write_json(root / 'results_summary.json', summary)
    report_path.write_text(report(summary))
    (root / 'X5-report.md').write_text(report(summary))
    print(json.dumps({'status': summary['status'], 'frontiers': summary['frontiers']}))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=Path('X5-report.md'))
    args = parser.parse_args()
    aggregate(args.root, args.report)
