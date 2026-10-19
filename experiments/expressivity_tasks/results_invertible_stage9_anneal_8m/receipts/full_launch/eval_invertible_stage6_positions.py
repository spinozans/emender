"""X6D: snapshot-only positional diagnosis; never writes to live run directories."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks.eval_invertible_stage5c_positions import COUNT_KEYS
from experiments.expressivity_tasks.run_invertible_stage2 import metric_counts
from experiments.expressivity_tasks.run_invertible_stage3_8m import digest, finish_metrics, write_json
from experiments.expressivity_tasks import run_invertible_stage6_truecur_8m as runner

ROOT = runner.ROOT
EVIDENCE = ROOT / 'receipts/positional_X6D'
SCRATCH = Path('/home/erikg/emender-scratch/X6D-positiondiagnosis')
NAMES = tuple(f'A_{arm}_seed{seed}' for arm in ('e97', 'gdn2') for seed in range(3)) + ('B_e97_seed0', 'B_gdn2_seed0')
DEPTHS = (1, 2, 4, 8, 16)
SOURCES = (Path(__file__), Path(runner.__file__),
           Path('experiments/expressivity_tasks/eval_invertible_stage5c_positions.py'),
           Path('experiments/expressivity_tasks/run_invertible_stage3_8m.py'),
           Path('experiments/expressivity_tasks/run_invertible_stage2.py'),
           Path('experiments/expressivity_tasks/run_invertible_stage5_curriculum_8m.py'),
           Path('experiments/expressivity_tasks/tasks/invertible_quadratic.py'),
           Path('ndm/models/hybrid_ladder.py'))


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(path.read_text())


def position_counts(pred, target, mask, losses):
    depth = target.shape[1] - 2
    if pred.shape != (512, depth + 2) or target.shape != pred.shape or mask.shape != pred.shape:
        raise ValueError('expected 512 frozen sequences')
    if mask[:, :2].any() or not mask[:, 2:].all() or losses.shape != (512, depth):
        raise ValueError('expected exactly depth driver targets')
    rows = []
    for j in range(depth):
        m = np.zeros_like(mask)
        m[:, j + 2] = True
        counts = metric_counts(pred, target, m, 61)
        rows.append({'driver_position': j + 1, 'token_index': j + 2,
                     **{k: counts[k] for k in COUNT_KEYS},
                     'loss': float(losses[:, j].astype(np.float64).mean())})
    return rows


def reconcile(aggregate, positions, recorded):
    sums = {k: sum(row[k] for row in positions) for k in COUNT_KEYS}
    if sums != {k: aggregate[k] for k in COUNT_KEYS}:
        raise ValueError('position totals do not reconcile')
    if positions[-1]['correct'] != aggregate['terminal_correct']:
        raise ValueError('terminal counts do not reconcile')
    if sum(row['correct'] for row in positions[8:]) != aggregate['post_suffix7_correct']:
        raise ValueError('suffix counts do not reconcile')
    keys = (*COUNT_KEYS, 'terminal_correct', 'terminal_count', 'post_suffix7_correct', 'post_suffix7_count')
    differences = {k: aggregate[k] - recorded[k] for k in keys if aggregate[k] != recorded[k]}
    return {'position_sums': sums, 'recorded_count_differences': differences,
            'recorded_counts_match': not differences,
            'recorded_loss_delta': aggregate['loss'] - recorded['loss'],
            'position_mean_loss_delta': sum(r['loss'] for r in positions) / len(positions) - aggregate['loss']}


def safe_copy(source, dest, expected):
    """Hash source before and after copying; never deserialize a live payload."""
    source, dest = Path(source), Path(dest)
    if source.resolve() == dest.resolve():
        raise ValueError('snapshot must not overwrite source')
    if digest(source) != expected:
        raise ValueError(f'pre-copy hash mismatch: {source}')
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, dest)
    if digest(dest) != expected or digest(source) != expected:
        raise ValueError(f'unstable snapshot: {source}')


def capture_json(path):
    # validation/checkpoint indexes are rewritten nonatomically by live trainers.
    for _ in range(20):
        raw = path.read_bytes()
        try:
            return json.loads(raw), hashlib.sha256(raw).hexdigest()
        except json.JSONDecodeError:
            time.sleep(.1)
    raise ValueError(f'cannot capture complete JSON: {path}')


def capture_lines(path):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    complete = b''.join(lines)
    return [json.loads(line) for line in lines], hashlib.sha256(complete).hexdigest()


def snapshot(root=ROOT, evidence=EVIDENCE, scratch=SCRATCH):
    if (evidence / 'inventory.json').exists():
        raise FileExistsError('registered inventory already exists; do not reselect')
    if not (evidence / 'registration.md').exists():
        raise ValueError('register diagnostic before snapshot/evaluation')
    inventory = {'task': 'X6D', 'started_utc': utc(), 'runs': {},
                 'registration_sha256': digest(evidence / 'registration.md')}
    for name in NAMES:
        live = root / name
        local = scratch / name
        local.mkdir(parents=True, exist_ok=True)
        sources, captured = {}, {}
        for filename in ('config.json', 'stages.json', 'validation.json', 'checkpoints.json', 'data_receipt.json'):
            captured[filename], sources[filename] = capture_json(live / filename)
            write_json(local / filename, captured[filename])
        transitions, sources['transitions.jsonl'] = capture_lines(live / 'transitions.jsonl')
        pruning, sources['pruning.jsonl'] = capture_lines(live / 'pruning.jsonl')
        stages = [s for s in captured['stages.json'] if s['depth'] in DEPTHS]
        if len(stages) != len(DEPTHS):
            raise ValueError(f'{name}: required completed stages missing')
        history = captured['validation.json']
        entries = {e['depth']: e for e in transitions if e['event'] == 'stage_entry' and e['depth'] <= 32}
        jobs, boundaries, missing, checkpoints = [], [], [], {}

        def row(depth, step, stage_step):
            return next(r for r in history if (r['depth'], r['step'], r['stage_step']) == (depth, step, stage_step))

        def copy_cp(cp):
            if cp['path'] not in checkpoints:
                dest = local / Path(cp['path']).name
                safe_copy(cp['path'], dest, cp['sha256'])
                checkpoints[cp['path']] = {**cp, 'snapshot_path': str(dest)}
            return checkpoints[cp['path']]

        def add(cp, depth, recorded, role, stage_depth, provenance='direct retained checkpoint'):
            jobs.append({'checkpoint': copy_cp(cp), 'pool_depth': depth, 'recorded': recorded,
                         'role': role, 'stage_depth': stage_depth, 'provenance': provenance})

        for stage in stages:
            d = stage['depth']
            for role, cp in (('best', stage['best']['checkpoint']), ('exit', stage['checkpoint'])):
                validation = row(d, cp['step'], cp['stage_step'])
                add(cp, d, validation['validation_by_depth'][str(d)], role, d)
            entry = entries[d]
            cp = entry['checkpoint']
            original_exists = Path(cp['path']).exists()
            carried = entry['carried_checkpoint']
            pruning_events = [{'time': p['time'], 'pruned': cp['path']} for p in pruning if cp['path'] in p['pruned']]
            if not original_exists:
                missing.append({'depth': d, 'checkpoint': cp, 'pruning_events': pruning_events})
            if original_exists:
                entry_cp, provenance = cp, 'direct retained stage entry'
            elif carried is not None:
                entry_cp = carried
                provenance = ('entry (sha-equal to prior carried checkpoint)' if cp['sha256'] == carried['sha256'] else
                              'surrogate entry: retained carried checkpoint; entry payload missing and SHA unequal')
            else:
                entry_cp = None
                provenance = 'unavailable: initial entry pruned; no retained predecessor'
            entry_row = row(d, entry['step'], 0)
            if entry_cp is not None:
                add(entry_cp, d, entry_row['validation_by_depth'][str(d)], 'entry', d, provenance)
            next_entry = entries.get(d * 2)
            if next_entry:
                next_cp = next_entry['checkpoint']
                carried_cp = stage['carried_checkpoint']
                next_metrics = row(d * 2, next_entry['step'], 0)['validation_by_depth'][str(d)]
                carried_metrics = row(d, carried_cp['step'], carried_cp['stage_step'])['validation_by_depth'][str(d)]
                direct = Path(next_cp['path']).exists()
                if direct:
                    add(next_cp, d, next_metrics, 'boundary_entry_previous_pool', d)
                boundaries.append({'from_depth': d, 'to_depth': d * 2, 'entry_checkpoint': next_cp,
                                   'carried_checkpoint': carried_cp, 'entry_payload_present': direct,
                                   'payload_sha_equal': next_cp['sha256'] == carried_cp['sha256'],
                                   'carried_recorded': carried_metrics, 'entry_recorded_previous_pool': next_metrics,
                                   'recorded_metrics_exactly_equal': carried_metrics == next_metrics,
                                   'entry_pruning_events': [{'time': p['time'], 'pruned': next_cp['path']}
                                                            for p in pruning if next_cp['path'] in p['pruned']]})
        data = captured['data_receipt.json']
        safe_copy(live / 'frozen_validation.npz', local / 'frozen_validation.npz', data['sha256'])
        relevant_rows = [r for r in history if r['depth'] <= 32 and
                         any((r['depth'], r['step'], r['stage_step']) ==
                             (j['stage_depth'], j['checkpoint']['step'], j['checkpoint']['stage_step'])
                             for j in jobs if j['role'] in ('best', 'exit')) or
                         (r['stage_step'] == 0 and r['depth'] <= 32)]
        metadata = {'name': name, 'captured_utc': utc(), 'source_sha256': sources,
                    'config': captured['config.json'], 'stages': stages, 'entries': entries,
                    'validation_rows': relevant_rows, 'data_receipt': data,
                    'missing_entry_checkpoints': missing, 'boundaries': boundaries,
                    'jobs': jobs, 'snapshot_data_path': str(local / 'frozen_validation.npz')}
        write_json(evidence / 'metadata' / f'{name}.json', metadata)
        inventory['runs'][name] = {'metadata': str(evidence / 'metadata' / f'{name}.json'),
                                   'sha256': digest(evidence / 'metadata' / f'{name}.json'),
                                   'jobs': len(jobs), 'missing_entries': len(missing)}
        print(json.dumps({'snapshot': name, 'jobs': len(jobs), 'missing_entries': len(missing)}), flush=True)
    inventory['finished_utc'] = utc()
    write_json(evidence / 'inventory.json', inventory)


def evaluate(evidence=EVIDENCE):
    import torch
    import torch.nn.functional as F

    inventory = read_json(evidence / 'inventory.json')
    if digest(evidence / 'registration.md') != inventory['registration_sha256']:
        raise ValueError('registration changed after snapshot')
    if (evidence / 'evaluation_summary.json').exists() or (evidence / 'positional').exists():
        raise FileExistsError('no implicit re-evaluation or overwrite of positional evidence')
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    summary = {'started_utc': utc(), 'receipts': [], 'aliases': [], 'runtime': {
        'host': platform.node(), 'python': platform.python_version(), 'torch': torch.__version__,
        'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES'),
        'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'co_resident': True, 'exclusive_lease': False, 'batch': 32,
        'precision': 'bf16 autocast; fp32 parameters', 'deterministic_algorithms': True,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()},
        'sources': {str(p): digest(p) for p in SOURCES}}
    started = time.monotonic()
    for name, info in inventory['runs'].items():
        meta_path = Path(info['metadata'])
        if digest(meta_path) != info['sha256']:
            raise ValueError('snapshot metadata changed')
        meta = read_json(meta_path)
        cfg = meta['config']
        data_path = Path(meta['snapshot_data_path'])
        if digest(data_path) != meta['data_receipt']['sha256']:
            raise ValueError('frozen NPZ mismatch')
        with np.load(data_path) as data:
            pools = {d: tuple(data[f'T{d}_{i}'] for i in range(3)) for d in DEPTHS}
        for d, pool in pools.items():
            hashes = [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]
            if hashes != meta['data_receipt']['depths'][str(d)]['array_sha256']:
                raise ValueError('frozen array mismatch')
        torch.manual_seed(cfg['seed'])
        model = runner.build_model(cfg['arm'], cfg['curriculum']).cuda()
        cache = {}
        for job in meta['jobs']:
            cp, d = job['checkpoint'], job['pool_depth']
            key = (cp['sha256'], d)
            if key in cache:
                prior = cache[key]
                comparison = reconcile(prior['aggregate'], prior['positions'], job['recorded'])
                summary['aliases'].append({'name': name, **job, 'receipt': prior['path'], 'reconciliation': comparison})
            else:
                if digest(cp['snapshot_path']) != cp['sha256']:
                    raise ValueError('snapshot checkpoint mismatch')
                raw = torch.load(cp['snapshot_path'], map_location='cpu', weights_only=False)
                if (raw['step'], raw['depth'], raw['stage_step'], raw['weights']) != (cp['step'], cp['depth'], cp['stage_step'], 'ScheduleFree eval x'):
                    raise ValueError('checkpoint identity mismatch')
                model.load_state_dict(raw['model'], strict=True)
                del raw
                model.eval()
                predictions, losses, totals, loss_sum = [], [], {}, 0.
                pool = pools[d]
                with torch.no_grad():
                    for start in range(0, 512, 32):
                        batch = tuple(a[start:start + 32] for a in pool)
                        logits, y, mask, _ = runner.forward_batch(model, batch, cfg['curriculum'])
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError('nonfinite logits')
                        pred = logits.argmax(-1).cpu().numpy()
                        predictions.append(pred)
                        loss_sum += F.cross_entropy(logits[mask].float(), y[mask], reduction='sum').item()
                        losses.append(F.cross_entropy(logits[mask].float(), y[mask], reduction='none').reshape(32, d).cpu().numpy())
                        for k, v in metric_counts(pred, batch[1], batch[2], 61).items():
                            totals[k] = totals.get(k, 0) + v
                positions = position_counts(np.concatenate(predictions), pool[1], pool[2], np.concatenate(losses))
                aggregate = finish_metrics(totals, loss_sum, 61)
                comparison = reconcile(aggregate, positions, job['recorded'])
                path = evidence / 'positional' / name / f"{Path(cp['path']).stem}_poolT{d}.json"
                receipt = {'task': 'X6D', 'name': name, **job, 'evaluated_utc': utc(),
                           'examples_per_position': 512, 'weights': 'ScheduleFree eval x; strict load; no optimizer transform',
                           'carry_probability_at_evaluation': 0.,
                           'carry_probability_at_checkpoint_training': runner.carry_probability(cp['stage_step']) if cfg['curriculum'] == 'B' else 0.,
                           'data_sha256': meta['data_receipt']['sha256'],
                           'array_sha256': meta['data_receipt']['depths'][str(d)]['array_sha256'],
                           'metadata_sha256': info['sha256'], 'positions': positions, 'aggregate': aggregate,
                           'reconciliation': comparison, 'runtime': summary['runtime'], 'sources': summary['sources']}
                write_json(path, receipt)
                prior = {'path': str(path), 'positions': positions, 'aggregate': aggregate}
                cache[key] = prior
                summary['receipts'].append({'path': str(path), 'sha256': digest(path)})
                print(json.dumps({'name': name, 'stage': job['stage_depth'], 'role': job['role'], 'pool': d,
                                  'joint': [r['correct'] for r in positions], 'comparison': comparison}), flush=True)
            if not comparison['recorded_counts_match'] or abs(comparison['recorded_loss_delta']) > 1e-6:
                write_json(evidence / 'discrepancy.json', {'job': job, 'reconciliation': comparison})
                raise RuntimeError('discrepancy retained; stopping without patch/rerun')
        del model
        torch.cuda.empty_cache()
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['finished_utc'] = utc()
    write_json(evidence / 'evaluation_summary.json', summary)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('mode', choices=('snapshot', 'evaluate'))
    args = parser.parse_args()
    if args.mode == 'snapshot':
        snapshot()
    else:
        evaluate()


if __name__ == '__main__':
    main()
