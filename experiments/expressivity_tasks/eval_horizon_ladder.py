"""X21E: registered, hash-verified horizon ladder; evaluation only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from experiments.expressivity_tasks import eval_invertible_stage18_freshdata as x18
from experiments.expressivity_tasks import run_invertible_stage17_carrywean_8m as x17
from experiments.expressivity_tasks.horizon_state_quantization import PRECISIONS, state_quantization

ROOT = Path('experiments/expressivity_tasks/results_stage21e_horizonladder')
SCRATCH = Path('/home/erikg/emender-scratch/X21E-horizonladder')
MULTIPLIERS = (1, 2, 4, 8, 16, 32)
BASE = x18.BASE
x7 = x18.x7


def utc():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    x7.write_json(path, data)


def register():
    path = ROOT / 'registration.json'
    if path.exists():
        raise FileExistsError('registration frozen')
    old = load(x18.ROOT / 'registration.json')
    candidates = [{**r, 'wean': False, 'receipt_source': str(x18.ROOT / 'registration.json')}
                  for r in old['champions']]
    for name, relative, pool_relative, wean in (
        ('X16E-curriculum', '16_ignition_8m/curriculum_extension/e97_extension_seed0',
         '16_ignition_8m/e97_curriculum_seed0', False),
        ('X17-wean', '17_carrywean_8m/infra_restart/e97_carrywean_seed0',
         '17_carrywean_8m/e97_carrywean_seed0', True)):
        directory = BASE / ('results_invertible_stage' + relative)
        result = load(directory / 'result.json')
        final = result['final']
        candidates.append({'name': name, 'architecture': 'e97', 'seed': 0, 'depth': 64,
            'checkpoint': final['checkpoint'], 'wean': wean,
            'frozen_pool': str(BASE / ('results_invertible_stage' + pool_relative) / 'frozen_validation.npz'),
            'frozen_metrics': {k: final[k] for k in ('aggregate', 'positions', 'perfect_prefix_depth')},
            'receipt_source': str(directory / 'result.json')})
    runs = []
    for candidate in candidates:
        r = {**candidate, 'receipt_source_sha256': x7.digest(candidate['receipt_source'])}
        try:
            cp = r['checkpoint']
            dest = SCRATCH / r['name'] / Path(cp['path']).name
            x18.safe_copy(cp['path'], dest, cp['sha256'])
            pool = SCRATCH / r['name'] / 'frozen_validation.npz'
            pool_sha = r.get('frozen_pool_sha256', x7.digest(r['frozen_pool']))
            x18.safe_copy(r['frozen_pool'], pool, pool_sha)
            r.update({'snapshot_path': str(dest), 'frozen_pool': str(pool),
                      'frozen_pool_sha256': pool_sha, 'usable': True})
        except (OSError, ValueError) as error:
            r.update({'usable': False, 'unusable_reason': repr(error)})
        runs.append(r)
    sources = x18.sources()
    for source in (Path(__file__), BASE / 'horizon_state_quantization.py',
                   Path('tests/test_eval_horizon_ladder.py'), ROOT / 'REGISTRATION.md'):
        sources[str(source)] = x7.digest(source)
    write(path, {'task': 'X21E', 'registered_utc': utc(), 'host': platform.node(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'predictions_sha256': x7.digest(ROOT / 'REGISTRATION.md'),
        'champions': runs, 'source_sha256': sources, 'examples': 512, 'batch': 32,
        'multipliers': MULTIPLIERS, 'precisions': PRECISIONS,
        'seed_rule': '202610220000 + actual_driver_horizon', 'training_updates': 0})
    print(json.dumps({'registration_sha256': x7.digest(path),
                      'usable': [r['name'] for r in runs if r['usable']]}))


def registration():
    reg = load(ROOT / 'registration.json')
    for path, sha in reg['source_sha256'].items():
        x18.require_hash(path, sha)
    return reg


def generate():
    reg = registration()
    if (ROOT / 'generation.json').exists():
        raise FileExistsError('generation frozen')
    depths = sorted({r['depth'] * m for r in reg['champions'] if r['usable'] for m in MULTIPLIERS})
    pools = {}
    for depth in depths:
        seed = 202610220000 + depth
        pool = x18.fresh_training_batch(x18.x12.x10.InvertibleQuadraticTask(61),
                                        512, depth, np.random.default_rng(seed))
        x18.validate_pool(pool, depth)
        if len(x18.row_set(pool[0])) != 512:
            raise ValueError('duplicate fresh inputs')
        path = ROOT / 'pools' / f'T{depth}.npz'
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **{f'T{depth}_{i}': a for i, a in enumerate(pool)})
        pools[str(depth)] = {'path': str(path), 'seed': seed, 'array_sha256': x18.hashes(pool),
            'archive_sha256': x7.digest(path), 'scalar_targets_verified': 512 * depth,
            'unique_full_inputs': 512}
    write(ROOT / 'generation.json', {'finished_utc': utc(), 'pools': pools,
        'registration_sha256': x7.digest(ROOT / 'registration.json')})


def metrics(pred, targets, losses):
    """Linear-in-horizon version of the canonical joint/coordinate instrument."""
    pred, targets = pred[:, 2:], targets[:, 2:]
    valid = (pred >= 0) & (pred < 3721)
    joint = valid & (pred == targets)
    xc = valid & (pred // 61 == targets // 61)
    yc = valid & (pred % 61 == targets % 61)
    positions = [{'driver_position': i + 1, 'count': len(pred),
        'correct': int(joint[:, i].sum()), 'x_correct': int(xc[:, i].sum()),
        'y_correct': int(yc[:, i].sum()), 'out_of_range': int((~valid[:, i]).sum()),
        'loss': float(losses[:, i].astype(np.float64).mean())}
        for i in range(pred.shape[1])]
    totals = {key: sum(p[key] for p in positions)
              for key in ('count', 'correct', 'x_correct', 'y_correct', 'out_of_range')}
    totals.update({'terminal_correct': int(joint[:, -1].sum()), 'terminal_count': len(pred),
        'post_suffix7_correct': int(joint[:, 8:].sum()), 'post_suffix7_count': joint[:, 8:].size})
    return {'aggregate': x18.x12.x10.x7.finish_metrics(totals, float(losses.astype(np.float64).sum()), 61),
            'positions': positions,
            'perfect_prefix_depth': next((p['driver_position'] - 1 for p in positions
                                          if p['correct'] != len(pred)), len(positions))}


def evaluate_pool(model, run, pool, directory, bindings):
    import torch
    import torch.nn.functional as F
    directory.mkdir(parents=True, exist_ok=True)
    predictions, losses = [], []
    forward = x17.forward_batch if run['wean'] else x7.forward_batch
    model.eval()
    with torch.no_grad():
        for start in range(0, 512, 32):
            batch_path = directory / f'batch_{start // 32:02d}.json'
            if batch_path.exists():
                raise FileExistsError('batch receipt already exists; no blind replay')
            begun = utc()
            timer = time.monotonic()
            try:
                batch = tuple(a[start:start + 32] for a in pool)
                logits, target, mask = forward(model, batch)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('nonfinite evaluation logits')
                pred = logits.argmax(-1).cpu().numpy()
                loss = F.cross_entropy(logits[mask].float(), target[mask], reduction='none')
                loss = loss.reshape(32, pool[0].shape[1] - 2).cpu().numpy()
                predictions.append(pred)
                losses.append(loss)
                result = metrics(pred, batch[1], loss)
                write(batch_path, {**bindings, 'started_utc': begun, 'finished_utc': utc(),
                    'status': 'complete', 'first_example': start, 'examples': 32,
                    'seconds': time.monotonic() - timer, 'training_updates': 0,
                    'input_target_mask_sha256': x18.hashes(batch),
                    'prediction_sha256': hashlib.sha256(pred.tobytes()).hexdigest(),
                    'loss_sha256': hashlib.sha256(loss.tobytes()).hexdigest(),
                    'aggregate': result['aggregate']})
                del logits, target, mask
            except Exception as error:
                write(batch_path, {**bindings, 'started_utc': begun, 'finished_utc': utc(),
                    'status': 'infrastructure_censored', 'first_example': start,
                    'error': repr(error), 'training_updates': 0})
                raise
    return metrics(np.concatenate(predictions), pool[1], np.concatenate(losses))


def controls_match(measured, historical):
    keys = ('count', 'correct', 'x_correct', 'y_correct', 'out_of_range')
    return (len(measured['positions']) == len(historical['positions']) and
            all(all(a[k] == b[k] for k in keys)
                for a, b in zip(measured['positions'], historical['positions'])))


def evaluate(precision):
    import torch
    reg = registration()
    generation = load(ROOT / 'generation.json')
    x18.require_hash(ROOT / 'registration.json', generation['registration_sha256'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('registered GPU0 and CUBLAS binding required')
    torch.manual_seed(0)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for run in reg['champions']:
        if not run['usable']:
            continue
        cp = run['checkpoint']
        x18.require_hash(run['snapshot_path'], cp['sha256'])
        saved = torch.load(run['snapshot_path'], map_location='cpu', weights_only=False)
        if saved['step'] != cp['step'] or saved['weights'] != 'ordinary live parameters':
            raise ValueError('step/weight representation mismatch')
        model = (x17.build_model() if run['wean'] else x18.x12.build_model(run['architecture'])).cuda()
        model.load_state_dict(saved['model'], strict=True)
        del saved
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        bindings = {'name': run['name'], 'precision': precision, 'champion_sha256': cp['sha256'],
            'registration_sha256': x7.digest(ROOT / 'registration.json'),
            'generation_sha256': x7.digest(ROOT / 'generation.json')}
        if precision == 'native':
            path = ROOT / 'controls' / run['name']
            x18.require_hash(run['frozen_pool'], run['frozen_pool_sha256'])
            frozen = x18.load_pool(run['frozen_pool'], run['depth'])
            measured = evaluate_pool(model, run, frozen, path, {**bindings, 'control': 'historical-frozen'})
            match = controls_match(measured, run['frozen_metrics'])
            write(path / 'result.json', {**bindings, 'metrics': measured, 'integer_counts_exact': match})
            if not match:
                raise ValueError('historical frozen integer counts differ')
        with state_quantization(model, run['architecture'], precision):
            for multiplier in MULTIPLIERS:
                depth = run['depth'] * multiplier
                directory = ROOT / 'evaluations' / run['name'] / precision / f'{multiplier:02d}x'
                if (directory / 'result.json').exists():
                    raise FileExistsError(directory)
                info = generation['pools'][str(depth)]
                x18.require_hash(info['path'], info['archive_sha256'])
                pool = x18.load_pool(info['path'], depth)
                if x18.hashes(pool) != info['array_sha256']:
                    raise ValueError('pool hash mismatch')
                bound = {**bindings, 'multiplier': multiplier, 'depth': depth,
                         'pool_archive_sha256': info['archive_sha256']}
                try:
                    result = evaluate_pool(model, run, pool, directory, bound)
                    unchanged = all(torch.equal(before[k], v) for k, v in model.state_dict().items())
                    if not unchanged:
                        raise ValueError('model tensors changed')
                    x18.require_hash(cp['path'], cp['sha256'])
                    x18.require_hash(run['snapshot_path'], cp['sha256'])
                    write(directory / 'result.json', {**bound, 'status': 'complete', 'metrics': result,
                        'weights_unchanged': True, 'training_updates': 0, 'finished_utc': utc(),
                        'batch_receipt_sha256': {p.name: x7.digest(p) for p in sorted(directory.glob('batch_*.json'))}})
                    print(json.dumps({'name': run['name'], 'precision': precision, 'T': depth,
                        'surviving': sum(p['correct'] >= 480 for p in result['positions']),
                        'first8': [p['correct'] for p in result['positions'][:8]],
                        'accuracy': result['aggregate']['accuracy']}), flush=True)
                except Exception as error:
                    write(directory / 'result.json', {**bound, 'status': 'infrastructure_censored',
                        'error': repr(error), 'training_updates': 0, 'finished_utc': utc()})
                    print(json.dumps({'censored': str(directory), 'error': repr(error)}), flush=True)
                    if isinstance(error, torch.cuda.OutOfMemoryError):
                        torch.cuda.empty_cache()
        del model, before
        torch.cuda.empty_cache()
    write(ROOT / f'runtime_{precision}.json', {'finished_utc': utc(), 'precision': precision,
        'python': platform.python_version(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': x18.gpu_snapshot(), 'training_updates': 0, 'registration_sha256': bindings['registration_sha256']})


def alignment(base, current):
    baseline = [p['driver_position'] for p in base if p['correct'] >= 480]
    shift = len(current) - len(base)
    rows = []
    for p in current:
        if p['correct'] < 480:
            continue
        position = p['driver_position']
        start = min(baseline, key=lambda b: abs(position - b)) if baseline else None
        suffix = min(baseline, key=lambda b: abs(position - (b + shift))) if baseline else None
        a = start is not None and position == start
        b = suffix is not None and position == suffix + shift
        rows.append({'position': position, 'correct': p['correct'], 'start_candidate': start,
            'suffix_candidate': suffix, 'start_offset': position - start if start is not None else None,
            'suffix_offset': position - (suffix + shift) if suffix is not None else None,
            'classification': 'ambiguous' if a and b else 'start-anchored' if a else 'suffix-relative' if b else 'unclassified',
            'structure_relative': 'not identifiable: no designated events'})
    return rows


def failure(base, rungs):
    positions = [p['driver_position'] for p in base if p['correct'] >= 480]
    if not positions:
        return {'status': 'already-failed-1x', 'multiplier': 1}
    for m in MULTIPLIERS:
        result = rungs.get(m)
        if result is None or result['status'] != 'complete':
            return {'status': 'infrastructure_censored', 'multiplier': m}
        if any(result['metrics']['positions'][p - 1]['correct'] < 480 for p in positions):
            return {'status': 'already-failed-1x' if m == 1 else 'observed-failure', 'multiplier': m}
    return {'status': 'right-censored', 'beyond_multiplier': 32}


def report():
    reg = registration()
    curves, shifts, failures, parity = {}, {}, {}, {}
    for r in reg['champions']:
        if not r['usable']:
            continue
        name = r['name']
        base = load(ROOT / 'evaluations' / name / 'native' / '01x' / 'result.json')['metrics']['positions']
        curves[name], shifts[name], failures[name] = {}, {}, {}
        for precision in PRECISIONS:
            rungs = {}
            for m in MULTIPLIERS:
                path = ROOT / 'evaluations' / name / precision / f'{m:02d}x' / 'result.json'
                if path.exists():
                    rungs[m] = load(path)
            if not rungs:
                continue
            curves[name][precision] = {str(m): {'depth': r['depth'] * m, 'status': result['status'],
                **({'accuracy': result['metrics']['aggregate']['accuracy'],
                    'surviving_columns': [p['driver_position'] for p in result['metrics']['positions'] if p['correct'] >= 480],
                    'perfect_prefix': result['metrics']['perfect_prefix_depth']} if result['status'] == 'complete' else {})}
                for m, result in rungs.items()}
            fail = failure(base, rungs)
            fail['training_horizon'] = r['depth']
            if 'multiplier' in fail:
                fail['depth'] = r['depth'] * fail['multiplier']
            failures[name][precision] = fail
            if precision == 'native':
                shifts[name] = {str(m): alignment(base, result['metrics']['positions'])
                    for m, result in rungs.items() if result['status'] == 'complete'}
            if precision == 'wrapper-fp32' and 1 in rungs and rungs[1]['status'] == 'complete':
                native = load(ROOT / 'evaluations' / name / 'native' / '01x' / 'result.json')['metrics']
                wrapper = rungs[1]['metrics']
                parity[name] = {'integer_counts_exact': controls_match(wrapper, native),
                    'joint_count_deltas': [a['correct'] - b['correct'] for a, b in zip(wrapper['positions'], native['positions'])],
                    'loss_delta': wrapper['aggregate']['loss'] - native['aggregate']['loss']}
        layer = x17.build_model().layers[0] if r['wean'] else x18.x12.build_model(r['architecture']).layers[0]
        h, n, v = ((layer.n_heads, layer.n_state, layer.head_v_dim) if r['architecture'] == 'e97'
                   else (layer.gdn.num_v_heads, layer.gdn.head_k_dim, layer.gdn.head_v_dim))
        failures[name]['bits_per_example_all_four_layers'] = {
            p: 4 * h * (n * v * bits + (32 if p.startswith('int') else 0))
            for p, bits in [('native', 32), ('bf16', 16), ('fp8', 8), ('int8', 8), ('int4', 4)]}
    write(ROOT / 'summary.json', {'reported_utc': utc(), 'curves': curves, 'alignment': shifts,
        'horizon_at_failure': failures, 'wrapper_fp32_control': parity,
        'structure_relative': 'not identifiable in iid task without designated events',
        'registration_sha256': x7.digest(ROOT / 'registration.json')})
    lines = ['# X21E first-tranche results', '',
        'Native baseline uses bf16 autocast but fp32 recurrent state accumulators.',
        'No training. Full position metrics and every minibatch receipt are retained.', '',
        '|Champion|State precision|Surviving columns at 1/2/4/8/16/32x|Failure/censoring|',
        '|---|---|---|---|']
    for name, precisions in curves.items():
        for p, rungs in precisions.items():
            counts = [len(rungs[str(m)].get('surviving_columns', [])) if str(m) in rungs and rungs[str(m)]['status'] == 'complete' else 'censored' for m in MULTIPLIERS]
            lines.append(f'|{name}|{p}|{counts}|{failures[name][p]}|')
    lines += ['', '## Alignment',
        'See summary.json for every surviving column and both start/suffix offsets.',
        'No designated events: structure-relative is not identifiable. Ties ambiguous.', '',
        '## Wrapper controls', json.dumps(parity, indent=2), '',
        'Architecture comparisons are descriptive: T16 cold AdamW pairs; T64 cold',
        'AdamW with seed2 vs seed0 confound. Staged overlays are not matched recipes.',
        'Failure is loss of a native1x >=480/512 column; >32x is right-censored,',
        'never proof of unbounded computation. Quantization includes integer scale metadata.',
        'Resilient DiLoCo R/NDP/V21S/ISP requirements: not applicable; no conformance claim.']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'generate', 'evaluate', 'report'))
    parser.add_argument('--precision', choices=PRECISIONS, default='native')
    args = parser.parse_args()
    evaluate(args.precision) if args.action == 'evaluate' else globals()[args.action]()
