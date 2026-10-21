"""X18: frozen champions, fresh in-distribution pools, read-only evaluation."""
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

from experiments.expressivity_tasks import run_invertible_stage12_separation_8m as x12
from experiments.expressivity_tasks.eval_invertible_stage6_positions import safe_copy, reconcile
from experiments.expressivity_tasks.run_invertible_stage2 import fresh_training_batch

ROOT = Path('experiments/expressivity_tasks/results_invertible_stage18_freshdata_8m')
SCRATCH = Path('/home/erikg/emender-scratch/X18-freshdata')
BASE = Path('experiments/expressivity_tasks')
# Supervisor-approved receipted champions, not a new search or uniform final selection.
CHAMPIONS = (
    ('X10-adamw', '10_optimizers', 'adamw_seed0', 'e97', 0, 16, 468000),
    ('X11-GDN2', '11_gdn2_adamw', 'gdn2_adamw_seed0', 'gdn2', 0, 16, 482500),
    ('X12-GDN2', '12_separation', 'gdn2_bf16_seed0', 'gdn2', 0, 64, 1000000),
    ('X13-seed2', '13_zerocheck', 'e97_bf16_seed2', 'e97', 2, 64, 1000000),
)
SEEDS = {16: 202610211816, 64: 202610211864}
x7 = x12.x7


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def hashes(pool):
    return [hashlib.sha256(a.tobytes()).hexdigest() for a in pool]


def require_hash(path, sha):
    if x7.digest(path) != sha:
        raise ValueError(f'hash mismatch: {path}')


def sources():
    import fla
    paths = {Path(__file__), Path('tests/test_eval_invertible_stage18_freshdata.py')}
    paths.update(BASE.glob('*.py'))
    paths.update((BASE / 'tasks').glob('*.py'))
    paths.update(Path('ndm').rglob('*.py'))
    paths.update(Path(fla.__file__).parent.rglob('*.py'))
    return {str(p): x7.digest(p) for p in sorted(paths)}


def register():
    if (ROOT / 'registration.json').exists():
        raise FileExistsError('registration frozen; do not reselect')
    runs = []
    for name, stage, directory, architecture, seed, depth, step in CHAMPIONS:
        live = BASE / f'results_invertible_stage{stage}_8m' / directory
        historical = read_json(live / 'config.json')['registration']
        # All model/data/kernel bytes must agree with the original training receipt.
        verified = {}
        for path, sha in historical['source_sha256'].items():
            if path.startswith('ndm/') or path.endswith(('/invertible_quadratic.py',
                    '/run_invertible_stage3_8m.py', '/run_invertible_stage7_deepfirst_8m.py')):
                require_hash(path, sha)
                verified[path] = sha
        for path, sha in historical.get('dependency_sha256', {}).items():
            require_hash(path, sha)
            verified[path] = sha
        row = next(r for r in read_json(live / 'validation.json') if r['step'] == step)
        cp = row['checkpoint']
        result = read_json(live / 'result.json')
        selected = result['best']['checkpoint'] if name.startswith(('X10', 'X11')) else result['final']['checkpoint']
        if cp != selected:
            raise ValueError('receipted champion selection mismatch')
        dest = SCRATCH / name / Path(cp['path']).name
        safe_copy(cp['path'], dest, cp['sha256'])
        archive = live / 'frozen_validation.npz'
        local_pool = SCRATCH / name / archive.name
        archive_sha = x7.digest(archive)
        safe_copy(archive, local_pool, archive_sha)
        with np.load(local_pool) as data:
            pool = tuple(data[f'T{depth}_{i}'] for i in range(3))
        if hashes(pool) != historical['pool_array_sha256'][str(seed)]:
            raise ValueError('historical frozen array identity mismatch')
        runs.append({'name': name, 'architecture': architecture, 'seed': seed, 'depth': depth,
                     'checkpoint': cp, 'snapshot_path': str(dest), 'frozen_pool': str(local_pool),
                     'frozen_pool_sha256': archive_sha, 'frozen_array_sha256': hashes(pool),
                     'frozen_metrics': {k: row[k] for k in ('aggregate', 'positions', 'perfect_prefix_depth')},
                     'historical_verified_sha256': verified,
                     'historical_files_sha256': {str(live / f): x7.digest(live / f)
                                                for f in ('config.json', 'validation.json', 'result.json')}})
    registration = {'task': 'X18', 'registered_utc': utc(), 'host': platform.node(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'authority': '/home/erikg/emender-scratch/fix-fanout/taskX18-freshdata.md; live checkpoint-selection ruling',
        'champions': runs, 'fresh_seeds': {str(k): v for k, v in SEEDS.items()},
        'source_sha256': sources(), 'examples': 512, 'batch': 32,
        'distribution': 'X2b fresh_training_batch(InvertibleQuadraticTask(61),512,T,default_rng(new_seed)); uniform initial coordinates and iid uniform drivers; T counts drivers, input length T+2',
        'scope': 'same-driver-distribution memorization-vs-execution check, NOT out-of-distribution extrapolation; no training or new checkpoint selection',
        'predictions': {
            'a_executing': 'Every champion retains >=480/512 on every historically perfect position. Supports execution/generalization across fresh sequences, not a unique circuit mechanism.',
            'b_memorizing': 'Any historically perfect position falls below300/512: fresh-data collapse; composition claims require re-scoping.',
            'c_intermediate': 'All other outcomes reported exactly, including all partial columns; do not relabel intermediate outcomes as either extreme.'},
        'runtime': 'Sequential co-resident GPU3 eval-only, explicitly authorized; no exclusive lease. model.eval/no_grad, ordinary live AdamW weights, strict loading, fp32 parameters/bf16 autocast, original E97 legacy/stock GDN2 forward and padding, batch32, deterministic algorithms, TF32 off, fixed seed0, CUBLAS :4096:8.',
        'novelty': 'Before forwards check zero full-input duplicates within fresh pools and against all four frozen pools. Replay each champion original training RNG input stream through its selected step, require terminal RNG state equals checkpoint numpy_rng, zero full-sequence overlap. Prefixes may repeat; this does not exclude learned finite-prefix lookup mechanisms.',
        'controls': 'Re-evaluate each own frozen pool first; exact historical integer counts and canonical aggregate loss required. No updates, optimizer construction or writes within original run directories. Hashes/tensors unchanged after forwards.',
        'readouts': 'All positions J/X/Y/OOR and full-state CE; frozen/fresh comparison, aggregate, terminal, suffix7, >=480 criterion. One512-example fresh pool per depth, shared across same-depth champions. No optional retry/reseed after outcomes.'}
    x7.write_json(ROOT / 'registration.json', registration)
    print(json.dumps({'registered_utc': registration['registered_utc'], 'sha256': x7.digest(ROOT / 'registration.json')}))


def registration():
    reg = read_json(ROOT / 'registration.json')
    for path, sha in reg['source_sha256'].items():
        require_hash(path, sha)
    return reg


def row_set(inputs):
    return {row.tobytes() for row in inputs}


def generate():
    reg = registration()
    if (ROOT / 'generation.json').exists():
        raise FileExistsError('fresh pools already frozen; no reseeding')
    started = utc()
    pools = {}
    for depth in (16, 64):
        rng = np.random.default_rng(reg['fresh_seeds'][str(depth)])
        initial_rng = rng.bit_generator.state
        pool = fresh_training_batch(x12.x10.InvertibleQuadraticTask(61), 512, depth, rng)
        keys = row_set(pool[0])
        if len(keys) != 512:
            raise ValueError('fresh pool contains duplicate full sequences')
        overlap = {}
        for run in reg['champions']:
            if run['depth'] != depth:
                continue
            require_hash(run['frozen_pool'], run['frozen_pool_sha256'])
            with np.load(run['frozen_pool']) as data:
                overlap[run['name']] = len(keys & row_set(data[f'T{depth}_0']))
        if any(overlap.values()):
            raise ValueError('fresh/frozen overlap')
        # Independently validate scalar arithmetic, masks and disjoint token roles.
        validate_pool(pool, depth)
        path = ROOT / f'fresh_T{depth}.npz'
        np.savez_compressed(path, **{f'T{depth}_{i}': a for i, a in enumerate(pool)})
        pools[str(depth)] = {'seed': reg['fresh_seeds'][str(depth)], 'path': str(path),
            'archive_sha256': x7.digest(path), 'array_sha256': hashes(pool),
            'initial_rng': initial_rng, 'final_rng': rng.bit_generator.state,
            'unique_sequences': len(keys), 'frozen_overlap': overlap,
            'shapes': [list(a.shape) for a in pool], 'dtypes': [str(a.dtype) for a in pool],
            'scalar_targets_verified': 512 * depth}
    x7.write_json(ROOT / 'generation.json', {'started_utc': started, 'finished_utc': utc(),
        'registration_sha256': x7.digest(ROOT / 'registration.json'), 'numpy': np.__version__, 'pools': pools})
    print('Fresh pools generated and hash-bound.')


def validate_pool(pool, depth):
    inputs, targets, mask = pool
    if (any(a.shape != (512, depth + 2) for a in pool) or mask[:, :2].any()
            or not mask[:, 2:].all() or not (61 <= inputs[:, 0]).all()
            or not (inputs[:, 0] < 122).all() or not (122 <= inputs[:, 1]).all()
            or not (inputs[:, 1] < 183).all() or not (0 <= inputs[:, 2:]).all()
            or not (inputs[:, 2:] < 61).all()):
        raise ValueError('invalid task inputs/masks')
    for i in range(512):
        x, y = int(inputs[i, 0]) - 61, int(inputs[i, 1]) - 122
        for j in range(depth):
            x, y = y, (x + y * y + int(inputs[i, j + 2])) % 61
            if targets[i, j + 2] != 61 * x + y:
                raise ValueError('scalar arithmetic mismatch')


def replay_inputs(rng, depth, updates):
    """Batch RNG calls only; preserve each original batch's initial-then-driver order."""
    draws = rng.integers(61, size=(updates, 32 * (depth + 2)), dtype=np.int64)
    initial = draws[:, :64].reshape(updates, 32, 2) + np.array([61, 122])
    drivers = draws[:, 64:].reshape(updates, 32, depth)
    return np.concatenate((initial, drivers), axis=2).reshape(-1, depth + 2)


def training_overlap(pool, seed, depth, steps, chunk=256):
    rng, matches = np.random.default_rng(seed), 0
    keys = row_set(pool[0])
    # Exact cheap three-token prefilter, then byte-exact full-sequence membership.
    def prefixes(a):
        return (a[:, 0] - 61) * 61**2 + (a[:, 1] - 122) * 61 + a[:, 2]
    wanted = prefixes(pool[0])
    for start in range(0, steps, chunk):
        inputs = replay_inputs(rng, depth, min(chunk, steps - start))
        candidates = inputs[np.isin(prefixes(inputs), wanted, kind='table')]
        matches += sum(row.tobytes() in keys for row in candidates)
    return matches, rng.bit_generator.state


def load_pool(path, depth):
    with np.load(path) as data:
        return tuple(data[f'T{depth}_{i}'] for i in range(3))


def evaluate_pool(model, pool):
    # Same canonical implementation, scoped loss reshape only, as X13's path audit.
    old_depth = x12.DEPTH
    try:
        x12.DEPTH = pool[0].shape[1] - 2
        return x12.evaluate(model, pool)
    finally:
        x12.DEPTH = old_depth


def gpu_snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu',
                                    '--format=csv'], text=True)


def evaluate():
    import torch
    reg = registration()
    generation = read_json(ROOT / 'generation.json')
    require_hash(ROOT / 'registration.json', generation['registration_sha256'])
    if (ROOT / 'runtime.json').exists() or list((ROOT / 'evaluations').glob('*.json')):
        raise FileExistsError('evaluation receipts exist; do not repeat')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '3' or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('registered co-resident GPU3/CUBLAS binding required')
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)
    before_gpu, started, started_utc = gpu_snapshot(), time.monotonic(), utc()
    for run in reg['champions']:
        name, depth = run['name'], run['depth']
        cp = run['checkpoint']
        require_hash(run['snapshot_path'], cp['sha256'])
        saved = torch.load(run['snapshot_path'], map_location='cpu', weights_only=False)
        if saved['step'] != cp['step'] or saved['weights'] != 'ordinary live parameters':
            raise ValueError('checkpoint step/weight representation mismatch')
        info = generation['pools'][str(depth)]
        require_hash(info['path'], info['archive_sha256'])
        fresh = load_pool(info['path'], depth)
        if hashes(fresh) != info['array_sha256']:
            raise ValueError('fresh arrays changed')
        replay_start = time.monotonic()
        overlap, final_rng = training_overlap(fresh, run['seed'], depth, cp['step'])
        if overlap or final_rng != saved['numpy_rng']:
            raise ValueError('training overlap or RNG replay mismatch')
        x7.write_json(ROOT / 'novelty' / f'{name}.json', {'champion_sha256': cp['sha256'],
            'generation_sha256': x7.digest(ROOT / 'generation.json'), 'seed': run['seed'],
            'depth': depth, 'updates_replayed': cp['step'], 'sequences_checked': cp['step'] * 32,
            'full_sequence_matches': overlap, 'terminal_rng_matches_checkpoint': True,
            'terminal_rng': final_rng, 'elapsed_seconds': time.monotonic() - replay_start})
        model = x12.build_model(run['architecture']).cuda()
        model.load_state_dict(saved['model'], strict=True)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if len(model.layers) != 4 or sum(p.numel() for p in model.parameters()) != x12.PARAMETERS[run['architecture']]:
            raise ValueError('model geometry mismatch')
        del saved
        require_hash(run['frozen_pool'], run['frozen_pool_sha256'])
        frozen = load_pool(run['frozen_pool'], depth)
        if hashes(frozen) != run['frozen_array_sha256']:
            raise ValueError('frozen arrays changed')
        eval_start = time.monotonic()
        control = evaluate_pool(model, frozen)
        if control != run['frozen_metrics']:
            raise ValueError(f'{name}: exact frozen metrics do not reproduce')
        result = evaluate_pool(model, fresh)
        torch.cuda.synchronize()
        if not all(torch.equal(before[k], v) for k, v in model.state_dict().items()):
            raise ValueError('model mutated during evaluation')
        require_hash(run['snapshot_path'], cp['sha256'])
        require_hash(cp['path'], cp['sha256'])
        receipt = {'name': name, 'finished_utc': utc(), 'champion_sha256': cp['sha256'],
            'registration_sha256': x7.digest(ROOT / 'registration.json'),
            'generation_sha256': x7.digest(ROOT / 'generation.json'),
            'training_updates': 0, 'weights_unchanged': True, 'frozen_exact_match': True,
            'frozen': control, 'fresh': result, 'gpu_eval_seconds': time.monotonic() - eval_start}
        for metrics in (control, result):
            check = reconcile(metrics['aggregate'], metrics['positions'], metrics['aggregate'])
            if not check['recorded_counts_match']:
                raise ValueError('accounting mismatch')
        x7.write_json(ROOT / 'evaluations' / f'{name}.json', receipt)
        print(json.dumps({'name': name, 'fresh_joint': [p['correct'] for p in result['positions']]}), flush=True)
        del model, before
        torch.cuda.empty_cache()
    x7.write_json(ROOT / 'runtime.json', {'started_utc': started_utc, 'finished_utc': utc(),
        'elapsed_seconds_including_cpu_replay': time.monotonic() - started,
        'gpu_before': before_gpu, 'gpu_after': gpu_snapshot(), 'co_resident': True,
        'cuda_visible_devices': '3', 'exclusive_lease': False, 'python': platform.python_version(),
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'training_updates': 0,
        'temporary_training_slowdown': 'possible; exact slowdown not measured; no counterfactual',
        'source_sha256': reg['source_sha256']})


def audit():
    reg = registration()
    generation = read_json(ROOT / 'generation.json')
    require_hash(ROOT / 'registration.json', generation['registration_sha256'])
    verdicts = {}
    for depth, info in generation['pools'].items():
        require_hash(info['path'], info['archive_sha256'])
        pool = load_pool(info['path'], int(depth))
        validate_pool(pool, int(depth))
        if hashes(pool) != info['array_sha256']:
            raise ValueError('pool hash mismatch')
    for run in reg['champions']:
        receipt = read_json(ROOT / 'evaluations' / f"{run['name']}.json")
        novelty = read_json(ROOT / 'novelty' / f"{run['name']}.json")
        if receipt['frozen'] != run['frozen_metrics'] or receipt['champion_sha256'] != run['checkpoint']['sha256']:
            raise ValueError('champion/control mismatch')
        require_hash(ROOT / 'registration.json', receipt['registration_sha256'])
        require_hash(ROOT / 'generation.json', receipt['generation_sha256'])
        if (receipt['training_updates'] or not receipt['weights_unchanged'] or not receipt['frozen_exact_match']
                or novelty['full_sequence_matches'] or not novelty['terminal_rng_matches_checkpoint']
                or novelty['updates_replayed'] != run['checkpoint']['step']
                or novelty['sequences_checked'] != 32 * run['checkpoint']['step']
                or novelty['champion_sha256'] != run['checkpoint']['sha256']):
            raise ValueError('non-readonly or non-fresh result')
        require_hash(ROOT / 'generation.json', novelty['generation_sha256'])
        for metrics in (receipt['frozen'], receipt['fresh']):
            if len(metrics['positions']) != run['depth'] or any(p['count'] != 512 for p in metrics['positions']):
                raise ValueError('position denominator/depth mismatch')
            check = reconcile(metrics['aggregate'], metrics['positions'], metrics['aggregate'])
            if abs(check['position_mean_loss_delta']) > 1e-6 + 1e-7 * metrics['aggregate']['loss']:
                raise ValueError('loss accounting mismatch')
        values = [fresh['correct'] for frozen, fresh in zip(receipt['frozen']['positions'], receipt['fresh']['positions'])
                  if frozen['correct'] == 512]
        verdicts[run['name']] = {'fresh_counts_at_frozen_perfect_positions': values,
                                'executing': min(values) >= 480, 'collapse': min(values) < 300}
    overall = ('b_memorizing' if any(v['collapse'] for v in verdicts.values()) else
               'a_executing' if all(v['executing'] for v in verdicts.values()) else 'c_intermediate')
    x7.write_json(ROOT / 'audit.json', {'audited_utc': utc(), 'verdict': overall, 'champions': verdicts,
        'generation_sha256': x7.digest(ROOT / 'generation.json'),
        'registration_sha256': x7.digest(ROOT / 'registration.json')})
    print(json.dumps({'verdict': overall, 'champions': verdicts}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('register', 'generate', 'evaluate', 'audit'))
    args = parser.parse_args()
    globals()[args.action]()
