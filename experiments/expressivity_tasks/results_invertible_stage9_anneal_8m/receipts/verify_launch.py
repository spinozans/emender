"""Freeze X9 launch evidence; never writes in live training directories."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import torch

from experiments.expressivity_tasks import run_invertible_stage9_anneal_8m as runner

root = runner.ROOT
snapshot = root / 'receipts/launch_snapshot'
snapshot.mkdir(exist_ok=False)
registration = json.loads((root / 'registration.json').read_text())
assert all(runner.digest(p) == h for p, h in registration['sources'].items())
report = {'captured_utc': datetime.now(timezone.utc).isoformat(),
          'registration_sha256': runner.digest(root / 'registration.json'), 'runs': {}, 'existing_runs_preserved': {}}
arrays = None
for name in runner.NAMES:
    live, dest = root / name, snapshot / name
    dest.mkdir()
    captured = {}
    for filename in ('config.json', 'data_receipt.json', 'validation.json', 'checkpoints.json'):
        raw = (live / filename).read_bytes()
        captured[filename] = json.loads(raw)
        (dest / filename).write_bytes(raw)
    config, data = captured['config.json'], captured['data_receipt.json']
    assert config['source_commit'] == (root / 'receipts/full_launch/base_commit.txt').read_text().strip()
    assert all(runner.digest(p) == h for p, h in config['source_sha256'].items())
    assert all(runner.digest(p) == h for p, h in config['fla_source_sha256'].items())
    assert config['depths'] == [16] and config['safety_ceiling'] == 500000
    assert not any(config[k] for k in ('ladder', 'carry_supervision', 'stagnation_kill'))
    assert runner.digest(live / 'frozen_validation.npz') == data['sha256']
    shutil.copyfile(live / 'frozen_validation.npz', dest / 'frozen_validation.npz')
    with np.load(dest / 'frozen_validation.npz') as npz:
        current = tuple(npz[f'T16_{i}'] for i in range(3))
    assert [hashlib.sha256(a.tobytes()).hexdigest() for a in current] == data['array_sha256']
    if arrays is None:
        arrays = current
    assert all(np.array_equal(a, b) for a, b in zip(arrays, current))
    baseline = captured['validation.json'][0]
    assert baseline['step'] == 0 and baseline['depth'] == 16
    assert len(baseline['positions']) == 16
    assert sum(p['correct'] for p in baseline['positions']) == baseline['aggregate']['correct']
    assert all(p['count'] == 512 for p in baseline['positions'])
    cp = baseline['checkpoint']
    assert runner.digest(cp['path']) == cp['sha256']
    payload = torch.load(cp['path'], map_location='cpu', weights_only=False)
    assert payload['step'] == payload['stage_step'] == 0 and payload['depth'] == 16
    assert not payload['optimizer']['state']
    old_name = name if name.startswith('e97') else 'gdn2_fp32_seed0'
    old_root = Path('experiments/expressivity_tasks/results_invertible_stage7_deepfirst_8m')
    old_entry = json.loads((old_root / old_name / 'validation.json').read_text())[0]['checkpoint']
    old_payload = torch.load(old_entry['path'], map_location='cpu', weights_only=False)
    assert all(torch.equal(v, old_payload['model'][k]) for k, v in payload['model'].items())
    del old_payload
    assert config['optimizer'] == 'AdamW' and config['betas'] == [.9, .999]
    assert config['warmup'] == 5000 and config['loss_weighting'].startswith('uniform')
    assert payload['weights'] == 'ordinary AdamW parameters'
    assert payload['optimizer']['param_groups'][0]['lr'] == 0.
    assert data['array_sha256'] == json.loads((old_root / old_name / 'data_receipt.json').read_text())['array_sha256']
    fla_dest = root / 'receipts/full_launch/fla'
    for source, sha in config['fla_source_sha256'].items():
        target = fla_dest / source.split('/fla/', 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        assert runner.digest(target) == sha
    del payload
    identities = {}
    for role in ('worker', 'trainer'):
        identity = json.loads((root / 'receipts' / name / f'{role}.json').read_text())
        pid = identity['pid']
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(') ', 1)[1].split()
        assert stat[0] not in ('Z', 'X') and stat[19] == identity['starttime']
        identities[role] = {**identity, 'ppid': int(stat[1]), 'sid': os.getsid(pid)}
    worker, trainer = identities['worker'], identities['trainer']
    assert worker['ppid'] == 1 and worker['sid'] == worker['pid']
    assert trainer['ppid'] == worker['pid'] and trainer['sid'] == worker['pid']
    gpu = config['cuda_visible_devices']
    lease_path = Path('.wg/gpu_leases') / gpu
    lease = dict(line.split('=', 1) for line in lease_path.read_text().splitlines())
    assert lease['pid'] == str(worker['pid']) and lease['starttime'] == worker['starttime']
    assert gpu == ('2' if name.startswith('e97') else '7')
    lines = (live / 'training.jsonl').read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    rows = [json.loads(line) for line in lines]
    assert len(rows) >= 100
    assert [r['step'] for r in rows] == list(range(1, len(rows) + 1))
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['depth'] == 16 for r in rows)
    assert all(r['lr'] == runner.learning_rate(r['step']) for r in rows)
    (dest / 'training_snapshot.jsonl').write_bytes(b''.join(lines))
    first, last = rows[99], rows[-1]
    rate = (last['step'] - first['step']) / (last['elapsed_s'] - first['elapsed_s']) if last['step'] > 100 else None
    report['runs'][name] = {'gpu': gpu, 'identities': identities, 'lease': lease, 'baseline': baseline,
        'last_observed_update': last, 'operational_updates_per_second_from100': rate,
        'entry_checkpoint_sha256': cp['sha256'], 'empty_initial_optimizer': True,
        'safety_ceiling_remaining_hours_at_observed_rate': (500000-last['step']) / rate / 3600 if rate else None,
        'snapshot_sha256': {str(p.relative_to(root)): runner.digest(p) for p in dest.iterdir()}}
assert [v['gpu'] for v in report['runs'].values()] == ['2']
# Observe, never signal or mutate, the original X6/X7 trainer identities.
for record in json.loads((root / 'receipts/existing_runs_before.json').read_text()):
    identity_path = Path(record['identity_path'])
    identity = json.loads(identity_path.read_text())
    assert identity == record['identity']
    pid = identity['pid']
    proc = Path(f'/proc/{pid}/stat')
    if proc.exists():
        stat = proc.read_text().rsplit(') ', 1)[1].split()
        assert stat[0] not in ('Z', 'X') and stat[19] == identity['starttime']
        state = 'same live identity'
    else:
        terminal = json.loads((identity_path.parent / 'terminal.json').read_text())
        state = {'naturally_terminal': terminal}
    report['existing_runs_preserved'][str(identity_path)] = {'identity': identity, 'observed': state}
runner.write_json(root / 'receipts/launch_verification.json', report)
print(json.dumps(report, indent=2))
