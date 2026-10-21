"""Immutable launch audit, run from repo root after all four trainers start."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np
import torch

from experiments.expressivity_tasks import run_invertible_stage14_nofreebie_8m as r
from scripts.watch_invertible_stage10_optimizers import process_alive
from scripts.watch_invertible_stage14_nofreebie import status

root = r.ROOT
receipt = root / 'receipts'
output = receipt / 'launch_verification.json'
if output.exists():
    raise FileExistsError('immutable launch audit already exists')
registration = r.x10.verify_registration(root)
code, message = status(root)
assert code is None, message
smi = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                               '--format=csv,noheader,nounits'], text=True)
(receipt / 'gpu_processes_at_verification.txt').write_text(smi)
gpus = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
                                '--format=csv,noheader,nounits'], text=True)
uuids = dict(line.replace(' ', '').split(',') for line in gpus.splitlines())
process_gpu = dict(line.replace(' ', '').split(',') for line in smi.splitlines())
manifest = {}
runs = {}
for name in r.NAMES:
    live = root / name
    dest = receipt / 'launch_snapshot' / name
    dest.mkdir(parents=True, exist_ok=False)
    captured = {}
    for filename in ('config.json', 'data_receipt.json', 'validation.json', 'checkpoints.json'):
        raw = (live / filename).read_bytes()
        captured[filename] = json.loads(raw)
        path = dest / filename
        path.write_bytes(raw)
        manifest[str(path)] = r.x7.digest(path)
    config = captured['config.json']
    assert config['registration_sha256'] == r.x7.digest(root / 'registration.json')
    assert config['source_commit'] == subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    gpu = registration['gpu_binding'][name]
    assert config['cuda_visible_devices'] == str(gpu)
    ids = {role: json.loads((receipt / name / f'{role}.json').read_text()) for role in ('worker', 'trainer')}
    assert all(process_alive(record) for record in ids.values())
    assert process_gpu[str(ids['trainer']['pid'])] == uuids[str(gpu)]
    lease_dir = Path(os.environ.get('GPU_LEASE_DIR', '.wg/gpu_leases'))
    lease_path = lease_dir / str(gpu)
    lease_raw = lease_path.read_text()
    lease = dict(line.split('=', 1) for line in lease_raw.splitlines())
    assert lease['pid'] == str(ids['worker']['pid'])
    assert lease['starttime'] == str(ids['worker']['starttime'])
    assert __import__('time').time() - int(lease['heartbeat']) < 120
    (dest / 'lease_at_verification.txt').write_text(lease_raw)
    arrays = np.load(live / 'frozen_validation.npz')
    hashes = [hashlib.sha256(arrays[f'T16_{i}'].tobytes()).hexdigest() for i in range(3)]
    assert hashes == registration['pool_array_sha256'][str(config['seed'])]
    assert hashes == captured['data_receipt.json']['array_sha256']
    assert r.x7.digest(live / 'frozen_validation.npz') == captured['data_receipt.json']['archive_sha256']
    lines = (live / 'training.jsonl').read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        lines.pop()
    training = [json.loads(line) for line in lines]
    assert training and [t['step'] for t in training] == list(range(1, len(training) + 1))
    assert all(t['lr'] == r.x10.learning_rate(t['step']) and math.isfinite(t['loss']) for t in training)
    rows = captured['validation.json']
    for row in rows:
        assert len(row['positions']) == 16
        assert row['positions'][0]['role'] == 'diagnostic-only non-composition'
        assert all(p['role'] == 'composition' for p in row['positions'][1:])
        assert row['aggregate']['correct'] == sum(p['correct'] for p in row['positions'][1:])
        assert row['aggregate']['scope'] == 'P2..P16 only'
    entry = rows[0]['checkpoint']
    assert r.x7.digest(entry['path']) == entry['sha256']
    saved = torch.load(entry['path'], weights_only=False, map_location='cpu')
    assert saved['step'] == 0 and not saved['optimizer']['state']
    group = saved['optimizer']['param_groups'][0]
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == .01
    assert saved['optimizer_family'] == 'adamw' and saved['weights'] == 'ordinary live parameters'
    del saved
    # A checkpoint may be in its atomic staging phase; never copy a live payload.
    retained = list(live.glob('*.pt'))
    assert len(retained) <= 4  # pruning follows checkpoint publication atomically
    runs[name] = {'gpu': gpu, 'gpu_uuid': uuids[str(gpu)], 'identities': ids,
                  'audited_updates': len(training), 'all_lrs_exact': True, 'losses_finite': True,
                  'evaluation_through_step': rows[-1]['step'], 'pool_hashes_match': True,
                  'entry_optimizer_empty': True, 'entry_checkpoint': entry,
                  'retained_checkpoint_count_at_snapshot': len(retained),
                  'P1': 'diagnostic-only non-composition', 'primary': 'P2..P16'}
    for path in dest.iterdir():
        manifest[str(path)] = r.x7.digest(path)
r.x7.write_json(output, {'verified_utc': datetime.now(timezone.utc).isoformat(),
    'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
    'registration_sha256': r.x7.digest(root / 'registration.json'),
    'verified_project_files': len(registration['source_sha256']),
    'verified_dependencies': len(registration['dependency_sha256']),
    'stage12_in_source_closure': any('stage12' in p for p in registration['source_sha256']),
    'watcher': {'code': code, 'message': message}, 'runs': runs})
manifest[str(output)] = r.x7.digest(output)
r.x7.write_json(receipt / 'launch_manifest.json', manifest)
print(json.dumps(runs, indent=2))
