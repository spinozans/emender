"""Read-only X6 launch evidence capture; run from repository root after update100."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import torch

from experiments.expressivity_tasks import run_invertible_stage6_truecur_8m as runner

root = runner.ROOT
registration = json.loads((root / 'registration.json').read_text())
source_commit = (root / 'receipts/full_launch/base_commit.txt').read_text().strip()
for path, expected in registration['source_sha256'].items():
    assert runner.digest(path) == expected, path
snapshots = root / 'receipts/launch_snapshot'
snapshots.mkdir(exist_ok=True)
rows = []
gpus = []
for spec in runner.matrix():
    name = spec['name']
    receipt = root / 'receipts' / name
    snapshot = snapshots / name
    snapshot.mkdir(exist_ok=True)
    row = dict(spec)
    worker = json.loads((receipt / 'worker.json').read_text())
    stat = Path(f"/proc/{worker['pid']}/stat").read_text().rsplit(') ', 1)[1].split()
    assert stat[0] not in ('Z', 'X') and stat[19] == worker['starttime']
    assert int(stat[1]) == 1 and int(stat[3]) == worker['pid']
    row['worker'] = {**worker, 'ppid': int(stat[1]), 'sid': int(stat[3])}
    if not (receipt / 'trainer.json').exists():
        assert spec['curriculum'] == 'B' and spec['seed'] in (1, 2)
        row['state'] = 'queued_exclusive_lease'
        rows.append(row)
        continue
    trainer = json.loads((receipt / 'trainer.json').read_text())
    stat = Path(f"/proc/{trainer['pid']}/stat").read_text().rsplit(') ', 1)[1].split()
    assert stat[0] not in ('Z', 'X') and stat[19] == trainer['starttime']
    assert int(stat[1]) == worker['pid'] and int(stat[3]) == worker['pid']
    row['trainer'] = {**trainer, 'ppid': int(stat[1]), 'sid': int(stat[3])}
    command = (receipt / 'command.txt').read_text()
    gpu = command.split('CUDA_VISIBLE_DEVICES=')[1].split()[0]
    gpus.append(gpu)
    lease = dict(line.split('=', 1) for line in Path('.wg/gpu_leases', gpu).read_text().splitlines() if '=' in line)
    assert lease['pid'] == str(worker['pid']) and lease['starttime'] == worker['starttime']
    row['gpu'], row['live_lease'] = gpu, lease
    out = root / name
    for filename in ('config.json', 'validation.json', 'transitions.jsonl', 'checkpoints.json', 'data_receipt.json', 'frozen_validation.npz'):
        shutil.copyfile(out / filename, snapshot / filename)
    config = json.loads((snapshot / 'config.json').read_text())
    assert config['source_commit'] == source_commit
    assert config['depths'] == list(runner.DEPTHS) and not config['stagnation_kill']
    assert config['hard_cap'] == 4000000 and config['stage_safety_ceiling'] == 500000
    assert config['lr'] == .001 and config['vocab'] == config['output_vocab'] == 3721
    assert config['trainable_parameters'] == registration['recipe'][f"architecture_parameters_{spec['curriculum']}"][spec['arm']]
    for path, expected in config['source_sha256'].items():
        assert runner.digest(path) == expected
    data = json.loads((snapshot / 'data_receipt.json').read_text())
    assert runner.digest(snapshot / 'frozen_validation.npz') == data['sha256']
    with np.load(snapshot / 'frozen_validation.npz') as pools:
        for depth in runner.DEPTHS:
            arrays = [pools[f'T{depth}_{i}'] for i in range(3)]
            assert [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays] == data['depths'][str(depth)]['array_sha256']
    entry = json.loads((snapshot / 'transitions.jsonl').read_text().splitlines()[0])['checkpoint']
    assert runner.digest(entry['path']) == entry['sha256']
    payload = torch.load(entry['path'], map_location='cpu', weights_only=False)
    assert all(key in payload for key in ('model', 'optimizer', 'numpy_rng', 'torch_rng', 'cuda_rng'))
    assert payload['step'] == 0 and payload['depth'] == 1 and payload['optimizer']['state'] == {}
    row['entry_checkpoint'] = entry
    training = []
    with (out / 'training.jsonl').open() as stream:
        for line in stream:
            if line.endswith('\n'):
                training.append(json.loads(line))
    assert len(training) >= 1000, (name, len(training))
    assert all(r['step'] == i + 1 and math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) and r['depth'] == 1 for i, r in enumerate(training))
    first, last = training[99], training[-1]
    row['updates'], row['first_rate_point'], row['last_rate_point'] = len(training), first, last
    row['wall_updates_per_second'] = (last['step'] - first['step']) / (last['elapsed_s'] - first['elapsed_s'])
    (snapshot / 'rate_endpoints.json').write_text(json.dumps([first, last], indent=2) + '\n')
    row['config_sha256'] = runner.digest(snapshot / 'config.json')
    row['state'] = 'training'
    rows.append(row)
assert len(gpus) == len(set(gpus)) == 8
# Exact base validation equality for A/B and architectures at the same seed.
for seed in range(3):
    paths = [snapshots / r['name'] / 'data_receipt.json' for r in rows if r['seed'] == seed and r['state'] == 'training']
    depths = [json.loads(p.read_text())['depths'] for p in paths]
    assert all(d == depths[0] for d in depths)
result = {'verified_at': time.time(), 'source_commit': source_commit,
    'registration_sha256': runner.digest(root / 'registration.json'), 'rows': rows,
    'all_12_workers_detached': True, 'active_trainers': 8, 'queued_workers': 4,
    'source_hashes_match': True, 'finite_consecutive_training_receipts': True,
    'entry_checkpoint_payloads_and_frozen_arrays_verified': True}
runner.write_json(root / 'receipts/launch_verification.json', result)
for row in rows:
    print(row['name'], row['state'], row.get('gpu'), row.get('updates'), row.get('wall_updates_per_second'))
print('VERIFIED eight active + four queued; all twelve PPID1/SIDworker')
