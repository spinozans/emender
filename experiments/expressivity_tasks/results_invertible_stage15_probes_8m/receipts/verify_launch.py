"""Immutable X15 launch snapshot; run once after both preregistered launches."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import time
import numpy as np
import torch
from experiments.expressivity_tasks import run_invertible_stage15_probes_8m as r
from scripts import invertible_stage15_gdn2_amendment as a
from scripts import watch_invertible_stage15_amended as w
from scripts.watch_invertible_stage10_optimizers import process_alive

root = r.ROOT
out = root / 'receipts/launch_snapshot'
out.mkdir(exist_ok=False)
registration = r.verify_registration(root)
amended = r.verify_registration(a.ROOT)
assert not any('stage12' in p for p in registration['source_sha256'])
assert not any('stage12' in p for p in amended['source_sha256'])
assert w.status(root)[0] is None
checks = {'created_at': time.time(), 'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          'registration_sha256': r.x7.digest(root / 'registration.json'),
          'amendment_sha256': r.x7.digest(a.ROOT / 'registration.json')}
zero = a.ROOT / 'gdn2_zeroshot'
result = json.loads((zero / 'result.json').read_text())
assert result['training_updates'] == 0 and result['weights_unchanged']
assert result['champion_sha256'] == r.CHAMPION_SHA256 == r.x7.digest(r.CHAMPION)
assert result['cuda_visible_devices'] == '6'
assert [p['correct'] for p in result['t16_control']['positions']] == [512,512,512,512,333,3]+[0]*10
assert len(result['final']['positions']) == 64
assert result['pool_array_sha256'] == registration['pool_array_sha256']['0']
checks['gdn2'] = {'state': 'complete', 'positions': [p['correct'] for p in result['final']['positions']],
                 'result_sha256': r.x7.digest(zero / 'result.json')}
for path in (zero / 'result.json', zero / 't16_control.json', a.ROOT / 'markers/gdn2_zeroshot.DONE'):
    shutil.copy2(path, out / path.name)
name = 'muon3x_seed0'
run = root / name
receipt = root / 'receipts' / name
identities = {role: json.loads((receipt / f'{role}.json').read_text()) for role in ('worker','trainer')}
assert all(process_alive(v) for v in identities.values())
config = json.loads((run / 'config.json').read_text())
assert config['cuda_visible_devices'] == '7'
assert config['registration_sha256'] == checks['registration_sha256']
data = json.loads((run / 'data_receipt.json').read_text())
assert data['array_sha256'] == registration['pool_array_sha256']['0']
assert data['archive_sha256'] == r.x7.digest(run / 'frozen_validation.npz')
archive = np.load(run / 'frozen_validation.npz')
assert set(archive.files) == {'T64_0','T64_1','T64_2'}
assert r.array_hashes(tuple(archive[f'T64_{i}'] for i in range(3))) == data['array_sha256']
rows = json.loads((run / 'validation.json').read_text())
assert [row['step'] for row in rows] == list(range(0, rows[-1]['step']+1, 500))
for row in rows:
    assert len(row['positions']) == 64
    assert sum(p['correct'] for p in row['positions']) == row['aggregate']['correct']
    assert sum(p['count'] for p in row['positions']) == row['aggregate']['count'] == 512*64
    assert math.isfinite(row['aggregate']['loss'])
    assert row['peak_allocated_bytes'] < registration['resource_bounds']['muon3x']['peak_allocated_bytes']
training = []
with (run / 'training.jsonl').open() as stream:
    for line in stream:
        try:
            training.append(json.loads(line))
        except ValueError:
            break  # Snapshot at the last complete live write only.
for step,row in enumerate(training,1):
    assert row['step'] == step and row['lr'] == r.learning_rate(step)
    assert all(math.isfinite(row[k]) for k in ('loss','grad_norm','optimizer_seconds'))
entry = rows[0]['checkpoint']
assert entry['sha256'] == r.x7.digest(entry['path'])
saved = torch.load(entry['path'],map_location='cpu',weights_only=False)
assert saved['step'] == 0 and not saved['optimizer']['muon']['state'] and not saved['optimizer']['adamw']['state']
assert saved['optimizer']['routes'] == registration['parameter_routes']
assert len(list(run.glob('step_*.pt'))) <= 3
for path in (run / 'config.json',run / 'validation.json',run / 'data_receipt.json',run / 'checkpoints.json'):
    # Freeze the bytes already audited where the file is a live producer.
    if path.name == 'validation.json':
        r.x7.write_json(out / path.name,rows)
    else:
        shutil.copy2(path,out / path.name)
shutil.copy2(Path('.wg/gpu_leases/7'),out / 'lease7.txt')
lease = dict(line.split('=',1) for line in (out / 'lease7.txt').read_text().splitlines())
assert lease['pid'] == str(identities['worker']['pid']) and lease['starttime'] == identities['worker']['starttime']
processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv'],text=True)
(out / 'gpu_processes.txt').write_text(processes)
assert any(line.startswith(str(identities['trainer']['pid'])+', GPU-8f63894f-5b64-755c-060f-7022c603e7d2') for line in processes.splitlines())
checks['muon'] = {'identities': identities,'training_updates_audited': len(training),
    'validation_step': rows[-1]['step'], 'validation_loss': rows[-1]['aggregate']['loss'],
    'ce_minus_ln3721': rows[-1]['aggregate']['loss']-math.log(3721),
    'max_position_correct': max(p['correct'] for p in rows[-1]['positions']),
    'entry_checkpoint_sha256': entry['sha256'], 'retained_checkpoints': len(list(run.glob('step_*.pt')))}
r.x7.write_json(out / 'verification.json',checks)
r.x7.write_json(root / 'receipts/launch_manifest.json',{str(p):r.x7.digest(p) for p in sorted(out.iterdir())})
print(json.dumps(checks,indent=2))
