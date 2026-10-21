"""CPU-only X10b immutable launch audit; run from the repository root."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.expressivity_tasks import run_invertible_stage10b_muonlr_8m as r
from scripts import invertible_stage10b_control as control
from scripts.watch_invertible_stage10b_muonlr import process_alive

root = r.ROOT
registration = r.verify_registration(root)
registration_commit = '502329ab'
subprocess.run(['git', 'merge-base', '--is-ancestor', registration_commit, 'HEAD'], check=True)
assert subprocess.check_output(['git', 'show', f'{registration_commit}:{root}/registration.json']) == (root/'registration.json').read_bytes()
report = {'snapshot_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'registration_commit': subprocess.check_output(['git','rev-parse',registration_commit],text=True).strip(),
          'registration_sha256': r.x7.digest(root/'registration.json'), 'arms': {}}
reference = torch.load(control.BASELINE_ROOT/'muon_seed0/step_000000.pt', map_location='cpu', weights_only=False)
for arm in r.ARMS:
    name = f'{arm}_seed0'
    out = root/name
    receipt = root/'receipts'/name
    snapshot = root/'receipts'/'launch_snapshot'/name
    snapshot.mkdir(parents=True, exist_ok=False)
    identities = {role: json.loads((receipt/f'{role}.json').read_text()) for role in ('worker','trainer')}
    assert all(process_alive(identity) for identity in identities.values())
    actual_worker = control.identity(identities['worker']['pid'])
    actual_trainer = control.identity(identities['trainer']['pid'])
    assert actual_worker['ppid'] == 1
    assert actual_trainer['ppid'] == identities['worker']['pid']
    card = r.CARDS[arm]
    assert card not in (0,1,2)
    assert registration['gpu_binding'][name] == card
    lease = Path(f'.wg/gpu_leases/{card}').read_text()
    fields = dict(line.split('=',1) for line in lease.splitlines())
    assert int(fields['pid']) == identities['worker']['pid']
    assert fields['starttime'] == identities['worker']['starttime']
    assert time.time()-int(fields['heartbeat']) < 120
    (snapshot/'lease.txt').write_text(lease)
    for file in ('config.json','data_receipt.json','validation.json'):
        (snapshot/file).write_bytes((out/file).read_bytes())
    config = json.loads((snapshot/'config.json').read_text())
    assert config['registration_sha256'] == r.x7.digest(root/'registration.json')
    assert config['source_commit'] == report['registration_commit']
    assert config['cuda_visible_devices'] == str(card)
    pool = np.load(out/'frozen_validation.npz')
    arrays = [hashlib.sha256(pool[f'T16_{i}'].tobytes()).hexdigest() for i in range(3)]
    assert arrays == registration['pool_array_sha256']['0']
    entry = torch.load(out/'step_000000.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,reference['model'][k]) for k,v in entry['model'].items())
    assert not entry['optimizer']['muon']['state'] and not entry['optimizer']['adamw']['state']
    validation = json.loads((snapshot/'validation.json').read_text())
    assert [v['step'] for v in validation] == list(range(0,validation[-1]['step']+1,500))
    for row in validation:
        assert len(row['positions']) == 16
        assert sum(p['correct'] for p in row['positions']) == row['aggregate']['correct']
        assert all(p['count']==512 for p in row['positions'])
    raw = (out/'training.jsonl').read_bytes()
    raw = raw[:raw.rfind(b'\n')+1]
    (snapshot/'training_prefix.jsonl').write_bytes(raw)
    rows = [json.loads(line) for line in raw.splitlines()]
    assert len(rows) >= 500
    for step,row in enumerate(rows,1):
        assert row['step'] == step
        assert row['lr'] == r.learning_rate(arm,step)
        assert all(math.isfinite(row[k]) for k in ('loss','grad_norm','lr','optimizer_seconds','update_seconds','elapsed_s'))
    report['arms'][arm] = {'card':card,'identities':identities,
        'consecutive_finite_updates':len(rows),'last_lr':rows[-1]['lr'],
        'entry_model_bit_identical_to_X10_muon':True,'empty_entry_optimizer':True,
        'validation_steps':[v['step'] for v in validation],
        'entry_counts':[p['correct'] for p in validation[0]['positions']],
        'entry_checkpoint_sha256':r.x7.digest(out/'step_000000.pt'),
        'snapshot_sha256':{p.name:r.x7.digest(p) for p in snapshot.iterdir()}}
r.x7.write_json(root/'receipts'/'launch_verification.json',report)
print(json.dumps(report,indent=2))
print('PASS X10b source/commit/lease/process/pool/initialization/empty-state/per-position/finite-LR launch audit')
