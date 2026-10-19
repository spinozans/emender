"""Read-only X9b launch validation, with a bounded immutable evidence snapshot."""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

from experiments.expressivity_tasks import run_invertible_stage9b_sf_anneal_8m as runner
from scripts.watch_invertible_stage9b_sf_anneal import process_alive

root = runner.ROOT.resolve()
name = runner.NAMES[0]
out = root / name
load = lambda p: json.loads(p.read_text())
reg = load(root / 'registration.json')
config = load(out / 'config.json')
assert config['cuda_visible_devices'] == '2' and config['optimizer'] == 'AdamWScheduleFree'
assert config['betas'] == [.9, .95] and config['internal_warmup'] == 0
assert config['loss_weighting'].startswith('uniform') and config['depths'] == [16]
assert not config['ladder'] and not config['carry_supervision'] and not config['stagnation_kill']
assert config['safety_ceiling'] == 500000 and config['trainable_parameters'] == 9900800
for p, h in reg['sources'].items():
    assert runner.digest(p) == h, p
    assert hashlib.sha256(subprocess.check_output(['git', 'show', config['source_commit'] + ':' + p])).hexdigest() == h
for key in ('source_sha256', 'fla_source_sha256', 'schedulefree_source_sha256'):
    for p, h in config[key].items():
        assert runner.digest(p) == h, p
assert subprocess.check_output(['git', 'show', config['source_commit'] + ':' + str((root / 'registration.json').relative_to(Path.cwd()))]) == (root / 'registration.json').read_bytes()
prior = load(root / 'receipts/existing_runs_before.json')
for p, old in prior.items():
    assert load(Path(p)) == old['identity'], p
    if old['alive']:
        assert process_alive(old['identity']), p
receipt = root / 'receipts' / name
worker, trainer = (load(receipt / (role + '.json')) for role in ('worker', 'trainer'))
assert process_alive(worker) and process_alive(trainer)
lease = dict(line.split('=', 1) for line in (receipt / 'lease.txt').read_text().splitlines())
assert int(lease['pid']) == worker['pid'] and lease['starttime'] == worker['starttime']
ps = subprocess.check_output(['ps', '-o', 'pid=,ppid=,sid=', '-p', f"{worker['pid']},{trainer['pid']}"], text=True)
identities = {int(parts[0]): [int(p) for p in parts] for line in ps.splitlines() if (parts := line.split())}
assert identities[worker['pid']][1:] == [1, worker['pid']]
assert identities[trainer['pid']][1:] == [worker['pid'], worker['pid']]
data = load(out / 'data_receipt.json')
x7 = root.parent / 'results_invertible_stage7_deepfirst_8m' / name
assert data['array_sha256'] == load(x7 / 'data_receipt.json')['array_sha256']
assert data['sha256'] == runner.digest(out / 'frozen_validation.npz')
rows = load(out / 'validation.json')
entry = rows[0]['checkpoint']
assert runner.digest(entry['path']) == entry['sha256']
cp = torch.load(entry['path'], map_location='cpu', weights_only=False)
assert cp['weights'] == 'ScheduleFree eval x' and not cp['optimizer']['state']
assert not cp['optimizer']['param_groups'][0]['train_mode']
x7_entry = torch.load(x7 / 'step_000000_T16_s00000.pt', map_location='cpu', weights_only=False)
assert all(torch.equal(v, x7_entry['model'][k]) for k, v in cp['model'].items())
assert rows[0]['positions'] == load(x7 / 'validation.json')[0]['positions']
training_bytes = (out / 'training.jsonl').read_bytes()
training_bytes = training_bytes[:training_bytes.rfind(b'\n') + 1]
training = [json.loads(line) for line in training_bytes.splitlines()]
assert len(training) >= 1000
for step, row in enumerate(training, 1):
    assert row['step'] == step and row['depth'] == 16
    assert row['lr'] == row['requested_lr'] == runner.learning_rate(step)
    assert all(math.isfinite(row[k]) for k in ('loss', 'grad_norm', 'lr'))
assert len(rows[-1]['positions']) == 16
trained_cp = rows[-1]['checkpoint']
if trained_cp is None:
    trained_cp = next(r['checkpoint'] for r in reversed(rows) if r['checkpoint'] and r['step'])
assert runner.digest(trained_cp['path']) == trained_cp['sha256']
trained = torch.load(trained_cp['path'], map_location='cpu', weights_only=False)
group = trained['optimizer']['param_groups'][0]
assert not group['train_mode'] and group['warmup_steps'] == 0
assert group['scheduled_lr'] == runner.learning_rate(trained['step'])
assert any('z' in state for state in trained['optimizer']['state'].values())
# Snapshot only bounded JSON, the frozen pool and a finite training prefix; never copy checkpoints.
snapshot = root / 'receipts/launch_snapshot'
snapshot.mkdir(exist_ok=False)
for p in ('config.json', 'data_receipt.json', 'frozen_validation.npz'):
    shutil.copy2(out / p, snapshot / p)
runner.write_json(snapshot / 'validation.json', rows)
(snapshot / 'training_prefix.jsonl').write_bytes(training_bytes)
runner.write_json(snapshot / 'checkpoints_verified.json', {'entry': entry, 'trained': trained_cp})
# Hash the source archive, including installed SF implementation and its license.
archive = root / 'receipts/full_launch'
manifest = {str(p.relative_to(archive)): runner.digest(p) for p in archive.rglob('*') if p.is_file()}
runner.write_json(root / 'receipts/full_launch_manifest.json', manifest)
summary = {'status': 'passed', 'verified_at': time.time(), 'source_commit': config['source_commit'],
    'worker': worker, 'trainer': trainer, 'ps': ps, 'card': 2, 'training_rows': len(training),
    'latest_training': training[-1], 'latest_validation_step': rows[-1]['step'],
    'entry': entry, 'trained_checkpoint': trained_cp, 'prior_live_identities_unchanged': sum(r['alive'] for r in prior.values()),
    'snapshot_sha256': {str(p.relative_to(snapshot)): runner.digest(p) for p in snapshot.iterdir()}}
runner.write_json(root / 'receipts/launch_verification.json', summary)
print(json.dumps(summary, indent=2))
