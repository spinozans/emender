"""Live X11 launch audit; refuse to overwrite the immutable first-launch snapshot."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from experiments.expressivity_tasks import run_invertible_stage11_gdn2_adamw_8m as r
from scripts import watch_invertible_stage11_gdn2_adamw as watch


def main():
    registration = r.x10.verify_registration(r.ROOT)
    receipt, out = r.ROOT / 'receipts' / r.NAME, r.ROOT / r.NAME
    config = json.loads((out / 'config.json').read_text())
    source_commit = config['source_commit']
    assert source_commit == subprocess.check_output(['git', 'log', '-1', '--format=%H', '--', str(r.ROOT / 'registration.json')], text=True).strip()
    assert config['registration_sha256'] == r.x7.digest(r.ROOT / 'registration.json')
    assert config['architecture'] == 'gdn2' and config['arm'] == 'adamw' and config['seed'] == 0
    worker = json.loads((receipt / 'worker.json').read_text())
    trainer = json.loads((receipt / 'trainer.json').read_text())
    assert watch.process_alive(worker) and watch.process_alive(trainer)
    gpu = registration['gpu_binding'][r.NAME]
    assert config['cuda_visible_devices'] == str(gpu)
    lease = dict(line.split('=', 1) for line in (Path('.wg/gpu_leases') / str(gpu)).read_text().splitlines())
    assert lease['pid'] == str(worker['pid']) and lease['starttime'] == str(worker['starttime'])
    assert time.time() - int(lease['heartbeat']) < 180
    devices = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    uuid = next(line.split(',')[1].strip() for line in devices.splitlines() if line.split(',')[0].strip() == str(gpu))
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    assert any(line.split(',')[0].strip() == str(trainer['pid']) and line.split(',')[1].strip() == uuid for line in processes.splitlines())
    validation = json.loads((out / 'validation.json').read_text())
    assert validation[-1]['step'] >= 500
    assert [row['step'] for row in validation] == list(range(0, validation[-1]['step'] + 1, 500))
    assert all(len(row['positions']) == 16 for row in validation)
    training = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines() if line.strip()]
    assert [row['step'] for row in training] == list(range(1, training[-1]['step'] + 1))
    assert all(row['lr'] == r.x10.learning_rate(row['step']) and math.isfinite(row['loss']) for row in training)
    archive = np.load(out / 'frozen_validation.npz')
    pool_hashes = [hashlib.sha256(archive[f'T16_{i}'].tobytes()).hexdigest() for i in range(3)]
    assert pool_hashes == registration['pool_array_sha256']['0']
    entry = out / 'step_000000.pt'
    saved = torch.load(entry, map_location='cpu', weights_only=False)
    assert saved['step'] == 0 and not saved['optimizer']['state']
    group, = saved['optimizer']['param_groups']
    assert group['betas'] == (.9, .999) and group['eps'] == 1e-8 and group['weight_decay'] == .01 and group['lr'] == 0.
    assert saved['weights'] == 'ordinary live parameters' and saved['optimizer_family'] == 'adamw'
    assert len(list(out.glob('*.pt'))) <= 3 and len(list(out.glob('*.pt.tmp'))) <= 1
    watcher_code, watcher_message = watch.status(r.ROOT)
    assert watcher_code is None
    snapshot = r.ROOT / 'receipts/launch_snapshot'
    snapshot.mkdir(exist_ok=False)
    for name in ('config.json', 'data_receipt.json', 'checkpoints.json'):
        (snapshot / name).write_bytes((out / name).read_bytes())
    r.x7.write_json(snapshot / 'validation.json', validation)
    (snapshot / 'training_prefix.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in training))
    (snapshot / 'gpu_devices.txt').write_text(devices)
    (snapshot / 'gpu_processes.txt').write_text(processes)
    result = {'verified_at_epoch': time.time(), 'source_commit': source_commit,
              'registration_sha256': r.x7.digest(r.ROOT / 'registration.json'),
              'worker': worker, 'trainer': trainer, 'gpu': gpu, 'gpu_uuid': uuid,
              'training_through_step': training[-1]['step'], 'validation_through_step': validation[-1]['step'],
              'entry_sha256': r.x7.digest(entry), 'entry_optimizer_empty': True,
              'all_logged_learning_rates_exact': True, 'all_position_counts_present': True,
              'pool_sha256_matches_x7b_x10': True, 'registered_hashes_verified_against_HEAD': True,
              'retained_checkpoint_count': len(list(out.glob('*.pt'))),
              'watcher': watcher_message, 'ceiling_not_yet_reached': True,
              'snapshot_sha256': {str(p): r.x7.digest(p) for p in sorted(snapshot.iterdir())}}
    r.x7.write_json(r.ROOT / 'receipts/launch_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
