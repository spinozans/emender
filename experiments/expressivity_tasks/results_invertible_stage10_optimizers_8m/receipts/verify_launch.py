"""Read-only launch checks; write immutable snapshots only when explicitly requested."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time

import torch

from experiments.expressivity_tasks import run_invertible_stage10_optimizers_8m as r


def live(record):
    assert record['host'] == socket.gethostname()
    stat = Path(f"/proc/{record['pid']}/stat").read_text().rsplit(') ', 1)[1].split()
    assert stat[0] not in ('Z', 'X') and stat[19] == str(record['starttime'])


def verify(snapshot):
    expected_commit = 'd9785a65530fd8c720c26063541c89522d7124eb'
    results = {}
    for arm in r.ARMS:
        root, name = r.ROOTS[arm], f'{arm}_seed0'
        r.verify_registration(root)
        out, receipt = root / name, root / 'receipts' / name
        config = json.loads((out / 'config.json').read_text())
        assert config['source_commit'] == expected_commit
        assert config['cuda_visible_devices'] == str(r.CARDS[arm])
        worker = json.loads((receipt / 'worker.json').read_text())
        trainer = json.loads((receipt / 'trainer.json').read_text())
        live(worker)
        live(trainer)
        assert trainer['ppid'] == worker['pid']
        assert worker['ppid'] == 1
        lease = dict(line.split('=', 1) for line in Path(f'.wg/gpu_leases/{r.CARDS[arm]}').read_text().splitlines())
        assert lease['pid'] == str(worker['pid']) and lease['starttime'] == str(worker['starttime'])
        data = json.loads((out / 'data_receipt.json').read_text())
        historical = Path('experiments/expressivity_tasks/results_invertible_stage' +
                          ('8_paydepth_8m/gdn2_bf16_seed0' if arm == 'gdn2_uniform' else '9_anneal_8m/e97_bf16_seed0'))
        historical_data = json.loads((historical / 'data_receipt.json').read_text())
        assert data['array_sha256'] == historical_data['array_sha256']
        rows = json.loads((out / 'validation.json').read_text())
        entry = rows[0]['checkpoint']
        assert entry['step'] == 0 and r.x7.digest(entry['path']) == entry['sha256']
        cp = torch.load(entry['path'], weights_only=False, map_location='cpu')
        old = torch.load(historical / 'step_000000_T16_s00000.pt', weights_only=False, map_location='cpu')
        assert all(torch.equal(value, old['model'][key]) for key, value in cp['model'].items())
        if arm == 'muon':
            assert not cp['optimizer']['muon']['state'] and not cp['optimizer']['adamw']['state']
        else:
            assert not cp['optimizer']['state']
        # Copy only complete lines; no live/mid-write files get staged.
        training = []
        for line in (out / 'training.jsonl').read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            training.append(row)
        assert training
        for step, row in enumerate(training, 1):
            assert row['step'] == step and row['lr'] == (r.learning_rate(step) if arm in ('muon', 'adamw') else .001)
            assert all(__import__('math').isfinite(row[key]) for key in ('loss', 'grad_norm', 'optimizer_seconds', 'elapsed_s'))
        initial_model_sha = hashlib.sha256(b''.join(cp['model'][key].numpy().tobytes() for key in sorted(cp['model']))).hexdigest()
        result = {'name': name, 'card': r.CARDS[arm], 'worker': worker, 'trainer': trainer,
                  'source_commit': expected_commit, 'registration_sha256': config['registration_sha256'],
                  'initial_model_matches': str(historical), 'initial_model_sha256': initial_model_sha,
                  'initial_checkpoint': entry, 'initial_optimizer_empty': True,
                  'frozen_pool_matches_historical': True, 'initial_position_counts': [p['correct'] for p in rows[0]['positions']],
                  'observed_steps': len(training), 'observed_finite': True, 'latest_update': training[-1],
                  'validation_steps': [row['step'] for row in rows], 'retained_checkpoints': len(list(out.glob('*.pt'))),
                  'wall_seconds_bound': config['registration']['resource_bounds'][arm]['wall_seconds']}
        results[arm] = result
        if snapshot:
            dest = root / 'receipts' / 'launch_snapshot' / name
            dest.mkdir(parents=True, exist_ok=False)
            for file in ('config.json', 'data_receipt.json'):
                (dest / file).write_bytes((out / file).read_bytes())
            r.x7.write_json(dest / 'validation.json', rows)
            r.x7.write_json(dest / 'checkpoints.json', json.loads((out / 'checkpoints.json').read_text()))
            (dest / 'training_prefix.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in training))
            r.x7.write_json(root / 'receipts' / f'launch_verification_{arm}.json', result)
        del cp, old
    assert len({results[arm]['initial_model_sha256'] for arm in ('muon', 'adamw', 'sf')}) == 1
    result = {'verified_at_epoch': time.time(), 'source_commit': expected_commit,
              'all_initializations_and_pools_match': True, 'runs': results,
              'preregistration_commit_is_ancestor': subprocess.run(['git', 'merge-base', '--is-ancestor', expected_commit, 'HEAD']).returncode == 0}
    if snapshot:
        r.x7.write_json(r.ROOTS['muon'] / 'receipts' / 'launch_verification.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', action='store_true')
    args = parser.parse_args()
    verify(args.snapshot)
