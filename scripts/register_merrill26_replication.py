"""Freeze X26 predictions, reconciled configs, source provenance and smoke gates."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import socket
import subprocess
import sys

import fla
import torch

from experiments.expressivity_tasks.run_merrill26_replication import (
    ROOT, COMPARISON_ROOT, CARDS, BATCH, LR, WEIGHT_DECAY, link_datasets,
)
from experiments.expressivity_tasks.run_merrill26_separators import digest, write_json


def register():
    if (ROOT / 'registration.json').exists():
        raise ValueError('X26 registration already exists; do not overwrite')
    link_datasets()
    smokes = {}
    for name in CARDS:
        arm, task, _ = name.split('_')
        row = json.loads((ROOT / 'receipts' / f'smoke_{arm}_{task}.json').read_text())
        if row['status'] != 'passed' or row['optimizer_updates'] != 0 or row['n'] != 300:
            raise ValueError('unqualified max-bin smoke: ' + name)
        smokes[name] = row
    sources = [
        'experiments/expressivity_tasks/merrill26_replication_models.py',
        'experiments/expressivity_tasks/run_merrill26_replication.py',
        'experiments/expressivity_tasks/run_merrill26_separators.py',
        'experiments/expressivity_tasks/tasks/merrill26_separators.py',
        'scripts/register_merrill26_replication.py', 'scripts/launch_merrill26_replication.sh',
        'scripts/watch_merrill26_replication.py', 'scripts/watch_merrill26_separators.py',
        'scripts/gpu_lease.sh', 'tests/test_merrill26_replication.py',
        str(COMPARISON_ROOT / 'registration.json')]
    sources += [str(p) for p in ROOT.rglob('*')
                if p.is_file() and 'datasets' not in p.parts and '__pycache__' not in p.parts]
    # FLA is unpinned in most upstream scripts; bind this actual installed backend.
    dependencies = {str(p): digest(p) for p in Path(fla.__file__).parent.rglob('*.py')}
    dataset_receipts = {str(p.resolve()): digest(p)
                        for task in ('graph', 'integer')
                        for p in (COMPARISON_ROOT / 'datasets' / task).glob('*.json')}
    dependencies.update(dataset_receipts)
    reg = {'task': 'X26', 'claim': 'partial replication under reconciled sources',
        'registered_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
        'python': sys.version, 'optimizer_updates_before_registration': 0,
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'protocol': str(ROOT / 'REGISTRATION.md'), 'runs': list(CARDS), 'cards': CARDS,
        'predictions': {'rnn_graph': '~100% ID, strong OOD length generalization',
                        'delta_graph': 'high ID, degraded OOD',
                        'rnn_integer': 'near-perfect ID', 'delta_integer': 'near-perfect ID'},
        'failure_contingency': 'native-tokenization control and field-by-field task audit; no automatic bug conclusion',
        'recipe': {'batch': BATCH, 'lr': LR, 'weight_decay': WEIGHT_DECAY,
                   'max_steps': 60000, 'eval_every': 500, 'clip': 1.,
                   'stop': 'three perfect ID validation evaluations or maximum steps',
                   'loss': 'BCEWithLogitsLoss; IMM auxiliary .3 on intermediate >0 prefix labels',
                   'curriculum': 'IMM T5->100 over150 eligible-pool epochs, no warmup'},
        'source_repository': json.loads((ROOT / 'sources' / 'manifest.json').read_text()),
        'shared_dataset_receipts': dataset_receipts, 'max_bin_zero_update_smokes': smokes,
        'versions': {'torch': torch.__version__, 'cuda': torch.version.cuda,
                     'flash-linear-attention': version('flash-linear-attention'),
                     'triton': version('triton'), 'numpy': version('numpy')},
        'source_sha256': {s: digest(s) for s in sorted(set(sources))},
        'dependency_sha256': dependencies}
    write_json(ROOT / 'registration.json', reg)


if __name__ == '__main__':
    register()
