"""Freeze X26b native-task control, failure triggers, provenance and gates."""
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys

import fla
import torch

from experiments.expressivity_tasks.run_merrill26_native_control import ROOT, CARDS, BATCH, LR, WEIGHT_DECAY
from experiments.expressivity_tasks.merrill26_replication_models import SOURCES
from experiments.expressivity_tasks.run_merrill26_separators import digest, write_json


def register():
    if (ROOT / 'registration.json').exists():
        raise ValueError('registration exists; never overwrite')
    old = ROOT.parent / 'results_stage26_replication'
    triggers = {}
    for arm in ('rnn', 'delta'):
        out = old / f'{arm}_integer_seed0'
        result = json.loads((out / 'result.json').read_text())
        terminal = json.loads((old / 'receipts' / out.name / 'terminal.json').read_text())
        assert result['status'] == terminal['status'] == 'complete' and result['step'] == 60000
        assert terminal['result_sha256'] == digest(out / 'result.json')
        history = json.loads((out / 'validation.json').read_text())
        best = next(row for row in history if row['step'] == result['best_step'])
        triggers[arm] = {'step': result['step'], 'best_step': result['best_step'],
                         'best_validation': best['validation']['accuracy'],
                         'hashes': {str(p): digest(p) for p in (out / 'result.json', out / 'validation.json',
                             old / 'receipts' / out.name / 'terminal.json')}}
    write_json(ROOT / 'receipts/trigger.json', triggers)
    smokes = {}
    for name in CARDS:
        arm = name.split('_')[0]
        row = json.loads((ROOT / 'receipts' / f'smoke_{arm}_integer.json').read_text())
        if row['status'] != 'passed' or row['n'] != 300 or row['batch'] != 128 or row['optimizer_updates'] != 0:
            raise ValueError('unqualified native smoke')
        smokes[name] = row
    manifest = json.loads((old / 'sources/manifest.json').read_text())
    for name, sha in manifest['selected_files_sha256'].items():
        if digest(SOURCES / name) != sha:
            raise ValueError('upstream snapshot changed: ' + name)
        blob = subprocess.check_output(['git', '-C', '/tmp/LinearRNN-x26', 'show',
                                        manifest['commit'] + ':' + name])
        import hashlib
        if hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError('upstream git blob mismatch: ' + name)
    files = ['experiments/expressivity_tasks/run_merrill26_native_control.py',
             'experiments/expressivity_tasks/run_merrill26_replication.py',
             'experiments/expressivity_tasks/run_merrill26_separators.py',
             'experiments/expressivity_tasks/merrill26_replication_models.py',
             'experiments/expressivity_tasks/tasks/merrill26_separators.py',
             'scripts/watch_merrill26_native_control.py', 'scripts/watch_merrill26_separators.py',
             'scripts/launch_merrill26_native_control.sh', 'scripts/gpu_lease.sh',
             'scripts/register_merrill26_native_control.py', 'tests/test_merrill26_native_control.py',
             str(old / 'sources/manifest.json')]
    files += [str(SOURCES / name) for name in manifest['selected_files_sha256']]
    files += [str(p) for p in ROOT.glob('*.md')]
    files += [str(p) for p in (ROOT / 'receipts').glob('*') if p.is_file()]
    dependencies = {str(p): digest(p) for p in Path(fla.__file__).parent.rglob('*.py')}
    data_receipt = json.loads((ROOT / 'receipts/data.json').read_text())
    dependencies.update({str(Path(p).resolve()): sha for p, sha in data_receipt['files'].items()})
    reg = {'task': 'X26b', 'claim': 'native task-code control, not exact Figure2 replication',
        'registered_utc': datetime.now(timezone.utc).isoformat(), 'optimizer_updates_before_registration': 0,
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'runs': list(CARDS), 'cards': CARDS, 'protocol': str(ROOT / 'REGISTRATION.md'),
        'predictions': {'learn_native': 'contract bundle divergence implicated, no single-field attribution',
                        'fail_native': 'reconciled protocol implicated; raw defaults next registered control',
                        'mixed': 'model-by-contract interaction; no forced dichotomy'},
        'recipe': {'batch': BATCH, 'lr': LR, 'weight_decay': WEIGHT_DECAY, 'clip': 1.,
                   'max_steps': 60000, 'eval_every': 500, 'warmup': 0, 'auxiliary_weight': .3,
                   'curriculum': '5->100/150 eligible epochs; tails carried128'},
        'trigger': triggers, 'source_repository': manifest, 'max_bin_zero_update_smokes': smokes,
        'data_receipt': data_receipt, 'versions': {'python': sys.version, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'flash-linear-attention': version('flash-linear-attention'),
            'triton': version('triton'), 'numpy': version('numpy')},
        'source_sha256': {p: digest(p) for p in sorted(set(files))}, 'dependency_sha256': dependencies}
    write_json(ROOT / 'registration.json', reg)


if __name__ == '__main__':
    register()
