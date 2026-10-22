"""One-shot X23E preregistration; refuses already-generated/trained experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import socket
import subprocess
import sys

from experiments.expressivity_tasks.run_merrill26_separators import ROOT, digest, write_json
from experiments.expressivity_tasks.tasks.merrill26_separators import SPLITS


def register():
    if (ROOT / 'registration.json').exists() or (ROOT / 'datasets').exists() or (ROOT / 'receipts' / 'launch.json').exists():
        raise ValueError('registration/data/launch exists; never overwrite preregistration')
    for path in ROOT.glob('*_seed0'):
        raise ValueError('experimental output already exists: ' + str(path))
    sources = subprocess.check_output(['git', 'ls-files', 'ndm'], text=True).splitlines()
    sources += [str(p) for p in Path('experiments/expressivity_tasks/tasks').glob('*.py')]
    sources += ['experiments/expressivity_tasks/run_merrill26_separators.py',
        'scripts/launch_merrill26_separators.sh', 'scripts/watch_merrill26_separators.py',
        'scripts/register_merrill26_separators.py', 'scripts/gpu_lease.sh',
        'tests/test_merrill26_separators.py',
        'experiments/expressivity_tasks/results_invertible_stage10_optimizers_8m/registration.json',
        'experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m/registration.json',
        'experiments/expressivity_tasks/results_invertible_stage11_gdn2_adamw_8m/REGISTRATION.md',
        'experiments/expressivity_tasks/run_invertible_stage10_optimizers_8m.py',
        'experiments/expressivity_tasks/run_invertible_stage11_gdn2_adamw_8m.py',
        'experiments/expressivity_tasks/run_invertible_stage3_8m.py',
        'experiments/expressivity_tasks/run_invertible_stage7_deepfirst_8m.py']
    sources += [str(p) for p in ROOT.rglob('*') if p.is_file()]
    # FLA is an installed dependency, not a tracked repo tree. Bind its actual
    # source bytes as well as the complete environment version receipt.
    fla = Path(importlib.util.find_spec('fla').origin).parent.resolve()
    dependencies = {str(p): digest(p) for p in sorted(fla.rglob('*.py'))}
    runs = [f'{arm}_{task}_seed0' for task in ('graph', 'integer', 'control', 'mod97')
            for arm in ('e97', 'gdn2')]
    cards = dict(zip(runs, [0, 1, 2, 3, 4, 5, 6, 6]))
    smokes = {}
    for name in runs:
        arm, task, _ = name.split('_')
        row = json.loads((ROOT / 'receipts' / f'smoke_{arm}_{task}.json').read_text())
        assert row['status'] == 'passed' and row['optimizer_updates'] == 0 and row['n'] == 300
        assert row['tokens'] == {'graph': 90904, 'integer': 2700, 'control': 301, 'mod97': 2700}[task]
        smokes[name] = row
    assert (ROOT / 'receipts' / 'tests_exit_code.txt').read_text().strip() == '0'
    reg = {'task': 'X23E', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version, 'optimizer_updates_before_registration': 0,
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'protocol': str(ROOT / 'REGISTRATION.md'), 'runs': runs, 'cards': cards,
        'primary': ['graph', 'integer', 'control'], 'secondary_adapted': ['mod97'],
        'predictions': {'graph': 'E97 ID and OOD >=99%; GDN ID >=99%, far-OOD drop >=5pp; all four outcomes informative',
            'integer': 'both ID and OOD >=99%', 'control': 'both ID and OOD >=99%',
            'mod97': 'secondary adapted final-entry classification: both ID/OOD >=99%'},
        'dataset_specs': SPLITS, 'training_seed': 23000, 'initialization_seed': 0,
        'architecture': {'dim': 256, 'depth': 4, 'mlp_ratio': 2, 'n_heads': 32, 'n_state': 32,
            'e97_expansion': 1., 'gdn_expansion': 8.625, 'e97_state': 'tanh', 'gdn_allow_neg_eigval': True,
            'only_shape_change': 'task-specific tied input/output vocabulary'},
        'recipe': {'optimizer': 'AdamW', 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .01,
            'batch': 64, 'clip_norm': 1., 'warmup': 5000, 'peak_lr': .001, 'final_lr': .00001,
            'schedule': 'linear warmup, cosine to60000', 'max_updates': 60000, 'eval_every': 500,
            'early_stop': '3 consecutive nonbaseline perfect validation evaluations',
            'selection': 'validation accuracy then loss; OOD only after selection'},
        'resource_bounds': {'wall_seconds': 604800, 'min_free_bytes': 400 * 1024**3,
            'max_rung_bytes': 40 * 1024**3, 'checkpoint_payloads': 'best plus atomic staging',
            'memory': 'physical CUDA allocator OOM is infrastructure censoring',
            'guard': 'independent 60s watcher, identity-verified process groups, TERM then KILL'},
        'max_bin_zero_update_smokes': smokes,
        'versions': {p: metadata.version(p) for p in ('torch', 'triton', 'numpy', 'flash-linear-attention')},
        'source_sha256': {p: digest(p) for p in sorted(set(sources))},
        'dependency_sha256': dependencies}
    write_json(ROOT / 'registration.json', reg)
    print(json.dumps({'registration': str(ROOT / 'registration.json'),
                      'sha256': digest(ROOT / 'registration.json'), 'registered_utc': reg['registered_utc']}))


if __name__ == '__main__':
    register()
