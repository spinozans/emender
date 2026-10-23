"""Freeze the append-only X25 amendment and reused zero-update qualification gates."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys

from experiments.expressivity_tasks.run_recipe_controlled_separators import ROOT, CARDS
from experiments.expressivity_tasks.run_merrill26_separators import digest, write_json


def register():
    if (ROOT / 'registration.json').exists():
        raise ValueError('X25 registration already exists; do not overwrite')
    base = ROOT.parent
    old23 = base / 'results_stage23e_separators' / 'registration.json'
    old24 = base / 'results_stage24_softplus' / 'registration.json'
    x23, x24 = (json.loads(p.read_text()) for p in (old23, old24))
    # Runtime dependencies and recurrence sources must still match the qualified X24 engine.
    for name, sha in x24['dependency_sha256'].items():
        if digest(name) != sha:
            raise ValueError('qualified dependency changed: ' + name)
    for name, sha in x24['source_sha256'].items():
        if name.startswith('ndm/') and digest(name) != sha:
            raise ValueError('qualified model/kernel changed: ' + name)
    smokes = {}
    reused = {}
    for name in CARDS:
        arm, task, _ = name.split('_')
        source = (old24.parent if arm == 'e97softplus' else old23.parent) / 'receipts' / f"smoke_{'e97' if arm == 'e97softplus' else arm}_{task}.json"
        row = json.loads(source.read_text())
        if row['status'] != 'passed' or row['optimizer_updates'] != 0:
            raise ValueError('unqualified smoke: ' + name)
        target = ROOT / 'receipts' / f'smoke_{arm}_{task}.json'
        shutil.copyfile(source, target)
        smokes[name] = row
        reused[name] = {'source': str(source), 'sha256': digest(source)}
    sources = set(x24['source_sha256']) | {str(old23), str(old24)}
    sources.update(['experiments/expressivity_tasks/run_recipe_controlled_separators.py',
        'scripts/launch_recipe_controlled_separators.sh', 'scripts/watch_recipe_controlled_separators.py',
        'scripts/register_recipe_controlled_separators.py', 'tests/test_recipe_controlled_separators.py'])
    sources.update(str(p) for p in ROOT.rglob('*') if p.is_file())
    for prior in (old23.parent, old24.parent):
        for filename in ('recipe_superseded.json', 'superseded_before.json',
                         'interruption_terminal.json', 'interruption_result.json', 'terminal.json'):
            sources.add(str(prior / 'receipts' / 'e97_graph_seed0' / filename))
        sources.update(str(prior / 'e97_graph_seed0' / f) for f in ('result.json', 'checkpoint.json'))
        sources.add(str(prior / 'markers' / 'e97_graph_seed0.CENSORED'))
    reg = {**x23, 'task': 'X25', 'registered_utc': datetime.now(timezone.utc).isoformat(),
        'host': socket.gethostname(), 'python': sys.version,
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'protocol': str(ROOT / 'REGISTRATION.md'), 'runs': list(CARDS), 'cards': CARDS,
        'primary': ['graph', 'integer'], 'secondary_adapted': [],
        'predictions': {'X23E_unchanged': x23['predictions'], 'X24_unchanged': x24['predictions']},
        'architecture': {**x23['architecture'], 'e97softplus_state': 'plain softplus, X24 legacy fused engine'},
        'recipe': {**x23['recipe'], 'weight_decay': 1e-4, 'warmup': 0, 'peak_lr': 1e-4,
                   'final_lr': 1e-4, 'schedule': 'constant paper Section6 learning rate'},
        'confounded_control_recipe': x23['recipe'],
        'amends': {str(p): digest(p) for p in (old23, old24)},
        'comparison': {'datasets': str(old23.parent / 'datasets'),
            'dataset_receipts_at_registration': {str(p): digest(p) for task in ('graph', 'integer')
                for p in (old23.parent / 'datasets' / task).glob('*.json')}},
        'max_bin_zero_update_smokes': smokes, 'reused_smoke_sources': reused,
        'kernel_match': x24['kernel_match'],
        'source_sha256': {name: digest(name) for name in sorted(sources)},
        'dependency_sha256': x24['dependency_sha256']}
    write_json(ROOT / 'registration.json', reg)


if __name__ == '__main__':
    register()
