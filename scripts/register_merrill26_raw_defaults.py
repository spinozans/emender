"""Freeze X26c trigger receipts and pure shipped-pipeline control before launch."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from experiments.expressivity_tasks.run_merrill26_raw_defaults import ROOT, CARDS
from experiments.expressivity_tasks.run_merrill26_native_control import ROOT as NATIVE
from experiments.expressivity_tasks.merrill26_replication_models import SOURCES
from experiments.expressivity_tasks.run_merrill26_separators import digest, write_json


def triggers():
    rows, hashes = {}, {}
    for stage in ('26_replication', '26b_native_control'):
        root = ROOT.parent / ('results_stage' + stage)
        for arm in ('rnn', 'delta'):
            name = f'{arm}_integer_seed0'
            out = root / name
            paths = [out / 'result.json', out / 'validation.json',
                     root / 'receipts' / name / 'terminal.json']
            r, terminal = json.loads(paths[0].read_text()), json.loads(paths[2].read_text())
            if (r['status'] != 'complete' or terminal['status'] != 'complete'
                    or r['step'] != 60000 or terminal['result_sha256'] != digest(paths[0])):
                raise ValueError('failure trigger missing completed matching receipts')
            metrics = {}
            for split, value in r['tests'].items():
                metrics[split] = {k: value[k] for k in ('accuracy', 'balanced_accuracy',
                    'class_recall', 'confusion_true_by_pred') if k in value}
            rows[f'{stage}/{name}'] = {'step': r['step'], 'best_step': r['best_step'],
                'metrics': metrics,
                'disclosure': ('X26 exact balanced pools: raw equals balanced accuracy; class recalls not logged'
                               if stage == '26_replication' else
                               'X26b learns partially ID; OOD near chance balanced despite high raw accuracy'),
                'hashes': {str(p): digest(p) for p in paths}}
            hashes.update({str(p.resolve()): digest(p) for p in paths})
    return rows, hashes


def register():
    if (ROOT / 'registration.json').exists():
        raise ValueError('registration exists; never overwrite')
    trigger, dependencies = triggers()
    write_json(ROOT / 'receipts/trigger.json', trigger)
    data = json.loads((NATIVE / 'receipts/data.json').read_text())
    for path, sha in data['files'].items():
        if digest(path) != sha:
            raise ValueError('X26b frozen data changed')
        dependencies[str(Path(path).resolve())] = sha
    manifest_path = SOURCES.parent / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    sources = []
    for path, sha in manifest['selected_files_sha256'].items():
        local = SOURCES / path
        blob = subprocess.check_output(['git', '-C', '/tmp/LinearRNN-x26', 'show',
                                       manifest['commit'] + ':' + path])
        if digest(local) != sha or hashlib.sha256(blob).hexdigest() != sha:
            raise ValueError('upstream snapshot mismatch: ' + path)
        sources.append(str(local))
    for arm in ('rnn', 'delta'):
        smoke = json.loads((ROOT / 'receipts' / f'smoke_{arm}_integer.json').read_text())
        if (smoke['status'] != 'passed' or smoke['batch'] != 256 or smoke['n'] != 300
                or smoke['optimizer_updates'] != 0):
            raise ValueError('max-bin smoke not qualified')
        venv = Path(f'/tmp/x26c-{arm}-venv')
        for p in [venv / 'pyvenv.cfg', *venv.glob('lib/*/site-packages/x26c_cuda_reuse.pth')]:
            dependencies[str(p)] = digest(p)
    files = ['experiments/expressivity_tasks/run_merrill26_raw_defaults.py',
        'experiments/expressivity_tasks/run_merrill26_native_control.py',
        'experiments/expressivity_tasks/run_merrill26_replication.py',
        'experiments/expressivity_tasks/run_merrill26_separators.py',
        'experiments/expressivity_tasks/merrill26_replication_models.py',
        'scripts/launch_merrill26_raw_defaults.sh', 'scripts/watch_merrill26_raw_defaults.py',
        'scripts/watch_merrill26_separators.py', 'scripts/gpu_lease.sh',
        'scripts/register_merrill26_raw_defaults.py', 'tests/test_merrill26_raw_defaults.py',
        str(manifest_path), str(NATIVE / 'receipts/data.json')]
    files += sources + [str(p) for p in ROOT.glob('*.md')]
    files += [str(p) for p in (ROOT / 'receipts').iterdir() if p.is_file()]
    write_json(ROOT / 'registration.json', {'task': 'X26c',
        'registered_utc': datetime.now(timezone.utc).isoformat(),
        'base_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'optimizer_updates_before_registration': 0, 'runs': list(CARDS), 'cards': CARDS,
        'source_commit': manifest['commit'], 'protocol': str(ROOT / 'REGISTRATION.md'),
        'predictions': {'raw-succeeds': 'Figure2-like ID and OOD with balanced accuracy decisively above chance: config/loop bundle divergence',
            'raw-fails': 'no Figure2-like result: environment/config fragility OR environment breaks assumptions; no impossibility claim',
            'mixed': 'ID-only learning or model disagreement: qualify OOD claims against matrix-length bins'},
        'data_receipt_sha256': digest(NATIVE / 'receipts/data.json'),
        'data_generation': data['command'], 'trigger': trigger,
        'loss_correction': 'Supervisor CE assertion was incorrect; shipped final BCE +0.3 auxiliary BCE retained',
        'source_sha256': {p: digest(p) for p in sorted(set(files))},
        'dependency_sha256': dependencies})


if __name__ == '__main__':
    register()
