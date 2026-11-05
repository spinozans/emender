"""Matched-exposure stopping amendment over the hash-bound original trainer.

The frozen module remains unchanged. Exact textual substitutions below alter
only stopping/reporting; fail closed if the historical implementation differs.
"""
import hashlib
import inspect
import json
from pathlib import Path
import sys

SESSION = Path('/home/erikg/emender')
BOUND = Path('/home/erikg/emender_c14_matched_bound')
REL = Path('experiments/expressivity_tasks/x43_fork_resolved')
OUT = SESSION / REL / 'receipts/c14_b_matched'
CHECKPOINT = Path('/home/erikg/c14_dense_bytes/emender_c14_b_matched_resume.pt')

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    registration = json.loads((OUT / 'registration.json').read_text())
    assert digest(Path(__file__).resolve()) == registration['runner_sha256']
    for name, value in registration['code_sha256'].items():
        assert digest(BOUND / REL / name) == value, name
    # Direct script invocation prevents the attended checkout entering sys.path.
    sys.path.insert(0, str(BOUND))
    from experiments.expressivity_tasks.x43_fork_resolved import c14_b_train as train
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_v2 import configure
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_control_model import StoreIntegrated
    assert Path(train.__file__).resolve().is_relative_to(BOUND)
    configure()
    train.OUT = OUT
    train.CHECKPOINT = CHECKPOINT
    train.CODE = tuple(registration['code_sha256'])
    train.StoreIntegrated = lambda: StoreIntegrated('unbudgeted')
    phase = sys.argv.pop(1)
    if phase == 'preflight':
        train.preflight()
        print('Frozen source, document receipts, cache shards and launch gates passed.', flush=True)
        return
    assert phase == 'train'
    source = inspect.getsource(train.main)
    substitutions = {
        'budget_active_seconds=2700': 'budget_active_seconds=None, matched_steps=35419, matched_targets=71677042',
        'if stopped or active()>=2700:': 'if stopped or steps>=35419:',
        "status='interrupted' if stopped else 'completed_active_budget'": "status='interrupted' if stopped else 'completed_matched_exposure'",
        "        save(status)\n    except": "        if status == 'completed_matched_exposure':\n            assert steps == 35419 and targets == 71677042\n        save(status)\n    except",
    }
    for before, after in substitutions.items():
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    (OUT / 'effective_train_main.txt').write_text(source)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), train.__dict__)
    train.main()

if __name__ == '__main__':
    main()
