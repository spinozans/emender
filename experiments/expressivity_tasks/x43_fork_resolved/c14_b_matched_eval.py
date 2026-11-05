"""Paired evaluation amendment for the completed matched-exposure twin."""
import inspect
import json
from pathlib import Path
import sys

# This file is invoked directly, so imports are pinned to the frozen checkout.
SESSION = Path('/home/erikg/emender')
BOUND = Path('/home/erikg/emender_c14_matched_bound')
REL = Path('experiments/expressivity_tasks/x43_fork_resolved')
OUT = SESSION / REL / 'receipts/c14_b_matched'

def main():
    assert sys.executable == '/home/erikg/emender/.venv/bin/python'
    import hashlib
    registration = json.loads((OUT / 'registration.json').read_text())
    for name, value in registration['code_sha256'].items():
        assert hashlib.sha256((BOUND / REL / name).read_bytes()).hexdigest() == value, name
    training = json.loads((OUT / 'training.json').read_text())
    assert training['status'] == 'completed_matched_exposure'
    assert (training['steps'], training['actual_bytes_targets']) == (35419, 71677042)
    for row in training['telemetry'].values():
        assert row['peak_occupancy'] < 257 and row['counts']['capacity_merges'] == 0
    sys.path.insert(0, str(BOUND))
    from experiments.expressivity_tasks.x43_fork_resolved import c14_b_control_eval as evaluation
    from experiments.expressivity_tasks.x43_fork_resolved.c14_b_control_model import StoreIntegrated
    assert Path(evaluation.__file__).resolve().is_relative_to(BOUND)
    evaluation.OUT = OUT
    evaluation.StoreIntegrated = lambda: StoreIntegrated('unbudgeted')
    # Same control-arm evaluation policy; synthetic budgeted decay is inapplicable.
    evaluation.synthetic_decay = lambda model: []
    source = inspect.getsource(evaluation.main)
    before = "assert training['status']=='completed_active_budget'"
    assert source.count(before) == 1
    source = source.replace(before, "assert training['status']=='completed_matched_exposure'")
    (OUT / 'effective_eval_main.txt').write_text(source)
    amendment = dict(status='evaluation_started', runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        historical_main_sha256=hashlib.sha256(inspect.getsource(evaluation.main).encode()).hexdigest(),
        effective_main_sha256=hashlib.sha256(source.encode()).hexdigest(),
        change='accept completed_matched_exposure status; frozen windows, strata and reduction unchanged')
    (OUT / 'eval_amendment.json').write_text(json.dumps(amendment, indent=2)+'\n')
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), evaluation.__dict__)
    evaluation.main()

if __name__ == '__main__':
    main()
