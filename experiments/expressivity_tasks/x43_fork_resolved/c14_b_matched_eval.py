"""Paired evaluation amendment for the completed matched-exposure twin."""
import inspect
import json
from pathlib import Path
import sys

# Frozen evaluation policy; explicitly certified current row-key kernel amendment.
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
    # Keep the historical evaluation and model policy, while loading only the
    # certified kernel installer from the attended checkout.
    import experiments.expressivity_tasks.x43_fork_resolved as package
    package.__path__ = [str(SESSION / REL), *package.__path__]
    phase2 = json.loads((SESSION / REL / 'receipts/scalability_phase2/registration.json').read_text())
    for name, value in phase2['sources'].items():
        assert hashlib.sha256((SESSION / name).read_bytes()).hexdigest() == value, name
    from experiments.expressivity_tasks.x43_fork_resolved.c14_phase2_staged import install_rowkeys
    install_rowkeys()
    # The metadata fast path launches directly; apply the same reachable tile.
    fast_source = inspect.getsource(evaluation.fast.forward)
    old_tile = 'BS=triton.next_power_of_2(cfg.slots)'
    assert fast_source.count(old_tile) == 1
    fast_source = fast_source.replace(old_tile, 'BS=triton.next_power_of_2(min(cfg.slots,max(1,2*t)))')
    exec(compile(fast_source, '<matched-eval-reachable-tile>', 'exec'), evaluation.fast.__dict__)
    (OUT / 'effective_eval_fast_forward.txt').write_text(fast_source)
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
        change='accept matched status; certified phase2 rowkeys kernel and reachable tile; frozen windows, strata and reduction unchanged')
    (OUT / 'eval_amendment.json').write_text(json.dumps(amendment, indent=2)+'\n')
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), evaluation.__dict__)
    evaluation.main()

if __name__ == '__main__':
    main()
