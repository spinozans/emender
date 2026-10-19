"""Preregistered W3 branch, schedule and conservative frozen-gate rules."""
import json

import pytest

from scripts.w3_attest_x9b_completion import attest
from scripts.w3_fix_regime_branch import NAME, fix_branch
from scripts.w3_gate_read import classify
from scripts.w3_matcheddose_preflight import file_sha
from scripts.w3_train_arm import learning_rate


def final_x9b(tmp_path, peak):
    run = tmp_path / NAME
    run.mkdir()
    (tmp_path / 'markers').mkdir()
    (tmp_path / 'receipts' / NAME).mkdir(parents=True)
    trajectory = []
    for step in range(0, 500001, 500):
        positions = [dict(driver_position=p, count=512, correct=512 if p == 1 else
                          peak if p == 2 and step == 250000 else 0) for p in range(1, 17)]
        trajectory.append(dict(step=step, positions=positions))
    (run / 'validation.json').write_text(json.dumps(trajectory))
    result = dict(name=NAME, status='complete', total_steps=500000, stop_reason='safety_ceiling', final=trajectory[-1])
    result_path = run / 'result.json'
    result_path.write_text(json.dumps(result))
    terminal = dict(name=NAME, exit_code=0, result_sha256=file_sha(result_path))
    for path in (tmp_path / 'markers' / f'{NAME}.DONE', tmp_path / 'receipts' / NAME / 'terminal.json'):
        path.write_text(json.dumps(terminal))
    (run / 'config.json').write_text(json.dumps(dict(name=NAME, safety_ceiling=500000,
        optimizer='AdamWScheduleFree', internal_warmup=0, eval_examples=512, eval_every=500)))
    frozen = run / 'frozen_validation.npz'
    frozen.write_bytes(b'fixture-frozen-validation-pool')
    (run / 'data_receipt.json').write_text(json.dumps(dict(sha256=file_sha(frozen), examples=512)))
    (tmp_path / 'completed-trajectory.json').write_text(json.dumps(attest(tmp_path)))
    return tmp_path


def selected(root):
    return fix_branch(root, root / 'completed-trajectory.json')


def test_partial_x9b_never_fixes_branch(tmp_path):
    (tmp_path / NAME).mkdir()
    (tmp_path / NAME / 'validation.json').write_text('live partial, not branch authority')
    with pytest.raises(ValueError, match='WAIT'):
        selected(tmp_path)


@pytest.mark.parametrize('peak,expected', [(49, 'adamw-cosine'), (50, 'sf-cosine')])
def test_completed_trajectory_branch_uses_max_anywhere_not_flat_final(tmp_path, peak, expected):
    result = selected(final_x9b(tmp_path, peak))
    assert result['regime'] == expected
    assert result['fixed_final_per_position'][1] == 0
    assert result['escape_without_consolidation'] is (peak >= 50)


def test_terminal_result_hash_mismatch_is_hard_stop(tmp_path):
    root = final_x9b(tmp_path, 512)
    (root / NAME / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='terminal digest mismatch'):
        selected(root)


def test_intermediate_49_to_50_tamper_is_rejected_by_terminal_digest(tmp_path):
    root = final_x9b(tmp_path, 49)
    path = root / NAME / 'validation.json'
    original_terminal = (root / 'receipts' / NAME / 'terminal.json').read_bytes()
    trajectory = json.loads(path.read_text())
    trajectory[500]['positions'][1]['correct'] = 50
    path.write_text(json.dumps(trajectory))
    assert (root / 'receipts' / NAME / 'terminal.json').read_bytes() == original_terminal
    with pytest.raises(ValueError, match='terminal digest mismatch'):
        selected(root)


def test_successful_original_done_without_supplemental_receipt_is_not_branch_evidence(tmp_path):
    root = final_x9b(tmp_path, 512)
    (root / 'completed-trajectory.json').unlink()
    with pytest.raises(ValueError, match='missing completed-trajectory terminal attestation'):
        selected(root)


def test_malformed_supplemental_receipt_fails_closed(tmp_path):
    root = final_x9b(tmp_path, 512)
    (root / 'completed-trajectory.json').write_text('{}')
    with pytest.raises(ValueError, match='malformed completed-trajectory'):
        selected(root)


@pytest.mark.parametrize('change', ['row-count', 'final-row'])
def test_attestor_rejects_incomplete_or_different_final_trajectory(tmp_path, change):
    root = final_x9b(tmp_path, 49)
    path = root / NAME / 'validation.json'
    trajectory = json.loads(path.read_text())
    if change == 'row-count':
        trajectory.pop(1)
    else:
        trajectory[-1]['positions'][1]['correct'] = 50
    path.write_text(json.dumps(trajectory))
    with pytest.raises(ValueError, match='trajectory coverage/binding'):
        attest(root)


def test_registered_lr_endpoints_and_fixed_stop():
    assert all(learning_rate(i, 'sf-constant') == 1e-5 for i in range(1, 513))
    for regime in ('sf-cosine', 'adamw-cosine'):
        values = [learning_rate(i, regime) for i in range(1, 513)]
        assert values[0] == 1e-5 and values[-1] == 1e-7
        assert all(a >= b for a, b in zip(values, values[1:]))
    with pytest.raises(ValueError):
        learning_rate(513, 'sf-cosine')


def reads(parent=(6, 6, 6), a=(6, 6, 6), b=(10, 10, 10), execution=64):
    return {name: dict(stage_b=[dict(valid=12, correct=n, invalid=0) for n in counts],
                       execution=dict(successes=execution))
            for name, counts in [('parent', parent), ('A', a), ('B', b)]}


def test_strong_local_rescue_uses_all_repetitions():
    assert classify(reads())['strong_regimen_rescue']
    result = classify(reads(parent=(6, 7, 6)))
    assert not result['strong_regimen_rescue']
    assert result['conservative_b_minus_parent'] == 3
    assert result['serving_nondeterminism']['parent']
    assert not classify(reads(a=(6, 8, 6)))['strong_regimen_rescue']


def test_prediction_falsifiers_and_no_execution_replica_rescue():
    result = classify(reads(b=(8, 8, 8)))
    assert result['large_matched_dose_regimen_rescue_prediction_failed']
    result = classify(reads(b=(9, 9, 9)))
    assert result['partial_acquisition_gate_failure'] and not result['dual_gate']['B']
    assert not classify(reads(execution=63))['strong_regimen_rescue']


def test_case_divergence_is_reported_even_when_tallies_match():
    data = reads()
    for index, repeat in enumerate(data['B']['stage_b']):
        repeat['per_episode'] = [dict(id='fixed-case', grade=dict(success=True), reason='finished' if index else 'turn_budget')]
    assert classify(data)['serving_nondeterminism']['B']
