"""Approved restart is distinct from verified replay, with a visible provenance boundary."""
import ast
import inspect
import json

import pytest

from scripts import invertible_stage16_17_restart as run
from scripts import watch_invertible_stage16_17_recovery as watcher


def test_restart_provenance_never_claims_at_death_restoration():
    resume = {'checkpoint': {'step': 160000}, 'last_full_step': 160667,
              'partial_step': 160668, 'divergence_receipt': 'failed/interrupted.json'}
    row = run.provenance(resume)
    assert row['mode'] == 'restart_from_checkpoint'
    assert row['historical_tail_superseded_from'] == 160001
    assert row['historical_last_full_step'] == 160667
    assert row['historical_partial_death_step'] == 160668
    assert row['restored_at_death_state'] is False and row['replay_verified'] is False
    assert row['divergence_receipt'] == resume['divergence_receipt']
    assert json.loads(json.dumps(row)) == row


def test_only_approved_divergent_arms_restart():
    assert run.BINDINGS == {'e97_w32_seed0': 1, 'e97_curriculum_seed0': 6, 'e97_carrywean_seed0': 4}
    for name in run.BINDINGS:
        assert run.root_for(name).name == 'infra_restart'
        assert run.replay.root_for(name).name == 'infra_resume'
    assert 'e97_w64_seed0' not in run.BINDINGS
    assert 'gdn2_w512_seed0' not in run.BINDINGS
    assert 'gdn2_w2048_seed0' not in run.BINDINGS


def test_restart_and_replay_share_fail_closed_disk_and_exact_restore_checks():
    assert run.disk_guard is run.replay.disk_guard
    assert run.write_json is run.replay.write_json
    assert run.assert_state_equal is run.replay.assert_state_equal
    assert run.prune_new is run.replay.prune_new
    assert run.valid_result is run.replay.valid_result
    assert run.DISK_MIN_BYTES == 200 * 1024**3


def test_restart_excludes_historical_tail_and_has_no_replay_claim():
    tree = ast.parse(inspect.getsource(run.train))
    calls = [node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert 'compare_replay' not in calls
    assert 'assert_state_equal' in calls and 'provenance' in calls
    text = inspect.getsource(run.train)
    assert "r['step'] <= saved['step']" in text
    assert "row['provenance'] = provenance(resume)" in text
    assert 'replay_verified.json' not in text
    assert 'restart_admitted.json' in text
    assert 'x17.train_update(model, opt, batch, phase_step, rng)' in text
    assert "x16.extension_rate(phase_step, transition['entry_lr'])" in text


def test_watcher_routes_each_arm_to_its_actual_admission(monkeypatch):
    for name in run.replay.BINDINGS:
        expected = 'infra_restart' if name in run.BINDINGS else 'infra_resume'
        assert watcher.active_root(name).name == expected
    monkeypatch.setattr(watcher.w16, 'arm_status', lambda root, name:
                        (1, 'FAIL') if name == 'e97_w32_seed0' else (None, 'RUNNING'))
    assert watcher.stage_status(16)[0] is None  # One failure cannot disarm the others.
    monkeypatch.setattr(watcher.w16, 'arm_status', lambda root, name: (0, 'DONE'))
    assert watcher.stage_status(16)[0] == 0


def test_registration_rejects_replay_success_instead_of_restarting(tmp_path, monkeypatch):
    original = tmp_path / 'infra_resume'
    original.mkdir()
    name = 'e97_w32_seed0'
    (original / name).mkdir()
    (original / name / 'interrupted.json').write_text(json.dumps({'replay_verified': True, 'error': ''}))
    monkeypatch.setattr(run.x16, 'ROOT', tmp_path)
    monkeypatch.setattr(run.x10, 'verify_registration', lambda _: {'source_sha256': {}, 'artifact_sha256': {}})
    monkeypatch.setattr(run.replay, 'verify', lambda _: {})
    monkeypatch.setattr(run, 'disk_guard', lambda: run.DISK_MIN_BYTES)
    with pytest.raises(ValueError, match='failed exact-replay'):
        run.register(16)
