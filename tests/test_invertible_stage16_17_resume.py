"""X16R/X17R recovery admission, replay and write-safety contracts."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import invertible_stage16_17_resume as run


def test_disk_guard_fails_closed_and_write_keeps_previous_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'validation.json'
    path.write_text('previous committed bytes')
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=run.DISK_MIN_BYTES - 1))
    with pytest.raises(run.DiskGuardError):
        run.write_json(path, {'new': True})
    assert path.read_text() == 'previous committed bytes'
    monkeypatch.setattr(run.shutil, 'disk_usage', lambda _: SimpleNamespace(free=run.DISK_MIN_BYTES))
    assert run.disk_guard() == run.DISK_MIN_BYTES


def test_full_rows_and_final_prefix_have_separate_scopes(tmp_path):
    path = tmp_path / 'training.jsonl'
    path.write_text('{"step": 10, "loss": 2, "grad_norm": 3, "lr": 0.1}\n'
                    '{"step": 11, "loss": 1, "grad_norm": 2, "lr": 0.1}\n'
                    '{"step": 12, "loss": 0.5, "grad_norm"')
    rows, summary = run.replay_reference(path, 10)
    assert rows == {11: {'loss': 1, 'grad_norm': 2, 'lr': .1}}
    assert summary == {'last_full_step': 11, 'partial_step': 12,
                       'prefix_diagnostics': {'loss': .5}, 'completed_replay_updates': 1}
    run.compare_replay(12, {'loss': .5, 'grad_norm': 100}, summary['prefix_diagnostics'])
    with pytest.raises(ValueError, match='replay mismatch'):
        run.compare_replay(11, {'loss': 1.000001}, rows[11])


@pytest.mark.parametrize('text', [
    '{"step": 12, "loss": 1, "grad_norm": 2, "lr": 0.1}\n',
    '{"step": 11, "loss": 1, "grad_norm": 2, "lr": 0.1}\n{"step": 13, "lo',
    '{"step": 11, "lo\n{"step": 12, "loss": 1}\n',
    '{"step": 11, "loss": 1, "grad_norm": 2, "lr": 0.1}\n'
    '{"step": 11, "loss": 1, "grad_norm": 2, "lr": 0.1}\n',
])
def test_missing_replay_or_nonfinal_truncation_fails_closed(tmp_path, text):
    path = tmp_path / 'training.jsonl'
    path.write_text(text)
    with pytest.raises(ValueError):
        run.replay_reference(path, 10)


def test_readable_step_only_prefix_is_not_claimed_as_full_diagnostics(tmp_path):
    path = tmp_path / 'training.jsonl'
    path.write_text('{"step": 11, "loss": 1, "grad_norm": 2, "lr": 0.1}\n'
                    '{"step": 12, "depth": 16, "phase_st')
    _, summary = run.replay_reference(path, 10)
    assert summary['prefix_diagnostics'] == {}
    assert summary['completed_replay_updates'] == 1


def test_checkpoint_preflight_checks_pruning_hash_optimizer_and_rng(tmp_path):
    import torch
    live, missing = tmp_path / 'step_000010.pt', tmp_path / 'step_000009.pt'
    saved = {'step': 10, 'optimizer_family': 'adamw', 'optimizer': {'state': {0: {'step': 10}},
        'param_groups': [{'lr': .1}]}, 'numpy_rng': {'state': 1},
        'torch_rng': torch.tensor([1], dtype=torch.uint8), 'cuda_rng': [torch.tensor([2])]}
    torch.save(saved, live)
    cp = {'path': str(live), 'sha256': run.x7.digest(live), 'step': 10}
    old = {'path': str(missing), 'sha256': 'prior-deletion-hash', 'step': 9}
    (tmp_path / 'checkpoints.json').write_text(json.dumps({str(live): cp, str(missing): old}))
    (tmp_path / 'pruning.jsonl').write_text(json.dumps({'removed': [old], 'retained': [cp]}) + '\n')
    assert run.checkpoint_preflight(tmp_path)[0] == cp
    (tmp_path / 'pruning.jsonl').write_text(json.dumps({'removed': [], 'retained': [cp]}) + '\n')
    with pytest.raises(ValueError, match='missing without'):
        run.checkpoint_preflight(tmp_path)
    (tmp_path / 'pruning.jsonl').write_text(json.dumps({'removed': [old], 'retained': [cp]}) + '\n')
    live.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash mismatch'):
        run.checkpoint_preflight(tmp_path)
    saved['optimizer']['state'] = {}
    torch.save(saved, live)
    cp['sha256'] = run.x7.digest(live)
    (tmp_path / 'checkpoints.json').write_text(json.dumps({str(live): cp, str(missing): old}))
    (tmp_path / 'pruning.jsonl').write_text(json.dumps({'removed': [old], 'retained': [cp]}) + '\n')
    with pytest.raises(ValueError, match='missing AdamW'):
        run.checkpoint_preflight(tmp_path)


def test_prune_new_never_deletes_original_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'disk_guard', lambda: run.DISK_MIN_BYTES)
    source, out = tmp_path / 'original', tmp_path / 'resume'
    source.mkdir()
    out.mkdir()
    paths = [source / 'entry.pt', out / 'old.pt', out / 'latest.pt']
    for p in paths:
        p.write_bytes(b'checkpoint')
    cps = {str(p): {'path': str(p), 'step': i, 'sha256': run.x7.digest(p)} for i, p in enumerate(paths)}
    run.prune_new(out, cps, {str(paths[-1])})
    assert paths[0].exists() and paths[-1].exists() and not paths[1].exists()
    receipt = json.loads((out / 'pruning.jsonl').read_text())
    assert receipt['removed'] == [cps[str(paths[1])]]


def test_restored_state_is_exact_including_rng():
    import torch
    run.assert_state_equal({'a': [torch.tensor([1]), {'rng': 7}]}, {'a': [torch.tensor([1]), {'rng': 7}]})
    with pytest.raises(ValueError, match='tensor mismatch'):
        run.assert_state_equal(torch.tensor([1]), torch.tensor([2]))
    with pytest.raises(ValueError, match='value mismatch'):
        run.assert_state_equal({'rng': 6}, {'rng': 7})


def test_original_schedules_ceilings_and_gpu_exclusion():
    assert list(run.BINDINGS.values()) == [0, 1, 2, 3, 6, 4]
    assert 7 not in run.BINDINGS.values() and 5 not in run.BINDINGS.values()
    assert run.x16.CEILING == 500000 and run.x17.CEILING == 1000000
    assert run.x17.learning_rate(160668) < run.x17.learning_rate(160000)
    assert run.x17.carry_probability(160668) == .5 * (1 - 160668 / 1000000)
    assert run.x17.carry_probability(1000000) == 0
    assert run.x16.extension_rate(0, .0006) == pytest.approx(.0006)
    assert run.x16.extension_rate(500000, .0006) == pytest.approx(.00001)
