"""X15b resume admission, exact state, replay rejection, disk guard and verdict tests."""
import json
import math
from types import SimpleNamespace
import pytest
import torch
from scripts import invertible_stage15_muon_resume as r
from scripts import watch_invertible_stage15_muon_resume as w


def test_disk_guard_fail_closed_boundary_and_stat_error(monkeypatch):
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=r.DISK_MIN_BYTES))
    assert r.disk_guard() == 200 * 1024**3
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=r.DISK_MIN_BYTES - 1))
    with pytest.raises(r.DiskGuardError):
        r.disk_guard()
    def unavailable(_):
        raise OSError('stat unavailable')
    monkeypatch.setattr(r.shutil, 'disk_usage', unavailable)
    with pytest.raises(OSError):
        r.disk_guard()


def test_checkpoint_hash_rejected_before_deserialization(tmp_path, monkeypatch):
    path = tmp_path / 'bad.pt'
    path.write_bytes(b'bad')
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('untrusted deserialization'))
    with pytest.raises(ValueError, match='hash mismatch'):
        r.load_checkpoint(path, 'wrong')


@pytest.mark.parametrize('missing', ['muon', 'adamw', 'torch_rng', 'cuda_rng'])
def test_missing_optimizer_or_rng_rejected(tmp_path, missing):
    saved = {'step': 433000, 'optimizer_family': 'muon3x',
        'optimizer': {'muon': {'state': {0: {'momentum': torch.ones(1)}}},
                      'adamw': {'state': {0: {'step': torch.ones(1)}}}},
        'numpy_rng': {'state': 1}, 'torch_rng': torch.ones(1), 'cuda_rng': [torch.ones(1)]}
    if missing in ('muon', 'adamw'):
        saved['optimizer'][missing]['state'] = {}
    else:
        saved[missing] = [] if missing == 'cuda_rng' else torch.empty(0)
    path = tmp_path / 'test.pt'
    torch.save(saved, path)
    with pytest.raises(ValueError, match='missing'):
        r.load_checkpoint(path, r.x7.digest(path))


def test_replay_full_and_truncated_last_row_and_exact_mismatch(tmp_path):
    path = tmp_path / 'training.jsonl'
    diagnostics = dict(zip(r.DIAGNOSTICS, (8.16, .1, r.original.learning_rate(433001), None, .12)))
    lines = [json.dumps({'step': step, **diagnostics, 'update_seconds': .04})
             for step in range(433001, 433297)]
    lines[-1] = lines[-1].rsplit('0.04', 1)[0] + '0.0'
    path.write_text('\n'.join(lines))
    refs = r.replay_reference(path)
    assert len(refs) == 296 and refs[433296] == diagnostics
    r.compare_replay(433296, diagnostics, refs)
    with pytest.raises(ValueError, match='replay mismatch'):
        r.compare_replay(433296, {**diagnostics, 'loss': 8.1600000001}, refs)
    path.write_text('\n'.join(lines[:-1]))
    with pytest.raises(ValueError, match='missing replay'):
        r.replay_reference(path)


def test_exact_state_equality_and_drift():
    r.assert_state_equal({'a': [torch.tensor([1.]), {'b': 3}]}, {'a': [torch.tensor([1.]), {'b': 3}]})
    with pytest.raises(ValueError):
        r.assert_state_equal(torch.tensor([1.00001]), torch.tensor([1.]))
    with pytest.raises(ValueError):
        r.assert_state_equal({'a': 1}, {'b': 1})


def test_original_schedule_never_restarted():
    assert r.CEILING == 1000000
    assert r.original.learning_rate(433001) == pytest.approx(.0018382515623657625, rel=.001)
    assert r.original.learning_rate(1000000) == .00003
    assert 1000000 - 433296 == 566704


def test_low_disk_train_never_loads_or_creates_output(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'ROOT', tmp_path)
    monkeypatch.setattr(r, 'OUT', tmp_path / 'muon3x_seed0')
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: {'artifact_sha256': {}})
    monkeypatch.setattr(r.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(r, 'load_checkpoint', lambda: pytest.fail('must not load on low disk'))
    with pytest.raises(r.DiskGuardError):
        r.train()
    assert not r.OUT.exists()


def test_addendum_reports_all64_and_no_false_terminal(tmp_path, monkeypatch):
    root = tmp_path / 'muon_resume'
    positions = [{'driver_position': j+1, 'correct': 12 if j == 0 else 0,
                  'loss': math.log(3721)} for j in range(64)]
    rows = [{'step': 433000, 'aggregate': {'loss': 8.16}, 'positions': positions}]
    r.x7.write_json(tmp_path / 'muon3x_seed0/validation.json', rows)
    monkeypatch.setattr(w.original, 'arm_status', lambda *_: (None, 'RUNNING'))
    root.mkdir()
    w.growth_report(root)
    text = (root / 'X15G-growth-report-addendum.md').read_text()
    assert '|P64|' in text and 'Terminal verdict pending/censored' in text and 'NOT proof' in text
    rows[0]['step'] = 1000000
    rows[0]['positions'][6]['correct'] = 50
    r.x7.write_json(root / 'muon3x_seed0/validation.json', rows)
    monkeypatch.setattr(w.original, 'arm_status', lambda *_: (0, 'DONE'))
    w.growth_report(root)
    assert 'Terminal registered escape verdict: ESCAPED' in (root / 'X15G-growth-report-addendum.md').read_text()
    rows[0]['positions'][6]['correct'] = 0
    r.x7.write_json(root / 'muon3x_seed0/validation.json', rows)
    w.growth_report(root)
    assert 'NO REGISTERED ESCAPE' in (root / 'X15G-growth-report-addendum.md').read_text()


def test_live_disk_abort_preserves_checkpoint_and_publishes_resumable_metadata(tmp_path, monkeypatch):
    import numpy as np
    root, source = tmp_path / 'resume', tmp_path / 'original'
    source.mkdir()
    root.mkdir()
    checkpoint = source / 'step_433000.pt'
    checkpoint.write_bytes(b'immutable original checkpoint')
    cp = {'path': str(checkpoint), 'sha256': r.x7.digest(checkpoint), 'step': 433000}
    model = torch.nn.Linear(1, 1)
    opt = torch.optim.SGD(model.parameters(), lr=.001)
    rng = np.random.default_rng(0)
    cuda_rng = [torch.ones(1, dtype=torch.uint8)]
    saved = {'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'numpy_rng': rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
        'cuda_rng': cuda_rng, 'config': {'arm': 'muon3x'}}
    registration = {'artifact_sha256': {}, 'pool_array_sha256': {'0': []},
                    'resource_bounds': {'muon3x': {'wall_seconds': 3600, 'peak_allocated_bytes': 10**9}}}
    r.x7.write_json(root / 'registration.json', registration)
    r.x7.write_json(source / 'validation.json', [{'step': 433000, 'aggregate': {'accuracy': 0., 'loss': 8.16},
        'checkpoint': cp, 'optimizer_seconds_total': 1.}])
    r.x7.write_json(source / 'checkpoints.json', {str(checkpoint): cp})
    for name, value in [('ROOT', root), ('OUT', root / 'muon3x_seed0'), ('SOURCE', source), ('CHECKPOINT', checkpoint)]:
        monkeypatch.setattr(r, name, value)
    monkeypatch.setattr(r.x10, 'verify_registration', lambda _: registration)
    monkeypatch.setattr(r, 'load_checkpoint', lambda: saved)
    monkeypatch.setattr(r, 'replay_reference', lambda _: {433001: {'loss': 8.16}})
    monkeypatch.setattr(r.original, 'build_model', lambda _: model)
    monkeypatch.setattr(r.original, 'make_optimizer', lambda *_: opt)
    monkeypatch.setattr(r.original, 'array_hashes', lambda _: [])
    monkeypatch.setattr(r.original.x7, 'frozen_pools', lambda *_: {'val': None})
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: cuda_rng)
    monkeypatch.setattr(r.original, 'train_update', lambda *_: {'loss': 8.16, 'optimizer_seconds': .01})
    monkeypatch.setattr(r.original, 'checkpoint', lambda *_: pytest.fail('checkpoint write on low disk'))
    calls = []
    def guard():
        calls.append(1)
        if len(calls) == 4:  # before the first post-update log write
            raise r.DiskGuardError('injected low disk')
    monkeypatch.setattr(r, 'disk_guard', guard)
    with pytest.raises(r.DiskGuardError):
        r.train()
    interruption = json.loads((r.OUT / 'interrupted.json').read_text())
    assert interruption['status'] == 'INFRA-INTERRUPTED'
    assert interruption['resumable_checkpoint'] == cp
    assert checkpoint.read_bytes() == b'immutable original checkpoint'
    assert not (r.OUT / 'validation.json').exists()
    assert not (r.OUT / 'result.json').exists()
