"""X15b amendment-2 checkpoint-branch provenance, guards and terminal reporting."""
import json
import math
import pytest
import torch
from scripts import invertible_stage15_muon_restart as r
from scripts import watch_invertible_stage15_muon_restart as w
from scripts.invertible_stage15_muon_resume import DiskGuardError


def test_original_million_update_schedule_and_explicit_branch():
    assert r.CEILING == 1000000
    assert r.original.learning_rate(1000000) == .00003
    assert 1000000 - 433000 == 567000
    assert 'superseded by re-execution' in r.PROVENANCE
    assert 'replay' not in r.PROVENANCE


def test_addendum_reports_all64_and_no_false_terminal(tmp_path, monkeypatch):
    root = tmp_path / 'muon_restart'
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


@pytest.mark.parametrize("abort", [True, False])
def test_restart_branch_boundary_new_updates_guard_and_checkpoint(tmp_path, monkeypatch, abort):
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
    monkeypatch.setattr(r.original, 'build_model', lambda _: model)
    monkeypatch.setattr(r.original, 'make_optimizer', lambda *_: opt)
    monkeypatch.setattr(r.original, 'array_hashes', lambda _: [])
    monkeypatch.setattr(r.original.x7, 'frozen_pools', lambda *_: {'val': None})
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'set_rng_state_all', lambda _: None)
    monkeypatch.setattr(torch.cuda, 'get_rng_state_all', lambda: cuda_rng)
    updates = []
    def update(*args):
        updates.append(args[-1])
        return {'loss': 8.16, 'lr': r.original.learning_rate(args[-1]), 'optimizer_seconds': .01}
    monkeypatch.setattr(r.original, 'train_update', update)
    monkeypatch.setattr(r, 'CEILING', 433002 if abort else 435000)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(r.original, 'evaluate', lambda *_: {'aggregate': {'accuracy': 0., 'loss': 8.17},
        'positions': [{'driver_position': j+1, 'correct': 0, 'loss': 8.17} for j in range(64)]})
    if abort:
        monkeypatch.setattr(r.original, 'checkpoint', lambda *_: pytest.fail('checkpoint write on low disk'))
    calls = []
    def guard():
        calls.append(1)
        if abort and len(calls) == 6:  # before the first post-update log write
            raise DiskGuardError('injected low disk')
    monkeypatch.setattr(r, 'disk_guard', guard)
    if abort:
        with pytest.raises(DiskGuardError):
            r.train()
        interruption = json.loads((r.OUT / 'interrupted.json').read_text())
        assert interruption['status'] == 'INFRA-INTERRUPTED'
        assert interruption['resumable_checkpoint'] == cp
        assert not (r.OUT / 'validation.json').exists()
        assert not (r.OUT / 'result.json').exists()
    else:
        r.train()
        assert updates == list(range(433001, 435001))
        result = json.loads((r.OUT / 'result.json').read_text())
        assert result['status'] == 'complete' and result['total_steps'] == 435000
        assert result['final']['step'] == 435000
        assert result['final']['checkpoint']['sha256'] == r.x7.digest(result['final']['checkpoint']['path'])
        pruned = r.OUT / 'step_434000.pt'
        receipts = [json.loads(line) for line in (r.OUT / 'pruning.jsonl').read_text().splitlines()]
        index = json.loads((r.OUT / 'checkpoints.json').read_text())
        assert not pruned.exists()
        assert len(receipts) == 1
        assert receipts[0]['path'] == str(pruned)
        assert receipts[0]['sha256'] == index[str(pruned)]['sha256']
        assert receipts[0]['step'] == 434000
    assert checkpoint.read_bytes() == b'immutable original checkpoint'
    rows = [json.loads(line) for line in (r.OUT / 'training.jsonl').read_text().splitlines()]
    assert rows[0]['event'] == 'restart_from_checkpoint' and rows[0]['step'] == 433000
    assert all(row['provenance'] == r.PROVENANCE for row in rows)
    receipt = json.loads((r.OUT / 'checkpoint_restored.json').read_text())
    assert receipt['model_optimizer_rng_exact'] and not receipt['at_death_restoration_claimed']
    assert receipt['fresh_updates_to_execute'] == 567000


def test_pruning_receipts_append_and_verify_at_deletion(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    import hashlib
    from pathlib import Path

    out = tmp_path / 'restart'
    out.mkdir()
    monkeypatch.setattr(r, 'OUT', out)
    monkeypatch.setattr(r, 'disk_guard', lambda: None)
    retained = out / 'step_435000.pt'
    retained.write_bytes(b'keep')
    original = tmp_path / 'step_433000.pt'
    original.write_bytes(b'immutable original')
    keep = {str(retained), str(original)}
    checkpoints = {}
    before = datetime.now(timezone.utc)
    real_unlink = Path.unlink
    def receipted_unlink(path, *args, **kwargs):
        receipt = json.loads((out / 'pruning.jsonl').read_text().splitlines()[-1])
        assert receipt['path'] == str(path)
        assert receipt['sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert receipt['event'] == 'pre_unlink'
        assert before <= datetime.fromisoformat(receipt['timestamp']) <= datetime.now(timezone.utc)
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', receipted_unlink)
    for step in (434000, 434500):
        path = out / f'step_{step:06d}.pt'
        path.write_bytes(str(step).encode())
        checkpoints[str(path)] = {'path': str(path), 'step': step, 'sha256': r.x7.digest(path)}
        r.prune_checkpoints(keep, checkpoints)
        assert not path.exists()
    receipts = [json.loads(line) for line in (out / 'pruning.jsonl').read_text().splitlines()]
    assert [row['step'] for row in receipts] == [434000, 434500]
    assert retained.read_bytes() == b'keep'
    assert original.read_bytes() == b'immutable original'


@pytest.mark.parametrize('failure', ['hash', 'receipt', 'guard'])
def test_pruning_failure_preserves_checkpoint(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'disk_guard', lambda: None)
    path = tmp_path / 'step_434000.pt'
    path.write_bytes(b'checkpoint')
    checkpoint = {'path': str(path), 'step': 434000, 'sha256': r.x7.digest(path)}
    if failure == 'hash':
        path.write_bytes(b'changed since write-time index receipt')
        error = ValueError
    elif failure == 'receipt':
        def fail_fsync(_):
            raise OSError('injected receipt persistence failure')
        monkeypatch.setattr(r.os, 'fsync', fail_fsync)
        error = OSError
    else:
        def fail_guard():
            raise DiskGuardError('injected low disk')
        monkeypatch.setattr(r, 'disk_guard', fail_guard)
        error = DiskGuardError
    with pytest.raises(error):
        r.prune_checkpoints(set(), {str(path): checkpoint})
    assert path.exists()
    if failure != 'receipt':
        assert not (tmp_path / 'pruning.jsonl').exists()
