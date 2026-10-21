"""Approved X15 correction is isolated from immutable original Muon sources."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scripts import invertible_stage15_gdn2_amendment as a
from scripts import watch_invertible_stage15_amended as w


def terminal(root, name):
    result = {'name': name, 'status': 'complete',
              'total_steps': 0 if name == 'gdn2_zeroshot' else 1000000,
              'stop_reason': 'zero_shot_complete' if name == 'gdn2_zeroshot' else 'safety_ceiling'}
    path = root / name / 'result.json'
    a.x7.write_json(path, result)
    marker = {'name': name, 'exit_code': 0, 'result_sha256': a.x7.digest(path)}
    a.x7.write_json(root / 'receipts' / name / 'terminal.json', marker)
    a.x7.write_json(root / 'markers' / (name + '.DONE'), marker)


def test_corrected_333_zero_shot_strict_load_historical_control_and_no_training(tmp_path, monkeypatch):
    original_root = a.original.ROOT
    monkeypatch.setattr(a, 'ROOT', tmp_path / 'amendment')
    assert a.original.ROOT == original_root
    historical = tmp_path / 'historical'
    cp = historical / 'run' / 'best.pt'
    cp.parent.mkdir(parents=True)
    cp.write_bytes(b'hash-bound fixture')
    sha = a.x7.digest(cp)
    monkeypatch.setattr(a, 'CHAMPION', cp)
    monkeypatch.setattr(a, 'CHAMPION_SHA256', sha)
    monkeypatch.setattr(a.x11, 'ROOT', historical)
    a.x7.write_json(historical / 'registration.json', {'source_sha256': {}})
    a.x7.write_json(cp.parent / 'result.json', {'best': {'checkpoint': {'sha256': sha}}})
    p16 = a.x7.frozen_pools(0, 16, 61)['val']
    p64 = a.x7.frozen_pools(0, 64, 61)['val']
    np.savez_compressed(cp.parent / 'frozen_validation.npz', **{f'T16_{j}': v for j,v in enumerate(p16)})
    counts = [512,512,512,512,333,3]+[0]*10
    control = {'positions': [{'driver_position': j+1,'correct': c,'loss': 1.} for j,c in enumerate(counts)]}
    a.x7.write_json(cp.parent / 'validation.json', [{'step': 482500, **control}])
    reg = {'pool_array_sha256': {'0': a.array_hashes(p64)}}
    monkeypatch.setattr(a, 'verify_registration', lambda _: reg)
    a.x7.write_json(a.ROOT / 'registration.json', reg)
    model = torch.nn.Linear(2, 2)
    saved = {k: v.clone() for k,v in model.state_dict().items()}
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(a.x11, 'build_model', lambda: model)
    monkeypatch.setattr(torch, 'load', lambda *args, **kw: {'step': 482500, 'model': saved})
    monkeypatch.setattr(a.x7, 'evaluate', lambda *args: control)
    calls = []
    def eval64(m, pool):
        calls.append(pool[0].shape)
        return {'positions': control['positions']*4}
    monkeypatch.setattr(a, 'evaluate', eval64)
    a.zero_shot()
    result = json.loads((a.ROOT / 'gdn2_zeroshot/result.json').read_text())
    assert result['training_updates'] == 0 and result['weights_unchanged']
    assert result['t16_control']['positions'][4]['correct'] == 333
    assert calls == [(512,66)]
    for k,v in model.state_dict().items():
        torch.testing.assert_close(v, saved[k], rtol=0, atol=0)


def test_amended_watcher_uses_corrected_gdn2_but_original_muon(tmp_path):
    terminal(tmp_path / 'gdn2_amendment', 'gdn2_zeroshot')
    terminal(tmp_path, 'muon3x_seed0')
    # Preserved original preflight failure must not mask corrected result.
    a.x7.write_json(tmp_path / 'markers/gdn2_zeroshot.FAIL', {})
    assert w.status(tmp_path)[0] == 0
    path = tmp_path / 'gdn2_amendment/gdn2_zeroshot/result.json'
    path.write_text('{}')
    assert w.status(tmp_path)[0] == 1
