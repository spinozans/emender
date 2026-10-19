"""X8's only scientific intervention: normalized driver-position training weights."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks import run_invertible_stage8_paydepth_8m as runner


def test_position_weights_and_loss_gradient_match_registered_objective():
    assert runner.POSITION_WEIGHTS == tuple(t / 136 for t in range(1, 17))
    assert sum(runner.POSITION_WEIGHTS) == 1.
    assert runner.POSITION_WEIGHTS[0] < .01
    torch.manual_seed(44)
    logits = torch.randn(2, 18, 5, requires_grad=True)
    y = torch.randint(5, (2, 18))
    mask = torch.zeros(2, 18, dtype=torch.bool)
    mask[:, 1:17] = True
    actual = runner.position_weighted_loss(logits, y, mask)
    ce = torch.nn.functional.cross_entropy(logits[:, 1:17].reshape(-1, 5), y[:, 1:17].reshape(-1), reduction='none').reshape(2, 16)
    expected = sum(ce[:, t - 1].mean() * t / 136 for t in range(1, 17))
    torch.testing.assert_close(actual, expected)
    grad, = torch.autograd.grad(actual, logits, retain_graph=True)
    ref, = torch.autograd.grad(expected, logits)
    torch.testing.assert_close(grad, ref)
    assert grad[:, 0].count_nonzero() == grad[:, 17].count_nonzero() == 0
    # Identical losses at each position preserve X7's aggregate loss scale.
    flat = torch.zeros_like(logits)
    torch.testing.assert_close(runner.position_weighted_loss(flat, y, mask), torch.tensor(5.).log())


def test_reject_incomplete_or_misbalanced_driver_masks():
    mask = torch.zeros(2, 18, dtype=torch.bool)
    mask[0, :15], mask[1, :17] = True, True
    with pytest.raises(ValueError, match='exactly 16'):
        runner.position_weighted_loss(torch.zeros(2, 18, 2), torch.zeros(2, 18, dtype=torch.long), mask)


def test_scratch_models_match_x7_e97_and_stock_gdn():
    assert runner.DEPTH == 16 and runner.CEILING == 500000 and runner.EVAL_EVERY == 500
    assert runner.NAMES == ('e97_bf16_seed0', 'gdn2_bf16_seed0')
    for arm in ('e97', 'gdn2'):
        torch.manual_seed(0)
        model = runner.build_model(arm, 'bf16')
        torch.manual_seed(0)
        ref = runner.x7_model(arm, 'bf16') if arm == 'e97' else runner.base_model(arm, 61)
        assert all(torch.equal(v, ref.state_dict()[k]) for k, v in model.state_dict().items())
        assert sum(p.numel() for p in model.parameters()) == {'e97': 9900800, 'gdn2': 9887056}[arm]
        if arm == 'gdn2':
            from fla.layers import GatedDeltaNet
            assert all(layer.gdn.forward.__func__ is GatedDeltaNet.forward and layer.gdn.allow_neg_eigval for layer in model.layers)
        else:
            assert all(layer.fused_inference and layer.recurrent_state_precision == 'legacy' for layer in model.layers)
    with pytest.raises(ValueError):
        runner.build_model('gdn2', 'fp32')


def test_safety_only_no_ladder_and_retention_survives_best_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'CEILING', 3)
    monkeypatch.setattr(runner, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    lengths, retained = [], []
    def forward(model, batch):
        lengths.append(batch[0].shape[1])
        y, mask = torch.from_numpy(batch[1]) * 0, torch.from_numpy(batch[2])
        return model.embed.weight[0, :2].expand(*y.shape, 2), y, mask
    monkeypatch.setattr(runner, 'forward_batch', forward)
    monkeypatch.setattr(runner, 'evaluate', lambda *_: {'aggregate': {'accuracy': 1., 'loss': 4. - len(lengths)},
        'positions': [{'driver_position': j, 'correct': 512} for j in range(1, 17)], 'perfect_prefix_depth': 16})
    # Exercise the real X6 prune and its manifest under the low-space policy.
    monkeypatch.setattr('shutil.disk_usage', lambda _: SimpleNamespace(free=100))
    real_prune = runner.prune
    def prune(out, checkpoints, required):
        retained.append(set(required))
        real_prune(out, checkpoints, required)
    monkeypatch.setattr(runner, 'prune', prune)
    out = tmp_path / 'run'
    runner.run(argparse.Namespace(arm='e97', precision='bf16', seed=0, out=str(out)))
    assert lengths == [18] * 3  # Despite perfect accuracy, only the safety ceiling stops.
    config = json.loads((out / 'config.json').read_text())
    assert not config['ladder'] and not config['carry_supervision'] and not config['stagnation_kill']
    assert config['position_weights'] == list(runner.POSITION_WEIGHTS)
    rows = json.loads((out / 'validation.json').read_text())
    assert [r['step'] for r in rows] == [0, 1, 2, 3]
    entry = rows[0]['checkpoint']['path']
    assert all(entry in required for required in retained)
    assert Path(entry).exists() and Path(rows[-1]['checkpoint']['path']).exists()
    assert not Path(rows[1]['checkpoint']['path']).exists()
    manifest = [json.loads(line) for line in (out / 'pruning.jsonl').read_text().splitlines()]
    assert entry in manifest[-1]['retained_hashes']
    assert json.loads((out / 'result.json').read_text())['stop_reason'] == 'safety_ceiling'
    cp = torch.load(entry, weights_only=False)
    assert cp['step'] == 0 and not cp['optimizer']['state']
    for p, sha in manifest[-1]['retained_hashes'].items():
        assert runner.digest(p) == sha


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires exclusive leased GPU')
def test_bf16_models_finite_weighted_backward_and_position_validation():
    pool = runner.frozen_pools(0, 16, 61)['val']
    for arm in ('e97', 'gdn2'):
        torch.manual_seed(0)
        model = runner.build_model(arm, 'bf16').cuda().train()
        logits, y, mask = runner.forward_batch(model, tuple(a[:32] for a in pool))
        loss = runner.position_weighted_loss(logits, y, mask)
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        metrics = runner.evaluate(model, pool)
        assert len(metrics['positions']) == 16
        assert sum(p['correct'] for p in metrics['positions']) == metrics['aggregate']['correct']
        print('X8_SMOKE=' + json.dumps({'arm': arm, 'weighted_training_loss': loss.item(), 'validation': metrics}))
        del model, logits, loss
        torch.cuda.empty_cache()


def test_two_run_watcher_pending_done_fail_and_dead_identity(tmp_path):
    spec = importlib.util.spec_from_file_location('x8watch', 'scripts/watch_invertible_stage8_paydepth.py')
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    assert tuple(watcher.NAMES) == runner.NAMES
    (tmp_path / 'markers').mkdir()
    stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    for name in watcher.NAMES:
        receipt = tmp_path / 'receipts' / name
        receipt.mkdir(parents=True)
        runner.write_json(receipt / 'worker.json', {'pid': os.getpid(), 'starttime': stat[19], 'host': socket.gethostname()})
    assert watcher.status(tmp_path)[0] is None
    runner.write_json(tmp_path / 'receipts' / watcher.NAMES[0] / 'trainer.json', {'pid': os.getpid(), 'starttime': '0', 'host': socket.gethostname()})
    assert watcher.status(tmp_path)[0] == 1
    for name in watcher.NAMES:
        (tmp_path / name).mkdir()
        runner.write_json(tmp_path / name / 'result.json', {})
        runner.write_json(tmp_path / 'markers' / f'{name}.DONE', {'name': name, 'exit_code': 0})
    assert watcher.status(tmp_path)[0] == 0
    (tmp_path / 'markers' / f'{watcher.NAMES[0]}.FAIL').touch()
    assert watcher.status(tmp_path)[0] == 1
