"""X9 regime bundle: ordinary AdamW with explicit warmup/cosine and X7 uniform loss."""
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

from experiments.expressivity_tasks import run_invertible_stage9_anneal_8m as runner


def test_schedule_exact_endpoints_and_monotonicity():
    assert runner.learning_rate(0) == 0.
    assert runner.learning_rate(1) == pytest.approx(2e-7)
    assert runner.learning_rate(4999) == pytest.approx(.001 * 4999 / 5000)
    assert runner.learning_rate(5000) == .001
    assert runner.learning_rate(252500) == pytest.approx(.000505)
    assert runner.learning_rate(500000) == 1e-5
    rates = [runner.learning_rate(i) for i in range(500001)]
    assert all(a < b for a, b in zip(rates[:5000], rates[1:5001]))
    assert all(a >= b for a, b in zip(rates[5000:], rates[5001:]))
    for bad in (-1, 500001):
        with pytest.raises(ValueError):
            runner.learning_rate(bad)


def test_ordinary_adamw_standard_betas_decay_and_actual_update():
    p = torch.nn.Parameter(torch.tensor([1.]))
    opt = runner.make_optimizer([p])
    assert type(opt) is torch.optim.AdamW and not opt.state
    group = opt.param_groups[0]
    assert group['betas'] == (.9, .999) and group['weight_decay'] == .01
    assert group['eps'] == 1e-8 and group['lr'] == 0.
    for step in (1, 5000, 252500, 500000):
        group['lr'] = runner.learning_rate(step)
        before = p.detach().clone()
        p.grad = torch.zeros_like(p)
        opt.step()
        torch.testing.assert_close(p, before * (1 - group['lr'] * .01))
    assert 'exp_avg' in opt.state[p] and 'z' not in opt.state[p]


def test_scratch_model_is_exact_x7_bf16():
    assert runner.DEPTH == 16 and runner.CEILING == 500000 and runner.EVAL_EVERY == 500
    assert runner.NAMES == ('e97_bf16_seed0',)
    torch.manual_seed(0)
    model = runner.build_model('e97', 'bf16')
    torch.manual_seed(0)
    ref = runner.x7_model('e97', 'bf16')
    assert all(torch.equal(v, ref.state_dict()[k]) for k, v in model.state_dict().items())
    assert sum(p.numel() for p in model.parameters()) == 9900800
    assert all(layer.fused_inference and layer.recurrent_state_precision == 'legacy' for layer in model.layers)
    for arm, precision in (('gdn2', 'bf16'), ('e97', 'fp32')):
        with pytest.raises(ValueError):
            runner.build_model(arm, precision)


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
    assert config['loss_weighting'].startswith('uniform') and config['optimizer'] == 'AdamW'
    training = [json.loads(line) for line in (out / 'training.jsonl').read_text().splitlines()]
    assert [row['lr'] for row in training] == [runner.learning_rate(i) for i in (1, 2, 3)]
    assert all(np.isfinite(row['loss']) for row in training)
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
    assert cp['weights'] == 'ordinary AdamW parameters'
    terminal = torch.load(rows[-1]['checkpoint']['path'], weights_only=False)
    assert all('exp_avg' in state and 'z' not in state for state in terminal['optimizer']['state'].values())
    for p, sha in manifest[-1]['retained_hashes'].items():
        assert runner.digest(p) == sha


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires exclusive leased GPU')
def test_bf16_model_finite_uniform_backward_and_position_validation():
    pool = runner.frozen_pools(0, 16, 61)['val']
    for arm in ('e97',):
        torch.manual_seed(0)
        model = runner.build_model(arm, 'bf16').cuda().train()
        logits, y, mask = runner.forward_batch(model, tuple(a[:32] for a in pool))
        loss = torch.nn.functional.cross_entropy(logits[mask].float(), y[mask])
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(logits).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        metrics = runner.evaluate(model, pool)
        assert len(metrics['positions']) == 16
        assert sum(p['correct'] for p in metrics['positions']) == metrics['aggregate']['correct']
        print('X9_SMOKE=' + json.dumps({'arm': arm, 'uniform_training_loss': loss.item(), 'validation': metrics}))
        del model, logits, loss
        torch.cuda.empty_cache()


def test_one_run_watcher_pending_done_fail_and_dead_identity(tmp_path):
    spec = importlib.util.spec_from_file_location('x9watch', 'scripts/watch_invertible_stage9_anneal.py')
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
