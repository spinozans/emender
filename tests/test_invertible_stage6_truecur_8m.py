"""X6 source-named protocol, leakage, carry/restore and detached watcher gates."""
import argparse
import importlib.util
import json
from pathlib import Path
from collections import namedtuple

import numpy as np
import pytest

from experiments.expressivity_tasks import run_invertible_stage6_truecur_8m as runner


def test_registered_matrix_and_no_stagnation():
    assert len(runner.matrix()) == len({r['name'] for r in runner.matrix()}) == 12
    assert runner.DEPTHS == (1, 2, 4, 8, 16, 32, 64, 128)
    assert runner.STAGE_CEILING == 500000 and runner.GLOBAL_CAP == 4000000
    assert runner.THRESHOLD == .9 and runner.EVAL_EVERY == 500
    assert not hasattr(runner, 'Stagnation')
    for step in (0, 6000, 100000, 104000, 499999):
        assert runner.stage_decision(.001, step, step) == 'continue'
    assert runner.stage_decision(.89, 500000, 500000) == 'ceiling_non_threshold'
    assert runner.stage_decision(.89, 500000, 4000000) == 'global_safety_cap'
    assert runner.stage_decision(.90, 500000, 4000000) == 'threshold'


@pytest.mark.parametrize('accuracy,step,total', [(float('nan'), 0, 0), (-1., 0, 0), (2., 0, 0), (.1, 500001, 500001), (.1, 0, 4000001)])
def test_invalid_decisions(accuracy, step, total):
    with pytest.raises(ValueError):
        runner.stage_decision(accuracy, step, total)


def test_carry_is_previous_not_current_state_and_zero_probability_ignores_targets():
    task = runner.InvertibleQuadraticTask(61)
    x, y, mask = task.generate_batch(8, 6, np.random.default_rng(0))
    carried, positions = runner.carry_inputs(x, y, 1., np.random.default_rng(1))
    neutral, neutral_positions = runner.carry_inputs(x, y, 0.)
    corrupted, _ = runner.carry_inputs(x, y + 100000, 0.)
    assert carried.shape == neutral.shape == (8, 9)
    assert np.array_equal(carried[:, positions], x)
    assert np.array_equal(carried[:, 3::2], y[:, 2:-1])
    assert np.all(neutral[:, 3::2] == runner.NEUTRAL)
    assert np.array_equal(neutral, corrupted)
    assert np.array_equal(positions, neutral_positions)
    assert mask.sum() == 32
    one = task.generate_batch(8, 3, np.random.default_rng(0))
    assert np.array_equal(runner.carry_inputs(*one[:2], 1., np.random.default_rng(0))[0], one[0])
    assert runner.carry_probability(0) == .5
    assert runner.carry_probability(250000) == .25
    assert runner.carry_probability(500000) == 0.


@pytest.mark.parametrize('arm,base', [('e97', 9900800), ('gdn2', 9887056)])
def test_input_only_neutral_does_not_expand_output_classes(arm, base):
    import torch
    torch.manual_seed(0)
    a = runner.build_model(arm, 'A')
    torch.manual_seed(0)
    b = runner.build_model(arm, 'B')
    assert sum(p.numel() for p in a.parameters()) == base
    assert sum(p.numel() for p in b.parameters()) == base + 256
    assert b.out_proj.weight.shape == (3721, 256)
    assert b.embed.weight is b.out_proj.weight
    assert b.embed(torch.tensor([[3721]])).shape == (1, 1, 256)
    for key, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[key])


def test_eval_never_supplies_true_feedback_and_counts_original_driver_indices(monkeypatch):
    import torch
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    observed = []
    def forward(model, x):
        observed.append(x.numpy().copy())
        return torch.zeros(*x.shape, 3721)
    monkeypatch.setattr(runner, 'padded_forward', forward)
    pool = runner.InvertibleQuadraticTask(61).generate_batch(2, 14, np.random.default_rng(0))
    result = runner.evaluate(torch.nn.Linear(1, 1), pool, 'B')
    assert np.all(observed[0][:, 3::2] == runner.NEUTRAL)
    assert result['count'] == 24 and result['post_suffix7_count'] == 8
    assert result['terminal_count'] == 2


def test_restore_best_restores_weights_and_optimizer_but_not_rng(tmp_path):
    import torch
    import schedulefree
    model = torch.nn.Linear(1, 1)
    opt = schedulefree.AdamWScheduleFree(model.parameters(), lr=.001)
    rng = np.random.default_rng(0)
    opt.train()
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    opt.eval()
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer_expected = {p: {k: v.clone() for k, v in state.items()} for p, state in opt.state.items()}
    cp = runner.checkpoint(tmp_path, model, opt, rng, 1, 1, 1, {})
    opt.train()
    opt.zero_grad()
    model(torch.ones(1, 1)).sum().backward()
    opt.step()
    opt.eval()
    rng.random(10)
    state = rng.bit_generator.state
    runner.restore_best({'checkpoint': cp}, model, opt)
    assert all(torch.equal(expected[k], v) for k, v in model.state_dict().items())
    assert opt.param_groups[0]['k'] == 1 and not opt.param_groups[0]['train_mode']
    assert all(torch.equal(value, opt.state[p][key]) for p, state in optimizer_expected.items() for key, value in state.items())
    assert rng.bit_generator.state == state


@pytest.mark.parametrize('mode', ['ceiling', 'threshold', 'global', 'retention'])
def test_run_carries_optimizer_and_continues_after_ceiling(tmp_path, monkeypatch, mode):
    import torch
    monkeypatch.setattr(runner, 'DEPTHS', (1, 2, 4))
    monkeypatch.setattr(runner, 'STAGE_CEILING', 2)
    monkeypatch.setattr(runner, 'GLOBAL_CAP', 3 if mode == 'global' else 6)
    monkeypatch.setattr(runner, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(runner, 'archive_pools', lambda out, seed: {d: d for d in runner.DEPTHS})
    lengths, evaluations, ids, restores = [], [], [], []
    def forward(model, batch, curriculum, probability=0., rng=None):
        lengths.append(batch[0].shape[1])
        y = torch.zeros_like(torch.from_numpy(batch[1]))
        mask = torch.from_numpy(batch[2])
        return model.embed.weight[0, :2].expand(*y.shape, 2), y, mask, batch[0].size
    monkeypatch.setattr(runner, 'forward_batch', forward)
    def evaluate(model, depth, curriculum):
        evaluations.append(depth)
        return {'accuracy': .9 if mode == 'threshold' else .1,
                'loss': 8. - len(evaluations) / 100 if mode == 'retention' else 8.}
    monkeypatch.setattr(runner, 'evaluate', evaluate)
    real_checkpoint = runner.checkpoint
    def checkpoint(out, model, opt, *args):
        ids.append(id(opt))
        return real_checkpoint(out, model, opt, *args)
    monkeypatch.setattr(runner, 'checkpoint', checkpoint)
    real_restore = runner.restore_best
    def restore(best, model, opt):
        restores.append(best['checkpoint']['step'])
        return real_restore(best, model, opt)
    monkeypatch.setattr(runner, 'restore_best', restore)
    if mode == 'retention':
        import shutil
        monkeypatch.setattr(shutil, 'disk_usage', lambda _: namedtuple('Usage', 'total used free')(1, 1, 1))
    else:
        monkeypatch.setattr(runner, 'prune', lambda *args: None)
    out = tmp_path / mode
    runner.run(argparse.Namespace(arm='e97', curriculum='A', seed=0, out=str(out)))
    result = json.loads((out / 'result.json').read_text())
    assert len(set(ids)) == 1
    assert 'frontier' not in result
    if mode == 'retention':
        events = [json.loads(line) for line in (out / 'transitions.jsonl').read_text().splitlines()]
        for event in events:
            if event['event'] in ('stage_entry', 'stage_exit'):
                assert Path(event['checkpoint']['path']).exists()
        for stage in result['stages']:
            assert Path(stage['best']['checkpoint']['path']).exists()
        assert result['stages'][0]['best']['stage_step'] == 2  # entry ceased being best
        assert (out / 'pruning.jsonl').exists()  # nonboundary superseded best was removed
    elif mode == 'ceiling':
        assert lengths == [3, 3, 4, 4, 6, 6]
        assert result['total_steps'] == 6
        assert restores == [0, 2]  # All validation ties: stage-entry best wins.
        assert [s['decision'] for s in result['stages']] == ['ceiling_non_threshold'] * 2 + ['global_safety_cap']
        assert evaluations == [1] * 3 + [1, 2] * 3 + [1, 2, 4] * 3
    elif mode == 'threshold':
        assert not lengths and not restores
        assert [s['steps_to_threshold'] for s in result['stages']] == [0, 0, 0]
        assert result['stop_reason'] == 'completed_stage_ladder'
    else:
        assert result['total_steps'] == 3 and len(result['stages']) == 2
        assert result['stop_reason'] == 'global_safety_cap'


def test_bounded_retention_prunes_10k_but_protects_required(tmp_path, monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, 'disk_usage', lambda _: namedtuple('Usage', 'total used free')(1, 1, 1))
    checkpoints = {}
    for name in ('entry', 'exit', 'best', 'latest', 'old_10k'):
        path = tmp_path / name
        path.write_text(name)
        checkpoints[str(path)] = {'sha256': runner.digest(path), 'step': 10000}
    required = {p for p in checkpoints if not p.endswith('old_10k')}
    runner.prune(tmp_path, checkpoints, required)
    assert not (tmp_path / 'old_10k').exists()
    assert all(Path(p).exists() for p in required)
    receipt = json.loads((tmp_path / 'pruning.jsonl').read_text())
    assert set(receipt['retained_hashes']) == required


def test_twelve_run_watcher_pending_done_fail(tmp_path):
    import os
    import socket
    spec = importlib.util.spec_from_file_location('x6_watcher', 'scripts/watch_invertible_stage6_truecur.py')
    watcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(watcher)
    assert set(watcher.NAMES) == {r['name'] for r in runner.matrix()}
    (tmp_path / 'markers').mkdir()
    stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
    for name in watcher.NAMES:
        receipt = tmp_path / 'receipts' / name
        receipt.mkdir(parents=True)
        (receipt / 'worker.json').write_text(json.dumps({'pid': os.getpid(), 'starttime': stat[19], 'host': socket.gethostname()}))
    assert watcher.status(tmp_path)[0] is None  # No trainer identity is valid for queued workers.
    for name in watcher.NAMES:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'result.json').write_text('{}')
        (tmp_path / 'markers' / f'{name}.DONE').write_text(json.dumps({'name': name, 'exit_code': 0}))
    assert watcher.status(tmp_path)[0] == 0
    (tmp_path / 'markers' / 'B_e97_seed2.FAIL').touch()
    assert watcher.status(tmp_path)[0] == 1
