"""X5 curriculum protocol, validation identity and optimizer-continuity checks."""
import argparse
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from experiments.expressivity_tasks import run_invertible_stage5_curriculum_8m as runner


def test_six_matched_from_scratch_arms_and_registered_constants():
    rows = runner.matrix()
    assert len(rows) == len({r['name'] for r in rows}) == 6
    assert [r['seed'] for r in rows if r['arm'] == 'e97'] == [0, 1, 2]
    assert [r['seed'] for r in rows if r['arm'] == 'gdn2'] == [0, 1, 2]
    assert runner.DEPTHS == (4, 8, 16, 32, 64, 128)
    assert runner.STAGE_BUDGET == 10000 and runner.EVAL_EVERY == 500
    assert runner.THRESHOLD == .9
    assert runner.MARGINAL == pytest.approx(8.221747728)
    assert 100 * runner.FLOOR == pytest.approx(.0268744961)


@pytest.mark.parametrize('step', [0, 500, 9500, 10000])
def test_threshold_inclusive_and_precedes_exhaustion(step):
    assert runner.stage_decision(.9, step) == 'threshold'
    assert runner.stage_decision(.9001, step) == 'threshold'
    assert runner.stage_decision(.899999, step) == ('budget_exhausted' if step == 10000 else 'continue')


@pytest.mark.parametrize('accuracy,step', [(float('nan'), 0), (-.1, 0), (1.1, 0), (.5, 10001), (.5, -1)])
def test_invalid_observations_fail_closed(accuracy, step):
    with pytest.raises(ValueError):
        runner.stage_decision(accuracy, step)


def test_frozen_validation_exact_x_series_identity_and_full_state_targets():
    a, b = runner.validation_pools(0), runner.validation_pools(0)
    for depth in runner.DEPTHS:
        assert a[depth][0].shape == (512, depth + 2)
        assert a[depth][2].sum() == 512 * depth
        for x, y in zip(a[depth], b[depth]):
            np.testing.assert_array_equal(x, y)
    prior = runner.frozen_pools(0, 128, 61)['val']
    for x, y in zip(a[128], prior):
        np.testing.assert_array_equal(x, y)


def test_capacity_prunes_only_nonessential_checkpoints(tmp_path, monkeypatch):
    cps = {}
    for step in [0, 500, 750, 1000, 1500, 10000]:
        path = tmp_path / f'{step}.pt'
        path.write_bytes(str(step).encode())
        cps[str(path)] = {'step': step, 'sha256': runner.digest(path),
                          'stage_step': 0 if step == 750 else step}
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr(runner.shutil, 'disk_usage', lambda _: usage(1000, 900, 100))
    runner.prune(tmp_path, cps, {str(tmp_path / '500.pt'), str(tmp_path / '1500.pt')})
    assert not (tmp_path / '1000.pt').exists()
    assert all((tmp_path / f'{s}.pt').exists() for s in [0, 500, 750, 1500, 10000])
    receipt = json.loads((tmp_path / 'pruning.jsonl').read_text())
    assert receipt['pruned'] == [str(tmp_path / '1000.pt')]


def test_runner_carries_optimizer_and_stops_on_first_failed_depth(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(runner, 'STAGE_BUDGET', 2)
    monkeypatch.setattr(runner, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(runner, 'archive_pools', lambda out, seed: {d: d for d in runner.DEPTHS})
    monkeypatch.setattr(runner, 'padded_forward', lambda m, x: m.embed.weight[0, :2].expand(*x.shape, 2))
    class Task:
        def __init__(self, p):
            pass
        def generate_batch(self, batch, length, rng):
            return (np.zeros((batch, length), dtype=np.int64),
                    np.zeros((batch, length), dtype=np.int64), np.ones((batch, length), dtype=bool))
    monkeypatch.setattr(runner, 'InvertibleQuadraticTask', Task)
    accuracies = iter([.1, .9, .8, .1, .8, .89, .8, .899])
    seen = []
    def evaluate(model, depth, p):
        assert not model.training or not seen  # evaluate() below also sets eval like the real helper.
        model.eval()
        seen.append(depth)
        return {'accuracy': next(accuracies), 'loss': 8.}
    # Real evaluate sets eval before forwarding; mimic that without model forward.
    def eval_model(model, depth, p):
        model.eval()
        return evaluate(model, depth, p)
    monkeypatch.setattr(runner, 'evaluate', eval_model)
    states = []
    def checkpoint(out, model, opt, rng, step, depth, stage_step, config):
        states.append((id(opt), step, len(opt.state)))
        assert not opt.param_groups[0]['train_mode']
        return {'path': str(out / f'{depth}_{stage_step}.pt'), 'sha256': 'fake', 'step': step}
    monkeypatch.setattr(runner, 'checkpoint', checkpoint)
    monkeypatch.setattr(runner, 'prune', lambda *a: None)
    out = tmp_path / 'run'
    runner.run(argparse.Namespace(arm='e97', seed=0, out=str(out)))
    result = json.loads((out / 'result.json').read_text())
    assert result['frontier'] == 4 and result['stopping_depth'] == 8
    assert result['total_steps'] == 3 and result['stop_reason'] == 'budget_exhausted'
    assert [s['steps_to_threshold'] for s in result['stages']] == [1, None]
    assert seen == [4, 4, 4, 8, 4, 8, 4, 8]
    assert len({s[0] for s in states}) == 1
    assert states[0][2] == 0 and all(s[2] > 0 for s in states[1:])
    transitions = [json.loads(line) for line in (out / 'transitions.jsonl').read_text().splitlines()]
    assert [t['event'] for t in transitions] == ['stage_entry', 'stage_exit', 'stage_entry', 'stage_exit']
    assert transitions[2]['previous_stage_exit'] == transitions[1]['checkpoint']
    assert all(not t['optimizer_reset'] for t in transitions if t['event'] == 'stage_entry')
