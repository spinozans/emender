"""X5b growth protocol, double-window calibration and completion watcher."""
import argparse
import json
from collections import namedtuple
from pathlib import Path

import numpy as np
import pytest

from experiments.expressivity_tasks import run_invertible_stage5b_growth_8m as runner


def test_six_matched_from_scratch_arms_and_registered_constants():
    rows = runner.matrix()
    assert len(rows) == len({r['name'] for r in rows}) == 6
    assert [r['seed'] for r in rows if r['arm'] == 'e97'] == [0, 1, 2]
    assert [r['seed'] for r in rows if r['arm'] == 'gdn2'] == [0, 1, 2]
    assert runner.DEPTHS == (4, 8, 16, 32, 64, 128)
    assert runner.STAGE_CEILING == 500000 and runner.GLOBAL_CAP == 2000000
    assert runner.WINDOW == 2000 and runner.EVAL_EVERY == 500
    assert runner.VAL_GAIN == .001 and runner.LOSS_GAIN == .001
    assert runner.STAGNATION_MIN_UPDATES == 100000
    assert runner.THRESHOLD == .9
    assert runner.MARGINAL == pytest.approx(8.221747728)
    assert 100 * runner.FLOOR == pytest.approx(.0268744961)


def test_runner_carries_optimizer_and_stops_on_first_failed_depth(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(runner, 'STAGE_CEILING', 2)
    monkeypatch.setattr(runner, 'EVAL_EVERY', 1)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(runner, 'archive_pools', lambda out, seed: {d: d for d in runner.DEPTHS})
    monkeypatch.setattr(runner, 'padded_forward', lambda m, x: m.embed.weight[0, :2].expand(*x.shape, 2))
    generated_lengths = []
    class Task:
        def __init__(self, p):
            pass
        def generate_batch(self, batch, length, rng):
            generated_lengths.append((batch, length))
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
    assert result['total_steps'] == 3 and result['stop_reason'] == 'stage_safety_ceiling'
    assert generated_lengths == [(32, 6), (32, 10), (32, 10)]
    assert result['input_tokens'] == 32 * (6 + 10 + 10)
    assert [s['steps_to_threshold'] for s in result['stages']] == [1, None]
    assert seen == [4, 4, 4, 8, 4, 8, 4, 8]
    assert len({s[0] for s in states}) == 1
    assert states[0][2] == 0 and all(s[2] > 0 for s in states[1:])
    transitions = [json.loads(line) for line in (out / 'transitions.jsonl').read_text().splitlines()]
    assert [t['event'] for t in transitions] == ['stage_entry', 'stage_exit', 'stage_entry', 'stage_exit']
    assert transitions[2]['previous_stage_exit'] == transitions[1]['checkpoint']
    assert all(not t['optimizer_reset'] for t in transitions if t['event'] == 'stage_entry')

    curve = json.loads((out / 'growth_curve.json').read_text())
    assert curve['deepest_threshold_crossed'] == 4 and curve['total_steps'] == 3
    assert curve['stages'][-1]['safety_stop'] is True


@pytest.mark.parametrize('step', [0, 500, 10000, 500000])
def test_threshold_inclusive_and_has_priority(step):
    assert runner.stage_decision(.9, step, 2000000, True) == 'threshold'
    assert runner.stage_decision(.899, step, step) == ('stage_safety_ceiling' if step == 500000 else 'continue')


def test_safety_caps_are_not_stagnation_and_no_10k_budget():
    assert runner.stage_decision(.5, 10000, 10000) == 'continue'
    assert runner.stage_decision(.5, 500000, 500000) == 'stage_safety_ceiling'
    assert runner.stage_decision(.5, 100, 2000000) == 'global_safety_cap'
    for step in (6000, 98000, 100000, 102000):
        assert runner.stage_decision(.5, step, 1500000, True) == 'continue'
    assert runner.stage_decision(.5, 104000, 1500000, True) == 'stagnation'


@pytest.mark.parametrize('accuracy,step,total', [(float('nan'), 0, 0), (-.1, 0, 0), (1.1, 0, 0), (.5, 500001, 500001), (.5, -1, 0), (.5, 1, 2000001)])
def test_invalid_observations_fail_closed(accuracy, step, total):
    with pytest.raises(ValueError):
        runner.stage_decision(accuracy, step, total)


def test_two_consecutive_disjoint_windows_without_floor(monkeypatch):
    monkeypatch.setattr(runner, 'STAGNATION_MIN_UPDATES', 0)
    detector = runner.Stagnation()
    assert detector.observe(0, [], .1) is None
    assert detector.observe(500, [8.] * 500, .1) is None
    first = detector.observe(2000, [8.] * 2000, .1)
    second = detector.observe(4000, [8.] * 4000, .1)
    third = detector.observe(6000, [8.] * 6000, .1)
    assert not first['joint_plateau'] and not second['stagnated']
    assert third['stagnated'] and third['consecutive_joint_plateaus'] == 2
    assert third['window_start'] == 4001 and third['window_end'] == 6000
    assert third['mean_train_loss'] == 8.


@pytest.mark.parametrize('val_gain,loss_ratio', [(.002, 1.), (0., .998), (.0011, .998)])
def test_either_learning_signal_resets_plateau_streak(val_gain, loss_ratio, monkeypatch):
    monkeypatch.setattr(runner, 'STAGNATION_MIN_UPDATES', 0)
    detector = runner.Stagnation()
    detector.observe(2000, [8.] * 2000, .1)
    assert detector.observe(4000, [8.] * 4000, .1)['consecutive_joint_plateaus'] == 1
    row = detector.observe(6000, [8.] * 4000 + [8. * loss_ratio] * 2000, .1 + val_gain)
    assert row['consecutive_joint_plateaus'] == 0 and not row['stagnated']
    assert detector.observe(8000, [8. * loss_ratio] * 8000, .1 + val_gain)['consecutive_joint_plateaus'] == 1


def test_worsening_loss_counts_plateau_and_means_use_only_current_window(monkeypatch):
    monkeypatch.setattr(runner, 'STAGNATION_MIN_UPDATES', 0)
    detector = runner.Stagnation()
    detector.observe(2000, [8.] * 2000, .1)
    row = detector.observe(4000, [8.] * 2000 + [9.] * 2000, .1)
    assert row['mean_train_loss'] == 9. and row['relative_train_loss_decrease'] == -.125
    assert detector.observe(6000, [8.] * 2000 + [9.] * 4000, .1)['stagnated']


def test_stage_detector_reset(monkeypatch):
    monkeypatch.setattr(runner, 'STAGNATION_MIN_UPDATES', 0)
    old = runner.Stagnation()
    for step in (2000, 4000, 6000):
        row = old.observe(step, [8.] * step, .1)
    assert row['stagnated']
    assert not runner.Stagnation().observe(2000, [8.] * 2000, .1)['stagnated']


def watcher_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location('growth_watcher', 'scripts/watch_invertible_stage5b_growth.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def watcher_tree(root, module):
    import os, socket
    for name in module.NAMES:
        receipt = root / 'receipts' / name
        receipt.mkdir(parents=True)
        stat = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(') ', 1)[1].split()
        (receipt / 'worker.json').write_text(json.dumps({'pid': os.getpid(), 'starttime': stat[19], 'host': socket.gethostname()}))
    (root / 'markers').mkdir()


def test_watcher_running_done_fail_and_missing_receipt(tmp_path):
    watcher = watcher_module()
    watcher_tree(tmp_path, watcher)
    assert watcher.status(tmp_path)[0] is None
    for name in watcher.NAMES:
        (tmp_path / name).mkdir()
        (tmp_path / name / 'result.json').write_text('{}')
        (tmp_path / 'markers' / f'{name}.DONE').write_text(json.dumps({'name': name, 'exit_code': 0}))
    assert watcher.status(tmp_path)[0] == 0
    (tmp_path / 'markers' / 'gdn2_seed2.FAIL').touch()
    assert watcher.status(tmp_path)[0] == 1


def test_watcher_dead_process_without_marker_and_pid_reuse(tmp_path, monkeypatch):
    watcher = watcher_module()
    watcher_tree(tmp_path, watcher)
    monkeypatch.setattr(watcher.time, 'sleep', lambda _: None)
    record = tmp_path / 'receipts' / 'e97_seed0' / 'worker.json'
    row = json.loads(record.read_text())
    row['starttime'] = '0'
    record.write_text(json.dumps(row))
    code, message = watcher.status(tmp_path)
    assert code == 1 and 'without a terminal marker' in message

@pytest.mark.parametrize('mode,expected,steps', [('stagnation', 'stagnation', 6), ('global', 'global_safety_cap', 3), ('complete', 'completed_curriculum', 0)])
def test_run_terminal_paths_emit_growth_receipts(tmp_path, monkeypatch, mode, expected, steps):
    import torch
    import torch.nn.functional as functional
    monkeypatch.setattr(runner, 'STAGNATION_MIN_UPDATES', 0)
    monkeypatch.setattr(runner, 'WINDOW', 2)
    monkeypatch.setattr(runner, 'EVAL_EVERY', 1)
    monkeypatch.setattr(runner, 'STAGE_CEILING', 10)
    monkeypatch.setattr(runner, 'GLOBAL_CAP', 3 if mode == 'global' else 100)
    monkeypatch.setattr(torch.nn.Module, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.Tensor, 'cuda', lambda self: self)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(runner, 'archive_pools', lambda out, seed: {d: d for d in runner.DEPTHS})
    monkeypatch.setattr(runner, 'padded_forward', lambda m, x: m.embed.weight[0, :2].expand(*x.shape, 2))
    monkeypatch.setattr(functional, 'cross_entropy', lambda logits, targets: logits.sum() * 0 + 8.)
    monkeypatch.setattr(runner, 'evaluate', lambda m, p, _: {'accuracy': .9 if mode == 'complete' else .1, 'loss': 8.})
    def checkpoint(out, model, opt, rng, step, depth, stage_step, config):
        return {'path': str(out / f'{depth}_{stage_step}.pt'), 'sha256': 'fake', 'step': step}
    monkeypatch.setattr(runner, 'checkpoint', checkpoint)
    monkeypatch.setattr(runner, 'prune', lambda *a: None)
    out = tmp_path / mode
    runner.run(argparse.Namespace(arm='e97', seed=0, out=str(out)))
    result = json.loads((out / 'result.json').read_text())
    assert result['stop_reason'] == expected and result['total_steps'] == steps
    curve = json.loads((out / 'growth_curve.json').read_text())
    assert len(curve['stages']) == (6 if mode == 'complete' else 1)
    if mode == 'stagnation':
        assert curve['stages'][0]['stagnation_window']['stagnated']
        assert not curve['stages'][0]['safety_stop']
    if mode == 'complete':
        assert result['frontier'] == 128
        assert sorted(result['stages'][-1]['retention'], key=int) == ['4', '8', '16', '32', '64', '128']


def test_100k_floor_discards_silent_phase_streak_and_resets_at_each_stage():
    losses = [8.] * 104000
    for _ in range(2):  # A newly entered stage has its own floor, not global step.
        detector = runner.Stagnation()
        for step in range(2000, 104001, 2000):
            row = detector.observe(step, losses[:step], .1)
            assert row['stagnated'] == (step == 104000)
            if step <= 100000:
                assert row['consecutive_joint_plateaus'] == 0
                assert not row['comparison_eligible']
            if step == 102000:
                assert row['consecutive_joint_plateaus'] == 1
        assert runner.stage_decision(.9, 104000, 104000, True) == 'threshold'
