"""Permanent calibration gate: every candidate must survive both successful T1 curves.

The rejected X5b rule is a negative control, not an admissible shipped candidate.
Read the original receipts, not a synthetic curve or a hand-picked loss summary.
"""
import json
from pathlib import Path

import pytest

from experiments.expressivity_tasks import run_invertible_stage5b_growth_8m as runner

RECEIPTS = (Path(__file__).resolve().parents[1] / 'experiments/expressivity_tasks'
            / 'results_invertible_stage5_instrument')
CANDIDATE_RULES = {'shipped_100k_floor': runner.Stagnation}


def replay(rule, arm, threshold):
    root = RECEIPTS / f'{arm}_T1'
    validation = json.loads((root / 'validation.json').read_text())
    training = [json.loads(line) for line in (root / 'training.jsonl').read_text().splitlines()]
    assert [r['stage_step'] for r in training] == list(range(1, len(training) + 1))
    assert all(r['depth'] == 1 for r in training + validation)
    losses = [r['loss'] for r in training]
    detector, best, first_stop, threshold_step = rule(), 0., None, None
    for row in validation:
        step = row['stage_step']
        accuracy = row['validation_by_depth']['1']['accuracy']
        best = max(best, accuracy)
        window = detector.observe(step, losses[:step], best)
        # Advancement/marker crossing has priority, including coincident plateau.
        if accuracy >= threshold:
            threshold_step = step
            break
        if window and window['stagnated'] and first_stop is None:
            first_stop = step
    assert threshold_step is not None, 'known-successful receipt is incomplete'
    return {'arm': arm, 'threshold': threshold, 'threshold_step': threshold_step,
            'first_stop': first_stop}


@pytest.mark.parametrize('candidate', CANDIDATE_RULES)
@pytest.mark.parametrize('arm', ['e97', 'gdn2'])
@pytest.mark.parametrize('threshold', [.9, .97])
def test_replay_stagnation_rule(candidate, arm, threshold):
    evidence = replay(CANDIDATE_RULES[candidate], arm, threshold)
    assert evidence['first_stop'] is None, evidence
    assert evidence['threshold_step'] == {
        ('e97', .9): 48500, ('e97', .97): 57000,
        ('gdn2', .9): 13000, ('gdn2', .97): 13750,
    }[arm, threshold]
    assert evidence['threshold_step'] < runner.STAGNATION_MIN_UPDATES


@pytest.mark.parametrize('arm', ['e97', 'gdn2'])
@pytest.mark.parametrize('threshold', [.9, .97])
def test_replay_x5b_no_floor_negative_control(monkeypatch, arm, threshold):
    monkeypatch.setattr(runner, 'STAGNATION_MIN_UPDATES', 0)
    evidence = replay(runner.Stagnation, arm, threshold)
    if arm == 'e97':
        assert evidence['first_stop'] == 10000
        assert evidence['first_stop'] < evidence['threshold_step'], evidence
    else:
        assert evidence['first_stop'] is None
