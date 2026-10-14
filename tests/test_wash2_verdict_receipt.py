"""Source-named checks for wash2_verdict_receipt.py's read-only accounting."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    'wash2_verdict_receipt',
    Path(__file__).resolve().parents[1] / 'scripts/rl-loop-v1/wash2_verdict_receipt.py')
receipt = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(receipt)


def summary():
    rows = [dict(id=str(i), reason='finished',
                 grade=dict(checks=dict(valid_first_frame=i < 12,
                                        first_action_correct=i < 6)))
            for i in range(14)]
    rows[12]['reason'] = 'generation_budget'
    rows[13]['reason'] = 'invalid_frame'
    return dict(rows=rows, valid_first_frames=12, correct_first_actions=6, successes=1)


def test_stage_tally_keeps_invalid_frames_separate_from_all_nonvalid_frames():
    tally = receipt.stage_tally(summary())
    assert (tally['valid'], tally['correct'], tally['invalid']) == (12, 6, 1)
    assert tally['nonvalid_first_frames'] == 2


def test_stage_tally_rejects_mismatched_counts_and_duplicate_cases():
    value = summary()
    value['correct_first_actions'] = 7
    with pytest.raises(ValueError, match='tally mismatch'):
        receipt.stage_tally(value)
    value = summary()
    value['rows'][1]['id'] = '0'
    with pytest.raises(ValueError, match='coverage'):
        receipt.stage_tally(value)


@pytest.mark.parametrize('count,reading', [
    (4, 'flat'), (6, 'flat'), (8, 'flat'), (9, 'improved'), (3, 'regressed')])
def test_trend_applies_two_episode_band_to_cross_checkpoint_reading(count, reading):
    trend = receipt.trend_read([dict(correct=count)] * 3)
    assert trend['interpretation'].startswith(reading)
    assert trend['identical_tallies']
    assert 'not independent stochastic draws' in trend['method']
    assert trend['mean_correct_minus_seed'] == count - 6


def test_prepare_execution_changes_only_model_slots(tmp_path, monkeypatch):
    import json
    root = tmp_path
    source = root / 'seed-execution-panel-20261012'
    source.mkdir()
    panel = dict(cases=[dict(family=f) for f in ('sum', 'edit', 'lookup', 'recovery')
                        for _ in range(24)],
                 case_module_sha256='case-sha', args_json='args.json', args_sha256='args-sha',
                 models=[dict(name='old')], automatic_retry=False,
                 frozen_extra=dict(unchanged=True))
    (source / 'panel.json').write_text(json.dumps(panel))
    (source / 'measurement-receipt.json').write_text(json.dumps(dict(panel_sha256='panel-sha')))
    out = root / 'bank-heldout-reads/wash2-u512-read-v4'
    out.mkdir(parents=True)
    monkeypatch.setattr(receipt, 'checkpoint', lambda _: dict(path='/pinned.pt', sha256='ckpt-sha'))
    monkeypatch.setattr(receipt, 'sha', lambda p: (
        'case-sha' if str(p).endswith('e97_native_execution_cases.py') else
        'args-sha' if str(p) == 'args.json' else 'panel-sha'))
    receipt.prepare_execution(root)
    actual = json.loads((out / 'panel.json').read_text())
    assert {k: v for k, v in actual.items() if k != 'models'} == {
        k: v for k, v in panel.items() if k != 'models'}
    assert actual['models'] == [dict(name=f'wash2-r{i}', checkpoint='/pinned.pt',
                                    sha256='ckpt-sha', mode='train') for i in range(4)]
    before = (out / 'panel.json').read_bytes()
    receipt.prepare_execution(root)
    assert (out / 'panel.json').read_bytes() == before


def test_varying_repetitions_are_not_labeled_deterministic():
    trend = receipt.trend_read([dict(correct=n) for n in (6, 7, 5)])
    assert not trend['identical_tallies']
    assert trend['mean_correct'] == 6
    assert 'differ despite greedy' in trend['determinism']
