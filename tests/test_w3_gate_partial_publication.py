"""Receipt-only checks for W3G's fail-stopped publication; never run models."""
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / 'receipts/w3-matcheddose/gate-read'


def load(relative):
    return json.loads((ROOT / relative).read_text())


def test_partial_verdict_does_not_classify_incomplete_b_or_invent_docnll():
    receipt = load('partial-verdict.json')
    assert receipt['status'] == 'INVALID/INCONCLUSIVE'
    assert receipt['rescue_question'] == 'OPEN'
    assert not any(receipt[key] for key in ('retry', 'replay', 'resume', 'checkpoint_promotion',
                                           'frozen_verdict_receipt_exists'))
    assert receipt['docnll']['status'] == 'not run for any checkpoint'
    assert receipt['docnll']['trajectory_only'] and receipt['docnll']['numeric_bound'] is None
    b = receipt['checkpoints']['B']
    assert 'raw_scalar_gate' not in b and 'stage_b' not in b and 'execution' not in b
    partial = b['stage_b_partial']
    assert not partial['gate_evaluable']
    assert (partial['completed_cases'], partial['scheduled_cases']) == (10, 14)
    assert partial['missing_repetitions'] == [2, 3]
    rows = partial['per_episode']
    assert len(rows) == len({row['id'] for row in rows}) == 10
    assert partial['descriptive_prefix_counts'] == dict(
        valid=sum(row['grade']['checks']['valid_first_frame'] for row in rows),
        correct=sum(row['grade']['checks']['first_action_correct'] for row in rows),
        invalid=sum(row['reason'].startswith('invalid_') for row in rows),
        successes=sum(row['grade']['success'] for row in rows))


@pytest.mark.parametrize('name', ['parent', 'A'])
def test_complete_stage_b_tables_match_archived_frozen_summaries(name):
    receipt = load('partial-verdict.json')['checkpoints'][name]
    for number, repetition in enumerate(receipt['stage_b'], 1):
        summary_path = ROOT / f'{name}/stage-b-{number}-summary.json'
        summary = json.loads(summary_path.read_text())
        assert hashlib.sha256(summary_path.read_bytes()).hexdigest() == repetition['summary_sha256']
        assert repetition['per_episode'] == summary['rows']
        assert len(summary['rows']) == 14
        assert repetition['valid'] == summary['valid_first_frames']
        assert repetition['correct'] == summary['correct_first_actions']
        assert repetition['invalid'] == sum(row['reason'].startswith('invalid_') for row in summary['rows'])
        assert repetition['successes'] == summary['successes']
        after = load(f'{name}/stage-b-{number}-model-after.json')
        assert all(after[key] for key in ('unchanged', 'no_gradients', 'all_parameters_bf16'))
    assert receipt['serving_nondeterminism']['observed']
    assert receipt['serving_nondeterminism']['case_grade_or_reason_differences']


@pytest.mark.parametrize('name', ['parent', 'A'])
def test_execution_preserves_all_replicas_and_lowest_rank_primary(name):
    receipt = load('partial-verdict.json')['checkpoints'][name]['execution']
    summary = load(f'{name}/execution/summary.json')
    assert receipt['primary_replica'] == 'wash2-r0'
    assert receipt['primary_ranks'] == [0, 1]
    assert receipt['successes'] == summary['models']['wash2-r0']['successes']
    totals = []
    for i in range(4):
        replica = f'wash2-r{i}'
        model = summary['models'][replica]
        assert model['episodes'] == len(model['outcomes']) == 96
        assert model['successes'] == sum(row['grade']['success'] for row in model['outcomes'])
        assert receipt['replicas'][replica]['successes'] == model['successes']
        totals.append(model['successes'])
    assert receipt['success_range'] == [min(totals), max(totals)]
    assert receipt['observed_total_spread'] == max(totals) - min(totals)
    assert load(f'{name}/smoke-summary.json')['status'] == 'passed'


def test_readonly_boundary_observation_records_earlier_public_text_collision():
    receipt = load('boundary-observation.json')
    assert receipt['case'] == 'choice-write-file' and receipt['case_order'] == 11
    assert not receipt['generation_performed'] and not receipt['regrading_performed']
    assert not receipt['gate_evaluable']
    assert receipt['valid_generated_frames'] == 3 and receipt['token_counts'] == [83, 62, 48]
    trigger = receipt['exact_trigger']
    assert trigger['private_turn_zero_based'] == 2
    assert trigger['earlier_public_summary_turn_zero_based'] == 1
    assert len(trigger['public_text']) == trigger['match_characters'] == 56
    assert hashlib.sha256(trigger['public_text'].encode()).hexdigest() == trigger['matching_text_sha256']
    assert receipt['actions'][1]['public_summary'] == trigger['public_text']
    assert [row['name'] for row in receipt['actions']] == ['read', 'read', 'finish']
    assert receipt['workspace_empty']
    assert receipt['pi_terminal']['pi_exit'] == 0 and receipt['pi_terminal']['close_verified']
    assert len(trigger['event_matches']) == 6


def test_supervision_receipts_bind_empty_window_and_fail_stop_without_collision():
    receipt = load('partial-verdict.json')['occupancy']
    assert receipt['guard_reading_snapshots'] == 1528 and receipt['foreign_occupants'] == 0
    window = load('supervision/window-before-read.json')
    assert window['grantable'] == '0,1,2,3,4,5,6,7' and not window['processes']
    terminal = load('supervision/terminal.json')
    assert terminal == receipt['terminal']
    assert terminal['exit_code'] == 1 and terminal['status'] == 'frozen-runner-failed-no-retry'
    assert not terminal['automatic_retry']
    pins = load('supervision/read-time-training-pins.json')
    assert pins['verified_before_any_read']
    assert {(row['arm'], row['point']) for row in pins['records']} == {
        ('A', 'entry'), ('A', 'final'), ('B', 'entry'), ('B', 'final')}
