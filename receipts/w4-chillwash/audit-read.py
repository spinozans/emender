#!/usr/bin/env python3
"""Read-only W4 publication audit. No model serving, scoring, grading or generation."""
import hashlib
import json
import math
from pathlib import Path
import shutil
from datetime import datetime, timezone

from scripts.w4_chillwash import RUN, RECEIPTS, W3, verify_registration, read_identity
from scripts.w4_train_chill import learning_rate
from scripts.w3_matcheddose_preflight import checked, file_sha, PARENT_SHA


def load(path):
    return json.loads(path.read_text())


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


reg = RECEIPTS / 'preregistration.json'
verify_registration(reg)
assert load(RUN / 'terminal.json')['exit_code'] == 0
assert (RUN / 'DONE').is_file() and not (RUN / 'STOPPED').exists()
training = RUN / 'training'
complete = load(training / 'complete-receipt.json')
checked(training / 'training.jsonl', complete['training_log_sha256'])
checked(training / 'start-receipt.json', complete['start_receipt_sha256'])
rows = [json.loads(line) for line in (training / 'training.jsonl').read_text().splitlines()]
expected = load(W3 / 'preflight.json')['exposure']['sequence']
assert len(rows) == 512
inputs = targets = 0
exposure = weighted = 0.
for update, (row, pack) in enumerate(zip(rows, expected, strict=True), 1):
    assert row['update'] == update
    assert row['sample_ids'] == pack['sample_ids']
    assert row['sampled_packs'] == [dict(pack_id=pack['pack_id'], record_ids=pack['record_ids'])]
    assert (row['tokens'], row['targets']) == (pack['inputs'], pack['targets'])
    assert row['actual_lr'] == row['scheduled_lr'] == learning_rate(update)
    assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm'])
    inputs += row['tokens']; targets += row['targets']
    exposure += row['targets'] * pack['loss_weight']
    weighted += row['actual_lr'] * row['targets'] * pack['loss_weight']
    assert (row['cumulative_inputs'], row['cumulative_targets']) == (inputs, targets)
    assert row['target_objective_weight'] == pack['loss_weight']
    assert row['cumulative_objective_weighted_exposure'] == exposure
    assert row['cumulative_lr_weighted_exposure'] == weighted
assert (inputs, targets) == (3571182, 154386)
assert complete['cumulative_objective_weighted_exposure'] == exposure
assert complete['cumulative_lr_weighted_exposure'] == weighted
pins = []
for directory, sha in [(training / 'entry-pinned', PARENT_SHA),
        (training / 'final-pinned', complete['checkpoint_sha256']),
        (RUN / 'reads/checkpoint-pinned', complete['checkpoint_sha256'])]:
    pin = load(directory / 'pin-receipt.json')
    assert pin['pinned_before_any_evaluation'] and pin['sha256'] == sha
    pins.append(checked(directory / pin['checkpoint'], sha))
identity = read_identity()
assert identity['sha256'] == complete['checkpoint_sha256']
read = RUN / 'reads'
observations = load(read / 'exploratory-observations.json')
summary = load(read / 'stage-b/summary.json')
assert observations['summary_sha256'] == file_sha(read / 'stage-b/summary.json')
assert observations['trip_log_sha256'] == file_sha(read / 'stage-b/boundary-trips.jsonl')
assert observations['registration_sha256'] == file_sha(reg)
assert len(summary['rows']) == 14
assert [row['id'] for row in summary['rows']] == load(W3 / 'preflight.json')['instruments']['stage_b']['case_ids']
trip_log = [json.loads(line) for line in (read / 'stage-b/boundary-trips.jsonl').read_text().splitlines()]
trips = []
for row in summary['rows']:
    episode = read / 'stage-b' / row['id']
    raw = load(episode / 'episode-private.json')
    for key in row:
        assert raw[key] == row[key]
    private = []
    for action in raw['actions']:
        if action:
            private.extend([action.get('reasoning_content') or '', action['arguments'].get('thought', '') if action['name'] == 'think' else ''])
    events = (episode / 'pi/pi-events-private.jsonl').read_text()
    matching = [i for i, value in enumerate(private) if value and value in events]
    assert matching == [trip['private_field_index'] for trip in row['boundary_trips']]
    assert row['boundary_collision'] == bool(matching)
    for trip in row['boundary_trips']:
        value = private[trip['private_field_index']]
        assert trip['case_id'] == row['id']
        assert trip['content_sha256'] == hashlib.sha256(value.encode()).hexdigest()
        assert trip['content_utf8_bytes'] == len(value.encode())
        assert trip['event_occurrences'] == events.count(value)
        trips.append(trip)
assert trips == trip_log == observations['boundary_trips']
assert summary['valid_first_frames'] == sum(row['grade']['checks']['valid_first_frame'] for row in summary['rows'])
assert summary['correct_first_actions'] == sum(row['grade']['checks']['first_action_correct'] for row in summary['rows'])
assert summary['successes'] == sum(row['grade']['success'] for row in summary['rows'])
assert summary['boundary_collision_cases'] == sum(row['boundary_collision'] for row in summary['rows'])
assert observations['classification'] == ('b' if trip_log else 'a')
after = load(read / 'stage-b/model-after-private.json')
assert all(after[key] for key in ('unchanged','no_gradients','all_parameters_bf16'))
assert 'gate' not in summary and summary['automatic_retries'] == 0
pin_time = datetime.fromisoformat(load(read / 'checkpoint-pinned/pin-receipt.json')['pinned_at']).timestamp()
assert pin_time < (read / 'stage-b-plan/plan-private.json').stat().st_mtime
assert pin_time < min((read / 'stage-b' / row['id'] / 'generations-private.json').stat().st_mtime for row in summary['rows'])
nll = load(read / 'docnll.json')
assert nll['trajectory_only'] and nll['weight_mode'] == 'saved' and nll['checkpoint_sha256'] == identity['sha256']
assert len(nll['records']) == 8 and sum(row['scored_tokens'] for row in nll['records']) == nll['scored_tokens'] == 198295
assert nll['mean_nll'] == sum(row['nll_sum'] for row in nll['records']) / 198295
occupancy = [json.loads(line) for line in (RUN / 'occupancy.jsonl').read_text().splitlines()]
assert not any(state['foreign_on_selected_gpu'] for state in occupancy)
launch = load(RUN / 'launcher.json')
assert launch['physical_gpu'] in ['2','3','4','5','6','7'] and launch['ppid'] == 1
assert load(RUN / 'watcher.json')['ppid'] == 1
# Preserve public measurements and model fingerprints, never private transcripts.
for relative in ['terminal.json', 'training/complete-receipt.json', 'training/training.jsonl',
        'training/final-pinned/pin-receipt.json', 'reads/checkpoint-pinned/pin-receipt.json',
        'reads/read-identity.json', 'reads/stage-b/summary.json', 'reads/stage-b/boundary-trips.jsonl',
        'reads/stage-b/model-before-private.json', 'reads/stage-b/model-after-private.json',
        'reads/docnll.json', 'reads/exploratory-observations.json']:
    shutil.copyfile(RUN / relative, RECEIPTS / relative.replace('/', '-'))
manifest = []
for path in sorted(RUN.rglob('*')):
    if path.is_file() and not path.is_symlink() and path.suffix != '.pt' and 'triton' not in path.parts:
        manifest.append(dict(path=str(path.relative_to(RUN)), bytes=path.stat().st_size, sha256=file_sha(path)))
save(RECEIPTS / 'raw-evidence-manifest.json', dict(root=str(RUN), files=manifest,
    exclusions='checkpoint payloads separately verified by full SHA256 in audit; no Triton cache or symlinks',
    private_transcripts='retained in artifact root only'))
save(RECEIPTS / 'publication-audit.json', dict(status='verified', audited_at=datetime.now(timezone.utc).isoformat(),
    audit_script_sha256=file_sha(Path(__file__)), registration_sha256=file_sha(reg),
    raw_manifest_sha256=file_sha(RECEIPTS / 'raw-evidence-manifest.json'), raw_files=len(manifest),
    all_512_actual_updates_matched=True, updates=512, inputs=inputs, targets=targets,
    cumulative_objective_weighted_exposure=exposure, cumulative_lr_weighted_exposure=weighted,
    checkpoint_pins=pins, complete_14_case_read=True, collision_cases=summary['boundary_collision_cases'],
    matching_private_fields=len(trips), classification=observations['classification'],
    docnll_saved_targets=198295, occupancy_snapshots=len(occupancy), foreign_on_selected_gpu=0,
    no_model_serving_scoring_regrading_or_generation=True))
print('W4_PUBLICATION_AUDIT: verified all512 updates, three pins, all14 cases/trip digests, docNLL and occupancy')
