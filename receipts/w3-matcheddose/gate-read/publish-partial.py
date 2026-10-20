"""Archive and tally existing W3G outputs only: no model load, generation or regrade."""
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import tiktoken

from scripts.w3_matcheddose_preflight import REPO, ROOT, checked, file_sha
from scripts.w3_gate_read import historical_reader
from scripts.w3_train_arm import verify_sources
from scripts.eval_e97_pi_native_baselines import actions

RUN = ROOT / 'w3-matcheddose'
OUT = REPO / 'receipts/w3-matcheddose/gate-read'
OUT.mkdir(mode=0o700, exist_ok=False)
reader = historical_reader()
preflight = json.loads((REPO / 'receipts/w3-matcheddose/preflight.json').read_text())
freeze = json.loads((REPO / 'receipts/w3-matcheddose/runtime-freeze-v2.json').read_text())
verify_sources(freeze)
for binding in preflight['instruments']['bindings']:
    checked(binding['path'], binding['sha256'])
for binding in freeze['docnll']['bindings']:
    checked(binding['path'], binding['sha256'])

def load(path):
    return json.loads(path.read_text())

def publish(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')

def archive(path, relative):
    target = OUT / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('xb') as stream:
        stream.write(path.read_bytes())
    assert file_sha(path) == file_sha(target)
    return checked(path, file_sha(target))

# Hash every retained raw evidence file, but do not duplicate private transcripts
# in Git. Checkpoint identities are explicitly checked separately below.
manifest = []
for path in sorted(RUN.rglob('*')):
    if not path.is_file() or 'reads' not in path.parts and 'gate-supervision' not in path.parts:
        continue
    if path.suffix == '.pt' or any(part.startswith('triton') for part in path.relative_to(RUN).parts):
        continue
    manifest.append(checked(path, file_sha(path)))
publish(OUT / 'raw-evidence-manifest.json', manifest)

terminal = load(RUN / 'gate-supervision/terminal.json')
assert terminal['status'] == 'frozen-runner-failed-no-retry' and terminal['exit_code'] == 1
assert not terminal['automatic_retry']
for path in sorted((RUN / 'gate-supervision').iterdir()):
    if path.is_file():
        archive(path, 'supervision/' + path.name)
occupancy = [json.loads(line) for line in (RUN / 'gate-supervision/occupancy.jsonl').read_text().splitlines()]
reading = [row for row in occupancy if row['phase'] == 'reading']
assert reading and not any(row['foreign'] for row in reading)
window = load(RUN / 'gate-supervision/window-before-read.json')
assert window['grantable'] == '0,1,2,3,4,5,6,7' and not window['processes']
assert all(row[2:] == ['4', '0'] for row in window['gpus'])

results = {}
for name in ('parent', 'A', 'B'):
    directory = RUN / 'reads' / name
    identity = load(directory / 'read-identity.json')
    checked(identity['checkpoint'], identity['sha256'])
    assert file_sha(directory / 'checkpoint-pinned/pin-receipt.json') == identity['pin_receipt_sha256']
    checked(directory / 'execution-panel.json', identity['execution_panel_sha256'])
    for relative in ('read-identity.json', 'checkpoint-pinned/pin-receipt.json', 'execution-panel.json'):
        archive(directory / relative, f'{name}/{relative}')
    results[name] = dict(identity=identity)
    repetitions = []
    for n in range(1, 4):
        plan_path = directory / f'stage-b-plan-{n}/plan-private.json'
        if not plan_path.exists():
            continue
        plan = load(plan_path)
        for key, value in preflight['instruments']['stage_b'].items():
            assert plan[key] == value, key
        assert plan['checkpoint_sha256'] == identity['sha256']
        archive(plan_path, f'{name}/stage-b-plan-{n}.json')
        for suffix in ('freeze', 'console'):
            archive(directory / f'stage-b-{suffix}-{n}.txt', f'{name}/stage-b-{suffix}-{n}.txt')
        after_path = directory / f'stage-b-{n}/model-after-private.json'
        after = load(after_path)
        assert all(after[key] for key in ('unchanged', 'no_gradients', 'all_parameters_bf16'))
        archive(after_path, f'{name}/stage-b-{n}-model-after.json')
        summary_path = directory / f'stage-b-{n}/summary.json'
        if not summary_path.exists():
            assert name == 'B' and n == 1
            continue
        summary = load(summary_path)
        assert summary['plan_sha256'] == file_sha(plan_path)
        assert summary['checkpoint_sha256'] == identity['sha256']
        assert [row['id'] for row in summary['rows']] == plan['case_ids']
        row = dict(repetition=n, **reader.stage_tally(summary), per_episode=summary['rows'], summary_sha256=file_sha(summary_path))
        assert row['successes'] == sum(item['grade']['success'] for item in row['per_episode'])
        repetitions.append(row)
        archive(summary_path, f'{name}/stage-b-{n}-summary.json')
    if name == 'B':
        assert not repetitions
        continue
    assert len(repetitions) == 3
    results[name]['stage_b'] = repetitions
    results[name]['stage_b_tally_range'] = {key: [min(row[key] for row in repetitions), max(row[key] for row in repetitions)] for key in ('valid', 'correct', 'invalid', 'successes')}
    changes = []
    for index, case in enumerate(repetitions[0]['per_episode']):
        rows = [row['per_episode'][index] for row in repetitions]
        if len({json.dumps((row['grade'], row['reason']), sort_keys=True) for row in rows}) > 1:
            changes.append(dict(id=case['id'], repetitions=[dict(repetition=n, grade=row['grade'], reason=row['reason']) for n, row in enumerate(rows, 1)]))
    results[name]['serving_nondeterminism'] = dict(observed=bool(changes), case_grade_or_reason_differences=changes)
    panel = load(directory / 'execution-panel.json')
    aggregate = load(directory / 'execution/summary.json')
    assert aggregate['panel_sha256'] == file_sha(directory / 'execution-panel.json')
    smoke = load(directory / 'preflight/smoke-summary.json')
    assert smoke['status'] == 'passed' and len(smoke['checks']) == 96
    assert smoke['panel_sha256'] == aggregate['panel_sha256']
    for rank in range(8):
        rank_path = directory / f'execution/rank-{rank}.json'
        report = load(rank_path)
        assert report['model']['sha256'] == identity['sha256']
        assert report['panel_sha256'] == aggregate['panel_sha256']
        archive(rank_path, f'{name}/execution/rank-{rank}.json')
    execution = reader.execution_replicas(aggregate, panel)
    execution['successes'] = aggregate['models']['wash2-r0']['successes']
    execution['summary_sha256'] = file_sha(directory / 'execution/summary.json')
    results[name]['execution'] = execution
    results[name]['raw_scalar_gate'] = dict(stage_b_each_repetition_pass=[row['valid'] >= 12 and row['correct'] >= 10 for row in repetitions], execution_primary_pass=execution['successes'] >= 64, dual_gate_pass=all(row['valid'] >= 12 and row['correct'] >= 10 for row in repetitions) and execution['successes'] >= 64)
    archive(directory / 'execution/summary.json', f'{name}/execution/summary.json')
    archive(directory / 'preflight/smoke-summary.json', f'{name}/smoke-summary.json')

# B's completed prefix is not a 14-case repetition, nor a gate verdict.
bdir = RUN / 'reads/B/stage-b-1'
plan = load(RUN / 'reads/B/stage-b-plan-1/plan-private.json')
prefix = []
for case in plan['case_ids']:
    path = bdir / case / 'episode-private.json'
    if not path.exists():
        break
    row = load(path)
    prefix.append({key: deepcopy(row[key]) for key in ('id', 'family', 'delay_tokens', 'model_generations', 'reason', 'failed', 'close_verified', 'model_failure_verified', 'grade')})
assert len(prefix) == 10
results['B']['stage_b_partial'] = dict(repetition=1, completed_cases=10, scheduled_cases=14, per_episode=prefix, descriptive_prefix_counts=dict(valid=sum(row['grade']['checks']['valid_first_frame'] for row in prefix), correct=sum(row['grade']['checks']['first_action_correct'] for row in prefix), invalid=sum(row['reason'].startswith('invalid_') for row in prefix), successes=sum(row['grade']['success'] for row in prefix)), gate_evaluable=False, missing_repetitions=[2, 3], execution='not run', docnll='not run')

# Decode only already saved token IDs. Never call generation or grade the tripped case.
enc = tiktoken.get_encoding('p50k_base')
panel = load(Path(plan['panel']))
tripped = bdir / 'choice-write-file'
generations = load(tripped / 'generations-private.json')['generations']
acts = actions(SimpleNamespace(generations=generations, tools=panel['tools']), enc)
assert [g['reason'] for g in generations] == ['valid', 'valid', 'valid']
assert [a['name'] for a in acts] == ['read', 'read', 'finish']
assert [a['arguments'] for a in acts] == [dict(path='result.txt'), dict(path='result.txt'), dict(message='done')]
private = acts[2]['reasoning_content']
assert private == acts[1]['content'] and len(private) == 56
assert private != acts[2]['content']
match_records = []
for index, line in enumerate((tripped / 'pi/pi-events-private.jsonl').read_text().splitlines(), 1):
    if private in line:
        event = json.loads(line)
        match_records.append(dict(line=index, type=event['type'], role=event.get('message', {}).get('role')))
assert [r['line'] for r in match_records] == [20, 22, 23, 27, 32, 43]
assert not list((tripped / 'workspace').rglob('*'))
transport = load(tripped / 'pi/transport-terminal.json')
assert transport['pi_exit'] == 0 and transport['close_verified'] and transport['reason'] == 'finished'
neighbors = []
for case in ('choice-read-explicit', 'choice-grep-symbol', 'choice-bash-command', 'choice-edit-file'):
    row = load(bdir / case / 'episode-private.json')
    valid = [action for action in row['actions'] if action]
    neighbors.append(dict(case=case, reason=row['reason'], actions=[dict(name=action['name'], arguments=action['arguments']) for action in valid]))
diagnostic = dict(schema='emender-w3g-readonly-boundary-observation-v1', checkpoint=results['B']['identity']['sha256'], case='choice-write-file', case_order=11, repetition=1, generation_performed=False, regrading_performed=False, valid_generated_frames=3, token_counts=[len(g['token_ids']) for g in generations], actions=[dict(turn=i, name=a['name'], arguments=a['arguments'], public_summary=a.get('content')) for i, a in enumerate(acts)], exact_trigger=dict(private_turn_zero_based=2, earlier_public_summary_turn_zero_based=1, match_characters=56, matching_text_sha256=__import__('hashlib').sha256(private.encode()).hexdigest(), public_text=acts[1]['content'], event_matches=match_records), interpretation='Private Analysis on third turn repeats second-turn public Summary. Frozen guard searches all private strings in the entire event stream, including earlier public events. This is a lexical collision with previously public text, not evidence that the transport serialized a private field. No malformed frame/reasoning-close marker was needed to trigger the guard.', workspace_empty=True, pi_terminal=transport, neighboring_observations=neighbors, gate_evaluable=False, source_bindings=[checked(p, file_sha(p)) for p in (tripped / 'generations-private.json', tripped / 'pi/pi-events-private.jsonl', tripped / 'pi/transport-terminal.json', bdir / 'model-after-private.json', REPO / 'scripts/eval_e97_pi_native_stage_b.py')])
publish(OUT / 'boundary-observation.json', diagnostic)
historical_path = ROOT / 'bank-heldout-reads/wash2-u512-verdict/verdict-receipt.json'
historical = load(historical_path)
report = dict(schema='emender-w3g-partial-publication-v1', status='INVALID/INCONCLUSIVE', published_at=datetime.now(timezone.utc).isoformat(), reason='B repetition1 fail-stopped at frozen private-reasoning substring guard before case11 row publication; remaining B instruments and all docNLL not run.', supervisor_ruling='Publish all preserved partial tables, inspect tripped case and neighbors read-only, document behavior without treating the guard as a gate verdict; rescue question OPEN; no retry/replay/resume without explicit operator authorization through supervisor.', checkpoints=results, historical_comparison=dict(receipt=checked(historical_path, file_sha(historical_path)), parent=historical['comparison_ledger']['wash_seed'], wash2_stage_b=historical['stage_b_panels'], wash2_primary_execution=historical['execution']['successes'], wash2_execution_replicas=historical['execution']['replicas']), docnll=dict(status='not run for any checkpoint', panel_prefix='5725d820', trajectory_only=True, numeric_bound=None), occupancy=dict(window_time=window['time'], guard_reading_snapshots=len(reading), foreign_occupants=0, terminal=terminal), frozen_verdict_receipt_exists=(RUN / 'reads/verdict-receipt.json').exists(), retry=False, replay=False, resume=False, checkpoint_promotion=False, rescue_question='OPEN', preregistered_predictions=load(REPO / 'receipts/w3-matcheddose/preregistration.json')['predictions_verbatim'], pending_operator_decision=['Accept inconclusive.', 'Authorize a timestamped amended-instrument B re-read with prior observations registered before any new read.'], scope='one matched seed/order, local quality read; no scale or optimizer-mechanism finding')
assert not report['frozen_verdict_receipt_exists']
publish(OUT / 'partial-verdict.json', report)
archive(Path(__file__), 'publish-partial.py')
for binding in manifest:
    checked(binding['path'], binding['sha256'])
print('W3G_PARTIAL_PUBLICATION_VERIFIED: original evidence unchanged, all complete outputs bound, no model load/generation/regrade/retry')
print(json.dumps({name: {key: value for key, value in row.items() if key not in ('identity', 'stage_b', 'execution')} for name, row in results.items()}, sort_keys=True))
