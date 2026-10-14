#!/usr/bin/env python3
"""Provenance-bound preparation and receipts for the wash-2 frozen read."""
import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.eval_e97_native_execution import publish, sha

REPO = Path(__file__).resolve().parents[2]


def load(path):
    return json.loads(Path(path).read_text())


def checkpoint(root):
    receipt = root / 'bank-heldout-reads/wash2-u512-pinned/pin-receipt.json'
    pin = load(receipt)
    path = receipt.parent / pin['checkpoint']
    if sha(path) != pin['sha256']:
        raise ValueError('pinned checkpoint identity')
    return dict(path=str(path), sha256=pin['sha256'], pin_receipt=str(receipt),
                pin_receipt_sha256=sha(receipt), training=pin['training'])


def prepare_execution(root):
    target = checkpoint(root)
    source = root / 'seed-execution-panel-20261012/panel.json'
    prior = load(root / 'seed-execution-panel-20261012/measurement-receipt.json')
    if sha(source) != prior['panel_sha256']:
        raise ValueError('F12 panel provenance')
    panel = load(source)
    if (len(panel['cases']) != 96 or
            Counter(c['family'] for c in panel['cases']) !=
            dict(sum=24, edit=24, lookup=24, recovery=24)):
        raise ValueError('F12 case coverage')
    if sha(REPO / 'scripts/e97_native_execution_cases.py') != panel['case_module_sha256']:
        raise ValueError('F12 case/grader module changed')
    if sha(panel['args_json']) != panel['args_sha256']:
        raise ValueError('F12 args identity')
    original = deepcopy(panel)
    panel['models'] = [dict(checkpoint=target['path'], sha256=target['sha256'],
                            mode='train', name=f'wash2-r{i}') for i in range(4)]
    # Only the four model slots change; every frozen instrument field is retained.
    if {k: v for k, v in panel.items() if k != 'models'} != {
            k: v for k, v in original.items() if k != 'models'}:
        raise ValueError('execution instrument changed')
    out = root / 'bank-heldout-reads/wash2-u512-read-v4'
    publish(out / 'panel.json', panel)
    publish(out / 'provenance.json', dict(
        source_panel=str(source), source_panel_sha256=sha(source),
        panel_sha256=sha(out / 'panel.json'), checkpoint=target,
        method='Unmodified eight-rank CLI; four identical model slots; 96 cases per slot.',
        stale_template_metadata='Inherited scope/diagnostic_derivation describe F12/E3, not this checkpoint; models are authoritative.',
        approved_deviation='Supervisor required genuine eight-GPU CLI first because run rejects WORLD_SIZE != 8; Stage-B uses GPU 0.'))


def stage_tally(summary):
    rows = summary['rows']
    if len(rows) != 14 or len({r['id'] for r in rows}) != 14:
        raise ValueError('Stage-B coverage')
    valid = sum(r['grade']['checks']['valid_first_frame'] for r in rows)
    correct = sum(r['grade']['checks']['first_action_correct'] for r in rows)
    if valid != summary['valid_first_frames'] or correct != summary['correct_first_actions']:
        raise ValueError('Stage-B tally mismatch')
    return dict(episodes=14, valid=valid, correct=correct,
                invalid=sum(r['reason'].startswith('invalid_') for r in rows),
                nonvalid_first_frames=14-valid, successes=summary['successes'],
                reasons=dict(Counter(r['reason'] for r in rows)))


def trend_read(tallies, seed_correct=6):
    counts = [t['correct'] for t in tallies]
    mean = sum(counts) / len(counts)
    identical = all(all(t.get(k) == tallies[0].get(k) for k in ('valid', 'correct', 'invalid'))
                    for t in tallies)
    delta = mean - seed_correct
    interpretation = ('flat within the historical +/-1-2 cross-checkpoint noise band' if abs(delta) <= 2
                      else 'improved beyond the historical +/-1-2 band' if delta > 2
                      else 'regressed beyond the historical +/-1-2 band')
    return dict(method='Three unchanged fresh greedy repetitions, not independent stochastic draws.',
                correct_counts=counts, mean_correct=mean, min_correct=min(counts), max_correct=max(counts),
                mean_correct_minus_seed=delta, historical_noise_band_episodes=2,
                identical_tallies=identical, interpretation=interpretation,
                determinism=('Identical tallies: exact observed checkpoint number on this fixed panel; '
                             'no within-checkpoint variance observed.' if identical else
                             'Tallies differ despite greedy decoding; do not infer independent samples.'),
                noise_application='Historical +/-1-2 applies to cross-checkpoint interpretation, not within-checkpoint sampling variance.')


def execution_replicas(aggregate, panel):
    """Use fixed lowest-rank pair, never best/mean replica, for the literal gate."""
    names = [f'wash2-r{i}' for i in range(4)]
    if set(aggregate['models']) != set(names):
        raise ValueError('execution replica coverage')
    models = aggregate['models']
    family_by_id = {c['id']: c['family'] for c in panel['cases']}
    outcomes = {}
    for name in names:
        model = models[name]
        rows = model['outcomes']
        if (model['episodes'] != 96 or len(rows) != 96 or
                {r['id'] for r in rows} != set(family_by_id)):
            raise ValueError('execution case coverage')
        outcomes[name] = {r['id']: r for r in rows}
    disagreements = []
    for case_id, family in family_by_id.items():
        grades = {name: outcomes[name][case_id]['grade'] for name in names}
        if any(g != grades[names[0]] for g in grades.values()):
            disagreements.append(dict(id=case_id, family=family, grades=grades,
                                      success_disagrees=len({g['success'] for g in grades.values()}) > 1,
                                      reasons={name: outcomes[name][case_id]['reason'] for name in names}))
    totals = [models[name]['successes'] for name in names]
    max_pairwise = max(sum(outcomes[a][c]['grade']['success'] != outcomes[b][c]['grade']['success']
                           for c in family_by_id) for a in names for b in names)
    return dict(
        primary_replica='wash2-r0', primary_ranks=[0, 1],
        primary_selection='Fixed rank order: lowest rank pair, designated before aggregate score comparison; on the first observed disagreement this was the conservative failing side. No best-score selection.',
        replicas={name: dict(episodes=models[name]['episodes'], successes=models[name]['successes'],
                             families=models[name]['families'], floor_passed=models[name]['successes'] >= 64)
                  for name in names},
        replica_grades_identical=not disagreements, disagreements=disagreements,
        disagreement_family_counts=dict(Counter(d['family'] for d in disagreements)),
        success_range=[min(totals), max(totals)], observed_total_spread=max(totals)-min(totals),
        max_pairwise_success_disagreements=max_pairwise,
        all_replicas_clear_floor=all(n >= 64 for n in totals),
        variance_note=f'Cross-replica nondeterminism observed: total spread {max(totals)-min(totals)}/96; '
                      f'maximum pairwise success-label disagreement {max_pairwise} cases. Near-floor execution '
                      'reads carry at least this observed band; replicas are not independent draws. No reruns or re-grading.')


def verdict(root):
    reads = root / 'bank-heldout-reads'
    identity = checkpoint(root)
    origin_path = (root.parent / 'e97-e4-corpus-arc-v1/e4-juncture-evals/'
                   'seg6-u384/stage-b-plan/plan-private.json')
    origin = load(origin_path)
    instruments = {}
    tallies = []
    for n in range(1, 4):
        read = reads / f'wash2-u512-read-v{n}'
        plan_path = read / 'stage-b-plan/plan-private.json'
        plan = load(plan_path)
        summary_path = read / 'stage-b/summary.json'
        summary = load(summary_path)
        if summary['plan_sha256'] != sha(plan_path) or summary['checkpoint_sha256'] != identity['sha256']:
            raise ValueError('Stage-B summary binding')
        after = load(read / 'stage-b/model-after-private.json')
        if not all(after[k] for k in ('unchanged', 'no_gradients', 'all_parameters_bf16')):
            raise ValueError('Stage-B model immutability')
        pins = {k: v for k, v in plan.items() if k in (
            'panel', 'panel_sha256', 'pi_bin', 'pi_inventory', 'pi_inventory_sha256',
            'cli_image', 'cli_image_sha256', 'args_json', 'args_sha256',
            'tokenizer_vocabulary_sha256', 'tool_manifest', 'tool_manifest_sha256',
            'stage_b_tools', 'stage_b_tools_sha256', 'case_ids')}
        if any(pins[k] != origin[k] for k in (
                'panel', 'pi_bin', 'pi_inventory', 'cli_image', 'cli_image_sha256')):
            raise ValueError('Stage-B seg6 provenance mismatch')
        if instruments and pins != instruments:
            raise ValueError('Stage-B repetitions used different instruments')
        instruments = pins
        tally = stage_tally(summary)
        publish(read / 'panel-tally.json', dict(
            **tally, method='greedy repetition', plan_sha256=sha(plan_path),
            summary_sha256=sha(summary_path), checkpoint_sha256=identity['sha256'],
            invalid_definition='Terminal reason starting invalid_; nonvalid_first_frames separately counts all missing first frames.',
            per_episode=summary['rows']))
        tallies.append(tally)
    execution = reads / 'wash2-u512-read-v4'
    panel = load(execution / 'panel.json')
    panel_sha = sha(execution / 'panel.json')
    smoke = load(execution / 'preflight/smoke-summary.json')
    if smoke['status'] != 'passed' or len(smoke['checks']) != 96 or smoke['panel_sha256'] != panel_sha:
        raise ValueError('execution smoke')
    aggregate = load(execution / 'results/summary.json')
    if aggregate['panel_sha256'] != panel_sha:
        raise ValueError('execution summary identity')
    replica_read = execution_replicas(aggregate, panel)
    models = list(aggregate['models'].values())
    baseline = aggregate['models'][replica_read['primary_replica']]
    rank_pins = {}
    seconds = []
    for rank in range(8):
        path = execution / f'results/rank-{rank}.json'
        report = load(path)
        if report['panel_sha256'] != panel_sha or report['model']['sha256'] != identity['sha256']:
            raise ValueError('execution rank binding')
        rank_pins[str(rank)] = sha(path)
        seconds.extend(r['seconds'] for r in report['results'])
    prior = load(root / 'seed-execution-panel-20261012/measurement-receipt.json')
    total = baseline['successes']
    execution_receipt = dict(
        episodes=96, successes=total, families=baseline['families'],
        per_episode=baseline['outcomes'], replica_count=4, **replica_read,
        raw_episodes=sum(m['episodes'] for m in models), raw_successes=sum(m['successes'] for m in models),
        normalization='No division or averaging: primary wash2-r0 is the literal 96-case gate score; all other replicas are reported separately.',
        seconds_per_case_mean=sum(seconds)/len(seconds),
        seed_successes=prior['successes'], seed_families=prior['families'],
        success_difference_from_seed=total-prior['successes'],
        family_differences_from_seed={f: v['successes']-prior['families'][f]['successes']
                                      for f, v in baseline['families'].items()},
        panel_sha256=panel_sha, summary_sha256=sha(execution / 'results/summary.json'),
        smoke_summary_sha256=sha(execution / 'preflight/smoke-summary.json'), rank_reports_sha256=rank_pins)
    publish(execution / 'panel-tally.json', execution_receipt)
    trend = trend_read(tallies)
    gate = dict(valid_min=12, correct_min=10, execution_min=64,
                per_panel=[dict(valid=t['valid'] >= 12, correct=t['correct'] >= 10,
                                valid_margin=t['valid']-12, correct_margin=t['correct']-10,
                                stage_b_passed=t['valid'] >= 12 and t['correct'] >= 10) for t in tallies],
                execution_passed=total >= 64, execution_margin=total-64,
                all_execution_replicas_passed=replica_read['all_replicas_clear_floor'])
    gate['dual_gate_passed'] = gate['execution_passed'] and all(p['stage_b_passed'] for p in gate['per_panel'])
    text = (f"At 2.1x wash-1 dose, Stage-B correct is {trend['correct_counts']} /14 "
            f"(mean {trend['mean_correct']:g}, change {trend['mean_correct_minus_seed']:+g} from seed 6); "
            f"{trend['interpretation']}. Valid counts are {[t['valid'] for t in tallies]}/14. "
            f"Primary fixed-rank execution is {total}/96 versus seed 64/96 "
            f"({'held' if total >= 64 else 'not held'} at >=64); replica range is {replica_read['success_range']}/96. "
            f"The dual gate {'passes' if gate['dual_gate_passed'] else 'fails'}; correct-action target remains 10/14. "
            "These are greedy repetitions on one frozen panel, not independent draws or proof of a dose effect.")
    receipt = dict(schema='emender-wash2-u512-verdict-v1', status='measurements-complete',
                   timestamp=datetime.now(timezone.utc).isoformat(), checkpoint=identity,
                   stage_b_panels=tallies, trend=trend, execution=execution_receipt, gate=gate,
                   comparison_ledger=dict(
                       wash_seed=dict(identity='cb50c658', valid=12, correct=6, invalid=1, execution='64/96'),
                       rebuild_pg41=dict(identity='631b08f5', valid=13, correct=5, invalid=0),
                       invalid_definition='Ledger invalid counts invalid-frame termination, not 14-valid; seed had two nonvalid first frames, one invalid_frame and one generation_budget.',
                       seed_source=str(reads / 'seed-execution-panel-receipt-20261012.json')),
                   instrument_pins=dict(stage_b=instruments,
                                        stage_b_provenance=dict(path=str(origin_path), sha256=sha(origin_path)),
                                        execution={k: v for k, v in panel.items() if k != 'cases'},
                                        execution_provenance=load(execution / 'provenance.json'),
                                        sources={p: sha(REPO / p) for p in (
                                            'scripts/eval_e97_pi_native_stage_b.py',
                                            'scripts/eval_e97_pi_native_baselines.py',
                                            'scripts/eval_e97_native_execution.py',
                                            'scripts/e97_native_execution_cases.py')}),
                   verdict=text, optimizer_updates=0, checkpoint_promotion=False,
                   scope='Local lambda01 quality read only; no resilient DiLoCo systems conformance claims.')
    publish(reads / 'wash2-u512-verdict/verdict-receipt.json', receipt)
    print(text, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare-execution', 'verdict'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare-execution':
        prepare_execution(args.root)
    else:
        verdict(args.root)


if __name__ == '__main__':
    main()
