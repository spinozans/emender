#!/usr/bin/env python3
"""Prepare pinned W3 reads and assemble unchanged frozen-gate case tallies."""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

from scripts.w3_matcheddose_preflight import REPO, ROOT, PARENT_SHA, checked, file_sha
from scripts.w3_train_arm import pin, publish, verify_sources

RUN = ROOT / 'w3-matcheddose'


def historical_reader():
    spec = importlib.util.spec_from_file_location('w3_wash2_verdict', REPO / 'scripts/rl-loop-v1/wash2_verdict_receipt.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(preflight_path):
    preflight = json.loads(preflight_path.read_text())
    freeze = json.loads((REPO / 'receipts/w3-matcheddose/runtime-freeze-v2.json').read_text())
    verify_sources(freeze)
    for binding in freeze['docnll']['bindings']:
        checked(binding['path'], binding['sha256'])
    for binding in preflight['instruments']['bindings']:
        checked(binding['path'], binding['sha256'])
    (RUN / 'reads').mkdir(mode=0o700, exist_ok=False)
    source_panel = Path(preflight['instruments']['execution_panel'])
    original = json.loads(source_panel.read_text())
    targets = dict(parent=dict(checkpoint=str(ROOT / 'wash-v1/checkpoints/wash_u256_y_seed_fresh_state.pt'),
                               checkpoint_sha256=PARENT_SHA))
    for arm in ('A', 'B'):
        complete = json.loads((RUN / f'arms/{arm}/complete-receipt.json').read_text())
        if (complete['status'], complete['updates'], complete['inputs'], complete['targets']) != (
                'complete', 512, 3571182, 154386):
            raise ValueError('incomplete/mismatched production arm; no gate read')
        targets[arm] = complete
    for name, target in targets.items():
        read = RUN / 'reads' / name
        read.mkdir(mode=0o700)
        checkpoint = pin(Path(target['checkpoint']), read / 'checkpoint-pinned', target['checkpoint_sha256'])
        panel = deepcopy(original)
        panel['models'] = [dict(checkpoint=str(checkpoint), sha256=target['checkpoint_sha256'],
                                mode='train', name=f'wash2-r{i}') for i in range(4)]
        if {k: v for k, v in panel.items() if k != 'models'} != {k: v for k, v in original.items() if k != 'models'}:
            raise ValueError('execution instrument changed')
        publish(read / 'execution-panel.json', panel)
        publish(read / 'read-identity.json', dict(checkpoint=str(checkpoint), sha256=target['checkpoint_sha256'],
                    pin_receipt_sha256=file_sha(read / 'checkpoint-pinned/pin-receipt.json'),
                    execution_panel_sha256=file_sha(read / 'execution-panel.json'),
                    primary_execution='wash2-r0', primary_ranks=[0, 1],
                    stage_b_repetitions=3, primary_weight_mode='train', saved_weight_mode='diagnostic-only'))


def classify(reads):
    """Prelaunch conservative repetition rule; never average away nondeterminism."""
    parent, a, b = [reads[name] for name in ('parent', 'A', 'B')]
    counts = {name: [r['correct'] for r in row['stage_b']] for name, row in reads.items()}
    b_min, a_max, p_max = min(counts['B']), max(counts['A']), max(counts['parent'])
    per_checkpoint_gate = {name: all(r['valid'] >= 12 and r['correct'] >= 10 for r in row['stage_b'])
                            and row['execution']['successes'] >= 64 for name, row in reads.items()}
    gain_parent, gain_a = b_min - p_max, b_min - a_max
    rescue = per_checkpoint_gate['B'] and b_min - 6 >= 4 and gain_parent >= 4 and gain_a >= 3
    large_prediction_failed = b_min <= 8 or gain_a < 3
    return dict(dual_gate=per_checkpoint_gate, strong_regimen_rescue=rescue,
                large_matched_dose_regimen_rescue_prediction_failed=large_prediction_failed,
                partial_acquisition_gate_failure=b_min == 9,
                both_arms_pass=per_checkpoint_gate['A'] and per_checkpoint_gate['B'],
                both_arms_pass_interpretation='If both arms pass, the old failure was not uniquely explained by the new regimen.',
                a_correct_range_prediction_holds=all(4 <= value <= 8 for value in counts['A']),
                b_correct_counts=counts['B'], conservative_b_minus_parent=gain_parent,
                conservative_b_minus_a=gain_a, conservative_b_minus_historical_parent6=b_min - 6,
                serving_nondeterminism={name: len({json.dumps([(p['id'], p['grade'], p['reason'])
                                           for p in repetition.get('per_episode', [])], sort_keys=True)
                                           for repetition in row['stage_b']}) > 1 or
                                        len({(r['valid'], r['correct'], r['invalid']) for r in row['stage_b']}) > 1
                                        for name, row in reads.items()},
                interpretation=('large local matched-dose regimen rescue' if rescue else
                                'large matched-dose regimen-rescue prediction fails' if large_prediction_failed else
                                'partial acquisition, gate failure' if b_min == 9 else
                                'no strong regimen-rescue finding; inspect all frozen gates and spreads'),
                scope='one matched seed/order; not robustness, unique optimizer mechanism or permission to scale')


def verdict(preflight_path):
    reader = historical_reader()
    preflight = json.loads(preflight_path.read_text())
    for binding in preflight['instruments']['bindings']:
        checked(binding['path'], binding['sha256'])
    reads = {}
    for name in ('parent', 'A', 'B'):
        directory = RUN / 'reads' / name
        identity_path = directory / 'read-identity.json'
        identity = json.loads(identity_path.read_text())
        checked(identity['checkpoint'], identity['sha256'])
        if file_sha(directory / 'checkpoint-pinned/pin-receipt.json') != identity['pin_receipt_sha256']:
            raise ValueError('pin receipt changed')
        stage = []
        for repetition in range(1, 4):
            plan_path = directory / f'stage-b-plan-{repetition}/plan-private.json'
            summary_path = directory / f'stage-b-{repetition}/summary.json'
            summary = json.loads(summary_path.read_text())
            plan = json.loads(plan_path.read_text())
            for key, expected in preflight['instruments']['stage_b'].items():
                if plan.get(key) != expected:
                    raise ValueError(f'Stage-B frozen instrument divergence: {key}')
            if plan['checkpoint_sha256'] != identity['sha256']:
                raise ValueError('Stage-B plan checkpoint divergence')
            if {row['id'] for row in summary['rows']} != set(plan['case_ids']):
                raise ValueError('Stage-B case coverage divergence')
            if (summary['plan_sha256'] != file_sha(plan_path)
                    or summary['checkpoint_sha256'] != identity['sha256']):
                raise ValueError('stage-B summary binding')
            after = json.loads((directory / f'stage-b-{repetition}/model-after-private.json').read_text())
            if not all(after[k] for k in ('unchanged', 'no_gradients', 'all_parameters_bf16')):
                raise ValueError('stage-B model mutation/precision divergence')
            stage.append(dict(**reader.stage_tally(summary), summary_sha256=file_sha(summary_path),
                              per_episode=summary['rows']))
        panel_path = directory / 'execution-panel.json'
        checked(panel_path, identity['execution_panel_sha256'])
        panel = json.loads(panel_path.read_text())
        aggregate_path = directory / 'execution/summary.json'
        aggregate = json.loads(aggregate_path.read_text())
        smoke = json.loads((directory / 'preflight/smoke-summary.json').read_text())
        if (smoke['status'] != 'passed' or len(smoke['checks']) != 96
                or smoke['panel_sha256'] != file_sha(panel_path)):
            raise ValueError('execution smoke binding')
        for rank in range(8):
            report = json.loads((directory / f'execution/rank-{rank}.json').read_text())
            if report['panel_sha256'] != file_sha(panel_path) or report['model']['sha256'] != identity['sha256']:
                raise ValueError('execution rank binding')
        if aggregate['panel_sha256'] != file_sha(panel_path):
            raise ValueError('execution summary binding')
        execution = reader.execution_replicas(aggregate, panel)
        primary = aggregate['models'][execution['primary_replica']]
        execution.update(successes=primary['successes'], per_episode=primary['outcomes'],
                         summary_sha256=file_sha(aggregate_path))
        reads[name] = dict(identity=identity, stage_b=stage, execution=execution)
    docnll_path = RUN / 'reads/docnll/docnll-gpu.json'
    docnll = json.loads(docnll_path.read_text())
    freeze = json.loads((REPO / 'receipts/w3-matcheddose/runtime-freeze-v2.json').read_text())
    verify_sources(freeze)
    for binding in freeze['docnll']['bindings']:
        checked(binding['path'], binding['sha256'])
    historical_docnll = json.loads(Path(freeze['docnll']['bindings'][0]['path']).read_text())
    if (docnll['panel_sha256'] != historical_docnll['panel_sha256']
            or docnll['args_sha256'] != historical_docnll['args_sha256']
            or set(docnll['models']) != {'substrate', 'parent', 'A', 'B'}):
        raise ValueError('docNLL instrument/model coverage divergence')
    for name, row in docnll['models'].items():
        expected_sha = freeze['docnll']['baseline']['sha256'] if name == 'substrate' else reads[name]['identity']['sha256']
        if row['sha256'] != expected_sha or row['weight_mode'] != 'saved' or row['totals']['scored_tokens'] != 198295:
            raise ValueError('docNLL point/dose divergence')
    report = dict(schema='emender-w3-verdict-v1', status='measurements-complete',
                  preflight_sha256=file_sha(preflight_path), checkpoints=reads,
                  verdict=classify(reads), docnll=dict(receipt=str(docnll_path), sha256=file_sha(docnll_path),
                  weight_mode='saved', trajectory_only=True, values={name: row['totals']['mean_nll']
                  for name, row in docnll['models'].items()}),
                  automatic_retry=False, best_checkpoint_selection=False, checkpoint_promotion=False)
    target = RUN / 'reads/verdict-receipt.json'
    publish(target, report)
    # Pin the final verdict bytes before any downstream interpretation/publication.
    pinned = RUN / 'reads/verdict-pinned'
    pinned.mkdir(mode=0o700, exist_ok=False)
    (pinned / target.name).write_bytes(target.read_bytes())
    publish(pinned / 'pin-receipt.json', dict(path=target.name, sha256=file_sha(target), source=str(target)))
    print(report['verdict']['interpretation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verdict'))
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.command == 'prepare' else verdict)(args.preflight)


if __name__ == '__main__':
    main()
