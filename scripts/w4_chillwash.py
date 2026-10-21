#!/usr/bin/env python3
"""W4-only registration, read-time pins and trajectory-only document read."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.w3_matcheddose_preflight import (
    ROOT, REPO, PARENT_SHA, AUTHORITY_SHA, PACK_SHA, checked, file_sha, digest, recount,
)
from scripts.w3_train_arm import verify_sources, verify_batch

RUN = ROOT / 'w4-chillwash'
RECEIPTS = REPO / 'receipts/w4-chillwash'
W3 = REPO / 'receipts/w3-matcheddose'


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write('\n')


def pin(source, directory, expected):
    directory.mkdir(mode=0o700, parents=True, exist_ok=False)
    checked(source, expected)
    target = directory / source.name
    subprocess.run(['cp', '--reflink=auto', str(source), str(target)], check=True)
    checked(target, expected)
    publish(directory / 'pin-receipt.json', dict(schema='emender-w4-pin-v1',
            checkpoint=target.name, sha256=expected, source=str(source),
            pinned_before_any_evaluation=True, pinned_at=datetime.now(timezone.utc).isoformat()))
    return target


def verify_registration(path):
    registration = json.loads(path.read_text())
    relative = str(path.resolve().relative_to(REPO))
    if digest(subprocess.check_output(['git', 'show', f'HEAD:{relative}'], cwd=REPO)) != file_sha(path):
        raise ValueError('W4 registration must be committed before launch')
    if registration['status'] != 'registered-before-launch':
        raise ValueError('missing W4 prelaunch registration')
    for binding in registration['bindings']:
        checked(binding['path'], binding['sha256'])
    freeze = json.loads((W3 / 'runtime-freeze-v2.json').read_text())
    verify_sources(freeze)
    for name, expected in registration['versions'].items():
        if version(name) != expected:
            raise ValueError(f'runtime version changed: {name}')
    return registration


def register(path):
    from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset, SFTSamplerIdentity
    from scripts.e97_open_swe_native_codec import vocabulary
    import tiktoken

    preflight = json.loads((W3 / 'preflight.json').read_text())
    freeze = json.loads((W3 / 'runtime-freeze-v2.json').read_text())
    checked(W3 / 'preflight.json', freeze['preflight_sha256'])
    verify_sources(freeze)
    exposure = recount(ROOT)
    if exposure != preflight['exposure']:
        raise ValueError('independent 512-update recount differs from W3')
    data = MaskedSFTPackedDataset(ROOT / 'wash-v2/authority', ROOT / 'wash-v2/packs',
        identity=SFTSamplerIdentity.from_metadata(freeze['sampler_identity']), rank=0,
        sampler_mode='epoch-permutation')
    for index, row in enumerate(exposure['sequence']):
        verify_batch(data, index, row)
    data.close()
    if vocabulary(tiktoken.get_encoding('p50k_base'))[1] != freeze['tokenizer_vocabulary_sha256']:
        raise ValueError('tokenizer identity changed')
    for binding in preflight['instruments']['bindings'] + freeze['docnll']['bindings']:
        checked(binding['path'], binding['sha256'])
    inventory = Path(preflight['instruments']['stage_b']['pi_inventory'])
    result = subprocess.run(['sha256sum', '-c', str(inventory)], capture_output=True, text=True, check=True)
    conversion = json.loads((W3 / 'parent-conversion.json').read_text())
    if conversion['status'] != 'passed' or conversion['files'][-1]['sha256'] != PARENT_SHA:
        raise ValueError('W3 parent conversion is not verified')
    # Reuse W3's verified conversion, never reconstruct a new parent.
    checked(conversion['files'][-1]['path'], PARENT_SHA)
    files = [*sorted((REPO / 'scripts').glob('w4_*.py')),
             REPO / 'scripts/w4_run_chillwash.sh', REPO / 'scripts/watch_w4_chillwash.py',
             REPO / 'tests/test_w4_chillwash.py', REPO / 'scripts/gpu_lease.sh',
             REPO / 'scripts/watch_w3_gate_window.py',
             Path('/home/erikg/emender-scratch/fix-fanout/taskW4-chillwash.md')]
    files += [W3 / name for name in ('preflight.json', 'runtime-freeze-v2.json',
                                    'parent-conversion.json', 'adamw-viability.json')]
    bindings = [dict(path=str(p), sha256=file_sha(p)) for p in files]
    bindings += preflight['instruments']['bindings'] + freeze['docnll']['bindings']
    versions = {name: version(name) for name in freeze['versions']}
    if versions != freeze['versions']:
        raise ValueError('W3 runtime versions changed')
    publish(path, dict(schema='emender-w4-registration-v1', status='registered-before-launch',
        registered_at=datetime.now(timezone.utc).isoformat(), bindings=bindings, versions=versions,
        parent_sha256=PARENT_SHA, authority_manifest_sha256=AUTHORITY_SHA, pack_manifest_sha256=PACK_SHA,
        conversion='reuse exact W3 verified cb50c658 fresh-state train-y seed; no reconversion',
        independent_recount='all payloads and 512 pack/sample/token/loss/valid/reset identities reverified',
        sequence_sha256=digest(json.dumps(exposure['sequence'], sort_keys=True).encode()),
        updates=512, inputs=3571182, targets=154386, world_size=1, allowed_physical_gpus=[2,3,4,5,6,7],
        optimizer=dict(name='torch.optim.AdamW', betas=[.9,.95], eps=1e-8, weight_decay=.01,
                       foreach=False, fused=False, fresh_state=True, warmup_steps=0),
        schedule='2e-8 + .5*(2e-6-2e-8)*(1+cos(pi*(update-1)/511)), update=1..512',
        protocol_amendment=dict(scope='THIS W4 exploratory read ONLY; frozen W3 strict fail-stop untouched',
            stage_b='one greedy pass, exact 14 cases/order and generation/tool budgets',
            boundary_guard='same lexical predicate; log every matching private field with case id and UTF8 SHA256; mark case boundary-collision and continue to next case; no retry',
            grading='unchanged underlying grades retained for every case, collision flag separate; counts are exploratory observations only',
            execution='optional replica omitted; one-GPU Stage-B plus one saved-weight docNLL panel pass only',
            docnll='one saved-mode pass, 8 documents / 198295 targets; exact frozen scoring functions; trajectory-only, no numeric bounds',
            read_pin='fresh complete-checkpoint copy and full SHA256 before any evaluation',
            repetitions=1, checkpoint_promotion=False),
        exploratory_question="Was arm B's case-11 boundary collision dose-related?",
        predictions={
            'a': 'If the chill run (2e-6) reads through all 14 Stage-B cases with zero boundary collisions, the collision is attributable to the 1e-5 modulation amplitude - dose-related, not recipe-fundamental.',
            'b': 'If collisions still appear (any amplitude), the anneal perturbs the scratchpad/public boundary in an amplitude-independent way - a genuine behavioral finding about the recipe class.',
            'c': 'Stage-B valid/correct counts are exploratory observations only; historical parent (12/6/1, 13/7/0, 12/6/1) and arm A (12/5/0) printed alongside; no verdict applied.'},
        interpretation_limit='The requested (a)/(b) classification is a registered exploratory hypothesis reading, not causal identification from one seed/read; the lexical guard can flag cross-turn public/private text reuse.',
        no_gate_verdict=True, no_scalar_thresholds=True, no_retry=True, no_best_selection=True,
        pi_payload_files_verified=result.stdout.count(': OK\n')))
    print('W4_REGISTERED: 512 matched batches; W3 identities unchanged; W4-only trip-and-continue')


def prepare(registration):
    verify_registration(registration)
    train = RUN / 'training'
    complete = json.loads((train / 'complete-receipt.json').read_text())
    if (complete['status'], complete['updates'], complete['inputs'], complete['targets']) != (
            'complete', 512, 3571182, 154386):
        raise ValueError('incomplete fixed training')
    checked(train / 'training.jsonl', complete['training_log_sha256'])
    checked(train / 'start-receipt.json', complete['start_receipt_sha256'])
    directory = RUN / 'reads'
    checkpoint = pin(Path(complete['checkpoint']), directory / 'checkpoint-pinned', complete['checkpoint_sha256'])
    publish(directory / 'read-identity.json', dict(checkpoint=str(checkpoint), sha256=complete['checkpoint_sha256'],
        pin_receipt_sha256=file_sha(directory / 'checkpoint-pinned/pin-receipt.json'),
        registration_sha256=file_sha(registration), primary_weight_mode='train', saved_mode='trajectory-only',
        exploratory_only=True, stage_b_passes=1, execution_replicas=0))


def read_identity():
    identity = json.loads((RUN / 'reads/read-identity.json').read_text())
    checked(identity['checkpoint'], identity['sha256'])
    checked(RUN / 'reads/checkpoint-pinned/pin-receipt.json', identity['pin_receipt_sha256'])
    return identity


def docnll(registration):
    import torch
    from ndm.e97 import load_e97_checkpoint

    verify_registration(registration)
    identity = read_identity()
    freeze = json.loads((W3 / 'runtime-freeze-v2.json').read_text())
    old = json.loads(Path(freeze['docnll']['bindings'][0]['path']).read_text())
    spec = importlib.util.spec_from_file_location('w4_frozen_docnll', ROOT / 'scripts/score_docnll_gpu.py')
    scorer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scorer)
    checked(old['panel'], old['panel_sha256'])
    checked(old['args_json'], old['args_sha256'])
    rows = scorer.load_panel(Path(old['panel']))
    if len(rows) != 8 or sum(len(row['token_ids'])-1 for row in rows) != 198295:
        raise ValueError('docNLL panel coverage')
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    loaded = load_e97_checkpoint(identity['checkpoint'], args_json=old['args_json'], device='cuda:0',
        dtype=torch.bfloat16, weight_mode='saved', use_triton=True, mmap=True)
    loaded.model.eval()
    records = []
    for row in rows:
        records.append(scorer.score_document(loaded.model, row['token_ids'], row['id'], torch.device('cuda:0')))
        print('W4_DOCNLL', row['id'], records[-1]['mean_nll'], flush=True)
    total = sum(row['nll_sum'] for row in records)
    publish(RUN / 'reads/docnll.json', dict(schema='emender-w4-docnll-v1', trajectory_only=True,
        checkpoint_sha256=identity['sha256'], weight_mode='saved', panel_sha256=old['panel_sha256'],
        args_sha256=old['args_sha256'], records=records, scored_tokens=198295, mean_nll=total/198295,
        scorer_sha256=file_sha(ROOT / 'scripts/score_docnll_gpu.py')))


def classification(case_count, collision_cases):
    if case_count != 14 or not 0 <= collision_cases <= case_count:
        raise ValueError('exploratory classification requires all 14 cases')
    return 'a' if collision_cases == 0 else 'b'


def summarize(registration):
    verify_registration(registration)
    identity = read_identity()
    read = RUN / 'reads'
    stage = json.loads((read / 'stage-b/summary.json').read_text())
    plan_path = read / 'stage-b-plan/plan-private.json'
    plan = json.loads(plan_path.read_text())
    frozen = json.loads((W3 / 'preflight.json').read_text())['instruments']['stage_b']
    for key, value in frozen.items():
        if plan.get(key) != value:
            raise ValueError(f'Stage-B instrument divergence: {key}')
    if (stage['checkpoint_sha256'] != identity['sha256'] or
            stage['plan_sha256'] != file_sha(plan_path) or
            [row['id'] for row in stage['rows']] != frozen['case_ids']):
        raise ValueError('Stage-B identity/coverage')
    after = json.loads((read / 'stage-b/model-after-private.json').read_text())
    if not all(after[key] for key in ('unchanged', 'no_gradients', 'all_parameters_bf16')):
        raise ValueError('Stage-B mutation/precision')
    collisions = [json.loads(line) for line in (read / 'stage-b/boundary-trips.jsonl').read_text().splitlines()]
    if collisions != [trip for row in stage['rows'] for trip in row['boundary_trips']]:
        raise ValueError('trip log differs from case records')
    collision_cases = sum(row['boundary_collision'] for row in stage['rows'])
    nll = json.loads((read / 'docnll.json').read_text())
    if (nll['checkpoint_sha256'], nll['scored_tokens'], nll['weight_mode']) != (identity['sha256'], 198295, 'saved'):
        raise ValueError('docNLL identity')
    publish(read / 'exploratory-observations.json', dict(schema='emender-w4-observations-v1',
        scope='exploratory observations only; no gate verdict or promotion', identity=identity,
        registration_sha256=file_sha(registration), stage_b=stage, boundary_trips=collisions,
        classification=classification(len(stage['rows']), collision_cases), collision_cases=collision_cases,
        invalid=sum(str(row['reason']).startswith('invalid_') for row in stage['rows']),
        historical_parent=['12/6/1','13/7/0','12/6/1'], historical_A='12/5/0',
        execution='optional exploratory replica omitted', docnll=nll,
        summary_sha256=file_sha(read / 'stage-b/summary.json'),
        trip_log_sha256=file_sha(read / 'stage-b/boundary-trips.jsonl')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('register','verify','prepare','docnll','summarize'))
    parser.add_argument('--registration', type=Path, default=RECEIPTS / 'preregistration.json')
    args = parser.parse_args()
    {'register':register, 'verify':verify_registration, 'prepare':prepare,
     'docnll':docnll, 'summarize':summarize}[args.command](args.registration)


if __name__ == '__main__':
    main()
