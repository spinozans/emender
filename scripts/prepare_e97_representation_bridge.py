#!/usr/bin/env python3
"""Freeze candidate cases and qualify a small native teacher slice, without training."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import signal
import tiktoken
from scripts.e97_representation_bridge import bridge_cases
from scripts.e97_grounded_curriculum import teacher_record
from scripts.audit_e97_grounded_expansion import check_teacher,reconstruct
from scripts.e97_native_execution_sandbox import NativeSandbox,validate_container
from scripts.eval_e97_native_execution import publish,sha


def read(p):return json.loads(Path(p).read_text())

def freeze(a):
    config=read(a.recipe)
    if config['optimizer_updates_authorized_by_this_preparation']!=0 or config['training_eligible'] is not False or config['native_preflight_records']!=48:raise ValueError('preparation only')
    for field in ('generation_panel','previous_evaluation','previous_training_cases'):
        if sha(config[field])!=config[field+'_sha256']:raise ValueError('source identity: '+field)
    train=bridge_cases(config['training_seed'],config['training_pairs_per_family'],'train')
    fresh=bridge_cases(config['fresh_seed'],config['fresh_pairs_per_family'],'fresh')
    composition=bridge_cases(config['composition_seed'],config['composition_pairs_per_family'],'composition')
    old=read(config['previous_evaluation'])['cases'];old_train=read(config['previous_training_cases'])
    known_answers={c['answer'] for c in old+old_train if c['family']!='edit'}
    train_answers={c['answer'] for c in train if c['family']!='edit'}
    if any(c['answer'] in known_answers or c['answer'] in train_answers for c in fresh+composition if c['family']!='edit'):raise ValueError('evaluation answer overlap')
    old_eval_answers={c['answer'] for c in old if c['family']!='edit'}
    if any(c['answer'] in old_eval_answers for c in train if c['family']!='edit'):raise ValueError('old evaluation answer in training')
    groups=[train,fresh,composition]
    if list(map(len,groups))!=[768,32,16]:raise ValueError('frozen candidate counts')
    path_sets=[{p for c in group for p in c['files']} for group in groups]
    if any(path_sets[i]&path_sets[j] for i in range(3) for j in range(i)):raise ValueError('cross-cohort paths')
    for group in groups:
        for x,y in zip(group[::2],group[1::2]):
            if x['prompt']!=y['prompt'] or set(x['files'])!=set(y['files']) or x['files']==y['files']:raise ValueError('counterfactual pairing')
            if x['family']!='edit' and (x['answer']==y['answer'] or x['answer'] in x['prompt'] or y['answer'] in y['prompt']):raise ValueError('observation-dependent answers')
            if x['family']=='edit' and x['expected_output']==y['expected_output']:raise ValueError('observation-dependent edits')
    selected=set(config['preflight_pair_indices'])
    preflight=[c for c in bridge_cases(config['preflight_seed'],96,'train') if int(c['id'].split('-world-')[0].rsplit('-',1)[1]) in selected]
    if len(preflight)!=48:raise ValueError('bounded preflight')
    extra_answers={c['answer'] for c in preflight if c['family']!='edit'}
    if extra_answers & {c['answer'] for c in fresh+composition if c['family']!='edit'}:raise ValueError('preflight/evaluation overlap')
    if {c['answer'] for c in fresh if c['family']!='edit'} & {c['answer'] for c in composition if c['family']!='edit'}:raise ValueError('evaluation cohort overlap')
    a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    for name,rows in (('training-candidates-private.json',train),('fresh-evaluation-private.json',fresh),('composition-evaluation-private.json',composition),('native-preflight-cases-private.json',preflight)):publish(a.output/name,rows)
    publish(a.output/'plan.json',dict(schema=config['schema'],recipe=config,recipe_sha256=sha(a.recipe),
        case_files={p.name:sha(p) for p in a.output.glob('*-private.json')},training_candidates=768,fresh_evaluation=32,composition_evaluation=16,
        native_preflight=48,training_eligible=False,optimizer_updates=0,scope='Candidate preparation; neither trained capability nor data admission'))
    print('REPRESENTATION_BRIDGE_CASES_FROZEN',flush=True)


def native(a):
    plan=read(a.output/'plan.json');config=plan['recipe']
    if sha(a.output/'plan.json')!=a.plan_sha:raise ValueError('plan identity')
    for name,digest in plan['case_files'].items():
        if sha(a.output/name)!=digest:raise ValueError('frozen cases changed')
    if sha(config['generation_panel'])!=config['generation_panel_sha256']:raise ValueError('native panel')
    panel=read(config['generation_panel']);cases=read(a.output/'native-preflight-cases-private.json');enc=tiktoken.get_encoding('p50k_base')
    count=targets=0;families=Counter()
    with (a.output/'native-receipts-private.jsonl').open('x') as f:
        for world in (0,1):
            selected=[c for c in cases if c['variant']==world];files={}
            for c in selected:
                if set(files)&set(c['files']):raise ValueError('fixture collision')
                files.update(c['files'])
            with NativeSandbox(panel,a.output/f'world-{world}') as sandbox:
                sandbox.request('setup',files=files)
                for case in selected:
                    record,receipt=teacher_record(case,panel,panel['models'][1]['system_message'],sandbox,enc)
                    check_teacher(case,record,receipt,panel);reconstruct(record,panel['tools'],enc)
                    if sum(m['role']=='assistant' for m in record['messages'])>8:raise ValueError('teacher exceeds autonomous turn budget')
                    f.write(json.dumps(dict(id=case['id'],record=record,receipt=receipt),sort_keys=True)+'\n');f.flush()
                    count+=1;targets+=record['targets'];families[case['family']]+=1
            folder=a.output/f'world-{world}';before=read(folder/'container-before.json');cleanup=read(folder/'cleanup.json')
            validate_container(before,panel['image_id'],before['Config']['Labels']['emender.native-qualification'])
            if cleanup!={'container_id':before['Id'],'removed':True}:raise ValueError('owned cleanup')
        f.flush();os.fsync(f.fileno())
    if count!=48 or dict(families)!=dict(lookup=12,sum=12,edit=12,recovery=12):raise ValueError('qualification coverage')
    publish(a.output/'native-summary.json',dict(native_teacher_qualification_passed=True,records=count,targets=targets,families=dict(families),
        actual_calls_and_observations_bound=True,independent_masks_and_oracles=True,owned_sandboxes_cleaned=2,
        plan_sha256=a.plan_sha,receipts_sha256=sha(a.output/'native-receipts-private.jsonl'),training_eligible=False,optimizer_updates=0,autonomous_model_evaluations=0))
    print('REPRESENTATION_BRIDGE_NATIVE_QUALIFIED',count,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('freeze');p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('native');p.add_argument('--output',type=Path,required=True);p.add_argument('--plan-sha',required=True)
    args=parser.parse_args()
    def terminate(*unused):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate)
    globals()[args.command](args)
