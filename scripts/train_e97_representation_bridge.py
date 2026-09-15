#!/usr/bin/env python3
"""Frozen32-update bridge controller using the unchanged exercised SFT trainer."""
import argparse
from collections import Counter
import math
from pathlib import Path
from scripts.build_e97_representation_bridge import bound_inputs,read
from scripts.prepare_e97_grounded_expansion import exposure as exposure_audit,LEARNING_PANEL,LEARNING_SHA
from scripts.prepare_e97_grounding_correction import prepare as prepare_stage
from scripts.prepare_e97_native_lr_screen import verify_inventory
from scripts.eval_e97_native_execution import publish,sha

MODELS=('pre-y','pre-x','bridge-y','bridge-x')
COHORTS={'prior-regression':16,'prior-fresh':16,'prior-transfer':16,'fresh':32,'composition':16}


def exposure(a):exposure_audit(a,authored_source='representation-bridge')


def prepare(a):
    config=read(a.recipe);bound_inputs(config)
    audit=read(a.data/'result-audit.json');e=read(a.data/'exposure-audit.json')
    if audit['passed'] is not True or audit['authority_sha256']!=sha(a.data/'authority/manifest.json'):raise ValueError('independent data audit')
    if e['passed'] is not True or e['schedule_sha256']!=sha(a.data/'expected-schedule.json') or e['unique_records']!=e['available_records']:raise ValueError('complete unique scheduled coverage')
    if any(v['unique_records']!=192 for v in e['authored_family_coverage'].values()) or len(e['authored_family_coverage'])!=4 or e['sources']['grounded-rehearsal']['unique_records']!=512:raise ValueError('curriculum exposure')
    if sha(LEARNING_PANEL)!=LEARNING_SHA:raise ValueError('retention freeze')
    prepare_stage(a)


def evaluation_cases(config,data):
    plan,_=bound_inputs(config);root=Path(config['preparation_root'])
    new=read(root/'fresh-evaluation-private.json')+read(root/'composition-evaluation-private.json')
    if read(data/'fresh-evaluation-cases.json')!=new:raise ValueError('new evaluation freeze')
    original=read(plan['recipe']['previous_evaluation'])
    cases=[dict(c,cohort='prior-'+c['cohort']) for c in original['cases']]+new
    if Counter(c['cohort'] for c in cases)!=COHORTS or len({c['id'] for c in cases})!=96 or any(c.get('supplied_calls') for c in cases):raise ValueError('unassisted frozen coverage')
    return original,cases


def evaluations(a):
    verify_inventory(a.phase);training=read(a.phase/'summary.json');recipe=read(a.phase/'recipe.json');config=recipe['correction_config']
    if training['status']!='passed' or training['updates']!=32 or training['recipe_sha256']!=sha(a.phase/'recipe.json'):raise ValueError('training receipt')
    target=training['checkpoint']
    if sha(target['path'])!=target['sha256'] or sha(config['parent_checkpoint'])!=config['parent_sha256']:raise ValueError('checkpoint identity')
    if sha(a.data/'fresh-evaluation-cases.json')!=recipe['fresh_evaluation_cases_sha256'] or sha(LEARNING_PANEL)!=LEARNING_SHA:raise ValueError('evaluation inputs')
    models=[dict(name=name,checkpoint=config['parent_checkpoint'] if name.startswith('pre-') else target['path'],
        sha256=config['parent_sha256'] if name.startswith('pre-') else target['sha256'],mode='train' if name.endswith('-y') else 'saved') for name in MODELS]
    execution,cases=evaluation_cases(config,a.data)
    for key in ('training_completion_sha256','training_summary_sha256'):execution.pop(key,None)
    execution.update(models=models,cases=cases,training_summary_sha256=sha(a.phase/'summary.json'),scope='48 prior regression/development plus32 fresh and16 composition cases; no repository independence claim')
    learning=read(LEARNING_PANEL)
    for key in ('current_update','program_sha256','schedule_sha256','evaluation_plan_sha256'):learning.pop(key,None)
    learning.update(models=models,source_panel_sha256=LEARNING_SHA,training_summary_sha256=sha(a.phase/'summary.json'),scope='Unchanged retention/development panel; matched expansion-parent and bridge x/y')
    out=a.phase/'evaluation';out.mkdir()
    for name,panel in (('execution',execution),('learning',learning)):
        (out/name).mkdir();publish(out/name/'panel.json',panel)
    print('REPRESENTATION_BRIDGE_EVALUATIONS_PREPARED',flush=True)


def gate_checks(execution,learning,panel):
    if len(panel['cases'])!=96 or len({c['id'] for c in panel['cases']})!=96 or Counter(c['cohort'] for c in panel['cases'])!=COHORTS:raise ValueError('case coverage')
    cases={c['id']:c for c in panel['cases']}
    if any(c['family'] not in ('lookup','sum','edit','recovery') or c.get('supplied_calls') for c in cases.values()):raise ValueError('unassisted families')
    cohorts={name:{i for i,c in cases.items() if c['cohort']==name} for name in COHORTS}
    for family in ('lookup','sum','edit','recovery'):
        cohorts['fresh-'+family]={i for i,c in cases.items() if c['cohort']=='fresh' and c['family']==family}
        if len(cohorts['fresh-'+family])!=8:raise ValueError('fresh family coverage')
    if set(execution['models'])!=set(MODELS) or set(learning['models'])!=set(MODELS):raise ValueError('matched model coverage')
    counts={}
    for name in MODELS:
        outcomes=execution['models'][name]['outcomes']
        if len(outcomes)!=96 or {r['id'] for r in outcomes}!=set(cases) or any(type(r['grade']['success']) is not bool for r in outcomes):raise ValueError('outcome coverage/type')
        counts[name]={cohort:sum(r['grade']['success'] for r in outcomes if r['id'] in ids) for cohort,ids in cohorts.items()}
        for cohort,metric in (('tool-retention','token_accuracy'),('conversation-retention','record_macro_nll'),('native-development','record_macro_nll')):
            value=learning['models'][name][cohort]['metrics']['assistant'][metric]
            if type(value) not in (int,float) or not math.isfinite(value) or value<0 or (metric=='token_accuracy' and value>1):raise ValueError('finite valid retention metrics')
    before=counts['pre-y'];after=counts['bridge-y'];checks=[]
    limits={'prior-regression':7,'prior-fresh':16,'prior-transfer':4,'fresh':min(32,max(24,before['fresh']+4)),
        'composition':min(16,max(4,before['composition']+2)),**{'fresh-'+family:4 for family in ('lookup','sum','edit','recovery')}}
    for cohort,limit in limits.items():checks.append(dict(model='bridge-y',cohort=cohort,metric='autonomous_completions',baseline=before[cohort],value=after[cohort],limit=limit,passed=after[cohort]>=limit))
    for name in ('bridge-y','bridge-x'):
        for cohort,metric,margin,minimum in (('tool-retention','token_accuracy',.98,True),('conversation-retention','record_macro_nll',.15,False),('native-development','record_macro_nll',.10,False)):
            baseline=learning['models']['pre-y'][cohort]['metrics']['assistant'][metric];value=learning['models'][name][cohort]['metrics']['assistant'][metric];limit=margin if minimum else baseline+margin
            checks.append(dict(model=name,cohort=cohort,metric=metric,baseline=baseline,value=value,limit=limit,passed=value>=limit if minimum else value<=limit))
    return checks,counts


def gate(a):
    out=a.phase/'evaluation';execution=read(out/'execution/summary.json');learning=read(out/'learning/summary.json')
    for name,report in (('execution',execution),('learning',learning)):
        if report['panel_sha256']!=sha(out/name/'panel.json'):raise ValueError('panel/report binding')
    checks,counts=gate_checks(execution,learning,read(out/'execution/panel.json'))
    publish(out/'gate.json',dict(schema='emender-e97-representation-bridge-evaluation-v1',status='measurements-complete',positive_bridge_evidence=all(c['passed'] for c in checks),checks=checks,counts=counts,
        automatic_expansion=False,checkpoint_promotion=False,independent_repository_claim=False,execution_summary_sha256=sha(out/'execution/summary.json'),
        learning_summary_sha256=sha(out/'learning/summary.json'),training_summary_sha256=sha(a.phase/'summary.json')))
    print('REPRESENTATION_BRIDGE_GATE',all(c['passed'] for c in checks),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('exposure');a.add_argument('--data',type=Path,required=True)
    a=s.add_parser('prepare');a.add_argument('--recipe',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--phase',type=Path,required=True)
    a=s.add_parser('evaluations');a.add_argument('--data',type=Path,required=True);a.add_argument('--phase',type=Path,required=True)
    a=s.add_parser('gate');a.add_argument('--phase',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
