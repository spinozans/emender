#!/usr/bin/env python3
"""New32-update tranche using the exercised trainer and matched48-case evaluation."""
import argparse
import math
from pathlib import Path
from scripts.prepare_e97_grounding_correction import prepare as prepare_stage,read,R
from scripts.prepare_e97_native_lr_screen import verify_inventory
from scripts.eval_e97_native_execution import publish,sha

EXECUTION_PANEL=R/'grounding-correction-v1-train/evaluation/execution/panel.json'
EXECUTION_SHA='7f03244fca9437ebdfbc7044287c9ddeff61d1953c8a80f062032e46f097ad67'
LEARNING_PANEL=R/'native-training-50m-agent-lr1e5-v1/segment-000880/evaluation/panel.json'
LEARNING_SHA='fe2d10d97c2ee6d31a35684b043f3e4bceaed49782f597612e38b3ca89935aad'
MODEL_NAMES=('pre-y','pre-x','expansion-y','expansion-x')


def prepare(a):
    if sha(EXECUTION_PANEL)!=EXECUTION_SHA or sha(LEARNING_PANEL)!=LEARNING_SHA:raise ValueError('frozen evaluation inputs')
    config=read(a.recipe)
    if config['schema']!='emender-e97-grounded-expansion-recipe-v1' or config['weight_mode']!='train' or config['experimental_precision_policy'] is not None:raise ValueError('tranche baseline')
    audit=read(a.data/'result-audit.json')
    if audit['passed'] is not True or audit['authority_sha256']!=sha(a.data/'authority/manifest.json'):raise ValueError('independent derivative audit required')
    if sha(a.data/'fresh-evaluation-cases.json')!='36921d4b718c3d8e0bf88c9fc1cc04ae15ada219229442eb420e5843a84c9331':raise ValueError('pre-training evaluation freeze')
    prepare_stage(a)


def evaluations(a):
    verify_inventory(a.phase);training=read(a.phase/'summary.json');recipe=read(a.phase/'recipe.json');config=recipe['correction_config']
    if training['status']!='passed' or training['recipe_sha256']!=sha(a.phase/'recipe.json'):raise ValueError('training receipt')
    target=training['checkpoint']
    if sha(target['path'])!=target['sha256'] or sha(config['parent_checkpoint'])!=config['parent_sha256']:raise ValueError('checkpoint binding')
    if sha(EXECUTION_PANEL)!=EXECUTION_SHA or sha(LEARNING_PANEL)!=LEARNING_SHA:raise ValueError('evaluation inputs changed')
    models=[]
    for name in MODEL_NAMES:
        checkpoint=config['parent_checkpoint'] if name.startswith('pre-') else target['path']
        digest=config['parent_sha256'] if name.startswith('pre-') else target['sha256']
        models.append(dict(name=name,checkpoint=checkpoint,sha256=digest,mode='train' if name.endswith('-y') else 'saved'))
    learning=read(LEARNING_PANEL)
    for key in ('current_update','program_sha256','schedule_sha256','evaluation_plan_sha256'):learning.pop(key,None)
    learning.update(models=models,source_panel_sha256=LEARNING_SHA,training_summary_sha256=sha(a.phase/'summary.json'),scope='Unchanged development/conversation/tool examples; matched correction-parent and expansion x/y')
    execution=read(EXECUTION_PANEL)
    if sha(a.data/'fresh-evaluation-cases.json')!=recipe['fresh_evaluation_cases_sha256']:raise ValueError('frozen fresh/transfer cases changed')
    cases=[dict(c,id='regression-'+c['id'],cohort='regression') for c in execution['cases']]+read(a.data/'fresh-evaluation-cases.json')
    if len(cases)!=48 or len({c['id'] for c in cases})!=48 or any(c.get('supplied_calls') for c in cases):raise ValueError('unassisted evaluation coverage')
    execution.pop('training_completion_sha256',None)
    execution.update(models=models,cases=cases,source_panel_sha256=EXECUTION_SHA,training_summary_sha256=sha(a.phase/'summary.json'),scope='Matched parent/expansion:16 regression,16 fresh values,16 structural variants; no independent repository claim')
    out=a.phase/'evaluation';out.mkdir()
    for name,panel in (('execution',execution),('learning',learning)):
        (out/name).mkdir();publish(out/name/'panel.json',panel)
    print('GROUNDED_EXPANSION_EVALUATIONS_PREPARED',flush=True)


def gate_checks(execution,learning,panel):
    cases={c['id']:c for c in panel['cases']}
    if len(cases)!=48 or len(panel['cases'])!=48:raise ValueError('panel case coverage')
    cohorts={name:{i for i,c in cases.items() if c['cohort']==name} for name in ('regression','fresh','transfer')}
    if any(len(ids)!=16 for ids in cohorts.values()) or set.union(*cohorts.values())!=set(cases):raise ValueError('cohort coverage')
    cohorts['new-edit-recovery']={i for i,c in cases.items() if c['cohort']!='regression' and c['family'] in ('edit','recovery')}
    if len(cohorts['new-edit-recovery'])!=16:raise ValueError('edit/recovery coverage')
    if set(execution['models'])!=set(MODEL_NAMES) or set(learning['models'])!=set(MODEL_NAMES):raise ValueError('matched model coverage')
    counts={}
    for model in MODEL_NAMES:
        outcomes=execution['models'][model]['outcomes']
        if len(outcomes)!=48 or {r['id'] for r in outcomes}!=set(cases) or any(type(r['grade']['success']) is not bool for r in outcomes):raise ValueError('execution outcome coverage/type')
        counts[model]={name:sum(r['grade']['success'] for r in outcomes if r['id'] in ids) for name,ids in cohorts.items()}
    prior=learning['models']['pre-y'];checks=[]
    for model in ('expansion-y','expansion-x'):
        for cohort,metric,margin,minimum in (('tool-retention','token_accuracy',.98,True),('conversation-retention','record_macro_nll',.15,False),('native-development','record_macro_nll',.10,False)):
            baseline=prior[cohort]['metrics']['assistant'][metric];value=learning['models'][model][cohort]['metrics']['assistant'][metric]
            if not math.isfinite(baseline) or not math.isfinite(value):raise ValueError('nonfinite retention measurement')
            limit=margin if minimum else baseline+margin
            checks.append(dict(model=model,cohort=cohort,metric=metric,value=value,limit=limit,passed=value>=limit if minimum else value<=limit))
    before=counts['pre-y'];after=counts['expansion-y']
    limits=dict(regression=before['regression'],fresh=min(16,max(8,before['fresh']+2)),transfer=min(16,max(4,before['transfer']+2)),**{'new-edit-recovery':min(16,before['new-edit-recovery']+2)})
    for cohort,limit in limits.items():checks.append(dict(model='expansion-y',cohort=cohort,metric='autonomous_completions',baseline=before[cohort],value=after[cohort],limit=limit,passed=after[cohort]>=limit))
    return checks,counts


def gate(a):
    out=a.phase/'evaluation';execution=read(out/'execution/summary.json');learning=read(out/'learning/summary.json')
    for name,report in (('execution',execution),('learning',learning)):
        if report['panel_sha256']!=sha(out/name/'panel.json'):raise ValueError('evaluation summary binding')
    checks,counts=gate_checks(execution,learning,read(out/'execution/panel.json'))
    result=dict(schema='emender-e97-grounded-expansion-evaluation-v1',status='measurements-complete',positive_expansion_evidence=all(c['passed'] for c in checks),checks=checks,counts=counts,
        automatic_expansion=False,checkpoint_promotion=False,independent_repository_claim=False,
        execution_summary_sha256=sha(out/'execution/summary.json'),learning_summary_sha256=sha(out/'learning/summary.json'),training_summary_sha256=sha(a.phase/'summary.json'))
    publish(out/'gate.json',result);print('GROUNDED_EXPANSION_GATE',result['positive_expansion_evidence'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--recipe',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--phase',type=Path,required=True)
    a=s.add_parser('evaluations');a.add_argument('--data',type=Path,required=True);a.add_argument('--phase',type=Path,required=True)
    a=s.add_parser('gate');a.add_argument('--phase',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
