#!/usr/bin/env python3
"""Bind a new 32-update correction stage to the unchanged numerical trainer."""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import shutil
from scripts.eval_e97_native_execution import publish, sha
from scripts.prepare_e97_native_training_segment import render
from scripts.prepare_e97_native_lr_screen import replace_once, verify_inventory

R=Path('/mnt/nvme2n1/erikg/e97_systematic_posttraining')


def read(p):return json.loads(Path(p).read_text())


def prepare(a):
    config=read(a.recipe);data=a.data;phase=a.phase;baseline=R/'native-real-data-smoke-v1'
    verify_inventory(baseline)
    if config['steps']!=32 or config['learning_rate']!=1e-5:raise ValueError('frozen correction budget')
    ds=read(data/'data-summary.json');authority=Path(ds['authority'])
    if ds['authority_sha256']!=sha(authority/'manifest.json') or read(authority/'manifest.json')['recipe_sha256']!=sha(a.recipe):
        raise ValueError('correction data binding')
    if sha(config['parent_checkpoint'])!=config['parent_sha256']:raise ValueError('parent identity')
    expected=read(data/'expected-schedule.json')
    if len(expected['steps'])!=32:raise ValueError('schedule budget')
    phase.mkdir(parents=True,exist_ok=False)
    (phase/'worktree').symlink_to(baseline/'worktree',target_is_directory=True)
    shutil.copyfile(baseline/'source.sha256',phase/'source.sha256')
    controller=Path(__file__).resolve().parents[1];base=read(baseline/'recipe.json')
    recipe={**base,'schema':'emender-e97-grounding-correction-stage-v1','purpose':config['purpose'],
            'parent_sha256':config['parent_sha256'],'lr':1e-5,'steps':32,'start_update':0,'run_root':str(phase),
            'save_every':128,'keep_checkpoints':32,'deadline_seconds':5400,
            'authority_manifest_sha256':ds['authority_sha256'],'pack_manifest_sha256':sha(data/'packs/manifest.json'),
            'schedule':dict(path=str(data/'expected-schedule.json'),sha256=sha(data/'expected-schedule.json'),sampler_key=974223),
            'correction_config':config,'correction_config_sha256':sha(a.recipe),'controller_source':str(controller),
            'fresh_evaluation_cases_sha256':sha(data/'fresh-evaluation-cases.json'),
            'acceptance':['finite complete checkpoint','exact source schedule','separate post-training behavioral and retention measurements']}
    publish(phase/'recipe.json',recipe)
    script=render((baseline/'run.sh').read_text(),baseline,phase,phase,data,authority,ds['authority_sha256'],
                  recipe['pack_manifest_sha256'],32,controller)
    old=re.findall(r'^PARENT=(.*)$',script,re.M)
    if len(old)!=1:raise ValueError('parent launcher anchor')
    script=replace_once(script,'PARENT='+old[0]+'\n','PARENT='+shlex.quote(config['parent_checkpoint'])+'\n')
    script=replace_once(script,base['parent_sha256'],config['parent_sha256'])
    (phase/'run.sh').write_text(script);(phase/'run.sh').chmod(0o500)
    args_path=Path(re.findall(r'^ARGS=(.*)$',script,re.M)[0])
    inputs=[(Path(config['parent_checkpoint']),config['parent_sha256']),(args_path,recipe['source_args_sha256']),
            (authority/'manifest.json',ds['authority_sha256']),(data/'packs/manifest.json',recipe['pack_manifest_sha256']),
            (data/'expected-schedule.json',recipe['schedule']['sha256']),
            (data/'fresh-evaluation-cases.json',recipe['fresh_evaluation_cases_sha256'])]
    (phase/'input.sha256').write_text(''.join(f'{digest}  {p}\n' for p,digest in inputs))
    (phase/'inventory.sha256').write_text(''.join(f'{sha(phase/n)}  {phase/n}\n' for n in ('recipe.json','run.sh','source.sha256','input.sha256')))
    for n in ('source.sha256','input.sha256','inventory.sha256'):(phase/n).chmod(0o400)
    print('CORRECTION_TRAINING_PREPARED',32,'updates',flush=True)


def evaluations(a):
    verify_inventory(a.phase);training=read(a.phase/'summary.json');recipe=read(a.phase/'recipe.json')
    if training['status']!='passed' or training['recipe_sha256']!=sha(a.phase/'recipe.json'):raise ValueError('training receipt')
    target=training['checkpoint']
    if sha(target['path'])!=target['sha256']:raise ValueError('terminal checkpoint identity')
    old=R/'native-training-50m-agent-lr1e5-v1/segment-000880/evaluation/panel.json'
    if sha(old)!='fe2d10d97c2ee6d31a35684b043f3e4bceaed49782f597612e38b3ca89935aad':raise ValueError('retention panel identity')
    learning=read(old);prior=learning['models'][2:]
    models=[{**prior[0],'name':'pre-y'},{**prior[1],'name':'pre-x'},
            {**prior[0],'name':'correction-y','checkpoint':target['path'],'sha256':target['sha256']},
            {**prior[1],'name':'correction-x','checkpoint':target['path'],'sha256':target['sha256']}]
    learning.update(models=models)
    execution=read(R/'native-execution-diagnostic-v2/panel.json')
    cases=[{**c,'id':'old-'+c['id']} for c in execution['cases']]
    if sha(a.data/'fresh-evaluation-cases.json')!=recipe['fresh_evaluation_cases_sha256']:raise ValueError('fresh cases changed')
    cases.extend({**c,'id':'fresh-'+c['id']} for c in read(a.data/'fresh-evaluation-cases.json'))
    execution.update(models=models,cases=cases,scope='Matched pre/post autonomous old and fresh-value same-family probes; no independent task-family claim')
    out=a.phase/'evaluation';out.mkdir()
    for name,panel in (('execution',execution),('learning',learning)):
        (out/name).mkdir();publish(out/name/'panel.json',panel)
    print('CORRECTION_EVALUATIONS_PREPARED',flush=True)


def gate(a):
    out=a.phase/'evaluation';execution=read(out/'execution/summary.json');learning=read(out/'learning/summary.json')
    for name,report in (('execution',execution),('learning',learning)):
        if report['panel_sha256']!=sha(out/name/'panel.json'):
            raise ValueError('evaluation panel binding')
    prior=learning['models']['pre-y'];checks=[]
    for model in ('correction-y','correction-x'):
        for cohort,metric,limit,minimum in (
            ('tool-retention','token_accuracy',.98,True),
            ('conversation-retention','record_macro_nll',prior['conversation-retention']['metrics']['assistant']['record_macro_nll']+.15,False),
            ('native-development','record_macro_nll',prior['native-development']['metrics']['assistant']['record_macro_nll']+.10,False)):
            value=learning['models'][model][cohort]['metrics']['assistant'][metric]
            checks.append(dict(model=model,cohort=cohort,metric=metric,value=value,limit=limit,
                               passed=math.isfinite(value) and (value>=limit if minimum else value<=limit)))
    fresh=[r for r in execution['models']['correction-y']['outcomes'] if r['id'].startswith('fresh-')]
    if len(fresh)!=8:raise ValueError('fresh execution coverage')
    successes=sum(r['grade']['success'] for r in fresh)
    checks.append(dict(model='correction-y',metric='fresh_autonomous_completions',value=successes,limit=4,passed=successes>=4))
    result=dict(schema='emender-e97-grounding-correction-evaluation-v1',status='measurements-complete',
                positive_correction_evidence=all(c['passed'] for c in checks),checks=checks,
                automatic_expansion=False,checkpoint_promotion=False,
                execution_summary_sha256=sha(out/'execution/summary.json'),learning_summary_sha256=sha(out/'learning/summary.json'),
                training_summary_sha256=sha(a.phase/'summary.json'))
    publish(out/'gate.json',result);print(json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('prepare');a.add_argument('--recipe',type=Path,required=True);a.add_argument('--data',type=Path,required=True);a.add_argument('--phase',type=Path,required=True)
    a=s.add_parser('evaluations');a.add_argument('--data',type=Path,required=True);a.add_argument('--phase',type=Path,required=True)
    a=s.add_parser('gate');a.add_argument('--phase',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)
