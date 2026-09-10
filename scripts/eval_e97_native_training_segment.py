#!/usr/bin/env python3
"""Bind a completed segment to the frozen panel and continuation policy."""
import argparse
import json
import math
from pathlib import Path
from ndm.data.masked_sft_dataset import sha256
from scripts.prepare_e97_native_lr_screen import publish


def read(path):return json.loads(Path(path).read_text())


def continuation_checks(models,policy):
    """Predeclared non-catastrophic guards, not a capability verdict."""
    parent=models['parent-y'];checks=[]
    for name in ('current-y','current-x'):
        measured=models[name]
        for cohort,metric,limit,direction in [
            ('tool-retention','token_accuracy',policy['minimum_tool_retention_token_accuracy'],'minimum'),
            ('conversation-retention','record_macro_nll',parent['conversation-retention']['metrics']['assistant']['record_macro_nll']+policy['maximum_conversation_nll_increase_over_parent'],'maximum'),
            ('native-development','record_macro_nll',parent['native-development']['metrics']['assistant']['record_macro_nll']+policy['maximum_native_development_nll_increase_over_parent'],'maximum')]:
            value=measured[cohort]['metrics']['assistant'][metric]
            if not math.isfinite(value) or not math.isfinite(limit):raise ValueError('nonfinite continuation metric')
            if value<0 or (metric=='token_accuracy' and value>1):raise ValueError('invalid metric range')
            passed=value>=limit if direction=='minimum' else value<=limit
            checks.append(dict(model=name,cohort=cohort,metric=metric,value=value,limit=limit,direction=direction,passed=passed))
    return checks


def prepare(args):
    phase=args.phase;training=read(phase/'summary.json');recipe=read(phase/'recipe.json');run=Path(recipe['run_root'])
    if training['status']!='passed' or training['recipe_sha256']!=sha256(phase/'recipe.json') or recipe['program_sha256']!=sha256(run/'program.json'):
        raise ValueError('training/program binding')
    if sha256(run/'base-panel.json')!=recipe['base_panel_sha256']:raise ValueError('frozen examples identity')
    if sha256(args.plan)!=args.plan_sha256:raise ValueError('evaluation execution plan identity')
    plan=read(args.plan);program=read(run/'program.json');panel=read(run/'base-panel.json')
    if plan['schema']!='emender-e97-native-long-evaluation-v1' or plan['program_sha256']!=sha256(run/'program.json'):
        raise ValueError('evaluation/program identity')
    current=training['checkpoint'];pilot=plan['pilot_control']
    panel['models']=[dict(name='parent-y',checkpoint=program['parent_checkpoint'],sha256=recipe['parent_sha256'],mode='train'),
        dict(name='pilot-1e-5-y',checkpoint=pilot['path'],sha256=pilot['sha256'],mode='train'),
        dict(name='current-y',checkpoint=current['path'],sha256=current['sha256'],mode='train'),
        dict(name='current-x',checkpoint=current['path'],sha256=current['sha256'],mode='saved')]
    verified=set()
    for model in panel['models']:
        identity=(model['checkpoint'],model['sha256'])
        if identity not in verified and sha256(Path(model['checkpoint']))!=model['sha256']:raise ValueError('model identity')
        verified.add(identity)
    panel.update(training_summary_sha256=sha256(phase/'summary.json'),program_sha256=sha256(run/'program.json'),
        evaluation_plan_sha256=args.plan_sha256,current_update=training['updates'])
    output=phase/'evaluation';output.mkdir(exist_ok=False)
    publish(output/'panel.json',panel)


def gate(args):
    phase=args.phase;root=phase/'evaluation';summary=read(root/'summary.json');panel=read(root/'panel.json')
    recipe=read(phase/'recipe.json');run=Path(recipe['run_root']);program=read(run/'program.json')
    training=read(phase/'summary.json')
    if training['status']!='passed' or training['recipe_sha256']!=sha256(phase/'recipe.json') or recipe['program_sha256']!=sha256(run/'program.json'):
        raise ValueError('training/program binding')
    for name,mode in [('current-y','train'),('current-x','saved')]:
        matches=[m for m in panel['models'] if m['name']==name]
        if len(matches)!=1 or matches[0]['sha256']!=training['checkpoint']['sha256'] or matches[0]['checkpoint']!=training['checkpoint']['path'] or matches[0]['mode']!=mode:
            raise ValueError('current model binding')
    if summary['status']!='passed' or summary['panel_sha256']!=sha256(root/'panel.json'):
        raise ValueError('evaluation completion identity')
    if panel['training_summary_sha256']!=sha256(phase/'summary.json') or panel['program_sha256']!=sha256(run/'program.json'):
        raise ValueError('evaluation training/program identity')
    if panel['examples']!=read(run/'base-panel.json')['examples'] or sha256(run/'base-panel.json')!=recipe['base_panel_sha256']:
        raise ValueError('changed examples')
    if set(summary['models'])!={model['name'] for model in panel['models']}:raise ValueError('model coverage')
    checks=continuation_checks(summary['models'],program['evaluation_policy'])
    accepted=all(c['passed'] for c in checks)
    publish(root/'gate.json',dict(schema='emender-e97-native-long-continuation-gate-v1',
        continue_training=accepted,training_summary_sha256=sha256(phase/'summary.json'),
        program_sha256=sha256(run/'program.json'),evaluation_summary_sha256=sha256(root/'summary.json'),
        panel_sha256=sha256(root/'panel.json'),checks=checks,checkpoint_promotion=False,
        scope='predeclared non-catastrophic x/y continuation guards; no task-execution or independent-generalization verdict'))
    print(json.dumps(dict(continue_training=accepted,checks=checks)),flush=True)
    if not accepted:raise SystemExit(2)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','gate'):
        q=sub.add_parser(name);q.add_argument('--phase',type=Path,required=True)
        if name=='prepare':q.add_argument('--plan',type=Path,required=True);q.add_argument('--plan-sha256',required=True)
    args=p.parse_args();{'prepare':prepare,'gate':gate}[args.command](args)


if __name__=='__main__':main()
