#!/usr/bin/env python3
"""Prepare/run distinct LR trials from the retained, successful smoke launcher.

No retries or promotion. Numerical training code is unchanged; only the LR and
isolated output/cache paths change. Evaluation reuses exact frozen examples.
"""
import argparse
import json
import math
from pathlib import Path
import shlex
import shutil

from ndm.data.masked_sft_dataset import sha256
from ndm.e97_atomic import publish_bytes_no_replace


def publish(path,value):
    publish_bytes_no_replace(path,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode(),mode=0o400)


def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('launcher anchor missing or ambiguous: '+old)
    return text.replace(old,new,1)


def render_trial(script,baseline,root,rate):
    if not math.isfinite(rate) or rate<=0:raise ValueError('invalid learning rate')
    script=replace_once(script,f'ROOT={baseline}\n',f'ROOT={shlex.quote(str(root))}\n')
    script=replace_once(script,'--lr 5e-5 --warmup-steps 0',f'--lr {rate!r} --warmup-steps 0')
    script=replace_once(script,'eval "$(bash scripts/gpu_lease.sh acquire 8 --no-wait)"',
                        'LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)\neval "$LEASE"')
    anchor=" assert checkpoint['sft_precision']['optimizer']=='bf16-sr-candidate'"
    script=replace_once(script,anchor,anchor+"\n assert checkpoint['sft_precision']['learning_rate']==recipe['lr']")
    script=replace_once(script," updates=8,world_size=8,assistant_target_tokens=", " learning_rate=recipe['lr'],updates=8,world_size=8,assistant_target_tokens=")
    return script


def verify_inventory(root):
    for line in (root/'inventory.sha256').read_text().splitlines():
        digest,path=line.split(maxsplit=1)
        if sha256(Path(path.strip()))!=digest:raise ValueError('retained launch identity mismatch')


def prepare(args):
    config=json.loads(args.config.read_text())
    if config['schema']!='emender-e97-native-lr-screen-v1':raise ValueError('screen schema')
    rates=config['new_trial_learning_rates_in_order']
    if len(rates)!=5 or len(set(rates))!=5 or config['baseline']['learning_rate'] in rates or any(not math.isfinite(v) or v<=0 for v in rates):
        raise ValueError('expected five distinct positive new rates')
    baseline=Path(config['baseline']['evidence_root']);verify_inventory(baseline)
    base=json.loads((baseline/'recipe.json').read_text())
    if sha256(baseline/'recipe.json')!=config['training_recipe_sha256'] or base['parent_sha256']!=config['parent_sha256']:
        raise ValueError('baseline recipe binding')
    result=json.loads((baseline/'summary.json').read_text())
    if result['status']!='passed' or result['checkpoints'][-1]['sha256']!=config['baseline']['checkpoint_sha256']:
        raise ValueError('baseline completion binding')
    evaluation=Path(config['evaluation']['panel_authority']).parent
    panel=json.loads((evaluation/'panel.json').read_text());evaluated=json.loads((evaluation/'results/summary.json').read_text())
    if evaluated['status']!='passed' or evaluated['panel_sha256']!=sha256(evaluation/'panel.json'):
        raise ValueError('baseline evaluation binding')
    args.root.mkdir(parents=True,exist_ok=False)
    trials=[]
    for index,rate in enumerate(config['new_trial_learning_rates_in_order']):
        root=args.root/f'trial-{index+1}';root.mkdir()
        (root/'worktree').symlink_to(baseline/'worktree',target_is_directory=True)
        for name in ('source.sha256','input.sha256'):shutil.copyfile(baseline/name,root/name)
        recipe={**base,'schema':'emender-e97-native-lr-trial-v1','lr':rate,
                'purpose':'exposure-matched LR screen; no automatic promotion',
                'screen_config_sha256':sha256(args.config),'baseline_recipe_sha256':config['training_recipe_sha256']}
        publish(root/'recipe.json',recipe)
        script=render_trial((baseline/'run.sh').read_text(),baseline,root,rate)
        (root/'run.sh').write_text(script);(root/'run.sh').chmod(0o500)
        for name in ('source.sha256','input.sha256'):(root/name).chmod(0o400)
        (root/'inventory.sha256').write_text(''.join(f'{sha256(root/name)}  {root/name}\n' for name in ('recipe.json','run.sh','source.sha256','input.sha256')))
        (root/'inventory.sha256').chmod(0o400)
        trials.append(dict(root=str(root),rate=rate))
    # Two fresh rates per evaluation. Last pair repeats the existing control,
    # retained separately rather than replacing the original baseline result.
    batches=[[0,1],[2,3],[4,'baseline']]
    manifest=dict(schema=config['schema'],config=config,config_sha256=sha256(args.config),
        baseline_evaluation_root=str(evaluation),panel_sha256=sha256(evaluation/'panel.json'),
        trials=trials,batches=batches,baseline_root=str(baseline),baseline_rate=base['lr'])
    publish(args.root/'manifest.json',manifest)
    publish(args.root/'base-panel.json',panel)
    (args.root/'eval-source').symlink_to(evaluation/'worktree',target_is_directory=True)


def prepare_eval(args):
    manifest=json.loads((args.root/'manifest.json').read_text())
    panel=json.loads((args.root/'base-panel.json').read_text());models=[]
    for index in manifest['batches'][args.batch]:
        item=dict(root=manifest['baseline_root'],rate=manifest['baseline_rate']) if index=='baseline' else manifest['trials'][index]
        root=Path(item['root']);verify_inventory(root)
        summary=json.loads((root/'summary.json').read_text());recipe=json.loads((root/'recipe.json').read_text())
        if summary['status']!='passed' or summary['recipe_sha256']!=sha256(root/'recipe.json') or recipe['lr']!=item['rate']:
            raise ValueError('trial evidence binding')
        if summary['assistant_target_tokens']!=manifest['config']['assistant_targets_per_trial'] or summary['source_target_totals']['native']!=manifest['config']['native_agent_targets_per_trial']:
            raise ValueError('exposure mismatch')
        checkpoint=summary['checkpoints'][-1]
        if sha256(Path(checkpoint['path']))!=checkpoint['sha256']:raise ValueError('terminal checkpoint identity')
        for mode,label in [('train','y'),('saved','x')]:
            models.append(dict(name=f"lr-{item['rate']!r}-{label}",checkpoint=checkpoint['path'],sha256=checkpoint['sha256'],mode=mode))
    panel={**panel,'models':models}
    output=args.root/f'eval-{args.batch+1}';output.mkdir(exist_ok=False);(output/'results').mkdir()
    publish(output/'panel.json',panel)


def finish(args):
    manifest=json.loads((args.root/'manifest.json').read_text());results={}
    original=json.loads((Path(manifest['baseline_evaluation_root'])/'results/summary.json').read_text())
    for batch in range(3):
        root=args.root/f'eval-{batch+1}';panel=json.loads((root/'panel.json').read_text())
        report=json.loads((root/'results/summary.json').read_text())
        if report['status']!='passed' or report['panel_sha256']!=sha256(root/'panel.json'):raise ValueError('evaluation identity')
        if panel['examples']!=json.loads((args.root/'base-panel.json').read_text())['examples']:raise ValueError('changed evaluation examples')
        if set(results)&set(report['models']):raise ValueError('duplicate measured model')
        results.update(report['models'])
    publish(args.root/'summary.json',dict(schema=manifest['schema'],status='measurements-complete',
        screen_manifest_sha256=sha256(args.root/'manifest.json'),measurements_including_control_repeat=results,
        original_baseline_measurements=original['models'],baseline_repeat_is_not_a_replacement=True,
        selected_learning_rate=None,checkpoint_promotion=False,
        scope='bounded LR screening; same examples and exposure; not task execution or independent generalization'))
    print('LR_SCREEN_MEASUREMENTS_COMPLETE',flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    for name in ('prepare','prepare-eval','finish'):
        q=sub.add_parser(name);q.add_argument('--root',type=Path,required=True)
        if name=='prepare':q.add_argument('--config',type=Path,required=True)
        if name=='prepare-eval':q.add_argument('--batch',type=int,choices=(0,1,2),required=True)
    a=p.parse_args();{'prepare':prepare,'prepare-eval':prepare_eval,'finish':finish}[a.command](a)


if __name__=='__main__':main()
