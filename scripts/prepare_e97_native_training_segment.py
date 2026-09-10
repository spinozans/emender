#!/usr/bin/env python3
"""Frozen successful SFT segments; no failure retries or model promotion."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path
import shlex
import shutil

from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset,SFTSamplerIdentity,sha256
from scripts.prepare_e97_native_lr_screen import publish,replace_once,verify_inventory


def read(path):return json.loads(Path(path).read_text())


def check_steps(events,planned,start,end):
    steps=[e for e in events if e['event']=='step']
    if [e['update'] for e in steps]!=list(range(start+1,end+1)):raise ValueError('update coverage')
    if len([e for e in events if e['event']=='complete'])!=1:raise ValueError('completion coverage')
    tokens=sum(e['global_tokens'] for e in planned[:start]);targets=sum(e['global_targets'] for e in planned[:start])
    for actual,expected in zip(steps,planned[start:end],strict=True):
        for key in ('update','rank_sample_ids','global_tokens','global_targets'):
            if actual[key]!=expected[key]:raise ValueError('scheduled '+key)
        tokens+=expected['global_tokens'];targets+=expected['global_targets']
        if actual['total_tokens']!=tokens or actual['total_targets']!=targets:raise ValueError('cumulative clocks')
        if not math.isfinite(actual['loss']) or not math.isfinite(actual['grad_norm']):raise ValueError('nonfinite update')
    return steps


def render(script,baseline,phase,run,data,mix,authority_sha,pack_sha,end,controller,start=0,anchor=None):
    marker='CUDA_VISIBLE_DEVICES=\'\' "$EMENDER_PYTHON" - "$ROOT" "$DATA" <<\'PY\'\n'
    if script.count(marker)!=1:raise ValueError('collector boundary')
    script=script.split(marker)[0]
    changes=[(f'ROOT={baseline}\n',f'ROOT={shlex.quote(str(phase))}\nRUN={shlex.quote(str(run))}\n'),
      ('DATA=/mnt/nvme2n1/erikg/e97_systematic_posttraining/native-training-mix-v1',f'DATA={shlex.quote(str(data))}'),
      ('MIX=/mnt/nvme2n1/erikg/sft/e97-4b-native-training-smoke-mix-v1',f'MIX={shlex.quote(str(mix))}'),
      ('69f26726e32523dd840e363a5b9b7663b5e9ed37bd48a68e0bdbc9827d98a2c7',authority_sha),
      ('51c8dc8cc06a895ea39be780d211af6d645e48117fe1be9c70f93088b0b393a6',pack_sha),
      ('--output-root "$ROOT/checkpoints"','--output-root "$RUN/checkpoints"'),
      ('--steps 8 --save-every 4 --keep-checkpoints 3',f'--steps {end} --save-every 128 --keep-checkpoints 32'),
      ('--sampler-key 974117','--sampler-key 974223'),('--lr 5e-5','--lr 1e-5'),
      ('eval "$(bash scripts/gpu_lease.sh acquire 8 --no-wait)"','LEASE=$(bash scripts/gpu_lease.sh acquire 8 --no-wait)\neval "$LEASE"')]
    for old,new in changes:script=replace_once(script,old,new)
    if start:
        script=replace_once(script,'--new-stage-from "$PARENT"','--resume "$RUN/checkpoints/latest.pt"')
        script=replace_once(script,'nvidia-smi --query-gpu=',
            f'[[ $(readlink -f "$RUN/checkpoints/latest.pt") == {shlex.quote(str(anchor))} ]]\nnvidia-smi --query-gpu=')
    script+=f'CUDA_VISIBLE_DEVICES=\'\' PYTHONPATH={shlex.quote(str(controller))} "$EMENDER_PYTHON" {shlex.quote(str(controller/"scripts/prepare_e97_native_training_segment.py"))} collect --phase {shlex.quote(str(phase))}\n'
    return script


def freeze_examples(config,run):
    """New fitting records are selected from planned first-128 consumption.

    The development/retention example bytes remain those of the LR screen.
    """
    import numpy as np
    import tiktoken
    from scripts.build_e97_native_training_mix import read_source
    from scripts.eval_e97_native_learning import select_unique,annotations
    data=Path(config['data_root']);summary=read(data/'summary.json');mix=Path(summary['authority'])
    manifest=read(mix/'manifest.json');rows=[json.loads(line) for line in (mix/manifest['outputs']['metadata']['path']).read_text().splitlines()]
    schedule=read(data/'expected-schedule.json')
    identity=SFTSamplerIdentity(authority_manifest_sha256=summary['authority_sha256'],pack_manifest_sha256=summary['pack_sha256'],sampler_key=974223,data_world_size=8,context_size=65536)
    consumed=set()
    for rank in range(8):
        dataset=MaskedSFTPackedDataset(mix,data/'packs',identity=identity,rank=rank,sampler_mode='epoch-permutation')
        try:
            for cursor in range(128):
                if schedule['steps'][cursor]['rank_sample_ids'][rank]!=[dataset.sample_id(cursor)]:raise ValueError('planned fitting identity')
                pack=dataset.packs[dataset.pack_id_at(cursor)];offset=int(pack['record_offset'])
                for i in dataset.pack_record_ids[offset:offset+int(pack['record_count'])]:
                    row=rows[int(i)]
                    if row['source']=='native':consumed.add(row['source_record_id'])
        finally:dataset.close()
    if sha256(Path(config['original_panel']))!=config['original_panel_sha256']:raise ValueError('original panel identity')
    panel=read(config['original_panel']);examples=[];encoding=tiktoken.get_encoding('p50k_base')
    with ExitStack() as stack:
        source=read_source(next(s for s in manifest['recipe']['sources'] if s['name']=='native'),stack)
        for ordinal,i in enumerate(select_unique(sorted(consumed),source['metadata'],'native-fitting',8)):
            row=source['records'][i];offset=int(row['offset']);n=int(row['tokens'])
            tb=source['maps']['tokens'][offset*4:(offset+n)*4];mb=source['maps']['mask'][offset:offset+n]
            tokens=np.frombuffer(tb,dtype='<u4').astype(np.int64);mask=np.frombuffer(mb,dtype='u1')
            if len(mask)!=n or sum(mb)!=int(row['targets']):raise ValueError('fitting record extent')
            marked=annotations(tokens,mask,True,encoding)
            if ordinal<2 and marked['opening_positions'][0]+4096>65536:raise ValueError('generation reserve')
            examples.append(dict(id=f'native-fitting:{i}',cohort='native-fitting',source_record_id=i,
                source_manifest_sha256=source['spec']['sha256'],source_bytes_sha256=hashlib.sha256(tb+mb).hexdigest(),
                tokens=tokens.tolist(),mask=mask.tolist(),generate=ordinal<2,**marked))
    examples.extend(e for e in panel['examples'] if e['cohort']!='native-fitting')
    panel.pop('models');panel.pop('smoke_summary_sha256',None)
    panel.update(examples=examples,original_panel_sha256=sha256(Path(config['original_panel'])),
        selection='fitting hash-ranked from planned first 128 updates; unchanged LR-screen development/retention',
        schedule_sha256=summary['schedule_sha256'])
    publish(run/'base-panel.json',panel)


def prepare(args):
    config=read(args.config);run=Path(config['run_root']);data=Path(config['data_root']);baseline=Path(config['baseline_root'])
    if config['schema']!='emender-e97-native-long-program-v1' or config['learning_rate']!=1e-5 or config['world_size']!=8 or config['segment_deadline_seconds']!=14400 or config['keep_checkpoints']!=32:
        raise ValueError('unsupported program policy')
    verify_inventory(baseline)
    if sha256(baseline/'recipe.json')!=config['baseline_recipe_sha256']:raise ValueError('baseline recipe')
    summary=read(data/'summary.json')
    if summary['status']!='data-and-schedule-ready' or summary['source_target_totals']['native']!=config['expected_native_targets'] or summary['total_assistant_targets']!=config['expected_total_assistant_targets']:
        raise ValueError('data budget/status')
    for path,key in [(Path(summary['authority'])/'manifest.json','authority_sha256'),(data/'packs/manifest.json','pack_sha256'),(data/'expected-schedule.json','schedule_sha256')]:
        if sha256(path)!=config[key] or summary[key]!=config[key]:raise ValueError('large data identity')
    endpoints=summary['planned_segment_endpoints']
    if args.end not in endpoints or endpoints!=config['segment_endpoints']:raise ValueError('segment endpoint')
    index=endpoints.index(args.end);start=0 if index==0 else endpoints[index-1]
    if not run.exists():
        if start:raise ValueError('missing predecessor run')
        run.mkdir(parents=True)
        publish(run/'program.json',config);freeze_examples(config,run)
        publish(run/'panel-identity.json',dict(sha256=sha256(run/'base-panel.json')))
    elif read(run/'program.json')!=config:raise ValueError('changed program')
    if sha256(run/'base-panel.json')!=read(run/'panel-identity.json')['sha256']:raise ValueError('incomplete or changed panel freeze')
    anchor=None
    if start:
        previous=run/f'segment-{start:06d}';receipt=read(previous/'summary.json')
        gate=read(previous/'evaluation/gate.json')
        if (receipt['status']!='passed' or receipt['recipe_sha256']!=sha256(previous/'recipe.json')
            or gate['continue_training'] is not True or gate['training_summary_sha256']!=sha256(previous/'summary.json')
            or gate['program_sha256']!=sha256(run/'program.json')
            or gate['evaluation_summary_sha256']!=sha256(previous/'evaluation/summary.json')
            or gate['panel_sha256']!=sha256(previous/'evaluation/panel.json')):
            raise ValueError('predecessor not accepted')
        anchor=(run/'checkpoints/latest.pt').resolve(strict=True)
        if str(anchor)!=receipt['checkpoint']['path'] or sha256(anchor)!=receipt['checkpoint']['sha256']:raise ValueError('committed latest identity')
    phase=run/f'segment-{args.end:06d}';phase.mkdir(exist_ok=False)
    (phase/'worktree').symlink_to(baseline/'worktree',target_is_directory=True)
    shutil.copyfile(baseline/'source.sha256',phase/'source.sha256')
    recipe={**read(baseline/'recipe.json'),'schema':'emender-e97-native-long-segment-v1',
        'purpose':'larger-exposure SFT; provisional LR1e-5; no promotion','lr':1e-5,'steps':args.end,'start_update':start,
        'run_root':str(run),'save_every':128,'keep_checkpoints':32,'deadline_seconds':14400,
        'authority_manifest_sha256':config['authority_sha256'],'pack_manifest_sha256':config['pack_sha256'],
        'program_sha256':sha256(run/'program.json'),'resume_anchor':str(anchor) if anchor else None,
        'schedule':dict(path=str(data/'expected-schedule.json'),sha256=config['schedule_sha256'],sampler_key=974223),
        'acceptance':['finite updates','exact source schedule and cumulative clocks','complete atomic SR checkpoint'],
        'controller_source':str(Path(__file__).resolve().parents[1]),
        'controller_script_sha256':sha256(Path(__file__)),
        'base_panel_sha256':sha256(run/'base-panel.json')}
    publish(phase/'recipe.json',recipe)
    inputs=[(Path(config['parent_checkpoint']),recipe['parent_sha256']),
            (Path(config['args_json']),recipe['source_args_sha256']),
            (Path(summary['authority'])/'manifest.json',config['authority_sha256']),
            (data/'packs/manifest.json',config['pack_sha256']),
            (data/'expected-schedule.json',config['schedule_sha256'])]
    if anchor:inputs.append((anchor,sha256(anchor)))
    (phase/'input.sha256').write_text(''.join(f'{digest}  {path}\n' for path,digest in inputs))
    controller=Path(__file__).resolve().parents[1]
    script=render((baseline/'run.sh').read_text(),baseline,phase,run,data,Path(summary['authority']),config['authority_sha256'],config['pack_sha256'],args.end,controller,start,anchor)
    (phase/'run.sh').write_text(script);(phase/'run.sh').chmod(0o500)
    for name in ('source.sha256','input.sha256'):(phase/name).chmod(0o400)
    (phase/'inventory.sha256').write_text(''.join(f'{sha256(phase/name)}  {phase/name}\n' for name in ('recipe.json','run.sh','source.sha256','input.sha256')))
    (phase/'inventory.sha256').chmod(0o400)
    print(json.dumps(dict(phase=str(phase),start=start,end=args.end)),flush=True)


def collect(args):
    import torch
    phase=args.phase;verify_inventory(phase);recipe=read(phase/'recipe.json');run=Path(recipe['run_root'])
    schedule=Path(recipe['schedule']['path'])
    if sha256(schedule)!=recipe['schedule']['sha256']:raise ValueError('schedule changed')
    planned=read(schedule)['steps'];start=recipe['start_update'];end=recipe['steps']
    events=[json.loads(line) for line in (phase/'console.log').read_text().splitlines() if line.startswith('{') and '"event":' in line]
    steps=check_steps(events,planned,start,end)
    checkpoints=[e for e in events if e['event']=='checkpoint']
    if [e['update'] for e in checkpoints]!=[end]:raise ValueError('checkpoint coverage')
    event=checkpoints[0];path=Path(event['checkpoint']).resolve(strict=True)
    if path.parent!=(run/'checkpoints').resolve() or sha256(path)!=event['checkpoint_sha256']:raise ValueError('checkpoint identity')
    checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    for key,value in [('sft_updates',end),('sft_total_tokens',steps[-1]['total_tokens']),
        ('assistant_target_tokens',steps[-1]['total_targets']),('sampler_cursor',end),('sampler_key',974223),
        ('learning_rate',recipe['lr']),('data_world_size',8),('diloco_k',4),('diloco_merge_enabled',False),
        ('parent_checkpoint_sha256',recipe['parent_sha256']),('authority_manifest_sha256',recipe['authority_manifest_sha256']),
        ('pack_manifest_sha256',recipe['pack_manifest_sha256']),('source_commit',recipe['source_commit'])]:
        if checkpoint[key]!=value:raise ValueError('checkpoint '+key)
    precision=checkpoint['sft_precision']
    if precision['optimizer']!='bf16-sr-candidate' or precision['learning_rate']!=recipe['lr']:raise ValueError('optimizer precision/LR')
    optimizer=checkpoint['optimizer_state_dict'];group=optimizer['param_groups'][0];ids=set(group['params'])
    if group['k']!=end or group['train_mode'] is not False or ids!=set(optimizer['state']) or ids!=set(optimizer['eval_live_y']):raise ValueError('optimizer clocks/coverage')
    if sum(s['z'].numel() for s in optimizer['state'].values())!=4045972080:raise ValueError('optimizer coordinate count')
    tensors=list(checkpoint['model_state_dict'].values())+list(optimizer['eval_live_y'].values())
    for state in optimizer['state'].values():
        if set(state)!={'z','exp_avg_sq'}:raise ValueError('optimizer slot layout')
        tensors.extend(state.values())
    for tensor in tensors:
        if tensor.is_floating_point() and tensor.dtype!=torch.bfloat16:raise ValueError('persistent dtype')
        flat=tensor.reshape(-1)
        for begin in range(0,flat.numel(),1048576):
            if not bool(torch.isfinite(flat[begin:begin+1048576]).all()):raise ValueError('nonfinite checkpoint')
    if (run/'checkpoints/latest.pt').resolve()!=path:raise ValueError('atomic latest')
    path.chmod(0o400)
    source=Counter()
    for row in planned[:end]:source.update(row['source_targets'])
    publish(phase/'summary.json',dict(schema=recipe['schema'],status='passed',recipe_sha256=sha256(phase/'recipe.json'),
        start_update=start,updates=end,segment_updates=end-start,source_target_totals=dict(source),
        assistant_target_tokens=steps[-1]['total_targets'],input_tokens=steps[-1]['total_tokens'],
        checkpoint=dict(path=str(path),sha256=event['checkpoint_sha256'],finite_complete_bf16_state=True),
        runtime_sample_ids_and_clocks_exact=True,step_losses=[e['loss'] for e in steps],
        peak_rank0_hbm_allocated=max(e['max_hbm_allocated'] for e in steps),
        numerical_fresh_continuation='not measured; exact restoration required; no bitwise pass claim',
        actual_training=True,checkpoint_promotion=False,provisional_learning_rate=recipe['lr']))
    print('LARGE_SFT_SEGMENT_COMPLETE',end,flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('prepare');q.add_argument('--config',type=Path,required=True);q.add_argument('--end',type=int,required=True)
    q=sub.add_parser('collect');q.add_argument('--phase',type=Path,required=True)
    args=p.parse_args();{'prepare':prepare,'collect':collect}[args.command](args)


if __name__=='__main__':main()
