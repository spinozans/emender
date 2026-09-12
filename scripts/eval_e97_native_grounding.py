#!/usr/bin/env python3
"""Separate exact-path selection, system-message sensitivity and supplied-read continuation."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import signal
import time
from scripts.eval_e97_native_execution import episode, generate_turn, publish, sha
from scripts.e97_native_execution_cases import context
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode, validate_generated_turn

SCHEMA = 'emender-e97-native-grounding-diagnostic-v1'
PATHS = ['/testbed/config.json', '/testbed/numbers.json', '/workspace/config.json',
         '/workspace/diagnostic__project__1.0/config.json', '/workspace/diagnostic__project__1.0', '/testbed']


def freeze(args):
    import tiktoken
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True, mode=0o700)
    base = args.base/'panel.json'
    if sha(base) != '601ca7525ef616930e912cd45f3565e8b596a422635bb0ed118876782beea854':
        raise ValueError('base panel identity')
    if sha(args.base/'preflight/smoke-summary.json') != '8e3ebb01005035d01e636fe7701ff5bbbcad3f960108c9bbffebf52b3c52165f':
        raise ValueError('qualified snapshot/executor identity')
    panel = json.loads(base.read_text())
    source_panel = args.training/'segment-000880/evaluation/panel.json'
    if sha(source_panel) != 'fe2d10d97c2ee6d31a35684b043f3e4bceaed49782f597612e38b3ca89935aad':
        raise ValueError('source-system provenance')
    old = json.loads(source_panel.read_text()); enc = tiktoken.get_encoding('p50k_base')
    e = next(e for e in old['examples'] if e['cohort']=='native-development' and e['generate'])
    prefix = enc.decode(e['tokens'][:e['opening_positions'][0]])
    systems = [json.loads(b[len('System:\n'):]) for b in prefix.split('\n\n') if b.startswith('System:\n')]
    if len(systems) != 1:
        raise ValueError('source system coverage')
    source_system = systems[0]
    models = []
    for model in panel['models'][2:]:
        for style in ('minimal', 'source'):
            models.append({**model, 'name':model['name']+'-'+style, 'system_style':style,
                           'system_message':source_system if style=='source' else context('system', panel['system'])})
    selected = []
    for c in panel['cases']:
        if c['family'] not in ('lookup', 'sum'):
            continue
        for assisted in (False, True):
            item = deepcopy(c); item['condition'] = 'supplied-read' if assisted else 'unassisted'
            item['id'] += '-'+item['condition']
            if assisted:
                path = next(iter(c['files']))
                item['supplied_calls'] = [dict(name='str_replace_editor', arguments=dict(command='view', path='/testbed/'+path))]
            selected.append(item)
    panel.update(schema=SCHEMA, models=models, cases=selected, paths=PATHS,
                 base_panel_sha256=sha(base), source_system_panel_sha256=sha(source_panel),
                 source_system_example=e['id'], source_system_sha256=hashlib.sha256(json.dumps(source_system,sort_keys=True).encode()).hexdigest(),
                 scope='24 first-turn path probes and 32 matched unassisted/supplied-read episodes; source system message includes its original think flag. No assisted result is autonomous success.')
    publish(args.output/'panel.json', panel)
    print('GROUNDING_PANEL_FROZEN', sha(args.output/'panel.json'), flush=True)


def panel_read(args):
    if sha(args.panel) != args.panel_sha:
        raise ValueError('panel identity')
    panel = json.loads(args.panel.read_text())
    if panel['schema'] != SCHEMA or panel['training_eligible'] is not False:
        raise ValueError('panel schema')
    return panel


def path_probe(loaded, target, panel, path, enc):
    e = NativeEpisode(panel['tools'], enc)
    e.append_source_message(target['system_message'])
    e.append_source_message(context('user', 'Use str_replace_editor with command view and path exactly '+json.dumps(path)+'. Do not change the path.'))
    text, ids, reason = generate_turn(loaded, e.prompt(), enc, panel['generation_budget'], time.monotonic()+panel['episode_seconds'])
    call = validate_generated_turn(text,enc).backend_call() if text is not None else None
    expected = dict(name='str_replace_editor', arguments=dict(command='view',path=path))
    return dict(path=path, valid_frame=text is not None, reason=reason, exact_requested_call=call==expected,
                argument_path_exact=bool(call and call['arguments'].get('path')==path),
                generated_token_ids=ids, generated_call=call, tool_dispatch=False)


def run(args):
    import torch
    import tiktoken
    from ndm.e97 import load_e97_checkpoint
    from scripts.e97_open_swe_native_codec import vocabulary
    panel = panel_read(args); rank = int(os.environ['RANK']); local = int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE']) != 8:
        raise ValueError('eight ranks required')
    target = panel['models'][rank//2]
    if sha(target['checkpoint']) != target['sha256'] or sha(panel['args_json']) != panel['args_sha256']:
        raise ValueError('model identity')
    torch.cuda.set_device(local); torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded = load_e97_checkpoint(target['checkpoint'], args_json=panel['args_json'], device=torch.device('cuda',local),
                                 dtype=torch.bfloat16, weight_mode=target['mode'], use_triton=True, mmap=True)
    loaded.model.eval(); enc = tiktoken.get_encoding('p50k_base')
    if vocabulary(enc)[1] != panel['tokenizer_vocabulary_sha256']:
        raise ValueError('tokenizer identity')
    path_results = [path_probe(loaded,target,panel,path,enc) for path in panel['paths'][rank%2::2]]
    publish(args.output/f'rank-{rank}-paths-private.json', path_results)
    results=[]; local_panel={**panel, 'system_message':target['system_message']}
    for c in panel['cases']:
        if c['pair_index']%2 != rank%2:
            continue
        r = episode(loaded,c,local_panel,enc,args.output/f'rank-{rank}-{c["id"]}')
        results.append({k:r[k] for k in ('id','family','reason','grade','assisted','autonomous_success','seconds')})
    publish(args.output/f'rank-{rank}.json',dict(rank=rank,model=target,panel_sha256=args.panel_sha,results=results,
        paths=[{k:r[k] for k in ('path','valid_frame','reason','exact_requested_call','argument_path_exact')} for r in path_results]))
    print(json.dumps(dict(event='grounding_rank_complete',rank=rank,model=target['name'])),flush=True)


def aggregate(args):
    p=panel_read(args); summary={}
    for index,target in enumerate(p['models']):
        paths=[]; rows=[]
        for rank in (index*2,index*2+1):
            r=json.loads((args.output/f'rank-{rank}.json').read_text())
            if r['rank']!=rank or r['model']!=target or r['panel_sha256']!=args.panel_sha:
                raise ValueError('shard identity')
            wanted={c['id'] for c in p['cases'] if c['pair_index']%2==rank%2}
            if len(r['results'])!=len(wanted) or {v['id'] for v in r['results']}!=wanted:
                raise ValueError('episode coverage')
            if [v['path'] for v in r['paths']]!=p['paths'][rank%2::2]:
                raise ValueError('path coverage')
            case_map={c['id']:c for c in p['cases']}
            for row in r['results']:
                assisted=bool(case_map[row['id']].get('supplied_calls'))
                if row['assisted'] is not assisted or row['autonomous_success'] is not (row['grade']['success'] and not assisted):
                    raise ValueError('assistance attribution')
            paths.extend(r['paths']); rows.extend(r['results'])
        conditions={}
        for assisted in (False,True):
            selected=[r for r in rows if r['assisted'] is assisted]
            conditions['supplied-read' if assisted else 'unassisted']=dict(episodes=len(selected),
                criterion_successes=sum(r['grade']['success'] for r in selected),
                autonomous_successes=sum(r['autonomous_success'] for r in selected),
                paired_success={f:all(r['grade']['success'] for r in selected if r['family']==f) for f in ('lookup','sum')},
                outcomes=selected)
        summary[target['name']]=dict(paths=paths,exact_paths=sum(v['argument_path_exact'] for v in paths),
                                    exact_calls=sum(v['exact_requested_call'] for v in paths),conditions=conditions)
    publish(args.output/'summary.json',dict(schema=SCHEMA,status='measurements-complete',panel_sha256=args.panel_sha,
        models=summary,training_eligible=False,checkpoint_promotion=False,scope=p['scope']))
    print('GROUNDING_MEASUREMENTS_COMPLETE',flush=True)


def main():
    def interrupted(signum,frame):raise TimeoutError('grounding diagnostic interrupted')
    signal.signal(signal.SIGTERM,interrupted)
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for name in ('base','training','output'):f.add_argument('--'+name,type=Path,required=True)
    for command in ('run','aggregate'):
        a=sub.add_parser(command);a.add_argument('--panel',type=Path,required=True);a.add_argument('--panel-sha',required=True);a.add_argument('--output',type=Path,required=True)
    a=p.parse_args();globals()[a.command](a)


if __name__=='__main__':main()
