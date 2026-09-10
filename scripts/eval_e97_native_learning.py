#!/usr/bin/env python3
"""Matched full-record likelihood and unassisted first-turn generation.

No optimizer or tool execution. Development/retention panels are diagnostic,
not independent final holdouts. Saved/x and live/y are explicitly distinguished.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left,bisect_right
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from ndm.data.masked_sft_dataset import MaskedSFTPackedDataset,SFTSamplerIdentity,sha256
from ndm.e97 import load_e97_checkpoint,advance_e97_cache_segment,generate_e97_from_cache
from ndm.e97_atomic import publish_bytes_no_replace
from scripts.build_e97_native_training_mix import read_source
from scripts.e97_open_swe_native_runtime_protocol import validate_generated_turn
from scripts.e97_open_swe_native_codec import parse_turn

SCHEMA='emender-e97-native-learning-panel-v1'


def publish(path,value):
    publish_bytes_no_replace(path,(json.dumps(value,indent=2,sort_keys=True)+'\n').encode(),mode=0o400)


def select_unique(ids,metadata,cohort,count):
    order=sorted(ids,key=lambda i:hashlib.sha256(f'native-learning-v1:{cohort}:{i}'.encode()).digest())
    selected=[];seen=set()
    for i in order:
        row=metadata[i]
        key=tuple(row['problem_key']) if 'problem_key' in row else (row.get('source_manifest_sha256'),row.get('source_record_id',row.get('identity',i)))
        if key in seen:continue
        seen.add(key);selected.append(i)
        if len(selected)==count:return selected
    raise ValueError('insufficient unique diagnostic records: '+cohort)


def annotations(tokens,mask,native,encoding):
    supervised=np.flatnonzero(mask).tolist()
    if not supervised or supervised[0]==0 or any(v not in (0,1) for v in mask):raise ValueError('invalid record mask')
    starts=np.flatnonzero((mask==1)&np.concatenate(([True],mask[:-1]==0))).tolist()
    choices=[];first_gold=None
    if native:
        pieces=[encoding.decode_single_token_bytes(int(t)) for t in tokens]
        offsets=np.concatenate(([0],np.cumsum([len(p) for p in pieces]))).tolist()
        ends=np.flatnonzero((mask==1)&np.concatenate((mask[1:]==0,[True]))).tolist()
        for start,last in zip(starts,ends,strict=True):
            body=b''.join(pieces[start:last+1]);parsed=parse_turn(body.decode())
            if first_gold is None:first_gold=parsed
            left=offsets[start]+body.index(b'\nAction: ')+len(b'\nAction: ')
            right=offsets[start]+body.index(b'\nArguments: ')
            # Include tokens touching the name; a leading space may share its BPE.
            ids=list(range(bisect_right(offsets,left)-1,bisect_left(offsets,right)))
            if not ids or any(not mask[i] for i in ids):raise ValueError('choice token annotation outside supervision')
            choices.extend(ids)
    return dict(opening_positions=starts,choice_positions=choices,first_gold=first_gold)


def freeze(args):
    smoke=args.smoke
    summary=json.loads((smoke/'summary.json').read_text());recipe=json.loads((smoke/'recipe.json').read_text())
    if summary['status']!='passed' or sha256(smoke/'recipe.json')!=summary['recipe_sha256']:raise ValueError('smoke evidence')
    mix=args.mix;manifest=json.loads((mix/'manifest.json').read_text())
    if sha256(mix/'manifest.json')!=recipe['authority_manifest_sha256']:raise ValueError('mixture identity')
    schedule_path=Path(recipe['schedule']['path']);schedule=json.loads(schedule_path.read_text())
    if sha256(schedule_path)!=recipe['schedule']['sha256']:raise ValueError('schedule identity')
    events=[json.loads(line) for line in (smoke/'training.jsonl').read_text().splitlines()]
    steps=[e for e in events if e['event']=='step']
    if len(steps)!=8:raise ValueError('expected eight consumed updates')
    meta=manifest['outputs']['metadata'];p=mix/Path(meta['path']).name
    if sha256(p)!=meta['sha256']:raise ValueError('mixture metadata identity')
    mixed_rows=[json.loads(line) for line in p.read_text().splitlines()];consumed=set()
    identity=SFTSamplerIdentity(authority_manifest_sha256=recipe['authority_manifest_sha256'],
        pack_manifest_sha256=recipe['pack_manifest_sha256'],sampler_key=recipe['schedule']['sampler_key'],data_world_size=8,context_size=65536)
    for rank in range(8):
        data=MaskedSFTPackedDataset(mix,args.packs,identity=identity,rank=rank,sampler_mode='epoch-permutation')
        try:
            for cursor,step in enumerate(steps):
                if step['rank_sample_ids'][rank]!=[data.sample_id(cursor)]:raise ValueError('consumed sample identity')
                pack=data.packs[data.pack_id_at(cursor)];start=int(pack['record_offset'])
                for i in data.pack_record_ids[start:start+int(pack['record_count'])]:
                    row=mixed_rows[int(i)]
                    if row['source']=='native':consumed.add(row['source_record_id'])
        finally:data.close()
    encoding=tiktoken.get_encoding('p50k_base');examples=[]
    with ExitStack() as stack:
        sources={s['name']:read_source(s,stack) for s in manifest['recipe']['sources']}
        cohorts=[('native-fitting','native',sorted(consumed)),
                 ('native-development','native',[i for i,r in enumerate(sources['native']['records']) if r['split']==1]),
                 ('conversation-retention','conversation',[i for i,r in enumerate(sources['conversation']['records']) if r['split']==1 and r['tokens']<=8192 and sources['conversation']['metadata'][i].get('has_think') is False]),
                 ('tool-retention','retention',[i for i,r in enumerate(sources['retention']['records']) if r['split']==1 and r['tokens']<=8192 and sources['retention']['metadata'][i].get('source') in ('pi-live','compositional')])]
        for cohort,name,ids in cohorts:
            source=sources[name]
            ids=[i for i in ids if source['records'][i]['targets']>0]
            for ordinal,i in enumerate(select_unique(ids,source['metadata'],cohort,8)):
                row=source['records'][i];offset=int(row['offset']);n=int(row['tokens'])
                tb=source['maps']['tokens'][offset*4:(offset+n)*4];mb=source['maps']['mask'][offset:offset+n]
                tokens=np.frombuffer(tb,dtype='<u4').astype(np.int64);mask=np.frombuffer(mb,dtype='u1')
                if len(mask)!=n or sum(mb)!=int(row['targets']):raise ValueError('source record extent/mask')
                marked=annotations(tokens,mask,source['native'],encoding)
                generate=source['native'] and ordinal<2
                if generate and marked['opening_positions'][0]+4096>65536:raise ValueError('generation context reserve')
                examples.append(dict(id=f'{cohort}:{i}',cohort=cohort,source_record_id=i,
                    source_manifest_sha256=source['spec']['sha256'],source_bytes_sha256=hashlib.sha256(tb+mb).hexdigest(),
                    tokens=tokens.tolist(),mask=mask.tolist(),generate=generate,**marked))
    models=[dict(name='parent-y',checkpoint=str(args.parent),sha256=recipe['parent_sha256'],mode='train')]
    for item in summary['checkpoints']:
        models.append(dict(name=f"u{item['updates']}-y",checkpoint=item['path'],sha256=item['sha256'],mode='train'))
    models.append({**models[-1],'name':'u8-x','mode':'saved'})
    if [m['name'] for m in models]!=['parent-y','u4-y','u8-y','u8-x']:raise ValueError('checkpoint panel')
    for model in models:
        if sha256(Path(model['checkpoint']))!=model['sha256']:raise ValueError('checkpoint identity')
    if sha256(args.args_json)!=recipe['source_args_sha256']:raise ValueError('model args identity')
    publish(args.output,dict(schema=SCHEMA,training_eligible=False,smoke_summary_sha256=sha256(smoke/'summary.json'),
        args_json=str(args.args_json),args_sha256=recipe['source_args_sha256'],models=models,examples=examples,
        generation_budget=4096,cohort_records=8,selection='hash-ranked unique problems/records before model scoring',
        scope='matched full-record teacher forcing and unassisted native first turns; no tool dispatch, independent holdout or decision-balanced D claim'))


@torch.no_grad()
def score(model,example,device):
    tokens=example['tokens'];mask=np.asarray(example['mask']);n=len(tokens)
    length=((n-2)//16+1)*16
    inputs=torch.zeros((1,length),device=device,dtype=torch.long)
    inputs[0,:n-1]=torch.tensor(tokens[:-1],device=device)
    with torch.autocast(torch.device(device).type,dtype=torch.bfloat16):logits=model(inputs,return_loss=False)
    if logits.shape[:2]!=(1,length):raise ValueError('logit shape')
    positions=np.flatnonzero(mask);opening=set(example['opening_positions']);choice=set(example['choice_positions'])
    totals={key:dict(nll_sum=0.,tokens=0,correct=0) for key in ('assistant','opening','choice')}
    for begin in range(0,len(positions),128):
        selected=positions[begin:begin+128].tolist();indices=torch.tensor(selected,device=device)
        values=logits[0,indices-1].float()
        if not torch.isfinite(values).all():raise ValueError('nonfinite logits')
        targets=torch.tensor([tokens[p] for p in selected],device=device)
        losses=F.cross_entropy(values,targets,reduction='none').tolist();correct=(values.argmax(-1)==targets).tolist()
        for p,loss,hit in zip(selected,losses,correct,strict=True):
            for key,include in (('assistant',True),('opening',p in opening),('choice',p in choice)):
                if include:
                    totals[key]['tokens']+=1;totals[key]['nll_sum']+=loss;totals[key]['correct']+=int(hit)
    return totals


@torch.no_grad()
def generate(loaded,example,encoding,budget):
    prefix=example['tokens'][:example['opening_positions'][0]]
    cache=advance_e97_cache_segment(loaded,prefix);ids=[];reason='budget';turn=None
    for _ in range(budget):
        new,cache=generate_e97_from_cache(loaded,cache,max_new_tokens=1,temperature=0.,top_k=0,top_p=0.,stop_token_ids=(218,))
        if not new:reason='empty';break
        ids.extend(new);text=encoding.decode(ids)
        if not ('Analysis: '.startswith(text) or text.startswith('Analysis: ')):
            reason='invalid_opening';break
        if text.count('\n')>4:reason='invalid_frame';break
        if text.count('\n')==4 and text.endswith('}'):
            try:turn=validate_generated_turn(text,encoding)
            except ValueError as error:reason=str(error)
            else:reason='valid';break
        if 218 in new:reason='separator_before_valid_turn';break
    message=parse_turn(encoding.decode(ids)) if turn is not None else None
    return dict(valid=turn is not None,reason=reason,generated_tokens=len(ids),token_ids=ids,
        choice_matches_source=message['name']==example['first_gold']['name'] if message else False,
        arguments_match_source=message['arguments']==example['first_gold']['arguments'] if message else False,
        tool_dispatch=False)


def run(args):
    if sha256(args.panel)!=args.panel_sha:raise ValueError('panel identity')
    panel=json.loads(args.panel.read_text())
    if panel['schema']!=SCHEMA or panel['training_eligible'] is not False:raise ValueError('panel schema')
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local=int(os.environ['LOCAL_RANK'])
    if world!=8:raise ValueError('eight evaluation ranks required')
    target=panel['models'][rank//2];lane=rank%2
    for path,digest in ((Path(target['checkpoint']),target['sha256']),(Path(panel['args_json']),panel['args_sha256'])):
        if sha256(path)!=digest:raise ValueError('model input identity')
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    loaded=load_e97_checkpoint(target['checkpoint'],args_json=panel['args_json'],device=device,dtype=torch.bfloat16,
                              weight_mode=target['mode'],use_triton=True,mmap=True)
    loaded.model.eval();encoding=tiktoken.get_encoding('p50k_base');results=[]
    for example in panel['examples'][lane::2]:
        began=time.monotonic();metrics=score(loaded.model,example,device)
        result=dict(id=example['id'],cohort=example['cohort'],metrics=metrics)
        if example['generate']:result['generation']=generate(loaded,example,encoding,panel['generation_budget'])
        result['seconds']=time.monotonic()-began;results.append(result)
        # Never print generated private reasoning in progress output.
        print(json.dumps(dict(event='evaluated',model=target['name'],id=example['id'],
            nll=metrics['assistant']['nll_sum']/metrics['assistant']['tokens'],
            generated_valid=result.get('generation',{}).get('valid'))),flush=True)
    publish(args.output/f'rank-{rank}.json',dict(schema=SCHEMA,panel_sha256=args.panel_sha,rank=rank,model=target,
        results=results,peak_hbm_allocated=torch.cuda.max_memory_allocated()))


def aggregate(args):
    if sha256(args.panel)!=args.panel_sha:raise ValueError('panel identity')
    panel=json.loads(args.panel.read_text());groups={}
    for rank in range(8):
        report=json.loads((args.output/f'rank-{rank}.json').read_text())
        if report['rank']!=rank or report['panel_sha256']!=args.panel_sha or report['model']!=panel['models'][rank//2]:raise ValueError('shard identity')
        groups.setdefault(report['model']['name'],[]).extend(report['results'])
    expected={e['id'] for e in panel['examples']};summary={}
    for name,rows in groups.items():
        if len(rows)!=len(expected) or {r['id'] for r in rows}!=expected:raise ValueError('coverage')
        cohorts={}
        for cohort in sorted({r['cohort'] for r in rows}):
            selected=[r for r in rows if r['cohort']==cohort];metrics={}
            for key in ('assistant','opening','choice'):
                values=[r['metrics'][key] for r in selected if r['metrics'][key]['tokens']]
                if values:
                    metrics[key]=dict(record_macro_nll=sum(v['nll_sum']/v['tokens'] for v in values)/len(values),
                        token_accuracy=sum(v['correct'] for v in values)/sum(v['tokens'] for v in values))
            generations=[r['generation'] for r in selected if 'generation' in r]
            cohorts[cohort]=dict(metrics=metrics,generations=len(generations),valid_first_turns=sum(g['valid'] for g in generations),
                matching_choices=sum(g['choice_matches_source'] for g in generations),matching_arguments=sum(g['arguments_match_source'] for g in generations))
        summary[name]=cohorts
    publish(args.output/'summary.json',dict(schema=SCHEMA,status='passed',panel_sha256=args.panel_sha,
        models=summary,scope=panel['scope'],capability_promotion=False))
    print(json.dumps(summary,sort_keys=True),flush=True)


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    f=sub.add_parser('freeze')
    for name in ('smoke','mix','packs','parent','args-json','output'):f.add_argument('--'+name,type=Path,required=True)
    for command in ('run','aggregate'):
        q=sub.add_parser(command);q.add_argument('--panel',type=Path,required=True);q.add_argument('--panel-sha',required=True);q.add_argument('--output',type=Path,required=True)
    a=p.parse_args();{'freeze':freeze,'run':run,'aggregate':aggregate}[a.command](a)


if __name__=='__main__':main()
