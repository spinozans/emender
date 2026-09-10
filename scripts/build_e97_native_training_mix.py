#!/usr/bin/env python3
"""Build a train-only, whole-record native/conversation/retention mixture.

Explicit internal-use authorization admits a derivative, not the immutable raw
candidate. Source bytes/masks are copied exactly. No licensing-review gate,
source commands, model updates or first-party task-registry changes.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
import json
import mmap
import os
from pathlib import Path
import random

import numpy as np
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA,RECORD_INDEX,sha256
from scripts.build_e97_open_swe_native_sft import rename_no_replace

LEGACY_ADMITTED={
    'd64f51abc615c097910900ffa7fc88f020ae5567b89958f607008078c271c7f6',
    '79e7981077fc387c3e3759728ddd0f9e1ac16f687c628542e8313fb735c4cfb7',
}
DTYPE=np.dtype([('offset','<u8'),('tokens','<u8'),('targets','<u8'),('split','u1'),('pad','V7')])


def record_order(ids,metadata,seed,native,unique_native=False,epochs=1):
    rng=random.Random(seed)
    if not native:
        for _ in range(epochs):
            order=list(ids);rng.shuffle(order)
            yield from order
        return
    groups=defaultdict(list)
    for i in ids:groups[tuple(metadata[i]['problem_key'])].append(i)
    keys=sorted(groups)
    for group in groups.values():rng.shuffle(group)
    cycle=0
    while keys:
        order=keys.copy();rng.shuffle(order)
        for key in order:yield groups[key][cycle%len(groups[key])]
        cycle+=1
        if unique_native:keys=[key for key in keys if len(groups[key])>cycle]


def read_source(spec,stack):
    root=Path(spec['root']);manifest_path=root/'manifest.json'
    if sha256(manifest_path)!=spec['sha256']:raise ValueError('source manifest identity')
    manifest=json.loads(manifest_path.read_text());native=spec['kind']=='native'
    if native:
        if manifest['schema']!='emender-open-swe-source-native-candidate-v1':raise ValueError('native schema')
        keys={'tokens':'tokens.bin','mask':'loss_mask.bin','index':'records.idx','metadata':'records.jsonl'}
    else:
        if manifest.get('schema')!=AUTHORITY_SCHEMA or manifest.get('status')!='complete':raise ValueError('legacy schema/status')
        eligibility=manifest.get('training_eligible')
        if eligibility is not True and not (eligibility is None and spec['sha256'] in LEGACY_ADMITTED):
            raise ValueError('not an explicitly admitted or hash-allowlisted historical source')
        keys={key:key for key in ('tokens','mask','index','metadata')}
    paths={}
    for name,key in keys.items():
        entry=manifest['outputs'][key];path=root/Path(entry['path']).name
        if path.stat().st_size!=entry['bytes'] or sha256(path)!=entry['sha256']:
            raise ValueError('source payload identity: '+name)
        paths[name]=path
    maps={}
    for name,path in paths.items():
        f=stack.enter_context(path.open('rb'))
        maps[name]=stack.enter_context(mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ))
    if len(maps['index'])%RECORD_INDEX.size:raise ValueError('index alignment')
    records=np.frombuffer(maps['index'],dtype=DTYPE).copy()
    metadata=None
    if native or spec.get('include_metadata_sources') or spec.get('exclude_think'):
        metadata=[json.loads(line) for line in maps['metadata'].read().splitlines()]
        if len(metadata)!=len(records):raise ValueError('metadata/index coverage')
    ids=[int(i) for i in np.flatnonzero((records['split']==0)&(records['targets']>0)&
                                      (records['tokens']>=2)&(records['tokens']<=65536))]
    exclusions=Counter(non_training_split=int((records['split']!=0).sum()),
                       whole_record_context_cap=int(((records['split']==0)&(records['tokens']>65536)).sum()))
    if native:
        validation={tuple(row['problem_key']) for row in metadata if row['split']==1}
        training={tuple(metadata[i]['problem_key']) for i in ids}
        if training&validation:raise ValueError('cross-split native problem')
        protected=set(manifest['protected_repositories'])
        basenames={repo.rsplit('/',1)[-1] for repo in protected}
        allowed=[]
        for i in ids:
            repo=metadata[i]['problem_key'][0]
            if repo in protected or repo.rsplit('/',1)[-1] in basenames:
                exclusions['protected_repo_or_same_basename']+=1
            else:allowed.append(i)
        ids=allowed
    if spec.get('include_metadata_sources'):
        wanted=set(spec['include_metadata_sources'])
        filtered=[i for i in ids if metadata[i].get('source') in wanted]
        exclusions['metadata_source_filter']=len(ids)-len(filtered);ids=filtered
    if spec.get('exclude_think'):
        filtered=[i for i in ids if metadata[i].get('has_think') is False]
        exclusions['thinking_record_or_missing_annotation']=len(ids)-len(filtered);ids=filtered
    available=sum(int(records[i]['targets']) for i in ids)
    if not ids or available*spec.get('epochs',1)<spec['target_tokens']:
        raise ValueError('insufficient eligible source targets within declared epochs')
    return dict(spec=spec,manifest=manifest,paths=paths,keys=keys,maps=maps,records=records,metadata=metadata,ids=ids,
                exclusions=dict(exclusions),available_targets=available,native=native)


def verify_authorization(recipe):
    if recipe.get('schema')!='emender-native-training-mix-recipe-v1' or recipe.get('operator_internal_training_authorized') is not True:
        raise ValueError('explicit internal-training authorization required')
    sources=recipe['sources']
    if {s['name'] for s in sources}!={'native','conversation','retention'} or len(sources)!=3:
        raise ValueError('expected exactly three named sources')
    native=next(s for s in sources if s['name']=='native')
    if native['kind']!='native' or any(s['kind']!='legacy' for s in sources if s['name']!='native'):
        raise ValueError('source kind mismatch')
    for source in sources:
        if type(source['target_tokens']) is not int or source['target_tokens']<=0:raise ValueError('positive integer quota required')
        epochs=source.get('epochs',1);unique=source.get('unique_native',False)
        if type(epochs) is not int or not 1<=epochs<=3 or (epochs!=1 and source['name']!='retention'):
            raise ValueError('only retention may repeat, for at most three epochs')
        if type(unique) is not bool or (unique and source['name']!='native'):
            raise ValueError('unique_native applies only to native source')
    for item in recipe['native_evidence']:
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise ValueError('native evidence bytes')
        report=json.loads(path.read_text())
        if report.get('status')!='passed':raise ValueError('native evidence did not pass')
        if report.get(item['binding_field'])!=native['sha256']:raise ValueError('native evidence source binding')
    if {e['binding_field'] for e in recipe['native_evidence']}!={'manifest_sha256','dataset_manifest_sha256'}:
        raise ValueError('require reconstruction and actual executor evidence')


def build(recipe_path,recipe_sha,output):
    if sha256(recipe_path)!=recipe_sha:raise ValueError('recipe identity')
    recipe=json.loads(recipe_path.read_text());verify_authorization(recipe)
    stage=output.with_name(output.name+'.partial')
    if output.exists() or stage.exists():raise FileExistsError('existing output or partial attempt')
    output.parent.mkdir(parents=True,exist_ok=True);stage.mkdir()
    names={'tokens':'tokens.uint32.bin','mask':'assistant_mask.uint8.bin','index':'records.idx','metadata':'records.jsonl'}
    receipts={};selected=[]
    with ExitStack() as stack:
        sources=[read_source(spec,stack) for spec in recipe['sources']]
        for source_index,source in enumerate(sources):
            spec=source['spec'];observed=0;unique=set();written=0
            order=record_order(source['ids'],source['metadata'],recipe['seed']+source_index,source['native'],
                               unique_native=spec.get('unique_native',False),epochs=spec.get('epochs',1))
            while observed<spec['target_tokens']:
                try:i=next(order)
                except StopIteration:raise ValueError('source exhausted before whole-record quota') from None
                row=source['records'][i];offset=int(row['offset']);n=int(row['tokens']);targets=int(row['targets'])
                mask=source['maps']['mask'][offset:offset+n]
                if len(mask)!=n or any(x not in (0,1) for x in mask) or sum(mask)!=targets or mask[0]:
                    raise ValueError('invalid or cross-record supervision')
                if (offset+n)*4>len(source['maps']['tokens']):raise ValueError('record token extent')
                selected.append((source_index,i));observed+=targets;unique.add(i);written+=1
            receipts[spec['name']]=dict(root=spec['root'],manifest_sha256=spec['sha256'],
                requested_targets=spec['target_tokens'],assistant_target_tokens=observed,
                records=written,unique_records=len(unique),repeated_records=written-len(unique),
                eligible_records=len(source['ids']),available_targets=source['available_targets'],
                exclusions=source['exclusions'],unique_native=spec.get('unique_native',False),
                maximum_epochs=spec.get('epochs',1) if not source['native'] or spec.get('unique_native',False) else None)
        # Never emit source-contiguous blocks into the greedy packer.
        random.Random(recipe['seed']^0x970041).shuffle(selected)
        handles={key:stack.enter_context((stage/name).open('wb')) for key,name in names.items()}
        offset=0;total_targets=0
        for record_id,(source_index,i) in enumerate(selected):
            source=sources[source_index];row=source['records'][i]
            start=int(row['offset']);n=int(row['tokens']);targets=int(row['targets'])
            token_bytes=source['maps']['tokens'][start*4:(start+n)*4]
            mask_bytes=source['maps']['mask'][start:start+n]
            handles['tokens'].write(token_bytes);handles['mask'].write(mask_bytes)
            handles['index'].write(RECORD_INDEX.pack(offset,n,targets,0))
            metadata=dict(id=f'native-mix-{record_id}',source=source['spec']['name'],
                          source_manifest_sha256=source['spec']['sha256'],source_record_id=i,
                          source_offset=start,offset=offset,tokens=n,targets=targets,split=0,
                          copied_bytes_sha256=hashlib.sha256(token_bytes+mask_bytes).hexdigest())
            if source['native']:
                original=source['metadata'][i]
                metadata.update(problem_key=original['problem_key'],trajectory_identity=original['trajectory_identity'])
            handles['metadata'].write((json.dumps(metadata,sort_keys=True)+'\n').encode())
            offset+=n;total_targets+=targets
        for handle in handles.values():handle.flush();os.fsync(handle.fileno())
        for source in sources:
            if sha256(Path(source['spec']['root'])/'manifest.json')!=source['spec']['sha256']:
                raise ValueError('source identity changed')
            for name,path in source['paths'].items():
                if sha256(path)!=source['manifest']['outputs'][source['keys'][name]]['sha256']:
                    raise ValueError('source payload changed during copy')
    # Independent verification of every emitted record, before publication.
    with (stage/names['tokens']).open('rb') as tokens,(stage/names['mask']).open('rb') as masks,(stage/names['index']).open('rb') as index:
        count=0
        for line in (stage/names['metadata']).read_text().splitlines():
            row=json.loads(line);n=row['tokens']
            if RECORD_INDEX.unpack(index.read(RECORD_INDEX.size))!=(row['offset'],n,row['targets'],0):raise ValueError('output index mismatch')
            tb=tokens.read(4*n);mb=masks.read(n)
            if len(tb)!=4*n or len(mb)!=n or sum(mb)!=row['targets'] or hashlib.sha256(tb+mb).hexdigest()!=row['copied_bytes_sha256']:
                raise ValueError('output copy mismatch')
            count+=1
        if tokens.read(1) or masks.read(1) or index.read(1) or count!=len(selected):raise ValueError('output coverage')
    for receipt in receipts.values():receipt['assistant_target_fraction']=receipt['assistant_target_tokens']/total_targets
    outputs={key:dict(path=name,bytes=(stage/name).stat().st_size,sha256=sha256(stage/name)) for key,name in names.items()}
    manifest=dict(schema=AUTHORITY_SCHEMA,status='complete',training_eligible=True,tokenizer='p50k_base',
        purpose='operator-authorized internal native-agent training mixture; not model promotion',
        recipe_sha256=recipe_sha,recipe=recipe,sources=receipts,outputs=outputs,
        counts=dict(records=len(selected),tokens=offset,assistant_target_tokens=total_targets,
                    train_records=len(selected),validation_records=0),
        sampling='native problem-shuffled rounds (unique when declared); legacy shuffled without replacement within declared epochs; global record shuffle',
        source_bytes_and_masks_unchanged=True,all_output_records_verified=True,
        licensing='provenance/notices retained; separate licensing review is not an internal-training launch gate',
        overlap_scope='train splits only; native problems disjoint from native development; protected canonical and same-basename repositories excluded',
        independent_holdout_claim=False)
    with (stage/'manifest.json').open('w') as stream:
        stream.write(json.dumps(manifest,sort_keys=True,indent=2)+'\n');stream.flush();os.fsync(stream.fileno())
    for path in stage.iterdir():path.chmod(0o400)
    stage.chmod(0o500)
    fd=os.open(stage,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    rename_no_replace(stage,output)
    print(json.dumps(dict(output=str(output),manifest_sha256=sha256(output/'manifest.json'),counts=manifest['counts'],sources=receipts)),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--recipe-sha256',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();build(a.recipe,a.recipe_sha256,a.output)


if __name__=='__main__':main()
