#!/usr/bin/env python3
"""Operator-authorized bridge derivative; original preparation/replay stays immutable."""
import argparse
from collections import Counter,defaultdict
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import signal
import tiktoken
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA,RECORD_INDEX
from scripts.build_e97_grounded_expansion import bytes_record
from scripts.build_e97_native_training_mix import read_source
from scripts.e97_grounded_curriculum import FAMILIES,teacher_record
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.eval_e97_native_execution import publish,sha


def read(p):return json.loads(Path(p).read_text())


def bound_inputs(config):
    if (config['schema']!='emender-e97-representation-bridge-training-v1' or config['operator_internal_training_authorized'] is not True
        or (config['steps'],config['learning_rate'],config['weight_mode'])!=(32,1e-5,'train') or config['experimental_precision_policy'] is not None):raise ValueError('new bounded authorization')
    root=Path(config['preparation_root'])
    if sha(root/'plan.json')!=config['plan_sha256'] or sha(root/'native-result-audit.json')!=config['native_audit_sha256']:raise ValueError('qualified preparation')
    plan=read(root/'plan.json');audit=read(root/'native-result-audit.json')
    if audit['passed'] is not True or audit['records_rechecked']!=48 or sha(root/'native-summary.json')!=audit['native_summary_sha256']:raise ValueError('native preflight audit')
    for name,digest in plan['case_files'].items():
        if sha(root/name)!=digest:raise ValueError('frozen case bytes')
    p=plan['recipe']
    for name in ('generation_panel','previous_evaluation','previous_training_cases'):
        if sha(p[name])!=p[name+'_sha256']:raise ValueError('source panel/cases')
    old=Path(config['rehearsal_data'])
    if sha(old/'authority/manifest.json')!=config['rehearsal_manifest_sha256'] or sha(old/'result-audit.json')!=config['rehearsal_audit_sha256']:raise ValueError('verified rehearsal authority')
    if read(old/'result-audit.json')['passed'] is not True:raise ValueError('rehearsal audit')
    return plan,read(p['generation_panel'])


def rehearsal_ids(cases,seed,pairs_per_family=64):
    by_family=defaultdict(lambda:defaultdict(list))
    if pairs_per_family!=64:raise ValueError('rehearsal budget')
    if len({c['id'] for c in cases})!=len(cases):raise ValueError('rehearsal duplicate ID')
    for c in cases:by_family[c['family']][c['pair_index']].append(c)
    if set(by_family)!=set(FAMILIES):raise ValueError('rehearsal families')
    selected=[]
    for family in FAMILIES:
        pairs=by_family[family]
        if len(pairs)<pairs_per_family:raise ValueError('rehearsal pair coverage')
        order=sorted(pairs,key=lambda i:hashlib.sha256(f'{seed}:{family}:{i}'.encode()).hexdigest())
        for index in order[:pairs_per_family]:
            pair=sorted(pairs[index],key=lambda c:c['variant'])
            if len(pair)!=2 or [c['variant'] for c in pair]!=[0,1] or pair[0]['prompt']!=pair[1]['prompt']:raise ValueError('whole rehearsal pairs')
            selected.extend(c['id'] for c in pair)
    return selected


def copied_row(source,i,name,provenance=None):
    r=source['records'][i];start=int(r['offset']);n=int(r['tokens'])
    return dict(tokens=bytes(source['maps']['tokens'][4*start:4*(start+n)]),mask=bytes(source['maps']['mask'][start:start+n]),source=name,source_record_id=i,provenance=provenance)


def replay_ids(source,config):
    if config['replay_order']!=['conversation','native','retention'] or config['replay_target_quotas']!=dict(conversation=300000,native=100000,retention=100000):raise ValueError('frozen replay quotas/order')
    result={}
    for j,name in enumerate(config['replay_order']):
        ids=[i for i in source['ids'] if source['metadata'][i]['source']==name]
        random.Random(config['replay_seed']+j).shuffle(ids);total=0;selected=[]
        for i in ids:
            selected.append(i);total+=int(source['records'][i]['targets'])
            if total>=config['replay_target_quotas'][name]:break
        if total<config['replay_target_quotas'][name]:raise ValueError('replay quota exhausted')
        result[name]=selected
    return result


def write_authority(records,config,recipe,output):
    authority=output/'authority';authority.mkdir(mode=0o700)
    names=dict(tokens='tokens.uint32.bin',mask='assistant_mask.uint8.bin',index='records.idx',metadata='records.jsonl');offset=targets=0;totals=Counter()
    with ExitStack() as stack:
        handles={k:stack.enter_context((authority/v).open('xb')) for k,v in names.items()}
        for i,row in enumerate(records):
            n=len(row['mask']);t=sum(row['mask'])
            if len(row['tokens'])!=4*n or not 2<=n<=65536 or not t or row['mask'][0] or any(v not in (0,1) for v in row['mask']):raise ValueError('whole record bytes/masks')
            handles['tokens'].write(row['tokens']);handles['mask'].write(row['mask']);handles['index'].write(RECORD_INDEX.pack(offset,n,t,0))
            meta=dict(id=f'representation-bridge-{i}',offset=offset,tokens=n,targets=t,split=0,source=row['source'],source_record_id=row['source_record_id'],
                provenance=row.get('provenance'),copied_bytes_sha256=hashlib.sha256(row['tokens']+row['mask']).hexdigest())
            handles['metadata'].write((json.dumps(meta,sort_keys=True)+'\n').encode());offset+=n;targets+=t;totals[row['source']]+=t
        for f in handles.values():f.flush();os.fsync(f.fileno())
    publish(authority/'manifest.json',dict(schema=AUTHORITY_SCHEMA,status='complete',training_eligible=True,tokenizer='p50k_base',
        purpose='Operator-authorized internal bridge derivative; original preparation and production admission unchanged',
        outputs={k:dict(path=v,bytes=(authority/v).stat().st_size,sha256=sha(authority/v)) for k,v in names.items()},
        counts=dict(records=len(records),tokens=offset,assistant_target_tokens=targets,train_records=len(records),validation_records=0),
        recipe=config,recipe_sha256=sha(recipe),source_target_totals=dict(totals),selection_sha256=sha(output/'selection.json'),
        verification_sha256=sha(output/'authored-verification-private.jsonl'),candidates_sha256=sha(output/'authored-candidates-private.jsonl'),
        evaluation_cases_sha256=sha(output/'fresh-evaluation-cases.json'),first_party_registry_admission=False))
    for p in authority.iterdir():p.chmod(0o400)
    publish(output/'data-summary.json',dict(status='verified-representation-bridge-ready',authority=str(authority),authority_sha256=sha(authority/'manifest.json'),
        authored_records=768,rehearsal_records=512,records=len(records),input_tokens=offset,assistant_targets=targets,source_target_totals=dict(totals),optimizer_updates=0))


def build(a):
    if sha('ndm/data/masked_sft_dataset.py')!='3d49215c5aad63186aecbc0fdf699c26776e8167bde2f3db4aa4e55f1e84889c':raise ValueError('immutable committed loader required')
    config=read(a.recipe);plan,panel=bound_inputs(config);root=Path(config['preparation_root']);train=read(root/'training-candidates-private.json')
    if len(train)!=768:raise ValueError('authored budget')
    a.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    shutil.copyfile(root/'training-candidates-private.json',a.output/'training-cases-private.json')
    publish(a.output/'fresh-evaluation-cases.json',read(root/'fresh-evaluation-private.json')+read(root/'composition-evaluation-private.json'))
    chosen=rehearsal_ids(read(plan['recipe']['previous_training_cases']),config['rehearsal_seed'],config['rehearsal_pairs_per_family'])
    records=[]
    with ExitStack() as stack:
        old=read_source(dict(root=str(Path(config['rehearsal_data'])/'authority'),sha256=config['rehearsal_manifest_sha256'],kind='legacy',include_metadata_sources=['grounded-expansion'],target_tokens=1),stack)
        ids={old['metadata'][i]['source_record_id']:i for i in old['ids']}
        for identity in chosen:
            i=ids[identity];row=copied_row(old,i,'grounded-rehearsal',dict(original_case_id=identity,authority_sha256=config['rehearsal_manifest_sha256']))
            records.append(row)
        replay=read_source(dict(root=config['replay_root'],sha256=config['replay_sha256'],kind='legacy',include_metadata_sources=config['replay_order'],target_tokens=1),stack)
        selection=replay_ids(replay,config)
        for name,ids in selection.items():
            records.extend(copied_row(replay,i,name,dict(authority_sha256=config['replay_sha256'])) for i in ids)
    publish(a.output/'selection.json',dict(rehearsal_case_ids=chosen,replay_record_ids=selection,recipe_sha256=sha(a.recipe)))
    enc=tiktoken.get_encoding('p50k_base');count=0
    with (a.output/'authored-verification-private.jsonl').open('x') as evidence,(a.output/'authored-candidates-private.jsonl').open('x') as candidates:
        for world in (0,1):
            cases=[c for c in train if c['variant']==world];files={}
            for c in cases:
                if set(files)&set(c['files']):raise ValueError('world fixture collision')
                files.update(c['files'])
            with NativeSandbox(panel,a.output/f'authored-world-{world}') as sandbox:
                sandbox.request('setup',files=files)
                for case in cases:
                    record,receipt=teacher_record(case,panel,panel['models'][1]['system_message'],sandbox,enc)
                    if record['assistant_units']>8:raise ValueError('teacher turn budget')
                    records.append(bytes_record(record,'representation-bridge',case['id']))
                    evidence.write(json.dumps(receipt,sort_keys=True)+'\n');evidence.flush()
                    candidates.write(json.dumps(dict(id=case['id'],candidate=record),sort_keys=True)+'\n');candidates.flush()
                    count+=1
                    if count%128==0:print('VERIFIED_BRIDGE_TRAJECTORIES',count,flush=True)
        for f in (evidence,candidates):f.flush();os.fsync(f.fileno())
    if count!=768:raise ValueError('complete new curriculum required')
    random.Random(config['shuffle_seed']).shuffle(records);write_authority(records,config,a.recipe,a.output)
    print('REPRESENTATION_BRIDGE_DATA_READY',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    def terminate(*unused):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate);existed=a.output.exists()
    try:build(a)
    except BaseException as exc:
        if not existed and a.output.exists():publish(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),optimizer_updates=0))
        raise
