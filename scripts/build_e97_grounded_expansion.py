#!/usr/bin/env python3
"""New bounded internal SFT derivative: verified paired curriculum and real corrections."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import numpy as np
import tiktoken
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA,RECORD_INDEX
from scripts.build_e97_native_training_mix import read_source
from scripts.e97_grounded_curriculum import fixtures,teacher_record
from scripts.e97_native_onpolicy_canary import candidate
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.eval_e97_native_execution import publish,sha


def read(p):return json.loads(Path(p).read_text())


def bytes_record(c,source,identity,provenance=None):
    tokens=np.asarray(c['token_ids'],dtype='<u4').tobytes();mask=bytes(c['assistant_mask'])
    if len(tokens)!=4*len(mask) or not 1<=sum(mask) or mask[0] or len(mask)>65536 or any(v not in (0,1) for v in mask):raise ValueError('whole masked native record')
    return dict(tokens=tokens,mask=mask,source=source,source_record_id=identity,provenance=provenance)


def load_candidates(config,enc):
    root=Path(config['canary_root'])
    for name,expected in config['canary_hashes'].items():
        if sha(root/name)!=expected:raise ValueError('canary authority')
    audit=read(root/'candidate-audit.json');summary=read(root/'results/summary.json');panel=read(root/'panel.json')
    results={r['id']:r for r in summary['results']};unique={};occurrences=[]
    for receipt in audit['records']:
        folder=root/'results'/receipt['id'];p=folder/'candidate-private.json';r=results[receipt['id']]
        if sha(p)!=receipt['sha256'] or sha(folder/'episode-private.json')!=r['episode_sha256']:raise ValueError('candidate/episode binding')
        c=read(p);episode=read(folder/'episode-private.json')
        if c['training_eligible'] is not False or r['continuation']['candidate_sha256']!=sha(p) or not r['continuation']['verified']:raise ValueError('original candidate verdict')
        if c!=candidate(c['messages'],panel['tools'],c['prefix_assistants_unsupervised'],enc):raise ValueError('native token/mask reconstruction')
        success=receipt['kind']=='autonomous-success'
        if success!=bool(episode['grade']['success']) or success!=bool(r['reward']):raise ValueError('immutable autonomous reward')
        if not success:
            teacher=read(folder/'teacher-private.json')
            if not teacher['verified'] or teacher['original_reward']!=0 or teacher['messages']!=c['messages']:raise ValueError('verified teacher suffix identity')
        row=bytes_record(c,'onpolicy-success' if success else 'onpolicy-teacher',receipt['id'],[receipt['id']])
        key=hashlib.sha256(row['tokens']+row['mask']).hexdigest()
        if key in unique:unique[key]['provenance'].append(receipt['id'])
        else:unique[key]=row
        occurrences.append(dict(id=receipt['id'],kind=receipt['kind'],candidate_sha256=sha(p),record_bytes_sha256=key))
    if len(occurrences)!=32 or len(unique)!=18 or sum(sum(r['mask']) for r in unique.values())!=3447:raise ValueError('frozen candidate inventory')
    return list(unique.values()),dict(occurrences=occurrences,unique_records=18,unique_targets=3447,original_eligibility_unchanged=True)


def freeze(config,output):
    if config['operator_internal_training_authorized'] is not True or config['steps']!=32 or config['learning_rate']!=1e-5:raise ValueError('new bounded tranche')
    if sha(config['generation_panel'])!=config['generation_panel_sha256']:raise ValueError('native panel identity')
    panel=read(config['generation_panel'])
    train=fixtures(config['seed'],config['training_pairs'],'train')
    fresh=fixtures(config['validation_seed'],dict.fromkeys(('lookup','sum','edit','recovery'),2),'fresh')
    transfer=fixtures(config['transfer_seed'],dict.fromkeys(('lookup','sum','edit','recovery'),2),'transfer')
    if len(train)!=2048 or len(fresh)!=16 or len(transfer)!=16:raise ValueError('curriculum counts')
    forbidden={c['answer'] for c in fresh+transfer+panel['cases'] if c['family']!='edit'}
    old_path=Path(config['previous_correction_data'])/'fresh-evaluation-cases.json'
    if sha(old_path)!=config['previous_evaluation_sha256']:raise ValueError('previous evaluation identity')
    old=read(old_path)
    forbidden.update(c['answer'] for c in old if c['family']!='edit')
    from scripts.build_e97_grounding_correction import fixtures as previous_fixtures
    canary_path=Path(config['canary_root'])/'panel.json'
    if sha(canary_path)!=config['canary_hashes']['panel.json']:raise ValueError('canary panel identity')
    known=previous_fixtures('grounding-correction-train-20260912-v1',1024)+read(canary_path)['cases']+old+panel['cases']
    known_answers={c['answer'] for c in known if c['family']!='edit'}
    if any(c['answer'] in known_answers for c in fresh+transfer if c['family']!='edit'):raise ValueError('new evaluation overlaps known correction/canary/diagnostic answers')
    if any(c['answer'] in forbidden or c['answer'] in c['prompt'] for c in train if c['family']!='edit'):raise ValueError('answer leakage/overlap')
    for group in (train,fresh,transfer):
        for i in range(0,len(group),2):
            a,b=group[i:i+2]
            if a['prompt']!=b['prompt'] or set(a['files'])!=set(b['files']) or a['files']==b['files']:raise ValueError('counterfactual pair')
            same_outcome=(a['answer']==b['answer']) if a['family']!='edit' else (a['expected_output']==b['expected_output'])
            if same_outcome:raise ValueError('environment-dependent outcome')
    # Both evaluation cohorts are published before any training demonstration is executed.
    publish(output/'fresh-evaluation-cases.json',fresh+transfer)
    publish(output/'training-cases-private.json',train)
    return train,panel


def build(args):
    config=read(args.recipe)
    if sha(Path('ndm/data/masked_sft_dataset.py'))!='3d49215c5aad63186aecbc0fdf699c26776e8167bde2f3db4aa4e55f1e84889c':raise ValueError('use committed immutable loader export; unrelated dirty loader excluded')
    args.output.mkdir(mode=0o700,parents=True,exist_ok=False)
    train,panel=freeze(config,args.output);enc=tiktoken.get_encoding('p50k_base')
    if sha(args.output/'training-cases-private.json')!='a21f68ad922f5d505980381ef17fa57bf624e1a179d986cf86d60ac8ee30540e' or sha(args.output/'fresh-evaluation-cases.json')!='36921d4b718c3d8e0bf88c9fc1cc04ae15ada219229442eb420e5843a84c9331':raise ValueError('preflight case identities')
    records,canary_audit=load_candidates(config,enc);publish(args.output/'canary-derivative-audit.json',canary_audit)
    source_system=panel['models'][1]['system_message'];count=0;reused=set();completion=None
    if getattr(args,'completed_from',None) is not None:
        from scripts.e97_grounded_resume import load_reuse
        retained,completion=load_reuse(args.completed_from,train,panel,enc)
        for row in retained:
            records.append(bytes_record(row['candidate'],'grounded-expansion',row['id']));reused.add(row['id'])
        count=len(reused);publish(args.output/'completion-provenance.json',completion)
        print('VERIFIED_RECORDS_REUSED',count,flush=True)
    with (args.output/'authored-verification-private.jsonl').open('x') as evidence, (args.output/'authored-candidates-private.jsonl').open('x') as candidates:
        if completion:
            evidence.write((args.completed_from/'authored-verification-private.jsonl').read_text());evidence.flush()
            candidates.write((args.completed_from/'authored-candidates-private.jsonl').read_text());candidates.flush()
        for world in (0,1):
            cases=[c for c in train if c['variant']==world and c['id'] not in reused];files={}
            if not cases:continue
            for c in cases:
                if set(files)&set(c['files']):raise ValueError('world fixture collision')
                files.update(c['files'])
            with NativeSandbox(panel,args.output/f'authored-world-{world}') as sandbox:
                sandbox.request('setup',files=files)
                for c in cases:
                    record,receipt=teacher_record(c,panel,source_system,sandbox,enc)
                    records.append(bytes_record(record,'grounded-expansion',c['id']))
                    evidence.write(json.dumps(receipt,sort_keys=True)+'\n');evidence.flush()
                    candidates.write(json.dumps(dict(id=c['id'],candidate=record),sort_keys=True)+'\n');candidates.flush()
                    count+=1
                    if count%128==0:print('VERIFIED_EXPANSION_TRAJECTORIES',count,flush=True)
        for f in (evidence,candidates):f.flush();os.fsync(f.fileno())
    if count!=2048:raise ValueError('complete frozen curriculum required')
    if completion:
        from scripts.e97_grounded_resume import verify_reuse
        verify_reuse(completion,args.output)
    with ExitStack() as stack:
        for index,(name,quota) in enumerate(config['replay_target_quotas'].items()):
            src=read_source(dict(root=config['replay_root'],sha256=config['replay_sha256'],kind='legacy',include_metadata_sources=[name],target_tokens=quota),stack)
            ids=src['ids'].copy();random.Random(981417+index).shuffle(ids);total=0
            for i in ids:
                r=src['records'][i];start=int(r['offset']);n=int(r['tokens'])
                records.append(dict(tokens=bytes(src['maps']['tokens'][start*4:(start+n)*4]),mask=bytes(src['maps']['mask'][start:start+n]),source=name,source_record_id=i,provenance=None))
                total+=int(r['targets'])
                if total>=quota:break
            if total<quota:raise ValueError('replay quota exhausted')
    random.Random(981417).shuffle(records);authority=args.output/'authority';authority.mkdir(mode=0o700)
    names=dict(tokens='tokens.uint32.bin',mask='assistant_mask.uint8.bin',index='records.idx',metadata='records.jsonl');offset=targets=0;totals={}
    with ExitStack() as stack:
        handles={k:stack.enter_context((authority/v).open('xb')) for k,v in names.items()}
        for i,row in enumerate(records):
            n=len(row['mask']);t=sum(row['mask'])
            if len(row['tokens'])!=4*n or not 1<=n<=65536 or not t or row['mask'][0] or any(v not in (0,1) for v in row['mask']):raise ValueError('source record')
            handles['tokens'].write(row['tokens']);handles['mask'].write(row['mask']);handles['index'].write(RECORD_INDEX.pack(offset,n,t,0))
            metadata=dict(id=f'grounded-expansion-{i}',offset=offset,tokens=n,targets=t,split=0,source=row['source'],source_record_id=row['source_record_id'],
                original_candidate_ids=row.get('provenance'),copied_bytes_sha256=hashlib.sha256(row['tokens']+row['mask']).hexdigest())
            handles['metadata'].write((json.dumps(metadata,sort_keys=True)+'\n').encode());offset+=n;targets+=t;totals[row['source']]=totals.get(row['source'],0)+t
        for f in handles.values():f.flush();os.fsync(f.fileno())
    publish(authority/'manifest.json',dict(schema=AUTHORITY_SCHEMA,status='complete',training_eligible=True,tokenizer='p50k_base',
        purpose='New operator-authorized internal grounded SFT tranche; raw candidate eligibility and production registry unchanged',
        outputs={k:dict(path=v,bytes=(authority/v).stat().st_size,sha256=sha(authority/v)) for k,v in names.items()},
        counts=dict(records=len(records),tokens=offset,assistant_target_tokens=targets,train_records=len(records),validation_records=0),
        recipe=config,recipe_sha256=sha(args.recipe),source_target_totals=totals,
        completion_provenance_sha256=sha(args.output/'completion-provenance.json') if completion else None,
        verification_sha256=sha(args.output/'authored-verification-private.jsonl'),canary_audit_sha256=sha(args.output/'canary-derivative-audit.json'),
        evaluation_cases_sha256=sha(args.output/'fresh-evaluation-cases.json'),independent_holdout_claim=False,first_party_registry_admission=False))
    for p in authority.iterdir():p.chmod(0o400)
    publish(args.output/'data-summary.json',dict(status='verified-grounded-expansion-ready',authority=str(authority),authority_sha256=sha(authority/'manifest.json'),
        authored_records=count,unique_canary_records=18,records=len(records),input_tokens=offset,assistant_targets=targets,source_target_totals=totals,
        optimizer_updates=0,first_party_registry_admission=False))
    print('GROUNDED_EXPANSION_DATA_READY',json.dumps(totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--completed-from',type=Path);a=p.parse_args()
    def terminate(_signum,_frame):raise SystemExit(143)
    signal.signal(signal.SIGTERM,terminate);existed=a.output.exists()
    try:build(a)
    except BaseException as exc:
        if not existed and a.output.exists():publish(a.output/'failure.json',dict(type=type(exc).__name__,message=str(exc),optimizer_updates=0))
        raise
