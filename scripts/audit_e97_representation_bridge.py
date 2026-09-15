#!/usr/bin/env python3
"""Independent bridge masks/oracles and byte-for-byte source-copy audit."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import tiktoken
from scripts.audit_e97_grounded_expansion import check_teacher,reconstruct,lines
from scripts.build_e97_representation_bridge import bound_inputs,read,rehearsal_ids
from scripts.build_e97_native_training_mix import read_source
from scripts.e97_native_execution_sandbox import validate_container
from scripts.eval_e97_native_execution import publish,sha


def audit(a):
    data=a.data;config=read(a.recipe);plan,panel=bound_inputs(config);root=Path(config['preparation_root'])
    manifest=read(data/'authority/manifest.json');summary=read(data/'data-summary.json');selection=read(data/'selection.json')
    if manifest['recipe']!=config or manifest['recipe_sha256']!=sha(a.recipe) or selection['recipe_sha256']!=sha(a.recipe):raise ValueError('recipe binding')
    if manifest['selection_sha256']!=sha(data/'selection.json') or summary['authority_sha256']!=sha(data/'authority/manifest.json'):raise ValueError('authority binding')
    if sha(data/'training-cases-private.json')!=plan['case_files']['training-candidates-private.json']:raise ValueError('training case freeze')
    if read(data/'fresh-evaluation-cases.json')!=read(root/'fresh-evaluation-private.json')+read(root/'composition-evaluation-private.json') or manifest['evaluation_cases_sha256']!=sha(data/'fresh-evaluation-cases.json'):raise ValueError('evaluation freeze')
    for file,key in (('authored-verification-private.jsonl','verification_sha256'),('authored-candidates-private.jsonl','candidates_sha256')):
        if sha(data/file)!=manifest[key]:raise ValueError('journal binding')
    cases={c['id']:c for c in read(data/'training-cases-private.json')};rows=lines(data/'authored-candidates-private.jsonl');receipts=lines(data/'authored-verification-private.jsonl')
    if len(rows)!=768 or len(receipts)!=768 or {c['id'] for c in rows}!=set(cases) or {c['id'] for c in receipts}!=set(cases):raise ValueError('authored coverage')
    receipts={c['id']:c for c in receipts};expected={};enc=tiktoken.get_encoding('p50k_base');families=Counter()
    for row in rows:
        identity=row['id'];record=row['candidate'];case=cases[identity]
        check_teacher(case,record,receipts[identity],panel)
        expected[('representation-bridge',identity)]=reconstruct(record,panel['tools'],enc);families[case['family']]+=1
        if record['assistant_units']>8:raise ValueError('teacher turn budget')
    if dict(families)!=dict(lookup=192,sum=192,edit=192,recovery=192):raise ValueError('family coverage')
    old_cases=read(plan['recipe']['previous_training_cases']);by_id={c['id']:c for c in old_cases}
    chosen=selection['rehearsal_case_ids']
    if chosen!=rehearsal_ids(old_cases,config['rehearsal_seed']) or len(set(chosen))!=512 or Counter(by_id[i]['family'] for i in chosen)!=dict(lookup=128,sum=128,edit=128,recovery=128):raise ValueError('paired rehearsal selection')
    with ExitStack() as stack:
        def payload(source,i):
            row=source['records'][i];start=int(row['offset']);n=int(row['tokens'])
            return bytes(source['maps']['tokens'][4*start:4*(start+n)]),bytes(source['maps']['mask'][start:start+n])
        old=read_source(dict(root=str(Path(config['rehearsal_data'])/'authority'),sha256=config['rehearsal_manifest_sha256'],kind='legacy',include_metadata_sources=['grounded-expansion'],target_tokens=1),stack)
        old_ids={old['metadata'][i]['source_record_id']:i for i in old['ids']}
        for case_id in chosen:
            i=old_ids[case_id];expected[('grounded-rehearsal',i)]=payload(old,i)
        replay=read_source(dict(root=config['replay_root'],sha256=config['replay_sha256'],kind='legacy',include_metadata_sources=['conversation','native','retention'],target_tokens=1),stack)
        for j,name in enumerate(('conversation','native','retention')):
            ids=[i for i in replay['ids'] if replay['metadata'][i]['source']==name];random.Random(config['replay_seed']+j).shuffle(ids)
            selected=[];total=0
            for i in ids:
                tb,mb=payload(replay,i);expected[(name,i)]=(tb,mb);selected.append(i);total+=sum(mb)
                if total>=config['replay_target_quotas'][name]:break
            if total<config['replay_target_quotas'][name] or selected!=selection['replay_record_ids'][name]:raise ValueError('independent replay selection')
        source=read_source(dict(root=str(data/'authority'),sha256=sha(data/'authority/manifest.json'),kind='legacy',include_metadata_sources=sorted({i[0] for i in expected}),target_tokens=1),stack)
        offset=targets=0;totals=Counter();seen=set()
        for i,row in enumerate(source['records']):
            m=source['metadata'][i];n=int(row['tokens']);t=int(row['targets']);key=(m['source'],m['source_record_id'])
            if key in seen or key not in expected or int(row['offset'])!=offset or int(row['split'])!=0 or any(m[k]!=v for k,v in dict(offset=offset,tokens=n,targets=t,split=0).items()):raise ValueError('index/metadata/identity')
            tb,mb=payload(source,i)
            if (tb,mb)!=expected[key] or sum(mb)!=t or mb[0] or not 2<=n<=65536 or any(x not in (0,1) for x in mb):raise ValueError('exact source bytes/masks')
            if hashlib.sha256(tb+mb).hexdigest()!=m['copied_bytes_sha256']:raise ValueError('record hash')
            if key[0]=='grounded-rehearsal' and m['provenance']!=dict(original_case_id=old['metadata'][key[1]]['source_record_id'],authority_sha256=config['rehearsal_manifest_sha256']):raise ValueError('rehearsal provenance')
            if key[0] in config['replay_order'] and m['provenance']!=dict(authority_sha256=config['replay_sha256']):raise ValueError('replay provenance')
            seen.add(key);offset+=n;targets+=t;totals[key[0]]+=t
        if seen!=set(expected) or len(source['maps']['tokens'])!=offset*4 or len(source['maps']['mask'])!=offset:raise ValueError('payload coverage')
    if summary['records']!=len(seen) or summary['input_tokens']!=offset or summary['assistant_targets']!=targets or summary['source_target_totals']!=dict(totals) or manifest['source_target_totals']!=dict(totals):raise ValueError('summary totals')
    if manifest['counts']!=dict(records=len(seen),tokens=offset,assistant_target_tokens=targets,train_records=len(seen),validation_records=0):raise ValueError('manifest counts')
    folders=sorted(data.glob('authored-world-*'))
    if len(folders)!=2:raise ValueError('sandbox count')
    for folder in folders:
        before=read(folder/'container-before.json');terminal=read(folder/'container-terminal.json');cleanup=read(folder/'cleanup.json');nonce=before['Config']['Labels']['emender.native-qualification']
        validate_container(before,panel['image_id'],nonce)
        if terminal['Id']!=before['Id'] or terminal['Config']['Labels']['emender.native-qualification']!=nonce or cleanup!=dict(container_id=before['Id'],removed=True):raise ValueError('owned cleanup')
    publish(data/'result-audit.json',dict(passed=True,authority_sha256=sha(data/'authority/manifest.json'),data_summary_sha256=sha(data/'data-summary.json'),
        records=len(seen),authored_records=768,rehearsal_records=512,input_tokens=offset,assistant_targets=targets,source_target_totals=dict(totals),
        actual_native_observations_bound=True,independent_oracles_and_masks=True,unchanged_replay_and_rehearsal=True,owned_sandboxes_cleaned=2,optimizer_updates=0))
    print('REPRESENTATION_BRIDGE_DATA_AUDIT_PASSED',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);audit(p.parse_args())
