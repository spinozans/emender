#!/usr/bin/env python3
"""Independent reconstruction/receipt/source-copy audit before packing or training."""
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import tiktoken
from scripts.build_e97_native_training_mix import read_source,DTYPE
from scripts.e97_native_execution_cases import context,grade
from scripts.e97_open_swe_native_codec import render
from scripts.e97_native_execution_sandbox import validate_container
from scripts.eval_e97_native_execution import publish,sha

RECIPE_SHA='12a6d13088dfdcfd1484e413ad13a402bc2ce99dfdb64651829f3c5775e5987f'
TRAIN_SHA='a21f68ad922f5d505980381ef17fa57bf624e1a179d986cf86d60ac8ee30540e'
EVAL_SHA='36921d4b718c3d8e0bf88c9fc1cc04ae15ada219229442eb420e5843a84c9331'


def read(p):return json.loads(Path(p).read_text())
def lines(p):return [json.loads(line) for line in Path(p).read_text().splitlines()]

def reconstruct(record,tools,enc):
    pieces,meta=render(dict(messages=record['messages'],tools=tools),enc)
    prefix=record['prefix_assistants_unsupervised']
    if record['training_eligible'] is not False or not 0<=prefix<meta['assistant_units']:raise ValueError('candidate eligibility/prefix')
    text=''.join(p for p,_ in pieces);ids=enc.encode_ordinary(text)
    boundaries={0:0};position=0
    for i,token in enumerate(ids):position+=len(enc.decode_single_token_bytes(token));boundaries[position]=i+1
    mask=bytearray(len(ids));position=0;assistant=0
    for text_part,target in pieces:
        end=position+len(text_part.encode())
        if target:
            if assistant>=prefix:
                if position not in boundaries or end not in boundaries:raise ValueError('target straddles token boundary')
                left,right=boundaries[position],boundaries[end];mask[left:right]=b'\1'*(right-left)
            assistant+=1
        position=end
    if ids!=record['token_ids'] or list(mask)!=record['assistant_mask'] or sum(mask)!=record['targets'] or assistant!=record['assistant_units']:raise ValueError('independent token/mask reconstruction')
    if hashlib.sha256(text.encode()).hexdigest()!=record['text_sha256'] or enc.decode_bytes(ids)!=text.encode():raise ValueError('text identity')
    return np.asarray(ids,dtype='<u4').tobytes(),bytes(mask)


def parse_view(text,path):
    header=f"Here's the result of running `cat -n` on {path}:\n"
    if not text.startswith(header):raise ValueError('view header')
    content=[]
    for i,line in enumerate(text[len(header):].splitlines(),1):
        number,body=line.split('\t',1)
        if int(number.strip())!=i:raise ValueError('view numbering')
        content.append(body)
    return json.loads('\n'.join(content))


def check_teacher(case,record,receipt,panel):
    messages=record['messages'];calls=receipt['calls'];cursor=2;call_index=0;finished=False
    system=panel['models'][1]['system_message'] if case['source_style'] else context('system',panel['system'])
    if messages[:2]!=[system,context('user',case['prompt'])]:raise ValueError('full causal initial context')
    snapshot=dict(case['files'])
    while cursor<len(messages):
        message=messages[cursor];cursor+=1
        if message['role']!='assistant' or len(message['tool_calls'])!=1:raise ValueError('authored assistant framing')
        function=message['tool_calls'][0]['function'];name=function['name'];args=json.loads(function['arguments'])
        if name=='finish':
            if args!={'message':case['answer']} or cursor!=len(messages):raise ValueError('verified finish')
            finished=True;break
        call=calls[call_index];call_index+=1
        if call['request']!={'name':name,'arguments':args} or messages[cursor]!=call['result']['message']:raise ValueError('actual call/observation binding')
        if name=='execute_bash' and call['result']['exit_code']!=0:raise ValueError('computation exit')
        if name=='str_replace_editor' and args.get('command')=='view' and args['path']=='/testbed/'+case['output_path']:
            snapshot[case['output_path']]=json.dumps(parse_view(call['result']['message']['content'],args['path']))
        cursor+=1
    if not finished or call_index!=len(calls):raise ValueError('complete native history')
    prefix=(1 if case['family']=='edit' else 2) if case['authored_failure_prefix'] else 0
    if record['prefix_assistants_unsupervised']!=prefix or receipt['prefix_assistants_unsupervised']!=prefix:raise ValueError('failure-prefix mask')
    for call in calls[:prefix]:
        if not call['result']['message']['content'].startswith('ERROR:'):raise ValueError('authored failure observation')
    integrity=receipt['external_integrity_receipt']['result']
    actual=json.loads(integrity['message']['content'].splitlines()[0])
    expected={'/testbed/'+p:hashlib.sha256(raw.encode()).hexdigest() for p,raw in case['files'].items()}
    if integrity['exit_code']!=0 or actual!=expected:raise ValueError('actual original-file integrity')
    if not grade(case,case['answer'],calls,snapshot)['success']:raise ValueError('independent task oracle')


def audit_partial(a):
    data=a.data;config_path=Path(__file__).resolve().parents[1]/'configs/pi/e97-grounded-expansion-v1.json'
    if sha(config_path)!=RECIPE_SHA or (data/'authority').exists():raise ValueError('partial attempt identity')
    config=read(config_path)
    if sha(data/'training-cases-private.json')!=TRAIN_SHA or sha(data/'fresh-evaluation-cases.json')!=EVAL_SHA:raise ValueError('frozen partial cases')
    panel_path=Path(config['generation_panel'])
    if sha(panel_path)!=config['generation_panel_sha256']:raise ValueError('native panel')
    panel=read(panel_path);enc=tiktoken.get_encoding('p50k_base');cases=read(data/'training-cases-private.json')
    planned=[c for world in (0,1) for c in cases if c['variant']==world]
    paths=[data/'authored-verification-private.jsonl',data/'authored-candidates-private.jsonl'];before={p.name:sha(p) for p in paths}
    receipts=lines(paths[0]);records=lines(paths[1])
    if not 0<len(records)<2048 or len(receipts)!=len(records):raise ValueError('partial complete-record journal coverage')
    families=Counter();worlds=Counter();targets=0
    for case,row,receipt in zip(planned,records,receipts):
        if case['id']!=row['id'] or receipt['id']!=row['id']:raise ValueError('partial planned prefix')
        check_teacher(case,row['candidate'],receipt,panel)
        _,mask=reconstruct(row['candidate'],panel['tools'],enc);targets+=sum(mask);families[case['family']]+=1;worlds[case['variant']]+=1
    cleanups=[]
    for world in worlds:
        folder=data/f'authored-world-{world}';initial=read(folder/'container-before.json');terminal=read(folder/'container-terminal.json');cleanup=read(folder/'cleanup.json')
        nonce=initial['Config']['Labels']['emender.native-qualification'];validate_container(initial,panel['image_id'],nonce)
        if terminal['Id']!=initial['Id'] or terminal['Config']['Labels']['emender.native-qualification']!=nonce or cleanup!={'container_id':initial['Id'],'removed':True}:raise ValueError('partial cleanup')
        cleanups.append(cleanup)
    if before!={p.name:sha(p) for p in paths}:raise ValueError('partial journals changed during audit')
    result=dict(verified_records_retained=True,data_authority_ready=False,training_eligible=False,completed=len(records),remaining=2048-len(records),
        completed_targets=targets,families=dict(families),worlds=dict(worlds),journal_sha256=before,cleanups=cleanups,
        completed_ids=[c['id'] for c in planned[:len(records)]],remaining_ids=[c['id'] for c in planned[len(records):]],
        failure_sha256=sha(data/'failure.json'),optimizer_updates=0)
    publish(data/'partial-result-audit.json',result)
    print('PARTIAL_GROUNDED_DATA_AUDIT',json.dumps({k:v for k,v in result.items() if not k.endswith('_ids')},sort_keys=True),flush=True)


def audit(a):
    data=a.data;summary=read(data/'data-summary.json');authority=data/'authority';manifest=read(authority/'manifest.json');config=manifest['recipe']
    config_path=Path(__file__).resolve().parents[1]/'configs/pi/e97-grounded-expansion-v1.json'
    if sha(config_path)!=RECIPE_SHA or config!=read(config_path):raise ValueError('frozen recipe contents')
    if manifest['recipe_sha256']!=RECIPE_SHA or summary['authority_sha256']!=sha(authority/'manifest.json') or summary['optimizer_updates']!=0:raise ValueError('data authority binding')
    if sha(data/'training-cases-private.json')!=TRAIN_SHA or sha(data/'fresh-evaluation-cases.json')!=EVAL_SHA:raise ValueError('frozen cases')
    panel_path=Path(config['generation_panel'])
    if sha(panel_path)!=config['generation_panel_sha256']:raise ValueError('native panel')
    panel=read(panel_path);enc=tiktoken.get_encoding('p50k_base')
    cases={c['id']:c for c in read(data/'training-cases-private.json')}
    receipts=lines(data/'authored-verification-private.jsonl');records=lines(data/'authored-candidates-private.jsonl')
    if len(cases)!=2048 or len(records)!=2048 or len(receipts)!=2048:raise ValueError('authored coverage')
    if {r['id'] for r in records}!=set(cases) or {r['id'] for r in receipts}!=set(cases):raise ValueError('authored duplicate/missing ID')
    receipts={r['id']:r for r in receipts};expected={};families=Counter()
    for row in records:
        identity=row['id'];record=row['candidate'];case=cases[identity]
        check_teacher(case,record,receipts[identity],panel)
        expected[('grounded-expansion',identity)]=reconstruct(record,panel['tools'],enc);families[case['family']]+=1
    if dict(families)!=dict(lookup=256,sum=256,edit=768,recovery=768):raise ValueError('family curriculum')
    canary=Path(config['canary_root'])
    for name,digest in config['canary_hashes'].items():
        if sha(canary/name)!=digest:raise ValueError('original canary authority')
    original=read(canary/'candidate-audit.json');canary_panel=read(canary/'panel.json');seen={};provenance={}
    for r in original['records']:
        p=canary/'results'/r['id']/'candidate-private.json'
        if sha(p)!=r['sha256']:raise ValueError('original candidate bytes')
        tb,mb=reconstruct(read(p),canary_panel['tools'],enc);key=hashlib.sha256(tb+mb).hexdigest()
        if key in seen:provenance[seen[key]].append(r['id']);continue
        identity=('onpolicy-success' if r['kind']=='autonomous-success' else 'onpolicy-teacher',r['id'])
        seen[key]=identity;expected[identity]=(tb,mb);provenance[identity]=[r['id']]
    if len(seen)!=18 or sum(sum(expected[i][1]) for i in seen.values())!=3447:raise ValueError('canary unique inventory')
    if sha(data/'authored-verification-private.jsonl')!=manifest['verification_sha256'] or sha(data/'canary-derivative-audit.json')!=manifest['canary_audit_sha256']:raise ValueError('receipt hashes')
    for world in (0,1):
        folder=data/f'authored-world-{world}';before=read(folder/'container-before.json');terminal=read(folder/'container-terminal.json');cleanup=read(folder/'cleanup.json')
        nonce=before['Config']['Labels']['emender.native-qualification'];validate_container(before,panel['image_id'],nonce)
        if terminal['Id']!=before['Id'] or terminal['Config']['Labels']['emender.native-qualification']!=nonce or cleanup!={'container_id':before['Id'],'removed':True}:raise ValueError('owned sandbox cleanup')
    with ExitStack() as stack:
        replay=read_source(dict(root=config['replay_root'],sha256=config['replay_sha256'],kind='legacy',include_metadata_sources=list(config['replay_target_quotas']),target_tokens=1),stack)
        for j,(name,quota) in enumerate(config['replay_target_quotas'].items()):
            ids=[i for i in replay['ids'] if replay['metadata'][i]['source']==name];random.Random(981417+j).shuffle(ids);total=0
            for i in ids:
                row=replay['records'][i];start=int(row['offset']);n=int(row['tokens']);mb=bytes(replay['maps']['mask'][start:start+n])
                expected[(name,i)]=(bytes(replay['maps']['tokens'][4*start:4*(start+n)]),mb);total+=sum(mb)
                if total>=quota:break
            if total<quota:raise ValueError('replay quota')
        source=read_source(dict(root=str(authority),sha256=sha(authority/'manifest.json'),kind='legacy',include_metadata_sources=list({i[0] for i in expected}),target_tokens=1),stack)
        offset=0;targets=0;totals=Counter();consumed=set()
        for i,row in enumerate(source['records']):
            m=source['metadata'][i];n=int(row['tokens']);t=int(row['targets']);identity=(m['source'],m['source_record_id'])
            if int(row['offset'])!=offset or int(row['split'])!=0 or m['offset']!=offset or m['tokens']!=n or m['targets']!=t or m['split']!=0 or identity in consumed:raise ValueError('record index/metadata/uniqueness')
            tb=bytes(source['maps']['tokens'][offset*4:(offset+n)*4]);mb=bytes(source['maps']['mask'][offset:offset+n])
            if (tb,mb)!=expected[identity] or sum(mb)!=t or mb[0] or len(mb)!=n or not 2<=n<=65536:raise ValueError('source bytes/masks')
            if m['copied_bytes_sha256']!=hashlib.sha256(tb+mb).hexdigest():raise ValueError('record bytes hash')
            if identity in provenance and m['original_candidate_ids']!=provenance[identity]:raise ValueError('duplicate provenance')
            consumed.add(identity);offset+=n;targets+=t;totals[identity[0]]+=t
        if consumed!=set(expected) or len(source['maps']['tokens'])!=4*offset or len(source['maps']['mask'])!=offset:raise ValueError('full payload coverage')
    if dict(totals)!=manifest['source_target_totals'] or dict(totals)!=summary['source_target_totals'] or summary['assistant_targets']!=targets or summary['input_tokens']!=offset or summary['records']!=len(consumed):raise ValueError('summary totals')
    counts=dict(records=len(consumed),tokens=offset,assistant_target_tokens=targets,train_records=len(consumed),validation_records=0)
    if manifest['counts']!=counts:raise ValueError('manifest counts')
    result=dict(passed=True,authority_sha256=sha(authority/'manifest.json'),data_summary_sha256=sha(data/'data-summary.json'),records=len(consumed),authored_records=2048,
        unique_canary_records=18,unique_canary_targets=3447,source_target_totals=dict(totals),input_tokens=offset,assistant_targets=targets,
        native_calls_observations_bound=True,independent_masks_and_oracles=True,unchanged_replay_bytes=True,owned_sandboxes_cleaned=2,optimizer_updates=0)
    publish(data/'result-audit.json',result);print('GROUNDED_EXPANSION_AUDIT_PASSED',json.dumps(result,sort_keys=True),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--partial',action='store_true');a=p.parse_args()
    audit_partial(a) if a.partial else audit(a)
