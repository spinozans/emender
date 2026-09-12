#!/usr/bin/env python3
"""Verified authored native trajectories plus unchanged admitted replay records."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import shlex
import numpy as np
import tiktoken
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA, RECORD_INDEX
from scripts.build_e97_native_training_mix import read_source
from scripts.e97_native_execution_cases import context, cases as eval_cases
from scripts.e97_native_execution_sandbox import NativeSandbox
from scripts.e97_open_swe_native_codec import compact, native_turn, render, encode, vocabulary
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode
from scripts.eval_e97_native_execution import publish, sha


def fixtures(seed, count):
    result=[]
    for i in range(count):
        h=hashlib.sha256(f'{seed}:{i}'.encode()).hexdigest(); base=f'train/{h[:8]}'
        path=f'/testbed/{base}/input.json'; family=('lookup','sum','edit','recovery')[i%4]
        a,b=int(h[24:30],16),int(h[30:36],16)
        if family=='lookup':
            data=dict(active='selected',values=dict(selected=h[4:24],other=h[36:56]))
            files={path:json.dumps(data)}; answer=data['values']['selected']
            prompt=f'Read {path}. Return exactly the value selected by its active key using finish. Do not change files.'
        elif family=='sum':
            data=dict(left=a,right=b);files={path:json.dumps(data)};answer=str(a+b)
            prompt=f'Read {path}. Return exactly the decimal sum of left and right using finish. Do not change files.'
        elif family=='edit':
            data=dict(ticket=h[4:24],count=a,owner='fixture-owner');files={path:json.dumps(data)};answer='done'
            prompt=f'Read {path}. Write /testbed/{base}/result.json with count increased by 7, preserving other fields and the original file. Verify the result, then finish with exactly done.'
        else:
            path=f'/testbed/{base}/catalog.json';data={'active_path':f'/testbed/{base}/value.json'}
            files={path:json.dumps(data),data['active_path']:json.dumps({'value':h[4:24]})};answer=h[4:24]
            prompt=f'First read /testbed/{base}/missing.json using str_replace_editor view and wait for its result. If absent, consult {path}, read active_path, and finish with exactly its value field. Do not change files.'
        result.append(dict(id=f'correction-{i:04d}',family=family,path=path,base=base,files=files,answer=answer,
                           prompt=prompt,source_style=bool((i//4)%2)))
    return result


def training_record(c, panel, source_system, sandbox, enc, lengths):
    e=NativeEpisode(panel['tools'],enc)
    e.append_source_message(source_system if c['source_style'] else context('system',panel['system']))
    e.append_source_message(context('user',c['prompt']));calls=[]
    def action(name, arguments):
        message=dict(role='assistant',content=None,reasoning_content=None,think=None,
                     tool_calls=[{'type':'function','function':{'name':name,'arguments':compact(arguments)}}])
        turn=e.accept_generated_turn(native_turn(message))
        if name=='finish':return
        reply=sandbox.request('execute',call=turn.backend_call())
        if 'dispatch_error' in reply:raise ValueError('authored dispatch failure')
        calls.append(dict(request=turn.backend_call(),result=reply['result']))
        e.append_observation(reply['result']['message'])
        return reply['result']['message']['content']
    def view(path):return action('str_replace_editor',dict(command='view',path=path))
    if c['family']=='recovery':
        if not view(f'/testbed/{c["base"]}/missing.json').startswith('ERROR:'):
            raise ValueError('required missing-file error absent')
    observation=view(c['path']);raw=c['files'][c['path']]
    if observation.startswith('ERROR:') or raw not in observation:raise ValueError('input observation mismatch')
    observed=json.loads(observation[observation.index(raw):observation.index(raw)+len(raw)])
    if c['family']=='lookup':answer=observed['values'][observed['active']]
    elif c['family']=='sum':
        program=f'import json; d=json.load(open({c["path"]!r})); print(d["left"]+d["right"])'
        text=action('execute_bash',dict(command='python -c '+shlex.quote(program),timeout=10))
        answer=str(observed['left']+observed['right'])
        if calls[-1]['result']['exit_code']!=0 or text.splitlines()[0]!=answer:
            raise ValueError('observed arithmetic verification')
    elif c['family']=='edit':
        wanted={**observed,'count':observed['count']+7};dest=f'/testbed/{c["base"]}/result.json'
        text=action('str_replace_editor',dict(command='create',path=dest,file_text=compact(wanted)))
        if text.startswith('ERROR:') or compact(wanted) not in view(dest):raise ValueError('edit verification')
        if raw not in view(c['path']):raise ValueError('original changed')
        answer='done'
    else:
        dest=observed['active_path'];text=view(dest);value=c['files'][dest]
        if text.startswith('ERROR:') or value not in text:raise ValueError('recovery observation mismatch')
        answer=json.loads(text[text.index(value):text.index(value)+len(value)])['value']
    if answer!=c['answer']:raise ValueError('independent expected answer mismatch')
    action('finish',dict(message=answer))
    pieces,_=render(dict(tools=panel['tools'],messages=e.source_messages()),enc)
    tb,mask,text=encode(pieces,enc,lengths)
    if not e.finished or text[:-1]!=e.text():raise ValueError('codec/runtime mismatch')
    return dict(tokens=tb.tobytes(),mask=mask.tobytes(),source='correction',source_record_id=c['id'],
                evidence=dict(id=c['id'],family=c['family'],source_style=c['source_style'],calls=calls,
                              answer_verified=True,text_sha256=hashlib.sha256(text.encode()).hexdigest()))


def build(args):
    recipe=json.loads(args.recipe.read_text())
    if recipe['operator_internal_training_authorized'] is not True or recipe['steps']!=32 or recipe['records']!=1024:
        raise ValueError('bounded authorized recipe')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True,mode=0o700)
    if sha(recipe['generation_panel'])!=recipe['generation_panel_sha256']:raise ValueError('panel identity')
    panel=json.loads(Path(recipe['generation_panel']).read_text());source_system=panel['models'][1]['system_message']
    fresh=eval_cases(recipe['validation_seed']);publish(args.output/'fresh-evaluation-cases.json',fresh)
    selected=fixtures(recipe['seed'],recipe['records'])
    forbidden={c['answer'] for c in fresh+panel['cases'] if c['family']!='edit'}
    if any(c['answer'] in forbidden for c in selected if c['family']!='edit'):raise ValueError('diagnostic answer overlap')
    enc=tiktoken.get_encoding('p50k_base');lengths,identity=vocabulary(enc)
    if identity!=panel['tokenizer_vocabulary_sha256']:raise ValueError('tokenizer')
    files={path.removeprefix('/testbed/'):text for c in selected for path,text in c['files'].items()}
    records=[];evidence=[]
    with NativeSandbox(panel,args.output/'authored-executor') as sandbox:
        sandbox.request('setup',files=files)
        for c in selected:
            row=training_record(c,panel,source_system,sandbox,enc,lengths)
            evidence.append(row.pop('evidence'));records.append(row)
            if len(records)%128==0:print('VERIFIED_CORRECTION_TRAJECTORIES',len(records),flush=True)
    publish(args.output/'correction-verification.json',dict(status='passed',records=evidence,model_loaded=False,
            recipe_sha256=sha(args.recipe),source_panel_sha256=recipe['generation_panel_sha256']))
    with ExitStack() as stack:
        for index,(name,quota) in enumerate(recipe['replay_target_quotas'].items()):
            spec=dict(root=recipe['replay_root'],sha256=recipe['replay_sha256'],kind='legacy',
                      include_metadata_sources=[name],target_tokens=quota)
            src=read_source(spec,stack);ids=src['ids'].copy();random.Random(974223+index).shuffle(ids);total=0
            for i in ids:
                r=src['records'][i];start=int(r['offset']);n=int(r['tokens'])
                records.append(dict(tokens=bytes(src['maps']['tokens'][start*4:(start+n)*4]),
                                    mask=bytes(src['maps']['mask'][start:start+n]),source=name,source_record_id=i))
                total+=int(r['targets'])
                if total>=quota:break
            if total<quota:raise ValueError('replay source exhausted')
    random.Random(974223).shuffle(records);authority=args.output/'authority';authority.mkdir(mode=0o700)
    names=dict(tokens='tokens.uint32.bin',mask='assistant_mask.uint8.bin',index='records.idx',metadata='records.jsonl')
    offset=targets=0;totals={}
    with ExitStack() as stack:
        handles={k:stack.enter_context((authority/v).open('wb')) for k,v in names.items()}
        for i,row in enumerate(records):
            n=len(row['mask']);t=sum(row['mask'])
            if len(row['tokens'])!=4*n or not t or row['mask'][0] or any(x not in (0,1) for x in row['mask']):raise ValueError('record mask')
            handles['tokens'].write(row['tokens']);handles['mask'].write(row['mask'])
            handles['index'].write(RECORD_INDEX.pack(offset,n,t,0))
            metadata=dict(id=f'grounding-mix-{i}',offset=offset,tokens=n,targets=t,split=0,
                          source=row['source'],source_record_id=row['source_record_id'],
                          copied_bytes_sha256=hashlib.sha256(row['tokens']+row['mask']).hexdigest())
            handles['metadata'].write((json.dumps(metadata,sort_keys=True)+'\n').encode())
            offset+=n;targets+=t;totals[row['source']]=totals.get(row['source'],0)+t
        for f in handles.values():f.flush();os.fsync(f.fileno())
    outputs={k:dict(path=v,bytes=(authority/v).stat().st_size,sha256=sha(authority/v)) for k,v in names.items()}
    publish(authority/'manifest.json',dict(schema=AUTHORITY_SCHEMA,status='complete',training_eligible=True,
        tokenizer='p50k_base',purpose='operator-authorized experimental verified correction and unchanged admitted replay',
        outputs=outputs,counts=dict(records=len(records),tokens=offset,assistant_target_tokens=targets,train_records=len(records),validation_records=0),
        recipe=recipe,recipe_sha256=sha(args.recipe),correction_verification_sha256=sha(args.output/'correction-verification.json'),
        source_target_totals=totals,independent_holdout_claim=False,first_party_registry_admission=False))
    for p in authority.iterdir():p.chmod(0o400)
    publish(args.output/'data-summary.json',dict(status='verified-correction-and-replay-ready',authority=str(authority),
        authority_sha256=sha(authority/'manifest.json'),source_target_totals=totals,records=len(records),input_tokens=offset,assistant_targets=targets))
    print('CORRECTION_DATA_READY',json.dumps(totals),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--recipe',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    build(p.parse_args())
