"""Paired training/fresh/structural-transfer tasks and verified native demonstrations."""
import copy
import hashlib
import json
import shlex
from scripts.e97_native_execution_cases import context
from scripts.e97_native_onpolicy_canary import candidate,observed_json
from scripts.e97_open_swe_native_codec import compact,native_turn
from scripts.e97_open_swe_native_runtime_protocol import NativeEpisode

FAMILIES=('lookup','sum','edit','recovery')


def fixtures(seed,pairs,cohort):
    if set(pairs)!=set(FAMILIES) or cohort not in ('train','fresh','transfer'):raise ValueError('curriculum cohort')
    rows=[];pair=0
    for family in FAMILIES:
        for index in range(pairs[family]):
            identity=hashlib.sha256(f'{seed}:{family}:{index}'.encode()).hexdigest()
            base=f'grounded/{cohort}/{identity[:12]}';path=f'/testbed/{base}/source.json';out=f'{base}/updated.json'
            missing=f'/testbed/{base}/absent.json';style=index%4
            selector=('active','chosen','pick','route')[style];table=('values','options','entries','choices')[style]
            field=('count','stock','visits','units')[style];delta=(1,3,7,11)[style]*(1 if index%3 else -1)
            routes=['active_path'] if index%2==0 else ['next_file','active_path']
            value_key=('value','token','answer','result')[style]
            for world in (0,1):
                h=hashlib.sha256(f'{seed}:{family}:{index}:{world}'.encode()).hexdigest()
                token=h[:(8,16,20,32)[style]];number=int(h[32:38],16);files={};output=None
                meta=dict(selector=selector,table=table,field_path=[field],delta=delta,routes=routes,value_key=value_key)
                if family=='lookup':
                    keys=['amber','violet','silver'];active=keys[(index+world)%3]
                    vals={k:hashlib.sha256(f'{h}:{k}'.encode()).hexdigest()[:len(token)] for k in keys};vals[active]=token
                    if cohort=='transfer':
                        data={'active_id':active,'records':[{'id':k,'payload':v} for k,v in vals.items()]}
                        expr='next(r["payload"] for r in d["records"] if r["id"]==d["active_id"])'
                        prompt=f'Read {path}. Select the record whose id equals active_id. Finish with exactly its payload. Do not change files.'
                    else:
                        data={selector:active,table:vals};expr=f'd[{table!r}][d[{selector!r}]]'
                        prompt=(f'Read {path}. Return exactly the {table} value selected by {selector} using finish. Do not change files.' if index%2==0 else
                            f'Inspect {path}; {selector} names the entry in {table} to select. Finish with that entry\'s exact string, without modifying files.')
                    answer=token;meta['expression']=expr
                elif family=='sum':
                    nums=[number,int(h[40:46],16)]
                    if index%2:nums[0]=-nums[0]
                    if cohort=='transfer':
                        nums.append(int(h[48:52],16));data={'measurements':nums,'note':token};expr='sum(d["measurements"])'
                        prompt=f'Read {path}. Sum every number in measurements. Finish with exactly the decimal total. Preserve all files.'
                    else:
                        keys=('left','right') if style<2 else ('first','second');data=dict(zip(keys,nums));data['note']=token
                        expr=f'd[{keys[0]!r}]+d[{keys[1]!r}]'
                        prompt=f'Read {path}. Finish with exactly the decimal sum of {keys[0]} and {keys[1]}. Preserve all files.'
                    answer=str(sum(nums));meta['expression']=expr
                elif family=='edit':
                    data={'ticket':token,'owner':f'owner-{identity[20:26]}','untouched':[world,number%17]}
                    if style>=2:data['metrics']={field:number};meta['field_path']=['metrics',field]
                    else:data[field]=number
                    if cohort=='transfer':data['change']=delta+world+2;actual_delta=data['change'];meta['delta_from']='change'
                    else:actual_delta=delta
                    output=copy.deepcopy(data);target=output
                    for key in meta['field_path'][:-1]:target=target[key]
                    target[meta['field_path'][-1]]+=actual_delta
                    field_text='.'.join(meta['field_path']);change='the number in change' if cohort=='transfer' else str(delta)
                    prompt=f'Read {path}. Write /testbed/{out} with {field_text} increased by {change}. Preserve every other field and the original source file. Read back the output to verify it, then finish with exactly done.'
                    answer='done'
                else:
                    meta['routes']=routes if cohort!='transfer' else ['selected_paths','active_path']
                    dest=f'/testbed/{base}/value.json'
                    if cohort=='transfer':
                        middle=f'/testbed/{base}/pointer.json';data={'active':'right','selected_paths':{'left':f'/testbed/{base}/decoy.json','right':middle}}
                        files[middle.removeprefix('/testbed/')]=json.dumps({'active_path':dest})
                        prompt=f'First try {missing} with str_replace_editor view and wait for the result. If missing, read {path}; active selects an entry in selected_paths. Read that file, follow active_path, and finish with exactly its {value_key}. Preserve all files.'
                    elif len(routes)==2:
                        middle=f'/testbed/{base}/pointer.json';data={routes[0]:middle}
                        files[middle.removeprefix('/testbed/')]=json.dumps({routes[1]:dest})
                        prompt=f'First read {missing} using str_replace_editor view and wait for its result. If absent, read {path}, follow next_file to another JSON, then follow active_path to the value file. Finish with exactly its {value_key}. Do not change files.'
                    else:
                        data={'active_path':dest}
                        prompt=f'First attempt {missing} using str_replace_editor view and wait for the result. If absent, consult {path}, follow active_path, and finish with exactly the observed {value_key}. Do not modify files.'
                    files[dest.removeprefix('/testbed/')]=json.dumps({value_key:token,'decoy':h[28:48]});answer=token
                files[path.removeprefix('/testbed/')]=json.dumps(data)
                rows.append(dict(id=f'{cohort}-{family}-{index:04d}-world-{world}',family=family,pair_index=pair,variant=world,
                    cohort=cohort,source_style=bool((index//4)%2),prompt=prompt,path=path,files=files,answer=answer,
                    expected_output=output,output_path=out,missing_path=missing,recipe=meta,
                    authored_failure_prefix=cohort=='train' and family in ('edit','recovery') and index%2==1))
            pair+=1
    return rows


def teacher_record(case,panel,source_system,sandbox,enc):
    e=NativeEpisode(panel['tools'],enc)
    e.append_source_message(source_system if case['source_style'] else context('system',panel['system']))
    e.append_source_message(context('user',case['prompt']));calls=[];prefix=0
    def action(name,args):
        message=dict(role='assistant',content=None,reasoning_content=None,think=None,
            tool_calls=[dict(type='function',function=dict(name=name,arguments=compact(args)))])
        turn=e.accept_generated_turn(native_turn(message))
        if name=='finish':return ''
        reply=sandbox.request('execute',call=turn.backend_call())
        if 'dispatch_error' in reply:raise ValueError('authored dispatch')
        calls.append(dict(request=turn.backend_call(),result=reply['result']))
        e.append_observation(reply['result']['message']);return reply['result']['message']['content']
    def view(path):return action('str_replace_editor',dict(command='view',path=path))
    if case['authored_failure_prefix']:
        if case['family']=='edit':
            text=action('str_replace_editor',dict(command='create',path=case['path'],file_text='{}'))
            if not text.startswith('ERROR:'):raise ValueError('required create-existing error absent')
            prefix=1
        else:
            for _ in range(2):
                if not view(case['missing_path']).startswith('ERROR:'):raise ValueError('required missing-file error absent')
            prefix=2
    elif case['family']=='recovery':
        if not view(case['missing_path']).startswith('ERROR:'):raise ValueError('required missing-file observation')
    path=case['path'];seen=observed_json(view(path),path)
    if seen!=json.loads(case['files'][path.removeprefix('/testbed/')]):raise ValueError('observed source mismatch')
    meta=case['recipe']
    if case['family']=='recovery':
        specs=meta.get('pointer_specs')
        for key in specs if specs is not None else meta['routes']:
            if specs is not None:
                from scripts.e97_representation_bridge import selected_value
                path=selected_value(seen,key)
            else:path=seen[key][seen['active']] if key=='selected_paths' else seen[key]
            if path.removeprefix('/testbed/') not in case['files']:raise ValueError('unbound observed pointer')
            seen=observed_json(view(path),path)
        expression=f'd[{meta["value_key"]!r}]'
    else:expression=meta.get('expression')
    program=f'import json,os; d=json.load(open({path!r})); '
    if case['family']=='edit':
        target='d'+''.join(f'[{key!r}]' for key in meta['field_path'])
        if 'delta_spec' in meta:
            from scripts.e97_representation_bridge import selection_expression
            amount=selection_expression(meta['delta_spec'])
        elif 'delta_path' in meta:amount='d'+''.join(f'[{key!r}]' for key in meta['delta_path'])
        else:amount=f'd[{meta["delta_from"]!r}]' if 'delta_from' in meta else repr(meta['delta'])
        dest='/testbed/'+case['output_path']
        program+=f'{target}+={amount}; f=os.fdopen(os.open({dest!r},os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600),"w"); json.dump(d,f); f.close(); print("written")'
    else:program+=f'print({expression})'
    text=action('execute_bash',dict(command='python -c '+shlex.quote(program),timeout=10))
    if calls[-1]['result']['exit_code']!=0:raise ValueError('authored computation failed')
    if case['family']=='edit':
        if observed_json(view(dest),dest)!=case['expected_output']:raise ValueError('authored output readback mismatch')
        if meta.get('verify_edit'):
            expected='expected'+''.join(f'[{key!r}]' for key in meta['field_path'])
            check=f'import copy,json; d=json.load(open({case["path"]!r})); expected=copy.deepcopy(d); {expected}+={amount}; actual=json.load(open({dest!r})); assert actual==expected; print("verified")'
            text=action('execute_bash',dict(command='python -c '+shlex.quote(check),timeout=10))
            if calls[-1]['result']['exit_code']!=0 or text.splitlines()[0]!='verified':raise ValueError('executed semantic edit verification')
        answer='done'
    else:answer=text.splitlines()[0]
    if answer!=case['answer']:raise ValueError('independent answer mismatch')
    action('finish',dict(message=answer))
    # External, read-only post-finish integrity verifier; not a model action or target.
    paths=['/testbed/'+p for p in case['files']]
    check='import hashlib,json; print(json.dumps({p:hashlib.sha256(open(p,"rb").read()).hexdigest() for p in '+repr(paths)+'},sort_keys=True))'
    verification=sandbox.request('execute',call=dict(name='execute_bash',arguments=dict(command='python -c '+shlex.quote(check),timeout=10)))
    if 'dispatch_error' in verification or verification['result']['exit_code']!=0:raise ValueError('external integrity verifier failed')
    actual=json.loads(verification['result']['message']['content'].splitlines()[0])
    wanted={'/testbed/'+p:hashlib.sha256(raw.encode()).hexdigest() for p,raw in case['files'].items()}
    if actual!=wanted:raise ValueError('original source bytes changed')
    record=candidate(e.source_messages(),panel['tools'],prefix,enc)
    if not e.finished or not record['targets'] or len(record['token_ids'])>65536:raise ValueError('whole native record bounds')
    return record,dict(id=case['id'],family=case['family'],variant=case['variant'],source_style=case['source_style'],
        source='authored-executor-verified',authored_failure_prefix=case['authored_failure_prefix'],prefix_assistants_unsupervised=prefix,
        answer_verified=True,source_bytes_unchanged=True,calls=calls,external_integrity_receipt=verification,
        candidate_text_sha256=record['text_sha256'])
