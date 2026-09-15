"""Training-only representation bridges and separately seeded evaluation candidates.

No admission, model sampling, or optimizer updates. Expected answers are constructed
from fixture values, independently of the teacher's emitted Python expressions.
"""
import copy
import hashlib
import json
from scripts.e97_grounded_curriculum import FAMILIES


def access(path, root='d'):
    if not isinstance(path,list) or not 1<=len(path)<=4 or any(type(k) not in (str,int) for k in path):
        raise ValueError('bounded JSON access path')
    return root+''.join(f'[{key!r}]' for key in path)


def selection_expression(spec):
    kind=spec['kind']
    if kind=='direct':return access(spec['path'])
    table=access(spec['table']);selector=access(spec['selector'])
    if kind=='map':return f'{table}[{selector}]'
    if kind=='rows':
        return f'next(row[{spec["value_field"]!r}] for row in {table} if row[{spec["key_field"]!r}]=={selector})'
    raise ValueError('selection representation')


def selected_value(document,spec):
    def get(path):
        access(path);value=document
        for key in path:value=value[key]
        return value
    if spec['kind']=='direct':return get(spec['path'])
    table=get(spec['table']);selector=get(spec['selector'])
    if spec['kind']=='map':return table[selector]
    if spec['kind']!='rows' or not isinstance(table,list):raise ValueError('pointer representation')
    matches=[row[spec['value_field']] for row in table if row[spec['key_field']]==selector]
    if len(matches)!=1:raise ValueError('pointer selection must be unique')
    return matches[0]


def bridge_cases(seed,pairs_per_family=96,cohort='train'):
    if cohort not in ('train','fresh','composition','preflight') or type(pairs_per_family) is not int or not 1<=pairs_per_family<=96:
        raise ValueError('bounded bridge cohort')
    cases=[];pair=0
    for family in FAMILIES:
        for index in range(pairs_per_family):
            h=hashlib.sha256(f'{seed}:{family}:{index}'.encode()).hexdigest()
            layout=index%3;naming=(index//3)%4;wording=(index//12)%2
            selector,table,key,value=(('active','values','id','payload'),('selected','records','key','value'),
                ('choice','options','name','token'),('pick','entries','label','answer'))[naming]
            counter,change=(('count','change'),('stock','adjustment'),('units','delta'),('visits','step'))[naming]
            base=f'bridge/{cohort}/{h[:12]}'
            basename={'lookup':'config','sum':'numbers','edit':'state','recovery':'catalog'}[family]
            if index//24%2:basename='input'
            path=f'/testbed/{base}/{basename}.json';out=f'{base}/'+('result.json' if index//6%2 else 'updated.json')
            missing=f'/testbed/{base}/missing.json';keys=['amber','silver','violet']
            rotation=(index//12)%3;ordered_keys=keys[rotation:]+keys[:rotation]
            values={k:hashlib.sha256(f'{h}:{k}'.encode()).hexdigest()[:(8,16,20,32)[naming]] for k in ordered_keys}
            observed_delta=(2+int(h[30:36],16)%99,-(2+int(h[36:42],16)%99))
            count=100000+int(h[12:18],16)%800000
            numbers=[count,-(1000+int(h[18:24],16)%80000)]
            numbers += [1000+int(hashlib.sha256(f'{h}:number:{j}'.encode()).hexdigest()[:6],16)%100000 for j in range(index%5)]
            for world in (0,1):
                active=keys[(index//3+world)%3];files={};meta={};expected_output=None
                if family=='lookup':
                    answer=values[active]
                    if cohort=='composition':
                        data={'request':{selector:active},'data':{table:[{key:k,value:v} for k,v in values.items()]}}
                        spec=dict(kind='rows',selector=['request',selector],table=['data',table],key_field=key,value_field=value)
                        instruction=f'In {path}, request.{selector} selects a row in data.{table} by its {key}. Finish with exactly that row\'s {value}.'
                    elif layout==0:
                        data={selector:active,table:values};spec=dict(kind='map',selector=[selector],table=[table])
                        instruction=f'Read {path}. {selector} names an entry in the {table} object. Finish with exactly the selected string.'
                    elif layout==1:
                        data={selector:active,table:[{key:k,value:v} for k,v in values.items()]};spec=dict(kind='rows',selector=[selector],table=[table],key_field=key,value_field=value)
                        instruction=f'Read {path}. Find the row in {table} whose {key} equals {selector}. Finish with exactly its {value}.'
                    else:
                        data={value:answer,'note':values[keys[(index//3+world+1)%3]]};spec=dict(kind='direct',path=[value])
                        instruction=f'Read {path} and finish with exactly the string in {value}, not note.'
                    meta['expression']=selection_expression(spec)
                elif family=='sum':
                    nums=list(numbers);nums[-1]+=world*173
                    answer=str(sum(nums))
                    if cohort=='composition':
                        data={'data':{table:[{value:n,'ignored':731} for n in nums]},'note':h[:8]}
                        expression=f'sum(row[{value!r}] for row in d["data"][{table!r}])'
                        instruction=f'Read {path}. Sum the {value} fields of every row in data.{table}, ignoring other fields. Finish with exactly the decimal total.'
                    elif layout==0:
                        # Named-pair bridge uses exactly two operands.
                        nums=nums[:2];nums[-1]=numbers[1]+world*173;answer=str(sum(nums))
                        a,b=('left','right') if naming%2==0 else ('a','b')
                        data={a:nums[0],b:nums[1],'ignored':731};expression=f'd[{a!r}]+d[{b!r}]'
                        instruction=f'Read {path}. Add {a} and {b}; finish with exactly the decimal result.'
                    elif layout==1:
                        data={table:nums,'ignored':731};expression=f'sum(d[{table!r}])'
                        instruction=f'Read {path}. Sum all numbers in the {table} list. Finish with exactly the decimal total.'
                    else:
                        data={table:[{key:f'row-{j}',value:n,'ignored':731} for j,n in enumerate(nums)]}
                        expression=f'sum(row[{value!r}] for row in d[{table!r}])'
                        instruction=f'Read {path}. Add the {value} fields from every row of {table}; finish with exactly the decimal total.'
                    meta['expression']=expression
                elif family=='edit':
                    data={'ticket':h[:20],'owner':'preserve-this','untouched':[17,29]}
                    field_path=['stats',counter] if layout==2 or cohort=='composition' else [counter]
                    old=count+world*173 if layout==0 and cohort!='composition' else count
                    if len(field_path)==2:data['stats']={counter:old}
                    else:data[counter]=old
                    if cohort=='composition':
                        delta=observed_delta[world];data['request']={selector:active};data['changes']={k:(delta if k==active else 19) for k in keys}
                        meta['delta_spec']=dict(kind='map',selector=['request',selector],table=['changes'])
                        amount=f'the entry of changes selected by request.{selector}'
                    elif layout==0:
                        delta=(1,-3,7,-11)[naming];meta['delta']=delta;data[change]=23
                        amount=f'the literal number {delta} (ignore the unrelated {change} field)'
                    else:
                        delta=observed_delta[world]
                        delta_path=['request',change] if layout==2 else [change]
                        if len(delta_path)==2:data['request']={change:delta}
                        else:data[change]=delta
                        meta['delta_path']=delta_path;amount='the observed number in '+'.'.join(delta_path)
                    expected_output=copy.deepcopy(data);target=expected_output
                    for name in field_path[:-1]:target=target[name]
                    target[field_path[-1]]+=delta
                    meta.update(field_path=field_path,verify_edit=True)
                    instruction=f'Read {path}. Write /testbed/{out}, adding {amount} to {".".join(field_path)}. Preserve all other fields and the original file. Read back and check the result before finishing with exactly done.'
                    answer='done'
                else:
                    destinations={k:f'/testbed/{base}/{k}.json' for k in keys};answer=values[active]
                    for k in keys:files[destinations[k].removeprefix('/testbed/')]=json.dumps({value:values[k],'note':'not the answer'})
                    if cohort=='composition':
                        middle=f'/testbed/{base}/pointer.json';files[middle.removeprefix('/testbed/')]=json.dumps({'next':destinations[active]})
                        data={'request':{selector:active},'data':{table:[{key:k,'path':middle if k==active else destinations[k]} for k in keys]}}
                        specs=[dict(kind='rows',selector=['request',selector],table=['data',table],key_field=key,value_field='path'),dict(kind='direct',path=['next'])]
                        route=f'request.{selector} selects the row of data.{table} by {key}; follow its path, then the next field in that file'
                    elif layout==0:
                        pointer=('active_path','current_file','next','path')[naming];data={pointer:destinations[active]};specs=[dict(kind='direct',path=[pointer])];route=f'follow {pointer}'
                    elif layout==1:
                        data={selector:active,table:destinations};specs=[dict(kind='map',selector=[selector],table=[table])];route=f'{selector} selects the path from the {table} object'
                    else:
                        data={selector:active,table:[{key:k,'path':destinations[k]} for k in keys]};specs=[dict(kind='rows',selector=[selector],table=[table],key_field=key,value_field='path')];route=f'{selector} selects a row of {table} by {key}; follow its path'
                    meta.update(pointer_specs=specs,value_key=value)
                    instruction=f'First try {missing} using str_replace_editor view and wait for the result. If missing, read {path}; {route}. Finish with exactly the final file\'s {value}.'
                if family!='edit':instruction+=' Preserve all files.'
                if wording:instruction='Use the file contents, not a guessed value. '+instruction
                files[path.removeprefix('/testbed/')]=json.dumps(data,indent=2 if index//6%2 else None)
                cases.append(dict(id=f'bridge-{cohort}-{family}-{index:04d}-world-{world}',family=family,pair_index=pair,variant=world,
                    cohort=cohort,source_style=bool(index//48%2),prompt=instruction,path=path,files=files,answer=answer,
                    expected_output=expected_output,output_path=out,missing_path=missing,recipe=meta,
                    authored_failure_prefix=cohort=='train' and family in ('edit','recovery') and index%4==3,
                    bridge_layout='composition' if cohort=='composition' else layout,naming_index=naming))
            pair+=1
    return cases
