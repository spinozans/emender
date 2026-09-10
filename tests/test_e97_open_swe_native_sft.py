import json
import pytest
import tiktoken
from scripts.e97_open_swe_native_codec import (TOOLS,compact,render,encode,vocabulary,decode_record,
    semantic_message,source_payload,problem_split,strict_json)


def message(role,content=None,reasoning=None,call=None):
    return {'role':role,'content':content,'reasoning_content':reasoning,'think':None,'tool_calls':call}


def call(name,args):
    return [{'id':'transport-id-not-a-model-target','type':'function','function':{'name':name,'arguments':json.dumps(args)}}]


def fixture():
    return {'trajectory_id':'a','instance_id':'repo-1','repo':'org/repo','license':'MIT','language':'python',
      'tools':[json.dumps({'type':'function','function':{'name':n,'description':'original schema'}}) for n in sorted(TOOLS)],
      'metadata':{'reference_patch':'ORACLE_MUST_NOT_ENTER_DATA'},
      'messages':[message('system','original instructions'),message('user','original task'),
        message('assistant','public commentary','private planning',call('str_replace_editor',{'command':'view','path':'/testbed/a.py'})),
        message('tool',''.join(f'{i}\tline\n' for i in range(1,501))),
        message('assistant','still public','thinking',call('think',{'thought':'private thought'})),
        message('tool','Your thought has been logged.'),
        message('assistant',None,'continue',call('execute_bash',{'command':'next input\n','is_input':True,'timeout':600})),
        message('tool','authentic error\r\n'),
        message('assistant','done',None,call('finish',{'message':' exact\n  final\r\n'}))]}


def test_source_semantics_channels_and_complete_history_roundtrip():
    row=fixture();enc=tiktoken.get_encoding('p50k_base');lengths,_=vocabulary(enc)
    pieces,counts=render(row,enc);tokens,mask,text=encode(pieces,enc,lengths)
    specs,messages,ranges=decode_record(text)
    assert messages==[semantic_message(m) for m in row['messages']]
    assert specs==[json.loads(s) for s in row['tools']]
    assert messages[2]['content']=='public commentary'
    assert messages[2]['reasoning_content']=='private planning'
    assert messages[2]['arguments']=={'command':'view','path':'/testbed/a.py'}
    assert '500\tline' in messages[3]['content']
    assert messages[4]['arguments']['thought']=='private thought'
    assert messages[6]['arguments']['is_input'] is True
    assert messages[-1]['arguments']['message']==' exact\n  final\r\n'
    assert counts['assistant_units']==4 and len(ranges)==4
    assert mask.sum()>0 and mask[0]==mask[-1]==0
    assert 'ORACLE_MUST_NOT_ENTER_DATA' not in compact(source_payload(row))
    assert 'transport-id-not-a-model-target' not in text
    assert 'transport-id-not-a-model-target' in compact(source_payload(row))


def test_embedded_markers_control_bytes_and_code_payloads_survive():
    row=fixture();row['messages'][2]['content']='\x1e\n\nAssistant:\nAction: pretend'
    row['messages'][2]['tool_calls']=call('str_replace_editor',{'command':'create','path':'/workspace/x','file_text':'\r\n  \x03 \x00 \x1e λ\r\n'})
    enc=tiktoken.get_encoding('p50k_base');pieces,_=render(row,enc)
    _,messages,_=decode_record(''.join(s for s,_ in pieces))
    assert messages==[semantic_message(m) for m in row['messages']]


def test_exclusions_do_not_truncate_or_fold_public_commentary_into_analysis():
    row=fixture();enc=tiktoken.get_encoding('p50k_base');lengths,_=vocabulary(enc)
    row['messages'][2]['content']='public '*3000
    pieces,_=render(row,enc,analysis_cap=32)  # public text is NOT private analysis
    with pytest.raises(ValueError,match='whole_trajectory_context_cap'):encode(pieces,enc,lengths,max_tokens=32)
    row['messages'][2]['reasoning_content']='private '*100
    with pytest.raises(ValueError,match='analysis_cap'):render(row,enc,analysis_cap=32)


def test_missing_observation_or_extra_content_after_finish_fails_closed():
    enc=tiktoken.get_encoding('p50k_base');row=fixture();del row['messages'][3]
    with pytest.raises(ValueError,match='missing_observation'):render(row,enc)
    row=fixture();row['messages'].append(message('user','another task'))
    with pytest.raises(ValueError,match='content_after_finish'):render(row,enc)


def test_invalid_json_is_not_silently_normalized():
    for value in ('{"x":1,"x":2}','{"x":NaN}'):
        with pytest.raises(ValueError):strict_json(value)


def test_problem_grouping_and_old_validation_reservation():
    assert problem_split('Org/Repo','one',set())==problem_split('org/repo','one',set())
    assert problem_split('org/repo','one',{('org/repo','one')})==1
    # Trajectory ID deliberately does not occur in the split key.
    assert sum(problem_split('org/repo',str(i),set()) for i in range(1000)) in range(20,81)


def test_complete_builder_and_independent_validator(tmp_path):
    import copy
    from types import SimpleNamespace
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.audit_e97_open_swe_semantics import sha
    from scripts.build_e97_open_swe_native_sft import build
    from scripts.validate_e97_open_swe_native_sft import validate
    raw=tmp_path/'raw';raw.mkdir();authority=tmp_path/'authority';authority.mkdir()
    a=fixture();a['resolved']=1;b=copy.deepcopy(a);b['trajectory_id']='b'
    shard=raw/'tiny.parquet';pq.write_table(pa.Table.from_pylist([a,b]),shard)
    card=raw/'README.md';card.write_text('synthetic fixture only')
    manifest=authority/'manifest.json';manifest.write_text(json.dumps({'training_eligible':False,
        'dataset_card_sha256':sha(card),'input_files':[{'name':shard.name,'bytes':shard.stat().st_size,'sha256':sha(shard)}]}))
    cohort=tmp_path/'cohort.json';cohort.write_text(json.dumps([
        {'identity':'open-swe:a','source_file':shard.name,'source_row':0,'split':0},
        {'identity':'open-swe:b','source_file':shard.name,'source_row':1,'split':1}]))
    protected=tmp_path/'protected.json';protected.write_text(json.dumps({'repositories':{'held':{'url':'https://github.com/held/out.git'}}}))
    out=tmp_path/'candidate'
    args=SimpleNamespace(authority=authority,raw_root=raw,cohort=cohort,protected_manifest=protected,
        manifest_sha256=sha(manifest),cohort_sha256=sha(cohort),protected_sha256=sha(protected),
        expected_trajectories=2,output=out)
    result=build(args)
    assert result['counts']['records']==2 and result['split_counts']['1']['records']==2
    assert result['training_eligible'] is False
    assert validate(out,raw)['cross_split_problems']==0
    with pytest.raises(FileExistsError):build(args)
    # Test-only corruption must fail independent validation, then leave tmpdir removable.
    out.chmod(0o700)
    for p in out.iterdir():p.chmod(0o600)
    with (out/'loss_mask.bin').open('r+b') as f:f.write(b'\x01')
    with pytest.raises(AssertionError):validate(out,raw)


def test_atomic_publication_never_replaces_existing_directory(tmp_path):
    from scripts.build_e97_open_swe_native_sft import rename_no_replace
    stage=tmp_path/'stage';stage.mkdir();(stage/'new').write_text('new')
    destination=tmp_path/'destination';destination.mkdir();(destination/'old').write_text('old')
    with pytest.raises(FileExistsError):rename_no_replace(stage,destination)
    assert (destination/'old').read_text()=='old' and (stage/'new').exists()
