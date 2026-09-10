import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from ndm.data.masked_sft_dataset import AUTHORITY_SCHEMA,RECORD_INDEX,sha256,MaskedSFTPackedDataset,SFTSamplerIdentity
from scripts.build_e97_native_training_mix import build,record_order,verify_authorization
from scripts.audit_e97_training_mix_schedule import audit
from types import SimpleNamespace


def source(root,native=False):
    root.mkdir()
    names={'tokens':'tokens.bin','mask':'loss_mask.bin','index':'records.idx','metadata':'records.jsonl'}
    rows=[];tokens=[];mask=[];index=[]
    for i,(repo,problem,split) in enumerate([('org/a','a',0),('org/a','a',0),('org/b','b',0),('org/dev','dev',1)]):
        values=[2,32750,20+i,4];bits=[0,1,1,0]
        tokens+=values;mask+=bits;index.append(RECORD_INDEX.pack(i*4,4,2,split))
        rows.append(dict(problem_key=[repo,problem],split=split,trajectory_identity=f't{i}',source='pi-live',has_think=False))
    (root/names['tokens']).write_bytes(np.asarray(tokens,dtype='<u4').tobytes())
    (root/names['mask']).write_bytes(bytes(mask));(root/names['index']).write_bytes(b''.join(index))
    (root/names['metadata']).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    outputs={names[k] if native else k:dict(path=v,bytes=(root/v).stat().st_size,sha256=sha256(root/v)) for k,v in names.items()}
    manifest=dict(schema='emender-open-swe-source-native-candidate-v1' if native else AUTHORITY_SCHEMA,
                  status='complete',training_eligible=not native,outputs=outputs,protected_repositories=['protected/repo'])
    (root/'manifest.json').write_text(json.dumps(manifest))
    return sha256(root/'manifest.json')


def setup(tmp_path):
    specs=[]
    for name in ('native','conversation','retention'):
        root=tmp_path/name;digest=source(root,name=='native')
        specs.append(dict(name=name,kind='native' if name=='native' else 'legacy',root=str(root),
                          sha256=digest,target_tokens=3 if name!='retention' else 2))
    evidence=[]
    for field in ('manifest_sha256','dataset_manifest_sha256'):
        path=tmp_path/(field+'.json');path.write_text(json.dumps({'status':'passed',field:specs[0]['sha256']}))
        evidence.append(dict(path=str(path),sha256=sha256(path),binding_field=field))
    recipe=dict(schema='emender-native-training-mix-recipe-v1',operator_internal_training_authorized=True,
                sources=specs,native_evidence=evidence,seed=97)
    path=tmp_path/'recipe.json';path.write_text(json.dumps(recipe))
    return path,recipe


def test_problem_cycles_do_not_overweight_multiple_attempts():
    metadata=[dict(problem_key=['r','a']),dict(problem_key=['r','a']),dict(problem_key=['r','b'])]
    order=record_order([0,1,2],metadata,17,True)
    first=[next(order) for _ in range(2)];second=[next(order) for _ in range(2)]
    assert {tuple(metadata[i]['problem_key']) for i in first}=={('r','a'),('r','b')}
    assert set(first)!=set(second)


def test_mix_copies_only_whole_training_records_and_loads_as_boundary_packs(tmp_path):
    path,recipe=setup(tmp_path);out=tmp_path/'mix'
    before={s['root']:sha256(Path(s['root'])/'manifest.json') for s in recipe['sources']}
    build(path,sha256(path),out)
    manifest=json.loads((out/'manifest.json').read_text());rows=[json.loads(x) for x in (out/'records.jsonl').read_text().splitlines()]
    assert manifest['training_eligible'] and manifest['all_output_records_verified']
    assert manifest['counts']['records']==5 and manifest['counts']['assistant_target_tokens']==10
    assert all(r['split']==0 and r['source_record_id']!=3 for r in rows)
    raw_tokens=(out/'tokens.uint32.bin').read_bytes();raw_mask=(out/'assistant_mask.uint8.bin').read_bytes()
    for row in rows:
        spec=next(s for s in recipe['sources'] if s['name']==row['source'])
        original=Path(spec['root']);start=row['source_record_id']*4;offset=row['offset']
        assert raw_tokens[offset*4:(offset+4)*4]==(original/'tokens.bin').read_bytes()[start*4:(start+4)*4]
        assert raw_mask[offset:offset+4]==(original/'loss_mask.bin').read_bytes()[start:start+4]
    assert all(sha256(Path(root)/'manifest.json')==digest for root,digest in before.items())
    packs=tmp_path/'packs'
    subprocess.run([sys.executable,'scripts/build_e97_sft_packs.py','--authority-root',str(out),
                    '--authority-manifest-sha256',sha256(out/'manifest.json'),'--output-root',str(packs),
                    '--context-size','16','--boundary-aware','--sampler-mode','epoch-permutation'],check=True,capture_output=True)
    identity=SFTSamplerIdentity(authority_manifest_sha256=sha256(out/'manifest.json'),
        pack_manifest_sha256=sha256(packs/'manifest.json'),sampler_key=97,data_world_size=1,context_size=16)
    data=MaskedSFTPackedDataset(out,packs,identity=identity,rank=0,sampler_mode='epoch-permutation')
    try:
        tokens,loss,valid,reset,lengths,counts=data.get_boundary_aware_batch(1)
        assert tokens.shape==(1,17) and loss.sum()==counts.sum()
        assert not (loss&reset[:,1:]).any()
    finally:data.close()
    schedule=tmp_path/'schedule.json'
    audit(SimpleNamespace(authority=out,packs=packs,output=schedule,
        authority_sha256=sha256(out/'manifest.json'),pack_sha256=sha256(packs/'manifest.json'),
        sampler_key=97,world_size=2,context_size=16,steps=1))
    report=json.loads(schedule.read_text())
    assert report['source_target_totals']=={'native':4,'conversation':4,'retention':2}
    assert all(len(ids)==1 for ids in report['steps'][0]['rank_sample_ids'])
    with pytest.raises(FileExistsError):build(path,sha256(path),out)


def test_unique_native_rounds_exhaust_all_records_without_repeating():
    metadata=[dict(problem_key=['r','a']) for _ in range(3)]+[dict(problem_key=['r','b'])]
    order=list(record_order(range(4),metadata,17,True,unique_native=True))
    assert sorted(order)==list(range(4))
    assert {tuple(metadata[i]['problem_key']) for i in order[:2]}=={('r','a'),('r','b')}
    assert order==list(record_order(range(4),metadata,17,True,unique_native=True))


def test_bounded_retention_replay_is_counted_and_copied_whole(tmp_path):
    path,recipe=setup(tmp_path)
    recipe['sources'][0]['unique_native']=True
    recipe['sources'][2].update(epochs=3,target_tokens=15)
    path.write_text(json.dumps(recipe));out=tmp_path/'mix'
    build(path,sha256(path),out)
    stats=json.loads((out/'manifest.json').read_text())['sources']['retention']
    assert (stats['assistant_target_tokens'],stats['records'],stats['unique_records'],stats['repeated_records'])==(16,8,3,5)
    assert stats['maximum_epochs']==3
    order=list(record_order([0,1,2],None,18,False,epochs=3))
    assert all(sorted(order[i:i+3])==[0,1,2] for i in (0,3,6))
    with pytest.raises(ValueError,match='insufficient'):
        recipe['sources'][2]['target_tokens']=19;path.write_text(json.dumps(recipe))
        build(path,sha256(path),tmp_path/'over-budget')


@pytest.mark.parametrize('name,change',[
    ('conversation',{'epochs':2}),('native',{'epochs':2}),
    ('retention',{'epochs':4}),('retention',{'epochs':True}),
    ('conversation',{'unique_native':True}),('native',{'unique_native':'yes'})])
def test_undeclared_or_invalid_repeat_policies_fail(tmp_path,name,change):
    _,recipe=setup(tmp_path)
    next(s for s in recipe['sources'] if s['name']==name).update(change)
    with pytest.raises(ValueError):verify_authorization(recipe)


def test_missing_authorization_or_evidence_fails(tmp_path):
    _,recipe=setup(tmp_path)
    for changed in ({**recipe,'operator_internal_training_authorized':False},{**recipe,'native_evidence':[]}):
        with pytest.raises(ValueError):verify_authorization(changed)


def test_thinking_conversations_are_excluded_whole_not_stripped(tmp_path):
    path,recipe=setup(tmp_path);spec=recipe['sources'][1];root=Path(spec['root'])
    rows=[json.loads(line) for line in (root/'records.jsonl').read_text().splitlines()]
    rows[0]['has_think']=True
    (root/'records.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    manifest=json.loads((root/'manifest.json').read_text())
    manifest['outputs']['metadata'].update(bytes=(root/'records.jsonl').stat().st_size,sha256=sha256(root/'records.jsonl'))
    (root/'manifest.json').write_text(json.dumps(manifest))
    spec.update(sha256=sha256(root/'manifest.json'),exclude_think=True)
    path.write_text(json.dumps(recipe));out=tmp_path/'mix';build(path,sha256(path),out)
    copied=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    assert {row['source_record_id'] for row in copied if row['source']=='conversation'}=={1,2}


def test_payload_corruption_is_not_admitted(tmp_path):
    path,recipe=setup(tmp_path)
    (Path(recipe['sources'][0]['root'])/'tokens.bin').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='payload identity'):build(path,sha256(path),tmp_path/'mix')
