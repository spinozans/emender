import copy
import json
from collections import Counter
from contextlib import ExitStack
from types import SimpleNamespace
import pytest
from scripts.train_e97_representation_bridge import gate_checks,MODELS,COHORTS
from scripts.build_e97_representation_bridge import rehearsal_ids,replay_ids,write_authority
from scripts.build_e97_native_training_mix import read_source
from scripts.eval_e97_native_execution import sha,publish


def panels():
    cases=[]
    for cohort,n in COHORTS.items():
        for family in ('lookup','sum','edit','recovery'):
            for i in range(n//4):cases.append(dict(id=f'{cohort}-{family}-{i}',cohort=cohort,family=family))
    execution={'models':{name:{'outcomes':[dict(id=c['id'],grade=dict(success=False)) for c in cases]} for name in MODELS}}
    for model in ('pre-y','pre-x','bridge-y','bridge-x'):
        limits={'prior-regression':6 if model.startswith('pre-') else 7,'prior-fresh':16,'prior-transfer':1 if model.startswith('pre-') else 4,'fresh':0 if model.startswith('pre-') else 24,'composition':0 if model.startswith('pre-') else 4}
        done=Counter()
        for c,r in zip(cases,execution['models'][model]['outcomes']):
            cohort=c['cohort']
            if cohort=='fresh':r['grade']['success']=not model.startswith('pre-') and int(c['id'].rsplit('-',1)[1])<6
            else:r['grade']['success']=done[cohort]<limits[cohort]
            done[cohort]+=1
    learning={'models':{name:{c:{'metrics':{'assistant':{m:v}}} for c,m,v in (
        ('tool-retention','token_accuracy',1.),('conversation-retention','record_macro_nll',1.5),('native-development','record_macro_nll',.9))} for name in MODELS}}
    return execution,learning,dict(cases=cases)


def test_bridge_gate_passes_and_counts_all_cases():
    checks,counts=gate_checks(*panels())
    assert len(checks)==15 and all(c['passed'] for c in checks)
    assert counts['pre-y']['prior-regression']==6 and counts['bridge-y']['prior-regression']==7
    assert counts['bridge-y']['fresh']==24 and counts['bridge-y']['composition']==4


@pytest.mark.parametrize('mutation',['duplicate','missing','integer','model','nan','boolean-metric','negative','over-one','assisted','family'])
def test_bridge_gate_rejects_invalid_evidence(mutation):
    e,l,p=panels();rows=e['models']['bridge-y']['outcomes']
    if mutation=='duplicate':rows[-1]=rows[0]
    elif mutation=='missing':rows.pop()
    elif mutation=='integer':rows[0]['grade']['success']=1
    elif mutation=='model':e['models']['unexpected']=e['models'].pop('pre-x')
    elif mutation in ('nan','boolean-metric','negative','over-one'):
        l['models']['pre-x']['tool-retention']['metrics']['assistant']['token_accuracy']={'nan':float('nan'),'boolean-metric':True,'negative':-.1,'over-one':1.1}[mutation]
    elif mutation=='assisted':p['cases'][0]['supplied_calls']=[{}]
    else:p['cases'][-1]['family']='unrecognized'
    with pytest.raises(ValueError):gate_checks(e,l,p)


def test_old_regression_and_fresh_family_failures_cannot_hide():
    e,l,p=panels();e['models']['bridge-y']['outcomes'][0]['grade']['success']=False
    assert any(not c['passed'] and c['cohort']=='prior-regression' for c in gate_checks(e,l,p)[0])
    e,l,p=panels()
    for c,row in zip(p['cases'],e['models']['bridge-y']['outcomes']):
        if c['cohort']=='fresh':row['grade']['success']=c['family']!='recovery'
    checks,counts=gate_checks(e,l,p)
    assert counts['bridge-y']['fresh']==24
    assert any(not c['passed'] and c['cohort']=='fresh-recovery' for c in checks)


def test_improvement_gate_clamps_at_total():
    e,l,p=panels()
    for model in MODELS:
        for row in e['models'][model]['outcomes']:row['grade']['success']=True
    checks,_=gate_checks(e,l,p)
    assert all(c['passed'] for c in checks)
    assert next(c['limit'] for c in checks if c['cohort']=='fresh')==32


def test_rehearsal_balanced_complete_pairs_order_independent():
    cases=[dict(id=f'{f}-{i}-{w}',family=f,pair_index=i,variant=w,prompt=f'{f}-{i}') for f in ('lookup','sum','edit','recovery') for i in range(80) for w in (0,1)]
    ids=rehearsal_ids(cases,'unit')
    assert len(ids)==len(set(ids))==512 and ids==rehearsal_ids(list(reversed(cases)),'unit')
    assert Counter(i.split('-')[0] for i in ids)==dict(lookup=128,sum=128,edit=128,recovery=128)
    bad=copy.deepcopy(cases)
    for c in bad:c['variant']=0
    with pytest.raises(ValueError,match='pairs'):rehearsal_ids(bad,'unit')
    with pytest.raises(ValueError,match='duplicate'):rehearsal_ids(cases+[cases[0]],'unit')


def test_replay_order_does_not_depend_on_recipe_dictionary_serialization():
    source=dict(ids=list(range(12)),metadata=[{'source':n} for n in ('conversation','native','retention') for _ in range(4)],records=[{'targets':100000}]*12)
    config=dict(replay_order=['conversation','native','retention'],replay_target_quotas=dict(conversation=300000,native=100000,retention=100000),replay_seed=991501)
    expected=replay_ids(source,config);config=json.loads(json.dumps(config,sort_keys=True))
    assert replay_ids(source,config)==expected and {k:len(v) for k,v in expected.items()}==dict(conversation=3,native=1,retention=1)


def test_authority_writer_preserves_bytes_and_provenance(tmp_path):
    config={'purpose':'unit'};recipe=tmp_path/'recipe.json';publish(recipe,config)
    for name in ('selection.json','fresh-evaluation-cases.json'):publish(tmp_path/name,{})
    for name in ('authored-verification-private.jsonl','authored-candidates-private.jsonl'):(tmp_path/name).write_text('{}\n')
    tokens=b'\x01\x00\x00\x00\x02\x00\x00\x00';mask=b'\x00\x01'
    rows=[dict(tokens=tokens,mask=mask,source='grounded-rehearsal',source_record_id=17,provenance={'original_case_id':'original'})]
    write_authority(rows,config,recipe,tmp_path)
    with ExitStack() as stack:
        src=read_source(dict(root=str(tmp_path/'authority'),sha256=sha(tmp_path/'authority/manifest.json'),kind='legacy',include_metadata_sources=['grounded-rehearsal'],target_tokens=1),stack)
        assert bytes(src['maps']['tokens'])==tokens and bytes(src['maps']['mask'])==mask
        assert src['metadata'][0]['provenance']=={'original_case_id':'original'}
    with pytest.raises(FileExistsError):write_authority(rows,config,recipe,tmp_path)
