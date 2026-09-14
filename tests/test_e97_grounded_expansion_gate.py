import copy
import json
from types import SimpleNamespace
import pytest
from scripts.prepare_e97_grounded_expansion import gate_checks,MODEL_NAMES


def reports():
    cases=[dict(id=f'{cohort}-{family}-{i}',cohort=cohort,family=family) for cohort in ('regression','fresh','transfer') for family in ('lookup','sum','edit','recovery') for i in range(4)]
    execution={'models':{name:{'outcomes':[dict(id=c['id'],grade={'success':name.startswith('expansion-') or (c['family']=='sum')}) for c in cases]} for name in MODEL_NAMES}}
    metrics={cohort:{'metrics':{'assistant':{'token_accuracy':1.,'record_macro_nll':1.}}} for cohort in ('tool-retention','conversation-retention','native-development')}
    learning={'models':{name:copy.deepcopy(metrics) for name in MODEL_NAMES}}
    return execution,learning,{'cases':cases}


def test_joint_behavior_and_retention_gate():
    e,l,p=reports();checks,counts=gate_checks(e,l,p)
    assert len(checks)==10 and all(c['passed'] for c in checks)
    assert counts['pre-y']['fresh']==4 and counts['expansion-y']['transfer']==16
    l['models']['expansion-x']['conversation-retention']['metrics']['assistant']['record_macro_nll']=1.151
    checks,_=gate_checks(e,l,p)
    assert sum(not c['passed'] for c in checks)==1


def test_old_regression_and_new_edits_cannot_be_hidden_by_sums():
    e,l,p=reports()
    for r in e['models']['expansion-y']['outcomes']:
        if '-edit-' in r['id'] or '-recovery-' in r['id']:r['grade']['success']=False
    checks,_=gate_checks(e,l,p)
    assert not next(c['passed'] for c in checks if c['cohort']=='new-edit-recovery')
    for r in e['models']['expansion-y']['outcomes']:
        if r['id'].startswith('regression-'):r['grade']['success']=False
    checks,_=gate_checks(e,l,p)
    assert not next(c['passed'] for c in checks if c['cohort']=='regression')


def test_evaluation_binds_correction_parent_not_old_u880(tmp_path,monkeypatch):
    import scripts.prepare_e97_grounded_expansion as module
    phase=tmp_path/'phase';phase.mkdir();data=tmp_path/'data';data.mkdir()
    def save(path,value):path.write_text(json.dumps(value));return module.sha(path)
    parent=tmp_path/'parent';parent.write_text('correction-live-y')
    post=tmp_path/'post';post.write_text('new-terminal')
    _,_,panel=reports();old=tmp_path/'old.json';learn=tmp_path/'learn.json'
    old_sha=save(old,{'cases':panel['cases'][:16],'models':[{'checkpoint':'wrong-u880'}]})
    learn_sha=save(learn,{'models':[{'checkpoint':'wrong-u880'}],'current_update':880})
    fresh_sha=save(data/'fresh-evaluation-cases.json',panel['cases'][16:])
    recipe_sha=save(phase/'recipe.json',dict(correction_config=dict(parent_checkpoint=str(parent),parent_sha256=module.sha(parent)),fresh_evaluation_cases_sha256=fresh_sha))
    save(phase/'summary.json',dict(status='passed',recipe_sha256=recipe_sha,checkpoint=dict(path=str(post),sha256=module.sha(post))))
    monkeypatch.setattr(module,'verify_inventory',lambda _:None)
    for name,value in (('EXECUTION_PANEL',old),('EXECUTION_SHA',old_sha),('LEARNING_PANEL',learn),('LEARNING_SHA',learn_sha)):monkeypatch.setattr(module,name,value)
    module.evaluations(SimpleNamespace(phase=phase,data=data))
    for kind in ('execution','learning'):
        actual=json.loads((phase/'evaluation'/kind/'panel.json').read_text())
        assert [m['checkpoint'] for m in actual['models']]==[str(parent),str(parent),str(post),str(post)]
        assert [m['mode'] for m in actual['models']]==['train','saved','train','saved']
    assert len(json.loads((phase/'evaluation/execution/panel.json').read_text())['cases'])==48


def test_exposure_distinguishes_unique_from_repeated(tmp_path,monkeypatch):
    import ndm.data.masked_sft_dataset as loader
    import scripts.prepare_e97_grounded_expansion as module
    (tmp_path/'authority').mkdir();(tmp_path/'packs').mkdir()
    rows=[dict(source='grounded-expansion',source_record_id=f,tokens=20,targets=10) for f in ('lookup','sum','edit','recovery')]
    (tmp_path/'authority/records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'authority/manifest.json').write_text(json.dumps({'outputs':{'metadata':{'path':'records.jsonl'}}}))
    (tmp_path/'packs/manifest.json').write_text('{}')
    (tmp_path/'training-cases-private.json').write_text(json.dumps([dict(id=r['source_record_id'],family=r['source_record_id']) for r in rows]))
    schedule=dict(authority_manifest_sha256=module.sha(tmp_path/'authority/manifest.json'),pack_manifest_sha256=module.sha(tmp_path/'packs/manifest.json'),
        sampler_key=974223,world_size=8,context_size=65536,unique_records=4,source_target_totals={'grounded-expansion':10240},
        steps=[dict(rank_sample_ids=[[i*8+r] for r in range(8)],global_tokens=640) for i in range(32)])
    (tmp_path/'expected-schedule.json').write_text(json.dumps(schedule))
    class Dataset:
        def __init__(self,*args,rank,**kwargs):self.rank=rank;self.packs=[{'record_offset':0,'record_count':4}];self.pack_record_ids=list(range(4))
        def sample_id(self,cursor):return cursor*8+self.rank
        def pack_id_at(self,cursor):return 0
        def close(self):pass
    monkeypatch.setattr(loader,'MaskedSFTPackedDataset',Dataset)
    module.exposure(SimpleNamespace(data=tmp_path))
    result=json.loads((tmp_path/'exposure-audit.json').read_text())
    assert result['unique_records']==4 and result['record_occurrences']==1024
    assert result['sources']['grounded-expansion']['unique_targets']==40
    assert result['target_exposures']==10240
    assert result['authored_family_coverage']['edit']['unique_records']==1


@pytest.mark.parametrize('bad',['missing','duplicate','nonboolean','nan','wrong-model'])
def test_gate_fails_closed_on_bad_coverage_or_metrics(bad):
    e,l,p=reports();rows=e['models']['pre-y']['outcomes']
    if bad=='missing':rows.pop()
    elif bad=='duplicate':rows[-1]=rows[0]
    elif bad=='nonboolean':rows[0]['grade']['success']='false'
    elif bad=='nan':l['models']['pre-y']['conversation-retention']['metrics']['assistant']['record_macro_nll']=float('nan')
    else:e['models']['other']=e['models'].pop('pre-x')
    with pytest.raises(ValueError):gate_checks(e,l,p)
