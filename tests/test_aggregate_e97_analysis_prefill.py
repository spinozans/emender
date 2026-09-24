import hashlib
import json
import pytest
from scripts.aggregate_e97_analysis_prefill import aggregate


def fixture(tmp_path):
    panel={'tasks':[{'id':'a'},{'id':'b'}]}; p=tmp_path/'panel.json'; p.write_text(json.dumps(panel))
    for rank,id in enumerate(['a','b']):
        event={'prefill':'Analysis:','prefill_token_ids':[32750,25],'model_generated_tokens':10,'stop_reason':'complete_turn'}
        result={'id':id,'kind':'read','success':True,'prefilled_first_turn_protocol_valid':True,'reads':['a.txt'],'missing_reads':[],
                'turns':[{'completion_tokens':12,'prefill_intervention':event}]}
        report={'schema':'emender-e97-analysis-prefill-shard-v1','training_eligible':False,
                'intervention':'supplied opening','checkpoint_sha256':'a'*64,'weight_mode':'train',
                'panel_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'world_size':2,'rank':rank,'device':f'cuda:{rank}',
                'controller_closure_sha256':'b'*64,'evaluator_sha256':'c'*64,'results':[result]}
        (tmp_path/f'rank-{rank:02d}.json').write_text(json.dumps(report))
    return p


def test_prefilled_not_autonomous_accounting(tmp_path):
    p=fixture(tmp_path); result=aggregate(tmp_path,p,2)
    assert result['prefilled_successes']==2 and result['admitted_read_calls']==2
    assert result['model_generated_tokens']==20 and result['supplied_tokens']==4
    assert 'first_turn_protocol_valid' not in result


@pytest.mark.parametrize('failure',['missing','autonomous','prefill','membership','device'])
def test_rejects_invalid_evidence(tmp_path,failure):
    p=fixture(tmp_path); shard=tmp_path/'rank-01.json'; d=json.loads(shard.read_text())
    if failure=='missing': shard.unlink()
    else:
        if failure=='autonomous': d['results'][0]['first_turn_protocol_valid']=True
        if failure=='prefill': d['results'][0]['turns'][0]['prefill_intervention']['prefill_token_ids']=[12502,25]
        if failure=='membership': d['results'][0]['id']='a'
        if failure=='device': d['device']='cuda:0'
        shard.write_text(json.dumps(d))
    with pytest.raises(ValueError): aggregate(tmp_path,p,2)
