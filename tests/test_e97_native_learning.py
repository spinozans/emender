import json
from types import SimpleNamespace
import numpy as np
import pytest
import tiktoken
import torch
from scripts.e97_open_swe_native_codec import native_turn
from scripts.eval_e97_native_learning import annotations,score,select_unique,aggregate,SCHEMA


def test_native_annotations_are_supervised_and_capture_action():
    enc=tiktoken.get_encoding('p50k_base')
    body=native_turn(dict(role='assistant',content='Public update',reasoning_content='Private thought',think=False,
        tool_calls=[dict(type='function',function=dict(name='str_replace_editor',arguments=json.dumps(dict(command='view',path='/testbed/α.py'))))]))
    prefix=enc.encode_ordinary('User: task\n\nAssistant:\n');target=enc.encode_ordinary(body)
    tokens=prefix+target;mask=np.array([0]*len(prefix)+[1]*len(target))
    result=annotations(tokens,mask,True,enc)
    assert result['opening_positions']==[len(prefix)]
    assert 'str_replace_editor' in enc.decode([tokens[p] for p in result['choice_positions']])
    assert result['first_gold']['arguments']['path']=='/testbed/α.py'
    with pytest.raises(ValueError):annotations([1,2],np.array([1,1]),False,enc)


def test_full_record_score_uses_shifted_supervision_and_distinct_metrics():
    class Model:
        def __call__(self,tokens,return_loss):
            assert tokens.shape==(1,16) and return_loss is False
            logits=torch.zeros((1,16,8));logits[0,0,3]=10;logits[0,2,5]=10
            return logits
    example=dict(tokens=[2,3,4,5],mask=[0,1,0,1],opening_positions=[1,3],choice_positions=[3])
    report=score(Model(),example,torch.device('cpu'))
    assert report['assistant']['correct']==report['opening']['correct']==2
    assert report['assistant']['tokens']==2 and report['choice']['tokens']==1
    assert report['assistant']['nll_sum']<.001


def test_selection_groups_problem_attempts():
    rows=[dict(problem_key=['a','1']),dict(problem_key=['a','1']),dict(problem_key=['b','2'])]
    selected=select_unique([0,1,2],rows,'dev',2)
    assert len({tuple(rows[i]['problem_key']) for i in selected})==2
    with pytest.raises(ValueError):select_unique([0,1,2],rows,'dev',3)


def test_aggregate_rejects_duplicate_example_coverage(tmp_path):
    from ndm.data.masked_sft_dataset import sha256
    models=[dict(name=str(i)) for i in range(4)]
    panel=dict(examples=[dict(id='a'),dict(id='b')],models=models,scope='test')
    path=tmp_path/'panel.json';path.write_text(json.dumps(panel));digest=sha256(path)
    for rank in range(8):
        (tmp_path/f'rank-{rank}.json').write_text(json.dumps(dict(rank=rank,panel_sha256=digest,model=models[rank//2],results=[dict(id='a')])) )
    with pytest.raises(ValueError,match='coverage'):
        aggregate(SimpleNamespace(panel=path,panel_sha=digest,output=tmp_path))
