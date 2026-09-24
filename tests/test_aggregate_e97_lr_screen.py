import json
import pytest
from scripts.aggregate_e97_lr_screen import aggregate
from scripts.e97_lr_screen_panel import make_panel


def fixtures(root):
    panel=make_panel()
    for rank in range(8):
        shard={'checkpoint_sha256': 'a'*64, 'weight_mode': 'train', 'panel_sha256': 'b'*64,
               'world_size':8, 'rank':rank, 'controller_closure_sha256':'c'*64,
               'evaluator_sha256':'d'*64,
               'results':[{'id':t['id'],'kind':t['kind'],'success':False,
                           'first_turn_protocol_valid':False,'reads':[],'missing_reads':[],'turns':[]}
                          for t in panel['tasks'][rank::8]]}
        (root/f'rank-{rank:02d}.json').write_text(json.dumps(shard))
    return panel


def test_complete_and_failed_behavior_is_valid_evidence(tmp_path):
    panel=fixtures(tmp_path)
    result=aggregate(tmp_path,panel,'b'*64)
    assert result['tasks']==24 and result['passed']==0


@pytest.mark.parametrize('field,value', [('checkpoint_sha256','e'*64),('weight_mode','saved'),('rank',3)])
def test_mixed_identities_rejected(tmp_path,field,value):
    panel=fixtures(tmp_path)
    path=tmp_path/'rank-00.json'
    data=json.loads(path.read_text()); data[field]=value; path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        aggregate(tmp_path,panel,'b'*64)


def test_missing_results_rejected(tmp_path):
    panel=fixtures(tmp_path)
    path=tmp_path/'rank-00.json'
    data=json.loads(path.read_text()); data['results']=[]; path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        aggregate(tmp_path,panel,'b'*64)
