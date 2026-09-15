import pytest
from scripts.eval_e97_pi_native_model_session import CASE_IDS,select_cases


def case(cid,family,cohort='fresh',supplied=None):
 return dict(id=cid,family=family,cohort=cohort,supplied_calls=[] if supplied is None else supplied)


def test_selects_fixed_previously_qualified_task_order():
 panel={'cases':[case(CASE_IDS[1],'edit'),case('other','sum'),case(CASE_IDS[0],'lookup')]}
 selected=select_cases(panel)
 assert [c['id'] for c in selected]==list(CASE_IDS)
 selected[0]['family']='changed'
 assert panel['cases'][2]['family']=='lookup'


def test_rejects_assisted_or_wrong_family():
 with pytest.raises(ValueError,match='fixed cases'):
  select_cases({'cases':[case(CASE_IDS[0],'lookup',supplied=[{}]),case(CASE_IDS[1],'edit')]})
 with pytest.raises(ValueError,match='fixed cases'):
  select_cases({'cases':[case(CASE_IDS[0],'lookup'),case(CASE_IDS[1],'sum')]})
