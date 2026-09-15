from copy import deepcopy
import pytest
from scripts.eval_e97_pi_repository_probes import measures


def data():
    return [dict(id=str(i),family='boundary' if i<2 else 'selection',route=r,success=True)
            for r in ('direct','pi') for i in range(4)]


def test_measurements_and_capability_are_separate():
    r=data(); ids=list(map(str,range(4)))
    assert measures(r,ids)==dict(per_case_parity=True,capability_gate_passed=True)
    for x in r: x['success']=False
    assert measures(r,ids)==dict(per_case_parity=True,capability_gate_passed=False)
    r[4]['success']=True; r[5]['success']=True
    assert measures(r,ids)==dict(per_case_parity=False,capability_gate_passed=False)
    r[6]['success']=True
    assert measures(r,ids)['capability_gate_passed'] is True


@pytest.mark.parametrize('mutation',['missing','duplicate','integer','family','route'])
def test_result_coverage_and_types(mutation):
    r=data()
    if mutation=='missing': r.pop()
    elif mutation=='duplicate': r[1]=deepcopy(r[0])
    elif mutation=='integer': r[0]['success']=1
    elif mutation=='family': r[4]['family']='other'
    else: r[0]['route']='unknown'
    with pytest.raises(ValueError): measures(r,list(map(str,range(4))))
