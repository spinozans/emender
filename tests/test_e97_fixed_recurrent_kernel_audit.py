import copy
import pytest
from scripts.audit_e97_fixed_recurrent_kernel import row_metrics,repeated_exactly


def example():
    selected=[dict(id='case',turn=0,generated=[1,2],recorded_logprobs=[-.1,-.2])]
    rows=[dict(id='case',turn=0,actor_replay=[-.2,-.3],teacher=[-.2,-.3],recorded=[-.1,-.2],ce_mean=.25)]
    return rows,selected


def test_current_policy_agreement_does_not_replace_historical_probabilities():
    rows,selected=example();original=copy.deepcopy(rows);metrics=row_metrics(rows,selected)
    assert metrics['pair_max']==0 and metrics['historical_replay_max']>.05
    assert metrics['ce_max']==0 and rows==original


def test_repeatability_is_exact_and_rejects_signed_zero_change():
    a,_=example();b=copy.deepcopy(a);assert repeated_exactly(a,b)
    b[0]['teacher'][0]+=.000001;assert not repeated_exactly(a,b)
    a[0]['teacher'][0]=0.;b=copy.deepcopy(a);b[0]['teacher'][0]=-0.
    assert not repeated_exactly(a,b)


@pytest.mark.parametrize('kind',['coverage','identity','historical','nonfinite'])
def test_bad_measurements_fail_closed(kind):
    rows,selected=example()
    if kind=='coverage':rows[0]['teacher'].pop()
    elif kind=='identity':rows[0]['id']='other'
    elif kind=='historical':rows[0]['recorded'][0]=-.2
    else:rows[0]['actor_replay'][0]=float('nan')
    with pytest.raises(ValueError):row_metrics(rows,selected)
