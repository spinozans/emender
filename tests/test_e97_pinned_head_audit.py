import copy
import pytest
from scripts.audit_e97_pinned_head import compare


def example():
    old=[dict(id='case',turn=0,recorded=[-.1],actor_replay=[-.1],teacher=[-.3],ce_mean=.3,padded_tokens=129)]
    rows=copy.deepcopy(old)
    a=dict(native_logprob=-.1,fp32_logprob=-.2,rounded_fp32_logprob=-.1,
        bf16_selected_logit=1.,fp32_selected_logit=1.,fp64_selected_logit=1.,
        fp32_vs_fp64_selected_abs=0.,native_vs_rounded_fp32_logit_abs_max=0.)
    t=dict(a,native_logprob=-.3,fp32_logprob=-.201,rounded_fp32_logprob=-.3,
        bf16_selected_logit=1.25,hidden_relative_l2=.001,hidden_max_absolute=.01)
    rows[0]['head_probe']=dict(actor=[a],teacher=[t])
    return rows,old


def test_head_shadow_success_does_not_replace_actual_failed_scores():
    rows,old=example();saved=copy.deepcopy(rows);binding,details=compare(rows,old)
    assert binding==0 and details[0]['actual_gap']>.05 and details[0]['fp32_gap']<.05
    assert rows==saved


def test_repeat_or_observer_binding_does_not_hide_mismatch():
    rows,old=example();rows[0]['teacher'][0]+=.01
    binding,_=compare(rows,old);assert binding>.0001


def test_auxiliary_native_probabilities_must_bind_too():
    rows,old=example();rows[0]['head_probe']['actor'][0]['native_logprob']+=.01
    binding,_=compare(rows,old);assert binding>.0001


@pytest.mark.parametrize('kind',['coverage','identity','nonfinite','dot'])
def test_bad_head_evidence_rejected(kind):
    rows,old=example()
    if kind=='coverage':rows[0]['head_probe']['actor'].clear()
    elif kind=='identity':rows[0]['recorded']=[-.2]
    elif kind=='nonfinite':rows[0]['head_probe']['actor'][0]['fp64_selected_logit']=float('nan')
    else:rows[0]['head_probe']['actor'][0]['fp32_vs_fp64_selected_abs']=.1
    with pytest.raises(ValueError):compare(rows,old)
