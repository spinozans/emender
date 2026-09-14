import json
import math
import pytest
import torch
from scripts.diagnose_e97_early_prompt import terminal_guard,ordered_first,selected_dot,round_bf16
from scripts.audit_e97_live_actor_capture import fingerprint


def test_temporal_first_is_not_topological_first():
    rows=[dict(site='input',exact=True,first_token=None),dict(site='qkv',exact=False,first_token=5),dict(site='readout',exact=False,first_token=1)]
    first,temporal=ordered_first(rows)
    assert first['site']=='qkv' and temporal['site']=='readout'
    assert ordered_first(rows[:1])==(None,None)


def test_terminal_guard_records_success_and_preserves_body_failure(tmp_path):
    model=torch.nn.Linear(2,2).bfloat16();before=fingerprint(model)
    with pytest.raises(ValueError,match='primary'):
        with terminal_guard(model,tmp_path,before,lambda:128,{},1024):raise ValueError('primary')
    result=json.loads((tmp_path/'terminal-safety.json').read_text())
    assert result['passed'] and result['before']==result['after'] and result['peak_hbm_allocated']==128


def test_terminal_guard_reports_parameter_mutation(tmp_path):
    model=torch.nn.Linear(2,2).bfloat16();before=fingerprint(model)
    with pytest.raises(RuntimeError,match='safety'):
        with terminal_guard(model,tmp_path,before,lambda:128,{},1024):
            with torch.no_grad():model.weight.add_(1)
    result=json.loads((tmp_path/'terminal-safety.json').read_text())
    assert not result['passed'] and not result['checks']['parameters_buffers_unchanged']


def test_terminal_guard_records_audit_failure(tmp_path):
    model=torch.nn.Linear(2,2).bfloat16();before=fingerprint(model)
    def broken():raise ValueError('memory query failed')
    with pytest.raises(RuntimeError,match='safety'):
        with terminal_guard(model,tmp_path,before,broken,{},1024):pass
    result=json.loads((tmp_path/'terminal-safety.json').read_text())
    assert result['audit_error']['message']=='memory query failed' and not result['passed']


def test_bf16_reference_rounding_avoids_float32_double_rounding():
    midpoint=1.+2.**-8
    assert round_bf16(midpoint)==1.
    assert round_bf16(math.nextafter(midpoint,math.inf))==1.+2.**-7
    assert round_bf16(-midpoint)==-1.
    assert math.copysign(1.,round_bf16(-2.**-200))==-1.
    assert round_bf16(2.**-134)==0.
    with pytest.raises(ValueError):round_bf16(float('nan'))


def test_fp64_selected_dot_uses_real_bf16_operands_and_bias():
    module=torch.nn.Linear(2,1).bfloat16()
    with torch.no_grad():module.weight.copy_(torch.tensor([[3.,4.]]));module.bias.fill_(1.)
    x=torch.tensor([1.,2.],dtype=torch.bfloat16);before=fingerprint(module)
    result,values=selected_dot(module,x,11.,12.,0)
    assert result['fp64_dot']==12. and result['bf16_rounded']==12.
    assert not result['actor_matches_rounded'] and result['teacher_matches_rounded']
    assert torch.equal(values['weight'],module.weight[0]) and values['input'].dtype==torch.bfloat16
    assert fingerprint(module)==before and module.weight.grad is None
    with pytest.raises(ValueError):selected_dot(module,x.float(),11.,12.,0)
