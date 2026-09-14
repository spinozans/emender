import pytest
import torch
import torch.nn.functional as F
from ndm.numerical_policy import RecomputedFP32Linear
from scripts.diagnose_e97_linear_operands import LinearCapture,hashes
from scripts.e97_tensor_capsule import unpack


def test_capture_actual_fp32_operation_and_first_bf16_store():
    module=RecomputedFP32Linear(3,4).bfloat16();x=torch.randn(1,2,3).bfloat16();original=F.linear
    with torch.no_grad():
        expected=original(x.float(),module.weight.float(),module.bias.float())
        with LinearCapture(module) as cap:
            result=module(x)
            second=module(x+1)
        assert cap.complete and F.linear is original
        assert torch.equal(cap.fp32,expected) and torch.equal(cap.body,result) and torch.equal(result,expected.bfloat16())
        assert torch.equal(second,original((x+1).float(),module.weight.float(),module.bias.float()).bfloat16())
        values=unpack(cap.capsule,'cpu')
        assert torch.equal(values['x'],x.float()) and torch.equal(values['w'],module.weight.float())
        assert hashes(values)==cap.input_hashes


def test_functional_observer_returns_original_result_object(monkeypatch):
    module=RecomputedFP32Linear(2,2,bias=False).bfloat16();x=torch.ones(1,1,2,dtype=torch.bfloat16)
    original=F.linear;seen=[]
    def spy(x,w,bias=None):
        result=original(x,w,bias);seen.append(result);return result
    monkeypatch.setattr(F,'linear',spy)
    with torch.no_grad(),LinearCapture(module) as cap:
        cap.before(module,(x,))
        result=F.linear(x.float(),module.weight.float())
        assert result is seen[0]
        body=result.bfloat16();assert cap.after(module,(x,),body) is None
    assert F.linear is spy


def test_observer_restores_after_original_failure(monkeypatch):
    module=RecomputedFP32Linear(2,2,bias=False).bfloat16()
    def broken(*args):raise RuntimeError('original failure')
    monkeypatch.setattr(F,'linear',broken)
    with pytest.raises(RuntimeError,match='original failure'):
        with torch.no_grad(),LinearCapture(module):module(torch.ones(1,1,2,dtype=torch.bfloat16))
    assert F.linear is broken and not module._forward_hooks and not module._forward_pre_hooks


def test_input_mutation_is_not_silently_accepted(monkeypatch):
    module=RecomputedFP32Linear(2,2,bias=False).bfloat16();original=F.linear
    def mutate(x,w,bias=None):
        result=original(x,w,bias);x.add_(1);return result
    monkeypatch.setattr(F,'linear',mutate)
    with pytest.raises(ValueError,match='mutated'):
        with torch.no_grad(),LinearCapture(module):module(torch.ones(1,1,2,dtype=torch.bfloat16))
    assert F.linear is mutate
