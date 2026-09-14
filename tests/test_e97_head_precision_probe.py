import pytest
import torch
import torch.nn.functional as F
from scripts.e97_head_precision_probe import HeadProbe,project


@pytest.mark.parametrize('numeric_audit',[False,True])
def test_probe_is_counterfactual_and_preserves_real_outputs_and_parameters(numeric_audit):
    torch.manual_seed(19)
    head=torch.nn.Linear(3,5).to(dtype=torch.bfloat16)
    saved={k:v.clone() for k,v in head.state_dict().items()}
    h=torch.randn(1,2,3,dtype=torch.bfloat16);next_h=torch.randn(1,1,3,dtype=torch.bfloat16)
    expected=head(h).detach();probe=HeadProbe([1,2],numeric_audit=numeric_audit);handle=head.register_forward_hook(probe.actor_hook)
    actual=head(h);head(next_h)
    with pytest.raises(ValueError,match='coverage'):probe.finish_actor()
    head(next_h);handle.remove();probe.finish_actor()
    assert torch.equal(actual,expected)
    teacher_h=torch.cat([h[:,-1,:],next_h[:,-1,:]],dim=0);targets=torch.tensor([1,2])
    probe.observe_teacher(head,teacher_h,head(teacher_h),targets)
    report=probe.report()
    reference=F.linear(teacher_h.float(),head.weight.float(),head.bias.float())
    expected_lp=torch.log_softmax(reference,-1).gather(1,targets[:,None]).flatten().tolist()
    assert [r['fp32_logprob'] for r in report['teacher']]==pytest.approx(expected_lp,abs=1e-6)
    assert all(r['hidden_relative_l2']==0 for r in report['teacher'])
    assert all(torch.equal(saved[k],v) for k,v in head.state_dict().items())
    assert all(p.dtype==torch.bfloat16 for p in head.parameters())


@pytest.mark.parametrize('numeric_audit',[False,True])
def test_vocab_chunks_and_autocast_do_not_round_shadow_projection_to_bf16(numeric_audit):
    torch.manual_seed(20)
    head=torch.nn.Linear(3,7,bias=False).to(dtype=torch.bfloat16);h=torch.randn(2,3,dtype=torch.bfloat16)
    targets=torch.tensor([1,6]);logits=head(h)
    with torch.autocast('cpu',dtype=torch.bfloat16):rows=project(head,h,logits,targets,vocab_chunk=2,numeric_audit=numeric_audit)
    reference=F.linear(h.float(),head.weight.float())
    assert [r['fp32_selected_logit'] for r in rows]==pytest.approx(reference.gather(1,targets[:,None]).flatten().tolist(),abs=1e-6)
    assert [r['bf16_selected_logit'] for r in rows]==logits.gather(1,targets[:,None]).flatten().tolist()
    if numeric_audit:
        import math
        expected=[math.fsum(float(a)*float(b) for a,b in zip(x,head.weight[t])) for x,t in zip(h,targets)]
        assert [r['fp64_selected_logit'] for r in rows]==expected
        quantized=torch.log_softmax(reference.bfloat16().float(),-1).gather(1,targets[:,None]).flatten().tolist()
        assert [r['rounded_fp32_logprob'] for r in rows]==pytest.approx(quantized,abs=1e-6)


@pytest.mark.parametrize('target,chunk',[(7,4096),(-1,4096),(0,0),(0,4097)])
def test_invalid_probe_bounds_fail_closed(target,chunk):
    head=torch.nn.Linear(3,7).to(dtype=torch.bfloat16);h=torch.ones(1,3,dtype=torch.bfloat16)
    with pytest.raises(ValueError):project(head,h,head(h),torch.tensor([target]),chunk)


def test_teacher_target_alignment_is_not_silently_rewritten():
    head=torch.nn.Linear(3,7).to(dtype=torch.bfloat16);h=torch.ones(1,1,3,dtype=torch.bfloat16)
    p=HeadProbe([1]);handle=head.register_forward_hook(p.actor_hook);head(h);head(h);handle.remove()
    with pytest.raises(ValueError,match='alignment'):p.observe_teacher(head,h[0],head(h[0]),torch.tensor([2]))


def test_final_bf16_store_can_amplify_a_small_hidden_difference():
    # Controlled final-store example, not an assertion about a particular GEMM.
    head=torch.nn.Linear(3,2,bias=False).bfloat16()
    with torch.no_grad():head.weight.copy_(torch.tensor([[20.,1.,1.],[20.,0.,0.]]))
    h=torch.tensor([[1.,.03125,.031005859375],[1.,.03125,.031494140625]],dtype=torch.bfloat16)
    logits=F.linear(h.float(),head.weight.float()).bfloat16();saved=logits.clone()
    a,b=project(head,h,logits,torch.tensor([0,0]),numeric_audit=True)
    assert abs(a['native_logprob']-b['native_logprob'])>.05
    assert abs(a['fp32_logprob']-b['fp32_logprob'])<.001
    assert abs(a['rounded_fp32_logprob']-b['rounded_fp32_logprob'])>.05
    assert a['fp32_vs_fp64_selected_abs']==b['fp32_vs_fp64_selected_abs']==0
    assert torch.equal(logits,saved)
