import pytest
import torch
import torch.nn.functional as F
from scripts.e97_head_precision_probe import HeadProbe,project


def test_probe_is_counterfactual_and_preserves_real_outputs_and_parameters():
    torch.manual_seed(19)
    head=torch.nn.Linear(3,5).to(dtype=torch.bfloat16)
    saved={k:v.clone() for k,v in head.state_dict().items()}
    h=torch.randn(1,2,3,dtype=torch.bfloat16);next_h=torch.randn(1,1,3,dtype=torch.bfloat16)
    expected=head(h).detach();probe=HeadProbe([1,2]);handle=head.register_forward_hook(probe.actor_hook)
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


def test_vocab_chunks_and_autocast_do_not_round_shadow_projection_to_bf16():
    torch.manual_seed(20)
    head=torch.nn.Linear(3,7,bias=False).to(dtype=torch.bfloat16);h=torch.randn(2,3,dtype=torch.bfloat16)
    targets=torch.tensor([1,6]);logits=head(h)
    with torch.autocast('cpu',dtype=torch.bfloat16):rows=project(head,h,logits,targets,vocab_chunk=2)
    reference=F.linear(h.float(),head.weight.float())
    assert [r['fp32_selected_logit'] for r in rows]==pytest.approx(reference.gather(1,targets[:,None]).flatten().tolist(),abs=1e-6)
    assert [r['bf16_selected_logit'] for r in rows]==logits.gather(1,targets[:,None]).flatten().tolist()


@pytest.mark.parametrize('target,chunk',[(7,4096),(-1,4096),(0,0),(0,4097)])
def test_invalid_probe_bounds_fail_closed(target,chunk):
    head=torch.nn.Linear(3,7).to(dtype=torch.bfloat16);h=torch.ones(1,3,dtype=torch.bfloat16)
    with pytest.raises(ValueError):project(head,h,head(h),torch.tensor([target]),chunk)


def test_teacher_target_alignment_is_not_silently_rewritten():
    head=torch.nn.Linear(3,7).to(dtype=torch.bfloat16);h=torch.ones(1,1,3,dtype=torch.bfloat16)
    p=HeadProbe([1]);handle=head.register_forward_hook(p.actor_hook);head(h);head(h);handle.remove()
    with pytest.raises(ValueError,match='alignment'):p.observe_teacher(head,h[0],head(h[0]),torch.tensor([2]))
