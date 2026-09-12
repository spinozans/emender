import math
import pytest
import torch
from ndm.e97_outcome_rl_candidate import group_advantages, policy_loss


@pytest.mark.parametrize('n,m',[(1,1),(5,128),(128,1),(129,3)])
def test_training_turn_target_alignment(n,m):
    from scripts.qualify_e97_native_rl_logprobs import turn_layout
    prefix=[2]*n;generated=[3]*m
    tokens,valid,reset,mask=turn_layout(prefix,generated,'cpu')
    assert (tokens.shape[1]-1)%128==0
    assert int(valid.sum())==n+m and int(reset.sum())==1 and reset[0,0]
    assert int(mask.sum())==m and tokens[:,1:][mask].tolist()==generated
    assert not (mask & ~valid[:,1:]).any() and not (mask & reset[:,1:]).any()
    assert not mask[:,:n-1].any() and not mask[:,n+m-1:].any()


def test_complete_groups_and_constant_rewards():
    groups=torch.tensor([0,0,1,1,2,2])
    r=torch.tensor([0.,0.,1.,1.,0.,1.],requires_grad=True)
    a=group_advantages(r,groups)
    assert a.tolist()==[0.,0.,0.,0.,-1.,1.] and not a.requires_grad
    with pytest.raises(ValueError):group_advantages(torch.tensor([1.]),torch.tensor([0]))


def test_assistant_mask_detaches_behavior_reference_and_rewards():
    new=torch.tensor([[float('nan'),-.5,-.7]],requires_grad=True)
    old=torch.tensor([[float('nan'),-.5,-.7]],requires_grad=True)
    ref=old.detach().clone().requires_grad_(True)
    adv=torch.tensor([1.],requires_grad=True);mask=torch.tensor([[False,True,True]])
    loss=policy_loss(new,old,ref,adv,mask,torch.tensor([2]),1)
    assert loss.item()==pytest.approx(-1.)
    loss.backward()
    assert new.grad.tolist()[0]==pytest.approx([0.,-.5,-.5])
    assert old.grad is None and ref.grad is None and adv.grad is None


def test_all_identical_groups_have_no_outcome_learning_signal():
    a=group_advantages(torch.tensor([1.,1.]),torch.tensor([0,0]))
    new=torch.tensor([[-1.],[-1.]],requires_grad=True)
    loss=policy_loss(new,new.detach(),new.detach(),a,torch.ones_like(new,dtype=torch.bool),torch.ones(2),2)
    loss.backward()
    assert loss.item()==0 and torch.count_nonzero(new.grad)==0


def test_turn_chunks_use_whole_episode_normalization():
    new=torch.tensor([[-.5,-.8,-1.]],requires_grad=True);old=new.detach()
    full=policy_loss(new,old,old,torch.tensor([1.]),torch.ones_like(new,dtype=torch.bool),torch.tensor([3]),1)
    pieces=sum(policy_loss(new[:,i:i+1],old[:,i:i+1],old[:,i:i+1],torch.tensor([1.]),
                           torch.ones((1,1),dtype=torch.bool),torch.tensor([3]),1) for i in range(3))
    assert full.item()==pytest.approx(pieces.item())


@pytest.mark.parametrize('adv,ratio,expected',[(1.,2.,-1.2),(-1.,.5,.8)])
def test_clip_direction(adv,ratio,expected):
    new=torch.tensor([[math.log(ratio)]],requires_grad=True);old=torch.zeros_like(new)
    loss=policy_loss(new,old,old,torch.tensor([adv]),torch.tensor([[True]]),torch.tensor([1]),1,kl_beta=0.)
    assert loss.item()==pytest.approx(expected)
    loss.backward();assert new.grad.item()==0


def test_repeatability_comparison_separates_probabilities_and_logits():
    from scripts.diagnose_e97_actor_logprob_repeatability import compare
    a={'steps':[{'logp':-.5,'selected_logit':2.,'logits_sha256':'a'}]}
    b={'steps':[{'logp':-.625,'selected_logit':2.,'logits_sha256':'b'}]}
    assert compare(a,b)==dict(logprob_max=.125,selected_logit_max=0.,differing_full_logit_digests=1)
    with pytest.raises(ValueError):compare(a,{'steps':[]})
    full={'steps':[{'logp':-.6,'selected_logit':None,'log_normalizer':None,'logits_sha256':None}]}
    result=compare(a,full)
    assert result['logprob_max']==pytest.approx(.1)
    assert result['selected_logit_max'] is None and result['differing_full_logit_digests'] is None


def test_actor_fingerprint_detects_parameter_and_buffer_changes():
    from scripts.audit_e97_live_actor_capture import fingerprint
    from scripts.qualify_e97_response_gradients import parameter_digest
    model=torch.nn.Linear(2,2).to(dtype=torch.bfloat16)
    model.register_buffer('probe',torch.tensor([1.]))
    initial=fingerprint(model)
    assert initial['parameters']==parameter_digest(model)
    model.probe.add_(1)
    changed=fingerprint(model)
    assert changed['parameters']==initial['parameters'] and changed['buffers']!=initial['buffers']
    with torch.no_grad():model.weight.add_(1)
    assert fingerprint(model)['parameters']!=initial['parameters']


def test_invalid_counts_and_ratio_overflow_fail_closed():
    x=torch.tensor([[0.,0.]])
    with pytest.raises(ValueError):policy_loss(x,x,x,torch.tensor([1.]),torch.tensor([[True,True]]),torch.tensor([1]),1)
    with pytest.raises(ValueError):policy_loss(x+1000,x,x,torch.tensor([1.]),torch.tensor([[True,True]]),torch.tensor([2]),1)
