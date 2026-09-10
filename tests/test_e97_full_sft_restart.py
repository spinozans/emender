from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
from scripts.qualify_e97_full_sft_restart import POLICY,SCHEMA,fixture,state_digest,validate_recipe
from scripts.train_e97_4b_pi_sft import build_optimizer


def recipe():
    return dict(schema=SCHEMA,policy=deepcopy(POLICY),world_size=8,saved_update=2,
                continuation_update=3,training_eligible=False,
                continuation_acceptance='bitwise-full-model-and-optimizer-state')


def test_frozen_recipe_has_no_tolerance_fallback():
    validate_recipe(recipe())
    for key,value in [('world_size',1),('saved_update',3),('training_eligible',True),
                      ('continuation_acceptance','approximately-equal')]:
        changed=recipe();changed[key]=value
        with pytest.raises(ValueError):validate_recipe(changed)
    changed=recipe();changed['policy']['lr']=.001
    with pytest.raises(ValueError):validate_recipe(changed)


def test_authored_packs_have_exact_masks_and_non_aligned_reset():
    previous=None
    for rank in range(8):
        for update in (1,2,3):
            tokens,loss,valid,reset=fixture(rank,update,'cpu')
            assert tokens.shape==(1,513) and loss.shape==(1,512)
            assert valid.sum()==258 and loss.sum()==256 and reset.sum()==2
            assert reset[0,0] and reset[0,129]
            assert not (loss&reset[:,1:]).any() and not (loss&~valid[:,1:]).any()
            assert not tokens[0,258:].any()
            assert all(torch.equal(a,b) for a,b in zip((tokens,loss,valid,reset),fixture(rank,update,'cpu')))
            if previous is not None:assert not torch.equal(tokens,previous)
            previous=tokens


@pytest.mark.parametrize('rank,update',[(-1,1),(8,1),(0,0),(0,4)])
def test_out_of_recipe_indices_fail(rank,update):
    with pytest.raises(ValueError):fixture(rank,update,'cpu')


def test_digest_covers_model_optimizer_and_basis():
    model=torch.nn.Linear(4,3).bfloat16()
    args=SimpleNamespace(**{**POLICY,'schedulefree_offload_pin_memory':0})
    opt=build_optimizer(model.parameters(),args,named_parameters=model.named_parameters())
    opt.initialize_state_();opt.train()
    initial=state_digest(model,opt)
    with torch.no_grad():next(model.parameters()).view(-1)[0]+=1
    changed=state_digest(model,opt)
    assert changed!=initial and changed==state_digest(model,opt)
    opt.state[next(model.parameters())]['exp_avg_sq'].fill_(1)
    assert state_digest(model,opt)!=changed
    trained=state_digest(model,opt);opt.eval()
    assert state_digest(model,opt)!=trained
    opt.train();assert state_digest(model,opt)==trained
