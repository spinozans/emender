import pytest
import torch
from ndm.models.ladder_lm import LadderLM


@pytest.mark.parametrize('chunk_size',[0,3])
def test_opt_in_fp32_ce_avoids_bf16_logit_loss_rounding(chunk_size):
    model=LadderLM(vocab_size=64,dim=16,depth=2,level='E97',expansion=1.,
        n_state=4,n_heads=4,use_gate=True,linear_state=False,use_triton=False,
        mlp_ratio=2.,mlp_multiple=4).bfloat16().train()
    # A controlled, differentiable head isolates CE from recurrence arithmetic.
    model.lm_head=torch.nn.Linear(16,64,bias=True,dtype=torch.bfloat16)
    with torch.no_grad():
        model.lm_head.weight.zero_();model.lm_head.bias.fill_(-20)
        model.lm_head.bias[9]=20;model.lm_head.bias[7]=0.5625
    model.loss_chunk_size=chunk_size
    tokens=torch.tensor([[2,3,4,7,8,9,10,11]])
    mask=torch.zeros(1,7,dtype=torch.bool);mask[0,2]=True
    valid=torch.ones_like(tokens,dtype=torch.bool)
    reset=torch.zeros_like(valid);reset[0,0]=True
    kwargs=dict(return_loss=True,loss_mask=mask,valid_mask=valid,reset_before=reset,loss_reduction='sum')
    assert model.loss_logits_fp32 is False
    legacy=model(tokens,**kwargs)
    assert float(legacy.detach())==19.5
    model.loss_logits_fp32=True
    candidate=model(tokens,**kwargs)
    assert candidate.dtype==torch.float32
    assert float(candidate.detach())==19.4375
    candidate.backward()
    assert model.lm_head.bias.grad[7]==-1
    assert model.lm_head.bias.grad[9]==1
    assert all(p.dtype==torch.bfloat16 for p in model.parameters())
