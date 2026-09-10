import pytest
import torch
from ndm.models.ladder_lm import LadderLM


@pytest.mark.parametrize('chunk_size,checkpoint_ce',[(0,False),(3,False),(3,True)])
def test_opt_in_fp32_ce_avoids_bf16_logit_loss_rounding(chunk_size,checkpoint_ce):
    model=LadderLM(vocab_size=64,dim=16,depth=2,level='E97',expansion=1.,
        n_state=4,n_heads=4,use_gate=True,linear_state=False,use_triton=False,
        mlp_ratio=2.,mlp_multiple=4).bfloat16().train()
    # A controlled, differentiable head isolates CE from recurrence arithmetic.
    model.lm_head=torch.nn.Linear(16,64,bias=True,dtype=torch.bfloat16)
    with torch.no_grad():
        model.lm_head.weight.zero_();model.lm_head.bias.fill_(-20)
        model.lm_head.bias[9]=20;model.lm_head.bias[7]=0.5625
    model.loss_chunk_size=chunk_size
    model.checkpoint_loss_chunks=checkpoint_ce
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


def test_checkpointed_ce_drops_retained_vocab_rows_and_preserves_gradients():
    torch.manual_seed(98)
    model=LadderLM(vocab_size=64,dim=16,depth=2,level='E97',expansion=1.,
        n_state=4,n_heads=4,use_gate=True,linear_state=False,use_triton=False,
        mlp_ratio=2.,mlp_multiple=4).bfloat16().train()
    model.loss_logits_fp32=True;model.loss_chunk_size=3
    tokens=torch.tensor([[2,3,4,7,8,9,10,11,12,13]])
    mask=torch.zeros(1,9,dtype=torch.bool);mask[0,[1,2,5,7]]=True
    valid=torch.ones_like(tokens,dtype=torch.bool)
    reset=torch.zeros_like(valid);reset[0,[0,4]]=True
    reference=None
    for checkpointed in (False,True):
        model.zero_grad(set_to_none=True);model.checkpoint_loss_chunks=checkpointed
        retained=[]
        def save(tensor):
            if tensor.ndim==2 and tensor.shape[1]==64 and tensor.shape[0]<=3:
                retained.append(tensor.numel())
            return tensor
        with torch.autograd.graph.saved_tensors_hooks(save,lambda tensor:tensor):
            loss=model(tokens,return_loss=True,loss_mask=mask,valid_mask=valid,
                       reset_before=reset,loss_reduction='sum')
        loss.backward()
        gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
        if not checkpointed:
            assert sum(retained)>0
            reference=(loss.detach().clone(),gradients)
        else:
            assert retained==[]
            assert torch.equal(loss.detach(),reference[0])
            assert all(torch.equal(g,reference[1][n]) for n,g in gradients.items())
