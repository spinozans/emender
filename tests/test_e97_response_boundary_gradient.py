"""Exercise actual LadderLM CE label/mask routing; CPU, not fused GPU backward."""
import pytest
import torch
from ndm.models.ladder_lm import LadderLM


@pytest.mark.parametrize('chunk_size',[0,3])
def test_response_first_token_is_predicted_before_consuming_it(chunk_size):
    torch.manual_seed(974120)
    model=LadderLM(vocab_size=64,dim=16,depth=2,level='E97',expansion=1.,
                   n_state=4,n_heads=4,use_gate=True,gate_activation='sigmoid',
                   linear_state=False,use_triton=False,mlp_ratio=2.,mlp_multiple=4).train()
    model.loss_chunk_size=chunk_size
    # Toy token IDs: context, Assistant header, newline, Analysis, colon, text.
    tokens=torch.tensor([[2,3,4,5,7,8,9,10]])
    mask=torch.zeros(1,7,dtype=torch.bool); mask[0,3]=True
    reset=torch.zeros_like(tokens,dtype=torch.bool); reset[:,0]=True
    valid=torch.ones_like(tokens,dtype=torch.bool)
    captured=[]
    def capture(module,args,output):
        output.retain_grad(); captured.append(output)
    handle=model.lm_head.register_forward_hook(capture)
    loss=model(tokens,return_loss=True,loss_mask=mask,reset_before=reset,
               valid_mask=valid,loss_reduction='sum')
    logits=torch.cat(captured,dim=1)
    torch.testing.assert_close(loss,-logits[0,3].log_softmax(-1)[7])
    loss.backward(); handle.remove()
    grad=torch.cat([output.grad for output in captured],dim=1)
    expected=torch.zeros_like(grad)
    expected[0,3]=logits[0,3].detach().softmax(-1); expected[0,3,7]-=1
    torch.testing.assert_close(grad,expected)
    assert grad[0,3,7]<0
    assert int(torch.count_nonzero(grad[:,:3]))==0
    assert int(torch.count_nonzero(grad[:,4:]))==0
