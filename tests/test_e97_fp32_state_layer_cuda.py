import os
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='leased CUDA required')
def test_fp32_state_across_real_512_projection_checkpoints():
    from ndm.models.e97 import E97SplitEditLayer
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.manual_seed(974223)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    model=E97SplitEditLayer(dim=16,n_heads=4,n_state=4,expansion=1.,use_conv=False,
        use_gate=True,gate_activation='silu',use_triton=True,projection_chunk_size=512,
        recurrent_state_precision='fp32').to(device='cuda',dtype=torch.bfloat16).train()
    original=[p.detach().clone() for p in model.parameters()]
    x=torch.randn(1,1040,16,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    saved=[]
    def pack(t):saved.append((tuple(t.shape),t.dtype));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        out,state=model(x)
    assert out.dtype==torch.bfloat16 and all(s.dtype==torch.float32 for s in state)
    # Outer projection checkpointing carries the real four-head matrix state.
    carries=[dtype for shape,dtype in saved if shape==(1,4,4,4)]
    assert carries and all(dtype==torch.float32 for dtype in carries)
    loss=out.float().square().mean()+sum(s.square().mean() for s in state)
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert float(x.grad[:,:512].float().norm())>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(p.dtype==torch.bfloat16 and p.grad.dtype==torch.bfloat16 for p in model.parameters())
    assert all(torch.equal(a,b) for a,b in zip(original,model.parameters()))
