"""X24 fused split-edit softplus against the existing E88 model oracle."""
import json
import os
from pathlib import Path

import pytest
import torch

from ndm.models.e88_fla_hybrid import E88FLAHybrid
from ndm.triton.e97_sequential import e97_split_edit_triton_apply

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason='leased CUDA required')


def relative_error(a, b):
    return (a.float() - b.float()).abs().max().item() / max(b.float().abs().max().item(), 1e-8)


@CUDA
@pytest.mark.parametrize('precision', ['legacy', 'fp32'])
@pytest.mark.parametrize('gated', [False, True])
def test_e97_softplus_recurrence_forward_backward_masks_and_checkpoints(precision, gated):
    torch.manual_seed(24001)
    B, T, H, N, V = 2, 48, 2, 8, 8
    dtype = torch.bfloat16 if precision == 'legacy' else torch.float32
    shapes = [(B,T,H,N), (B,T,H,V), (B,T,H,N), (B,T,H),
              (B,T,H,N), (B,T,H,V), (B,T,H,V), (B,H,N,V)]
    inputs = [torch.randn(s, device='cuda') * .2 for s in shapes]
    inputs[3] = torch.sigmoid(inputs[3])
    inputs[4] = torch.sigmoid(inputs[4])
    inputs[5] = torch.sigmoid(inputs[5])
    tri = [v.to(torch.float32 if i == 7 else dtype).detach().requires_grad_()
           for i, v in enumerate(inputs)]
    ref = [v.detach().float().requires_grad_() for v in tri]
    k,v,q,d,e,w,g,s = tri
    reset = torch.zeros(B,T,device='cuda',dtype=torch.bool); reset[:,19] = True
    valid = torch.ones_like(reset); valid[0,31:35] = False
    saved = []
    def pack(t):
        if t.ndim == 5:
            saved.append(t.dtype)
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        state, out = e97_split_edit_triton_apply(True,k,v,q,d,g if gated else None,s,
            apply_gate=gated, erase_gate=e, value_write_gate=w, softplus_state=True,
            reset_before=reset, valid_mask=valid, recurrent_state_precision=precision)
    assert state.dtype == torch.float32
    assert saved and all(t == dtype for t in saved)
    k,v,q,d,e,w,g,s = ref
    oracle = E88FLAHybrid(dim=16,n_heads=2,n_state=8,state_activation='softplus')
    outputs=[]
    for t in range(T):
        s = torch.where(reset[:,t,None,None,None], torch.zeros_like(s), s)
        read = k[:,t] * e[:,t]
        delta = v[:,t] * w[:,t] - torch.einsum('bhnv,bhn->bhv',s,read)
        new = oracle._apply_state_activation(d[:,t,:,None,None] * s + k[:,t,:,:,None] * delta[:,:,None,:])
        s = torch.where(valid[:,t,None,None,None],new,s)
        o = torch.einsum('bhnv,bhn->bhv',s,q[:,t])
        if gated:
            o = o * torch.nn.functional.silu(g[:,t])
        outputs.append(torch.where(valid[:,t,None,None],o,torch.zeros_like(o)))
    expected = torch.stack(outputs,1)
    assert relative_error(state,s) < 1e-5
    assert relative_error(out,expected) < (5e-3 if precision == 'legacy' else 1e-5)
    (out.float().square().sum() + state.square().sum()).backward()
    (expected.square().sum() + s.square().sum()).backward()
    for i,(a,b) in enumerate(zip(tri,ref)):
        if i == 6 and not gated:
            continue
        assert torch.isfinite(a.grad).all()
        assert relative_error(a.grad,b.grad) < (.06 if precision == 'legacy' else 2e-4), i


@CUDA
@pytest.mark.parametrize('training,length', [(True,32),(False,11)])
def test_e97_softplus_stable_extremes_and_unaligned_eval(training,length):
    k = torch.ones(1,length,1,1,device='cuda')
    v = torch.full_like(k,100.)
    d = torch.ones(1,length,1,device='cuda')
    zero = torch.zeros_like(k)
    state,out = e97_split_edit_triton_apply(training,k,v,k,d,apply_gate=False,
        erase_gate=zero,value_write_gate=k,softplus_state=True,
        recurrent_state_precision='fp32')
    assert torch.isfinite(out).all() and state.item() == pytest.approx(100.*length)
    state,out = e97_split_edit_triton_apply(training,k,-v,k,d,apply_gate=False,
        erase_gate=zero,value_write_gate=k,softplus_state=True,
        recurrent_state_precision='fp32')
    assert torch.isfinite(out).all() and state.item() < 1e-30


@CUDA
@pytest.mark.parametrize('projection_chunk_size', [0, 16])
def test_e97_softplus_full_layer_matches_existing_eager_fp32_oracle(monkeypatch, projection_chunk_size):
    """Zero-update launch gate includes projection chunks, outputs, state, gradients."""
    from ndm.models.e97 import E97SplitEditLayer
    from ndm.triton import e97_sequential
    kernel_calls = []
    original = e97_sequential.e97_split_edit_triton_apply
    def observed(*args, **kwargs):
        assert kwargs['softplus_state'] is True
        kernel_calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(e97_sequential, 'e97_split_edit_triton_apply', observed)
    torch.manual_seed(24002)
    torch.backends.cuda.matmul.allow_tf32 = False
    kwargs = dict(dim=32,n_heads=4,n_state=8,expansion=1.,use_conv=False,
                  state_activation='softplus',projection_chunk_size=projection_chunk_size,use_gate=True)
    fused = E97SplitEditLayer(**kwargs,use_triton=True).cuda().train()
    eager = E97SplitEditLayer(**kwargs,use_triton=False).cuda().train()
    eager.load_state_dict(fused.state_dict()); eager.disable_autocast = True
    fused.fused_inference = True
    x = torch.randn(2,48,32,device='cuda',dtype=torch.bfloat16).requires_grad_()
    y = x.detach().float().requires_grad_()
    with torch.autocast('cuda',dtype=torch.bfloat16):
        out,state = fused(x)
    assert len(kernel_calls) == (3 if projection_chunk_size else 1) and all(kernel_calls)
    ref,ref_state = eager(y)
    state,ref_state = torch.stack(state),torch.stack(ref_state)
    metrics = {'output_relative_error':relative_error(out,ref),
               'state_relative_error':relative_error(state,ref_state)}
    (out.float().square().mean()+state.square().mean()).backward()
    (ref.square().mean()+ref_state.square().mean()).backward()
    metrics['input_gradient_relative_error'] = relative_error(x.grad,y.grad)
    metrics['parameter_gradient_relative_error'] = max(relative_error(p.grad,r.grad)
        for p,r in zip(fused.parameters(),eager.parameters()) if p.grad is not None)
    assert state.dtype == torch.bfloat16  # legacy training chunk boundaries
    assert metrics['output_relative_error'] < .03
    assert metrics['state_relative_error'] < .03
    assert metrics['input_gradient_relative_error'] < .08
    assert metrics['parameter_gradient_relative_error'] < .08
    fused.eval(); eager.eval()
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
        evaluation,_ = fused(x.detach())
    with torch.no_grad():
        expected,_ = eager(y.detach())
    assert False in kernel_calls  # evaluation also used the fused facade
    metrics['eval_output_relative_error'] = relative_error(evaluation,expected)
    assert metrics['eval_output_relative_error'] < .03
    receipt = os.environ.get('X24_KERNEL_RECEIPT')
    if receipt:
        Path(receipt).parent.mkdir(parents=True,exist_ok=True)
        cases = {} if not projection_chunk_size else json.loads(Path(receipt).read_text())['cases']
        cases[str(projection_chunk_size)] = {'metrics':metrics,'fused_facade_calls':len(kernel_calls)}
        Path(receipt).write_text(json.dumps({'status':'passed','optimizer_updates':0,
            'seed':24002,'shape':[2,48,32],'oracle':'existing E88 eager FP32 softplus layer',
            'policy':'legacy: FP32 in-kernel live / bf16 training boundaries and replay',
            'tolerances':{'output_state':.03,'gradients':.08},'cases':cases,
            'metrics':{key:max(case['metrics'][key] for case in cases.values()) for key in metrics}},
            indent=2)+'\n')
