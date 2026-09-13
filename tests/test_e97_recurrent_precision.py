import json
from types import SimpleNamespace
import pytest
import torch
from ndm.recurrent_precision import configure_recurrent_precision,restore_checkpoint_precision,validate_state_precision


def test_policy_is_atomic_and_rejects_unknown_or_mixed_modes():
    model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.Linear(2,2))
    for m in model:m.recurrent_state_precision='legacy'
    saved=[p.detach().clone() for p in model.parameters()]
    assert configure_recurrent_precision(model,'fp32')=='fp32'
    assert configure_recurrent_precision(model)=='fp32'
    with pytest.raises(ValueError):configure_recurrent_precision(model,'fp16')
    assert all(m.recurrent_state_precision=='fp32' for m in model)
    assert all(torch.equal(a,b) for a,b in zip(saved,model.parameters()))
    model[0].recurrent_state_precision='legacy'
    with pytest.raises(ValueError,match='mixed'):configure_recurrent_precision(model,'fp32')
    with pytest.raises(ValueError):configure_recurrent_precision(torch.nn.Linear(2,2),'fp32')


def test_legacy_fp32_carry_does_not_imply_fp32_checkpoints():
    from ndm.recurrent_precision import recurrent_checkpoint_dtype
    state=torch.zeros(1,dtype=torch.float32)
    assert recurrent_checkpoint_dtype('legacy',state,torch.bfloat16)==torch.bfloat16
    assert recurrent_checkpoint_dtype('fp32',state,torch.bfloat16)==torch.float32
    with pytest.raises(ValueError,match='FP32 initial state'):
        recurrent_checkpoint_dtype('fp32',state.bfloat16(),torch.bfloat16)


def test_e97_facade_forwards_the_same_policy(monkeypatch):
    import ndm.triton.e88_triton_optimized as engine
    from ndm.triton.e97_sequential import e97_split_edit_triton_apply
    captured={}
    def fake(*args,**kwargs):captured.update(kwargs);return None,None
    monkeypatch.setattr(engine,'e88_triton_optimized_apply',fake)
    gate=torch.zeros(1)
    e97_split_edit_triton_apply(False,None,None,None,None,erase_gate=gate,value_write_gate=gate,recurrent_state_precision='fp32')
    assert captured['recurrent_state_precision']=='fp32'
    e97_split_edit_triton_apply(False,None,None,None,None,erase_gate=gate,value_write_gate=gate)
    assert captured['recurrent_state_precision']=='legacy'


def test_sft_policy_inherits_and_records_the_single_switch():
    from scripts.train_e97_4b_pi_sft import configure_precision
    model=torch.nn.Linear(2,2).bfloat16();model.recurrent_state_precision='legacy';model.loss_chunk_size=128
    args=SimpleNamespace(recurrent_state_precision='fp32',gradient_checkpoint_group_size=3,
        mlp_checkpoint_chunk_size=4096,lr=1e-5,weight_decay=.01,warmup_steps=0)
    assert configure_precision(model,args)['recurrent_state_precision']=='fp32'
    args.recurrent_state_precision=None
    assert configure_precision(model,args)['recurrent_state_precision']=='fp32'
    args.recurrent_state_precision='legacy'
    assert 'recurrent_state_precision' not in configure_precision(model,args)
    assert model.weight.dtype==torch.bfloat16


def test_precision_metadata_survives_external_architecture_config():
    base=dict(level='E97',layer_kwargs=json.dumps(dict(use_silu=True)))
    checkpoint=dict(sft_precision=dict(recurrent_state_precision='fp32'))
    restored=restore_checkpoint_precision(base,checkpoint)
    assert restored['layer_kwargs']==dict(use_silu=True,recurrent_state_precision='fp32')
    assert isinstance(base['layer_kwargs'],str)
    with pytest.raises(ValueError,match='mismatch'):
        restore_checkpoint_precision(dict(layer_kwargs=dict(recurrent_state_precision='legacy')),checkpoint)
    assert restore_checkpoint_precision(base,{})==base
    with pytest.raises(ValueError):restore_checkpoint_precision(base,dict(sft_precision=dict(recurrent_state_precision='fp16')))


def test_checkpoint_loader_retains_fp32_policy_with_old_args(tmp_path):
    from ndm.e97 import build_e97_model,load_e97_checkpoint
    config=dict(level='E97',dim=8,depth=1,n_heads=2,n_state=4,expansion=1.,use_gate=1,gate_activation='silu',use_conv=0,mlp_ratio=0.,projection_chunk_size=16)
    model=build_e97_model(config,vocab_size=32,use_triton=False)
    checkpoint=tmp_path/'checkpoint.pt';args=tmp_path/'args.json';args.write_text(json.dumps(config))
    torch.save(dict(model_state_dict=model.state_dict(),sft_precision=dict(recurrent_state_precision='fp32')),checkpoint)
    loaded=load_e97_checkpoint(checkpoint,args_json=args,device='cpu',weight_mode='saved',use_triton=False)
    assert configure_recurrent_precision(loaded.model)=='fp32'
    assert loaded.config['layer_kwargs']['recurrent_state_precision']=='fp32'
    assert list(model.state_dict())==list(loaded.model.state_dict())
    with pytest.raises(NotImplementedError,match='CUDA/Triton'):loaded.model(torch.tensor([[1,2]]))


@pytest.mark.parametrize('state_dtype',[torch.bfloat16,torch.float32])
def test_cpu_reference_distinguishes_state_and_projection_storage(state_dtype):
    from ndm.triton.e88_triton_forward import e88_torch_reference
    k=torch.full((16,1,1,4),.125,dtype=torch.bfloat16);s=torch.zeros(1,1,4,4,dtype=state_dtype)
    mode='fp32' if state_dtype==torch.float32 else 'legacy'
    out,final,checkpoints=e88_torch_reference(s,k,k,k,torch.full((16,1,1),.9,dtype=torch.bfloat16),recurrent_state_precision=mode)
    assert out.dtype==torch.bfloat16 and final.dtype==checkpoints.dtype==state_dtype


def _inputs(device,T=64,N=64):
    torch.manual_seed(974223)
    def leaf(shape,gate=False):
        x=torch.randn(shape,device=device)*.1
        return (x.sigmoid() if gate else x).to(torch.bfloat16).detach().requires_grad_()
    shape=(T,1,2,N)
    s=(torch.randn((1,2,N,N),device=device)*.01).requires_grad_()
    k,v,q=leaf(shape),leaf(shape),leaf(shape)
    decay=torch.full((T,1,2),.9,device=device,dtype=torch.bfloat16,requires_grad=True)
    g,e,w=leaf(shape),leaf(shape,True),leaf(shape,True)
    return s,k,v,q,decay,g,e,w


def _oracle(s,k,v,q,d,g,e,w,reset,valid):
    import torch.nn.functional as F
    S=s;rows=[]
    for t in range(len(k)):
        S=torch.where(reset[t,:,None,None,None],torch.zeros_like(S),S)
        K=F.silu(k[t].float());Q=F.silu(q[t].float());V=F.silu(v[t].float())
        K=K/(K.norm(dim=-1,keepdim=True)+1e-6);Q=Q/(Q.norm(dim=-1,keepdim=True)+1e-6)
        read=K*e[t].float();write=V*w[t].float()
        delta=write-(S*read[...,None]).sum(-2)
        pre=d[t].float()[...,None,None]*S+K[...,None]*delta[...,None,:]
        next_state=2*torch.sigmoid(2*pre)-1
        S=torch.where(valid[t,:,None,None,None],next_state,S)
        out=(S*Q[...,None]).sum(-2)*F.silu(g[t].float())
        rows.append(torch.where(valid[t,:,None,None],out,torch.zeros_like(out)).to(k.dtype))
    return torch.stack(rows),S


def _relative(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='leased CUDA required')
def test_cuda_fp32_state_forward_backward_and_connected_chunk_handoffs(monkeypatch,record_property):
    import os
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    from ndm.triton.e88_triton_backward import e88_triton
    inputs=_inputs('cuda');s,k,v,q,d,g,e,w=inputs;T=len(k)
    reset=torch.zeros(T,1,dtype=torch.bool,device='cuda');reset[32]=True
    valid=torch.ones_like(reset);valid[-3:]=False
    saved=[];allocations=[];original=torch.empty
    def alloc(*args,**kwargs):
        result=original(*args,**kwargs);allocations.append((tuple(result.shape),result.dtype));return result
    def pack(tensor):saved.append((tuple(tensor.shape),tensor.dtype));return tensor
    monkeypatch.setattr(torch,'empty',alloc)
    def forward(state,lo,hi):
        return e88_triton(state,k[lo:hi],v[lo:hi],q[lo:hi],d[lo:hi],g[lo:hi],
            normalize_kq=True,apply_silu_qkv=True,erase_gate=e[lo:hi],value_write_gate=w[lo:hi],
            reset_before=reset[lo:hi],valid_mask=valid[lo:hi],recurrent_state_precision='fp32')
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda tensor:tensor):
        out,final=forward(s,0,T)
    assert out.dtype==torch.bfloat16 and final.dtype==torch.float32
    assert ((T//16+1,1,2,64,64),torch.float32) in saved
    loss=out.float().square().mean()+final.square().mean();loss.backward()
    grads=[x.grad.clone() for x in inputs]
    assert grads[0].dtype==torch.float32 and all(x.dtype==torch.bfloat16 for x in grads[1:])
    assert ((2*17*64*64,),torch.float32) in allocations # per-program backward state replay scratch
    reference=[x.detach().clone().requires_grad_() for x in inputs]
    ro,rs=_oracle(*reference,reset,valid)
    (ro.float().square().mean()+rs.square().mean()).backward()
    state_error=_relative(final,rs);state_gradient_error=_relative(grads[0],reference[0].grad)
    projection_gradient_error=max(_relative(a,b.grad) for a,b in zip(grads[1:],reference[1:]))
    record_property('state_relative_l2',state_error);record_property('state_gradient_relative_l2',state_gradient_error)
    record_property('projection_gradient_relative_l2',projection_gradient_error)
    assert state_error<=1e-4 and state_gradient_error<=.005 and projection_gradient_error<=.02
    assert all(torch.isfinite(x).all() for x in grads)
    for x in inputs:x.grad=None
    state=s;parts=[]
    for lo in range(0,T,16):
        part,state=forward(state,lo,lo+16);parts.append(part)
    joined=torch.cat(parts)
    (joined.float().square().mean()+state.square().mean()).backward()
    assert _relative(state,final)<=1e-4 and _relative(joined,out)<=.001
    assert max(_relative(x.grad,gold) for x,gold in zip(inputs,grads))<=.005
    assert s.grad is not None and float(s.grad.norm())>0 # no TBPTT detach at internal handoffs


@pytest.mark.skipif(not torch.cuda.is_available(),reason='leased CUDA required')
@pytest.mark.parametrize('state_dtype',[torch.bfloat16,torch.float32])
def test_cuda_state_precision_storage_legacy_control(state_dtype):
    from ndm.triton.e88_triton_backward import e88_triton
    s,k,v,q,d,g,e,w=_inputs('cuda',T=16,N=8);s=s.detach().to(state_dtype).requires_grad_();saved=[]
    def pack(x):saved.append((x.ndim,x.dtype));return x
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda x:x):
        out,state=e88_triton(s,k,v,q,d,g,erase_gate=e,value_write_gate=w)
    (out.float().sum()+state.float().sum()).backward()
    assert state.dtype==s.grad.dtype==state_dtype
    # Legacy retains BF16 checkpoints even with FP32 carried inference state.
    assert (5,torch.bfloat16) in saved
