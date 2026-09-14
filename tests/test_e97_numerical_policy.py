import json
import os
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from ndm.numerical_policy import POLICY,RecomputedFP32Linear,configure_numerical_policy,restore_numerical_policy


def exercise(device,readout,bias,in_features=7,out_features=5):
    torch.manual_seed(27183)
    layer=RecomputedFP32Linear(in_features,out_features,bias=bias,device=device,dtype=torch.bfloat16)
    layer.fp32_readout=readout
    x=torch.randn(1,33,in_features,device=device,dtype=torch.bfloat16,requires_grad=True)
    originals=[x,layer.weight]+([layer.bias] if bias else [])
    copies=[p.detach().clone().requires_grad_() for p in originals]
    saved=[]
    def pack(t):saved.append((t.dtype,t.data_ptr()));return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        with torch.autocast(device_type=x.device.type,dtype=torch.bfloat16):out=layer(x)
    assert len(saved)==len(originals) and all(d==torch.bfloat16 for d,_ in saved)
    assert {p for _,p in saved}=={p.data_ptr() for p in originals}
    with torch.autocast(device_type=x.device.type,enabled=False):
        ref=F.linear(copies[0].float(),copies[1].float(),copies[2].float() if bias else None)
        if not readout:ref=ref.bfloat16()
    assert out.dtype==(torch.float32 if readout else torch.bfloat16)
    gradient=torch.randn_like(out)
    actual=torch.autograd.grad(out,originals,gradient)
    expected=torch.autograd.grad(ref,copies,gradient)
    assert torch.equal(out,ref)
    errors=[]
    for a,b,p,q in zip(actual,expected,originals,copies):
        assert a.dtype==torch.bfloat16 and torch.isfinite(a).all()
        errors.append(float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-30)))
        assert torch.equal(p,q)
    assert max(errors)<=.005
    return max(errors)


@pytest.mark.parametrize('readout',[False,True])
@pytest.mark.parametrize('bias',[False,True])
def test_recomputed_linear_saved_storage_and_gradients(readout,bias):exercise('cpu',readout,bias)


def test_policy_preserves_module_parameters_keys_and_inherits():
    model=torch.nn.Module();model.body=torch.nn.Linear(7,7).bfloat16();model.lm_head=torch.nn.Linear(7,5).bfloat16()
    model.recurrent_state_precision='legacy'
    original_modules=list(model.modules());ids={k:id(p) for k,p in model.named_parameters()};keys=list(model.state_dict())
    assert configure_numerical_policy(model) is None and type(model.body)==torch.nn.Linear
    assert configure_numerical_policy(model,POLICY)==POLICY
    assert configure_numerical_policy(model)==POLICY and model.recurrent_state_precision=='fp32'
    assert list(model.modules())==original_modules and list(model.state_dict())==keys
    assert {k:id(p) for k,p in model.named_parameters()}==ids
    x=torch.randn(1,3,7,dtype=torch.bfloat16)
    assert model.body(x).dtype==torch.bfloat16 and model.lm_head(x).dtype==torch.float32
    from scripts.train_e97_4b_pi_sft import configure_precision
    args=SimpleNamespace(gradient_checkpoint_group_size=3,mlp_checkpoint_chunk_size=4096,lr=1e-5,weight_decay=.01,warmup_steps=0)
    model.loss_chunk_size=128
    policy=configure_precision(model,args)
    assert policy['numerical_policy']==POLICY and policy['recurrent_state_precision']=='fp32'
    args.recurrent_state_precision='legacy'
    with pytest.raises(ValueError):configure_precision(model,args)


def test_checkpoint_roundtrip_and_conflict(tmp_path):
    from ndm.e97 import build_e97_model,load_e97_checkpoint
    config=dict(level='E97',dim=8,depth=1,n_heads=2,n_state=4,expansion=1.,use_gate=1,gate_activation='silu',use_conv=0,mlp_ratio=0.)
    model=build_e97_model(dict(config,numerical_policy=POLICY),vocab_size=32,use_triton=False).bfloat16()
    p=tmp_path/'checkpoint.pt';a=tmp_path/'args.json';a.write_text(json.dumps(config))
    torch.save(dict(model_state_dict=model.state_dict(),sft_precision=dict(numerical_policy=POLICY,recurrent_state_precision='fp32')),p)
    loaded=load_e97_checkpoint(p,args_json=a,device='cpu',dtype=torch.bfloat16,use_triton=False,weight_mode='saved')
    assert loaded.model.numerical_policy==POLICY and isinstance(loaded.model.lm_head,RecomputedFP32Linear)
    assert all(torch.equal(v,loaded.model.state_dict()[k]) for k,v in model.state_dict().items())
    with pytest.raises(ValueError):restore_numerical_policy({},dict(sft_precision=dict(numerical_policy=POLICY)))
    with pytest.raises(ValueError):build_e97_model(dict(config,numerical_policy=POLICY,layer_kwargs=dict(recurrent_state_precision='legacy')),vocab_size=32)
    with pytest.raises(ValueError):configure_numerical_policy(model,'unknown')


def test_scratch_dtype_and_higher_order_guards(monkeypatch):
    import ndm.numerical_policy as policy
    layer=RecomputedFP32Linear(7,5).bfloat16();x=torch.randn(2,7,dtype=torch.bfloat16,requires_grad=True)
    with pytest.raises(ValueError):layer(x.float())
    with pytest.raises(NotImplementedError):torch.autograd.grad(layer(x).float().sum(),x,create_graph=True)
    monkeypatch.setattr(policy,'MAX_WEIGHT_BYTES',4)
    with pytest.raises(ValueError,match='scratch'):layer(x)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='leased CUDA required')
@pytest.mark.parametrize('readout,out_features',[(False,11520),(True,50281)])
def test_cuda_production_linear_arithmetic_and_backward(readout,out_features,record_property):
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')))
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    error=exercise('cuda',readout,False,3840,out_features)
    record_property('gradient_relative_l2',error)
    record_property('largest_fp32_weight_bytes',3840*out_features*4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='leased CUDA required')
def test_cuda_checkpointed_model_forward_backward_matches_autograd_reference(record_property):
    import copy
    from ndm.e97 import build_e97_model
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK','0')));torch.manual_seed(71391)
    model=build_e97_model(dict(level='E97',dim=128,depth=2,n_heads=2,n_state=64,expansion=1.,
        use_gate=1,gate_activation='silu',use_conv=0,mlp_ratio=2.5,projection_chunk_size=512,checkpoint_interval=16,
        numerical_policy=POLICY),vocab_size=256,use_triton=True).cuda().bfloat16().train()
    model.gradient_checkpointing=True;model.gradient_checkpoint_group_size=2
    model.loss_chunk_size=128;model.loss_logits_fp32=True;model.checkpoint_loss_chunks=True
    for layer in model.layers:layer.mlp.checkpoint_chunk_size=4096
    reference=copy.deepcopy(model)
    class ReferenceLinear(torch.nn.Linear):
        def forward(self,x):
            with torch.autocast(device_type='cuda',enabled=False):
                value=F.linear(x.float(),self.weight.float(),None if self.bias is None else self.bias.float())
            return value if self.fp32_readout else value.bfloat16()
    for layer in reference.modules():
        if isinstance(layer,RecomputedFP32Linear):layer.__class__=ReferenceLinear
    tokens=torch.randint(0,256,(1,1041),device='cuda');valid=torch.ones_like(tokens,dtype=torch.bool);valid[:,-3:]=False
    reset=torch.zeros_like(valid);reset[:,0]=True;reset[:,512]=True
    mask=valid[:,:-1]&valid[:,1:]&~reset[:,1:]
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    seen=[];handles=[]
    def observe(module,inputs,output):
        assert output.dtype==(torch.float32 if module.fp32_readout else torch.bfloat16)
        seen.append(output.dtype)
        return None
    for module in model.modules():
        if isinstance(module,RecomputedFP32Linear):handles.append(module.register_forward_hook(observe))
    try:
        losses=[]
        for m in (model,reference):
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss=m(tokens,return_loss=True,loss_mask=mask,reset_before=reset,valid_mask=valid,loss_reduction='sum')/mask.sum()
            losses.append(float(loss));loss.backward()
    finally:
        for handle in handles:handle.remove()
    assert seen and torch.float32 in seen and torch.bfloat16 in seen
    errors=[]
    for p,q in zip(model.parameters(),reference.parameters()):
        if q.grad is None:assert p.grad is None;continue
        assert p.grad is not None and p.grad.dtype==torch.bfloat16 and torch.isfinite(p.grad).all()
        errors.append(float((p.grad.float()-q.grad.float()).norm()/q.grad.float().norm().clamp_min(1e-30)))
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    record_property('loss_delta',abs(losses[0]-losses[1]));record_property('gradient_relative_l2_max',max(errors))
    assert abs(losses[0]-losses[1])<=1e-4 and max(errors)<=.02
