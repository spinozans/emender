"""Opt-in composite arithmetic candidate; persistent weights/hidden stores stay BF16."""
import torch
import torch.nn.functional as F

POLICY='fp32-linear-v1'
UNIFORM_POLICY='fp32-linear-v2'
POLICIES=(POLICY,UNIFORM_POLICY)
MAX_WEIGHT_BYTES=1024**3


def validate_numerical_policy(mode):
    if mode is not None and mode not in POLICIES:raise ValueError('unsupported numerical policy')
    return mode


def _guard(x,w,b):
    if x.dtype!=torch.bfloat16 or w.dtype!=torch.bfloat16 or (b is not None and b.dtype!=torch.bfloat16):
        raise ValueError('fp32-linear-v1 requires BF16 stored inputs and parameters')
    if w.numel()*4>MAX_WEIGHT_BYTES:raise ValueError('FP32 linear weight scratch exceeds 1GiB')
    if x.device.type=='cuda' and (torch.backends.cuda.matmul.allow_tf32 or torch.get_float32_matmul_precision()!='highest'):
        raise ValueError('fp32-linear-v1 requires highest FP32 matmul precision, without TF32')


class _RecomputedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,w,b,readout):
        _guard(x,w,b)
        ctx.save_for_backward(x,w,b)
        with torch.autocast(device_type=x.device.type,enabled=False):
            out=F.linear(x.float(),w.float(),None if b is None else b.float())
        return out if readout else out.to(x.dtype)

    @staticmethod
    def backward(ctx,gradient):
        if torch.is_grad_enabled():raise NotImplementedError('higher-order linear derivatives are not qualified')
        x,w,b=ctx.saved_tensors
        _guard(x,w,b)
        dx=dw=db=None
        with torch.autocast(device_type=x.device.type,enabled=False):
            g=gradient.float()
            if ctx.needs_input_grad[0]:
                dx=F.linear(g,w.T.float()).to(x.dtype)
            if ctx.needs_input_grad[1]:
                dw=(g.reshape(-1,w.shape[0]).T @ x.reshape(-1,w.shape[1]).float()).to(w.dtype)
            if b is not None and ctx.needs_input_grad[2]:
                db=g.reshape(-1,w.shape[0]).sum(0).to(b.dtype)
        return dx,dw,db,None


class RecomputedFP32Linear(torch.nn.Linear):
    """No new parameters/buffers; casts are transient and recreated for backward."""
    def forward(self,x):
        return _RecomputedLinear.apply(x,self.weight,self.bias,getattr(self,'fp32_readout',False))


def configure_numerical_policy(model,mode=None):
    from ndm.recurrent_precision import configure_recurrent_precision
    inherited=validate_numerical_policy(getattr(model,'numerical_policy',None))
    mode=validate_numerical_policy(mode if mode is not None else inherited)
    if mode is None:return None
    modules=[m for m in model.modules() if isinstance(m,torch.nn.Linear)]
    head=getattr(model,'lm_head',None)
    if head not in modules or any(type(m) not in (torch.nn.Linear,RecomputedFP32Linear) for m in modules):
        raise ValueError('numerical policy requires a plain Linear readout and projections')
    configure_recurrent_precision(model,'fp32')
    for module in model.modules():
        if hasattr(module,'recurrent_state_precision'):
            module.uniform_recurrent_workspace=mode==UNIFORM_POLICY
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    # Preserve module and Parameter identities, hooks, names, aliases and state-dict
    # keys. No output-replacement hook or duplicate FP32 Parameter is installed.
    for module in modules:
        module.__class__=RecomputedFP32Linear
        module.fp32_readout=module is head
    model.numerical_policy=mode
    return mode


def restore_numerical_policy(config,checkpoint):
    saved=validate_numerical_policy(checkpoint.get('sft_precision',{}).get('numerical_policy'))
    explicit=validate_numerical_policy(config.get('numerical_policy'))
    if saved is not None and explicit is not None and saved!=explicit:
        raise ValueError('checkpoint/config numerical policy conflict')
    chosen=saved if saved is not None else explicit
    if saved is not None and checkpoint['sft_precision'].get('recurrent_state_precision')!='fp32':
        raise ValueError('saved numerical policy requires FP32 recurrent state')
    if chosen is not None:
        from ndm.recurrent_precision import validate_state_precision
        import json
        layers=config.get('layer_kwargs') or {}
        if isinstance(layers,str):layers=json.loads(layers)
        state=layers.get('recurrent_state_precision')
        if state is not None and validate_state_precision(state)!='fp32':
            raise ValueError('numerical policy/config recurrent precision conflict')
        return dict(config,numerical_policy=chosen)
    return config
