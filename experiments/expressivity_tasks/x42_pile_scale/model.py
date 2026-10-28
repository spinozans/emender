"""Scaled X34 pre-norm/RoPE/SDPA skeleton; architecture supplied after X40."""
import importlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from ..x34_lm_pilot.model import rope

class Block(nn.Module):
    def __init__(self, width, heads, lane=None):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.qkv = nn.Linear(width,3*width)
        self.out = nn.Linear(width,width)
        self.ff = nn.Sequential(nn.Linear(width,4*width),nn.GELU(),nn.Linear(4*width,width))
        self.lane = lane

    def forward(self, x, ids):
        b,t,d = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4)
        x = x+self.out(F.scaled_dot_product_attention(rope(q),rope(k),v,is_causal=True).transpose(1,2).reshape(b,t,d))
        h = self.norm2(x)
        x = x+self.ff(h)
        if self.lane is not None:
            x = x+self.lane(h,ids)
        return x

class LM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg['width']
        self.embed = nn.Embedding(50257,d)
        self.blocks = nn.ModuleList()
        for i in range(cfg['layers']):
            lane = None
            if i in cfg.get('lane_positions',[]):
                module, name = cfg['lane_factory'].split(':')
                lane = getattr(importlib.import_module(module),name)(cfg)
            self.blocks.append(Block(d,cfg['heads'],lane))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,50257,bias=False)
        self.checkpointing = cfg['activation_checkpointing']

    def forward(self, ids):
        x = self.embed(ids)
        for block in self.blocks:
            x = checkpoint(block,x,ids,use_reentrant=False) if self.checkpointing and self.training else block(x,ids)
        return self.head(self.norm(x))


def make_model(cfg):
    if cfg.get("model_factory"):
        module, name = cfg["model_factory"].split(":")
        return getattr(importlib.import_module(module), name)(cfg)
    if cfg.get("architecture") == "pure":
        raise ValueError("pure verdict requires a pure model_factory; attention backbone prohibited")
    return LM(cfg)


def memory(cfg):
    # Meta allocation counts actual tensors without allocating 1.3B host weights.
    with torch.device('meta'):
        model = make_model(cfg)
    n = sum(p.numel() for p in model.parameters())
    # fp32 master + fp32 gradient + fp32 moments + autocast bf16 copy = 18 bytes.
    state = n*18
    # Conservative checkpointed block workspace plus fp32 logits/CE, CUDA reserve.
    b,t,d = cfg['microbatch'],cfg['context'],cfg['width']
    activations = b*t*d*(cfg['layers']*(4 if cfg["activation_checkpointing"] else 64)+128)*4 + b*t*50257*12
    activations += cfg.get("extra_workspace_bytes", 0) + getattr(model,"extra_workspace_bytes",0)
    reserve = 6*(1 << 30)
    total = state+activations+reserve
    assert total < 44*(1 << 30), 'single 48GB card headroom gate failed'
    return dict(parameters=n, bf16_weights=n*2, fp32_master=n*4, fp32_gradients=n*4,
                adam_moments=n*8, activation_workspace=activations, cuda_reserve=reserve,
                total_bytes=total, headroom_48GiB=48*(1 << 30)-total,
                analytical_only=True, checkpointing=cfg['activation_checkpointing'])
