"""Small causal transformer with one unchanged X31 replay lane."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..x31_tape_cell_fused.cell_fused import TapeCell, TapeHeadConfig
from ..x31_tape_cell_fused.train_gates import require_parity


def parity_gate():
    from pathlib import Path
    require_parity(Path(__file__).resolve().parents[1] / 'x31_tape_cell_fused/parity_part8.json')


def rope(x):
    t, d = x.shape[-2:]
    angles = torch.outer(torch.arange(t, device=x.device).float(),
                         10000 ** (-torch.arange(0, d, 2, device=x.device).float()/d))
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((a*angles.cos()-b*angles.sin(),
                        a*angles.sin()+b*angles.cos()), -1).flatten(-2)


def trace_metrics(state, trace, pressure):
    # X31 freed is one bit per slot/token; allocation precedes freeing.
    alloc = (trace.alloc_target >= 0).long()
    freed = (trace.freed != 0).sum(-1)
    live = (alloc-freed).cumsum(1)
    before_free = live+(trace.freed == 2).sum(-1)
    final = state.native.live.sum(-1)
    if not torch.equal(live[:, -1], final):
        raise AssertionError('trace liveness reconstruction failed')
    return dict(peak_live_per_head=before_free.amax((0, 1)).tolist(),
                final_live_per_head=final.amax(0).tolist(),
                pressure_mean=pressure.mean().item(), pressure_max=pressure.max().item(),
                pressure_fraction=(pressure != 0).float().mean().item(),
                routing=trace.counts())


class Block(nn.Module):
    def __init__(self, lane=None, width=1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(256), nn.LayerNorm(256)
        self.qkv, self.out = nn.Linear(256, 768), nn.Linear(256, 256)
        self.ff = nn.Sequential(nn.Linear(256, 1024), nn.GELU(), nn.Linear(1024, 256))
        self.lane = lane
        if lane == 'tape_lm':
            self.cell = TapeCell(256, TapeHeadConfig(n_heads=4, budget=2944))
            self.read_out = nn.Linear(512, 256)
        elif lane == 'base_lm':
            self.extra = nn.Sequential(nn.Linear(256, width), nn.GELU(), nn.Linear(width, 256))

    def forward(self, x, ids):
        b, t, _ = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b,t,3,4,64).permute(2,0,3,1,4)
        a = F.scaled_dot_product_attention(rope(q), rope(k), v, is_causal=True)
        x = x+self.out(a.transpose(1,2).reshape(b,t,256))
        h = self.norm2(x)
        x = x+self.ff(h)
        metrics = None
        if self.lane == 'tape_lm':
            read, state, trace, pressure = self.cell(h.float(), ids)
            metrics = trace_metrics(state, trace, pressure)
            x = x+self.read_out(read.flatten(2))
        elif self.lane == 'base_lm':
            x = x+self.extra(h)
        return x, metrics


class LM(nn.Module):
    def __init__(self, arm, vocab=32768):
        super().__init__()
        if arm not in ('tape_lm', 'base_lm'):
            raise ValueError(arm)
        # Count the exact replacement including frozen X31 parameters.
        if arm == 'tape_lm':
            parity_gate()
        lane_params = 256*256*2+256*512+256*12+256*2+512+12+4+512*256+256
        width = round((lane_params-256)/513)
        self.embed = nn.Embedding(vocab, 256)
        blocks=[]
        for i in range(6):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3401+i)
                blocks.append(Block(arm if i == 2 else None, width))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(256)
        self.head = nn.Linear(256, vocab, bias=False)

    def forward(self, ids):
        x, telemetry = self.embed(ids), []
        for block in self.blocks:
            x, metrics = block(x, ids)
            if metrics is not None:
                telemetry.append(metrics)
        return self.head(self.norm(x)), telemetry


def parameter_receipt(vocab=32768):
    result = {}
    for arm in ('tape_lm', 'base_lm'):
        model = LM(arm, vocab)
        result[arm] = dict(total=sum(p.numel() for p in model.parameters()),
                           trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    ratio = abs(result['tape_lm']['total']/result['base_lm']['total']-1)
    assert ratio <= .1
    result['relative_total_difference'] = ratio
    return result
