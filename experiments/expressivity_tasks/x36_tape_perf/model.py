"""X34-compatible LM: telemetry eval-only; optional sparse replay traffic."""
import torch
from ..x34_lm_pilot import model as original
from .cell_fused import TapeCell


class Block(original.Block):
    def forward(self,x,ids):
        b,t,_=x.shape
        q,k,v=self.qkv(self.norm1(x)).reshape(b,t,3,4,64).permute(2,0,3,1,4)
        a=original.F.scaled_dot_product_attention(original.rope(q),original.rope(k),v,is_causal=True)
        x=x+self.out(a.transpose(1,2).reshape(b,t,256))
        h=self.norm2(x)
        x=x+self.ff(h)
        metrics=None
        if self.lane=='tape_lm':
            read,state,trace,pressure=self.cell(h.float(),ids)
            if not self.training:
                metrics=original.trace_metrics(state,trace,pressure)
            x=x+self.read_out(read.flatten(2))
        elif self.lane=='base_lm':
            x=x+self.extra(h)
        return x,metrics


class LM(original.LM):
    def __init__(self,arm,vocab=32768,optimized_replay=True):
        super().__init__(arm,vocab)
        if arm=='tape_lm':
            old=self.blocks[2]
            with torch.random.fork_rng(devices=[]):
                new=Block(arm)
                if optimized_replay:
                    new.cell=TapeCell(256,old.cell.cfg)
            new.load_state_dict(old.state_dict())
            self.blocks[2]=new
