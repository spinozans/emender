"""Scale adapter for existing X39 semantic replay cell and X36 per-head scans.

No changes to earlier packages. This is unvalidated GPU integration, gated by
X40 and upstream parity before any use. Pure removes attention; hybrid retains
X34 attention. Both reuse the same semantic read/write wire.
"""
import math
import torch
from torch import nn
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig
from ..x36_tape_perf.cell_fused import TapeCell
from ..x39_memory_usage.semantic import SemanticCell, scan
from .model import Block, LM


class Lane(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        c=cfg.get('search_constants',{})
        self.input=nn.Linear(cfg['width'],256)
        self.cell=SemanticCell(TapeCell(256,TapeHeadConfig(n_heads=4,budget=2944)),50257)
        w=int(c.get('w_in_width',64))
        self.cell.W_in=nn.Sequential(nn.Linear(256,w,bias=False),nn.Linear(w,64,bias=False))
        self.output=nn.Linear(512,cfg['width'])
        self.configs=[]
        for h in range(4):
            horizon=c.get(f'horizon{h}',[1000,10000,100000,1000000][h])
            self.configs.append(TapeHeadConfig(n_heads=1,budget=int(c.get(f'budget{h}',[64,256,736,1472][h])),
                decay_rate=math.exp(-1/horizon),pop_threshold=c.get('pop_threshold',.1)))
        self.lambda_=c.get('gate_cost_lambda',.01)
        # Meta construction must not materialize a tensor on a real device.
        with torch.no_grad():
            self.cell.match_coef.fill_(c.get('match_coef',-1))
            self.cell.W_write.bias.fill_(c.get('write_gate_bias',0))
        self.gate_mean=None
        self.last_profile=None

    def forward(self,h,ids):
        # Certified recurrent kernels operate in fp32; outer dense layers autocast.
        x=self.input(h)
        with torch.autocast(device_type=x.device.type,enabled=False):
            x=x.float()
            z=self.cell.W_in(x)
            b,t=ids.shape
            values=self.cell.W_v(x).reshape(b,t,4,128)
            pop=self.cell.W_pop(x).sigmoid()
            append=self.cell.W_append(x)
            logits=self.cell.W_write(x)
            reads=[]; gates=[]; profiles=[]
            for head,policy in enumerate(self.configs):
                features=z+self.cell.features[head](ids)
                k=self.cell.geometry[head]['key'](features).detach().unsqueeze(2)
                q=self.cell.geometry[head]['query'](features).unsqueeze(2)
                sl=slice(head,head+1)
                (r,state,trace,pressure),_,sigmoid=scan(k,q,values[:,:,sl].contiguous(),
                    torch.ones((b,t,1),device=x.device),pop[:,:,sl].contiguous(),
                    append[:,:,sl].contiguous(),None,policy,
                    self.cell.append_err_coef[sl].contiguous(),logits[:,:,sl].contiguous(),
                    self.cell.match_coef[sl])
                reads.append(r);gates.append(sigmoid.mean())
                if not self.training:
                    from ..x34_lm_pilot.model import trace_metrics
                    profiles.append(trace_metrics(state,trace,pressure))
            self.gate_mean=torch.stack(gates).mean()
            self.last_profile=profiles
            read=torch.cat(reads,2).flatten(2)
        return self.output(read)


class PureBlock(nn.Module):
    def __init__(self,width,lane):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.ff=nn.Sequential(nn.Linear(width,6*width),nn.GELU(),nn.Linear(6*width,width))
        self.lane=lane

    def forward(self,x,ids):
        h=self.norm(x)
        x=x+self.ff(h)
        return x+self.lane(h,ids) if self.lane is not None else x


class Emender(LM):
    def __init__(self,cfg):
        if cfg['architecture'] not in ('pure','hybrid'):
            raise ValueError('explicit pure/hybrid architecture required')
        local=dict(cfg,lane_positions=[])
        super().__init__(local)
        # Avoid allocating a second large backbone: replace one block at a time.
        positions=cfg.get('lane_positions',[cfg['layers']//2])
        for i in range(cfg['layers']):
            lane=Lane(cfg) if i in positions else None
            if cfg['architecture']=='pure':
                self.blocks[i]=PureBlock(cfg['width'],lane)
            else:
                self.blocks[i].lane=lane
        self.extra_workspace_bytes = sum(cfg['microbatch']*max(cfg['context'],4096)*sum(max(p.budget,8192) for p in block.lane.configs)*8 + cfg['microbatch']*sum(max(p.budget,8192) for p in block.lane.configs)*256*16 for block in self.blocks if block.lane is not None)
        self.consumed_search_constants=set(cfg.get('search_constants',{}))

    def auxiliary_loss(self):
        return sum((block.lane.lambda_*block.lane.gate_mean for block in self.blocks
                    if block.lane is not None),self.embed.weight.new_zeros(()))

    def occupancy_profile(self,stream,lengths):
        import numpy as np
        self.eval()
        lanes=[b.lane for b in self.blocks if b.lane is not None]
        if not lanes:
            raise RuntimeError('no memory lanes configured')
        from dataclasses import replace
        original=[lane.configs for lane in lanes]
        for lane in lanes:
            lane.configs=[replace(policy,budget=max(8192,2*max(lengths))) for policy in lane.configs]
        curves=[];pressure=[]
        for length in lengths:
            # Same validation prefix at each length; state resets every forward.
            ids=torch.from_numpy(np.array(stream.data[:length],dtype=np.int64)).unsqueeze(0).to(self.embed.weight.device)
            if ids.shape[1]!=length:
                raise RuntimeError('validation split too short for length profile')
            self(ids)
            rows=[head for lane in lanes for head in lane.last_profile]
            curves.append([head['peak_live_per_head'][0] for head in rows])
            pressure.append([head['pressure_fraction'] for head in rows])
        for lane,policies in zip(lanes,original):
            lane.configs=policies
        slopes=np.polyfit(np.asarray(lengths),np.asarray(curves),1)[0]
        # Report capacity censoring as a failure, never as a low-slope reward.
        rich=all(pressure[i][j]==0 for i in range(len(lengths)) for j in range(len(curves[0])))
        return dict(occupancy_length_slope=float(max(slopes)),per_head_slopes=slopes.tolist(),
                    pressure=pressure,rich_capacity=rich,occupancy=curves,lengths=lengths)


def factory(cfg):
    return Emender(cfg)
