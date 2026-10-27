"""Head-specific frozen geometry over the read-only X36 replay lane."""
from dataclasses import asdict
import torch
from torch import nn
from ..x31_tape_cell_fused.cell_fused import TapeHeadConfig
from ..x34_lm_pilot.model import LM as Skeleton, Block as OriginalBlock, rope, trace_metrics
from ..x30_tape_cell.routing_trace import CASE_NAMES
from ..x36_tape_perf.cell_fused import TapeCell as IntegratedCell, tape_scan

ARMS = ('b', 'c', 'd')
POLICIES = [dict(budget=b, decay_rate=r, pop_threshold=p) for b,r,p in
            zip((64,256,736,1472),(.999,.9999,.99999,.999999),(.01,.1,.5,.9))]


def metrics(state, trace, pressure):
    out = trace_metrics(state, trace, pressure)
    out['routing_per_head'] = [{name: int((trace.case[:,:,h] == code).sum())
                               for code,name in CASE_NAMES.items()}
                              for h in range(trace.case.shape[2])]
    out['pressure_mean_per_head'] = pressure.mean((0,1)).tolist()
    out['pressure_max_per_head'] = pressure.amax((0,1)).tolist()
    out['pressure_fraction_per_head'] = (pressure != 0).float().mean((0,1)).tolist()
    out['pop_count_per_head'] = trace.pop_fired.sum((0,1)).tolist()
    out['freed_pop_per_head'] = (trace.freed == 1).sum((0,1,3)).tolist()
    out['freed_decay_per_head'] = (trace.freed == 2).sum((0,1,3)).tolist()
    return out


class DifferentiatedCell(IntegratedCell):
    def __init__(self, arm, vocab):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__(256, TapeHeadConfig(n_heads=4, budget=2944))
        self.arm = arm
        # Preserve the X34 learned content paths initialized by the skeleton.
        del self.W_k, self.W_q
        self.features = nn.ModuleList()
        self.geometry = nn.ModuleList()
        for head in range(4):
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(3701+head)
                embedding = nn.Embedding(vocab,64)
                key, query = nn.Linear(64,64), nn.Linear(64,64)
                query.load_state_dict(key.state_dict())
            embedding.requires_grad_(False)
            key.requires_grad_(False); query.requires_grad_(False)
            self.features.append(embedding)
            self.geometry.append(nn.ModuleDict(dict(key=key,query=query)))
        self.W_in = nn.ModuleList()
        if arm in ('c','d'):
            for _ in range(4):
                layer = nn.Linear(256,64, bias=False)
                with torch.no_grad():
                    layer.weight.copy_(torch.eye(256)[:64])
                self.W_in.append(layer)
        self.configs = ([TapeHeadConfig(n_heads=1, **p) for p in POLICIES]
                        if arm == 'd' else [self.cfg])
        self.telemetry = None

    def forward(self, x, token_ids, state=None):
        if state is not None:
            raise ValueError('X37 LM resets tape at each packed block/story')
        b,t = token_ids.shape
        keys, queries = [], []
        for head in range(4):
            z = self.features[head](token_ids)
            if self.W_in:
                z = z+self.W_in[head](x)
            keys.append(self.geometry[head]['key'](z).detach())
            queries.append(self.geometry[head]['query'](z))
        k,q = torch.stack(keys,2),torch.stack(queries,2)
        v = self.W_v(x).reshape(b,t,4,128)
        beta = ((self.W_write(x).sigmoid().detach()>self.cfg.write_enable_threshold)
                & (token_ids!=0).unsqueeze(-1)).float()
        pop, logit = self.W_pop(x).sigmoid(), self.W_append(x)
        if self.arm != 'd':
            read, final, trace, pressure = tape_scan(k,q,v,beta,pop,logit,None,
                                                   self.cfg,self.append_err_coef)
            self.telemetry = metrics(final,trace,pressure) if not self.training else None
        else:
            reads, heads = [], []
            for head,cfg in enumerate(self.configs):
                sl = slice(head,head+1)
                r,final,trace,pressure = tape_scan(k[:,:,sl].contiguous(),q[:,:,sl].contiguous(),
                    v[:,:,sl].contiguous(),beta[:,:,sl].contiguous(),pop[:,:,sl].contiguous(),
                    logit[:,:,sl].contiguous(),None,cfg,self.append_err_coef[sl].contiguous())
                reads.append(r)
                if not self.training:
                    heads.append(metrics(final,trace,pressure))
            read = torch.cat(reads,2)
            self.telemetry = None
            if heads:
                self.telemetry = {name: sum((m[name] for m in heads), []) for name in
                    ('peak_live_per_head','final_live_per_head','routing_per_head',
                     'pressure_mean_per_head','pressure_max_per_head',
                     'pressure_fraction_per_head','pop_count_per_head',
                     'freed_pop_per_head','freed_decay_per_head')}
                self.telemetry['routing'] = {name:sum(m['routing'][name] for m in heads)
                                             for name in CASE_NAMES.values()}
        return read


class Block(OriginalBlock):
    def forward(self,x,ids):
        b,t,_ = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,t,3,4,64).permute(2,0,3,1,4)
        a = torch.nn.functional.scaled_dot_product_attention(rope(q),rope(k),v,is_causal=True)
        x = x+self.out(a.transpose(1,2).reshape(b,t,256))
        h = self.norm2(x)
        x = x+self.ff(h)
        read = self.cell(h.float(),ids)
        return x+self.read_out(read.flatten(2)), self.cell.telemetry


class LM(Skeleton):
    def __init__(self,arm,vocab=32768):
        super().__init__('tape_lm',vocab)
        old = self.blocks[2]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(3700)
            cell = DifferentiatedCell(arm,vocab)
        for name in ('W_v','W_write','W_pop','W_append','append_err_coef'):
            setattr(cell,name,getattr(old.cell,name))
        old.__class__ = Block
        old.cell = cell


def counts(model):
    return dict(total=sum(p.numel() for p in model.parameters()),
                trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))


def parameter_receipt(model,vocab):
    actual = counts(model)
    with torch.random.fork_rng(devices=[]):
        control = Skeleton('base_lm',vocab)
        old = sum(p.numel() for p in control.blocks[2].extra.parameters())
        width = round((actual['total']-(counts(control)['total']-old)-256)/513)
        control.blocks[2].extra = nn.Sequential(nn.Linear(256,width),nn.GELU(),nn.Linear(width,256))
    base = counts(control)
    ratio = abs(actual['total']/base['total']-1)
    assert ratio < .001
    return dict(arm=actual,base_lm=base,base_replacement_width=width,
                relative_total_difference=ratio,
                policies=[asdict(c) for c in model.blocks[2].cell.configs])


def input_snapshot(cell):
    return [layer.weight.detach().clone() for layer in cell.W_in]


def input_metrics(cell,initial):
    weights = [layer.weight.detach() for layer in cell.W_in]
    return dict(distance_from_init=[float((w-start).norm()) for w,start in zip(weights,initial)],
                pairwise_distance={f'{i}-{j}':float((weights[i]-weights[j]).norm())
                                   for i in range(len(weights)) for j in range(i+1,len(weights))},
                gradient_norm=[None if layer.weight.grad is None else float(layer.weight.grad.norm())
                               for layer in cell.W_in])
