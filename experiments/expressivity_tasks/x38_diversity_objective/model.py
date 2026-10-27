"""Arm (c)'s lane, with a differentiable objective on its existing read."""
import torch
from torch.nn import functional as F
from ..x37_head_differentiation.model import (
    LM as GeometryLM, parameter_receipt, input_snapshot, input_metrics,
)

ARMS = ('0', '0.001', '0.01', '0.1')


def read_diversity(read):
    """Return sum of six valid-position mean signed cosines and coverage."""
    normalized = F.normalize(read.float(), dim=-1, eps=1e-8)
    valid = read.float().norm(dim=-1) > 1e-8
    pairs, coverage = [], []
    for i in range(read.shape[2]):
        for j in range(i+1, read.shape[2]):
            mask = valid[..., i] & valid[..., j]
            cosine = (normalized[..., i, :]*normalized[..., j, :]).sum(-1)
            pairs.append((cosine*mask).sum()/mask.sum().clamp_min(1))
            coverage.append(mask.float().mean())
    return torch.stack(pairs).sum(), torch.stack(pairs), torch.stack(coverage)


class LM(GeometryLM):
    def __init__(self, arm='0.01', vocab=32768):
        if arm not in ARMS:
            raise ValueError(arm)
        super().__init__('c', vocab)
        self.diversity_lambda = float(arm)
        self.read = None
        self.blocks[2].cell.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self.read = output
        if not self.training:
            _, pairs, coverage = read_diversity(output)
            module.telemetry['read_cosine_pairs'] = pairs.tolist()
            module.telemetry['read_cosine_mean'] = float(pairs.mean())
            module.telemetry['read_nonzero_pair_coverage'] = coverage.tolist()
            module.telemetry['slot_usage_js'] = None
            module.telemetry['slot_usage_status'] = 'disabled outside story evaluation'
            if getattr(self,'measure_slots',False):
                module.telemetry.update(slot_usage(module,*inputs,output))

    def objective(self, logits, targets):
        ce = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        penalty, _, _ = read_diversity(self.read)
        self.read = None  # do not retain a training graph between steps
        return ce + self.diversity_lambda*penalty


@torch.no_grad()
def slot_usage(cell, x, ids, expected):
    """Eval-only shadow replay; un-decayed pre-write soft-strength read mass.

    Slot indices mean physical allocation slots, not semantic concept labels.
    Replay read parity is asserted before accepting this telemetry.
    """
    import math
    from ..x36_tape_perf.cell_fused import tape_scan
    from ..x30_tape_cell.naive_reference import init_state
    b,t = ids.shape
    keys, queries = [], []
    for h in range(4):
        z = cell.features[h](ids) + cell.W_in[h](x)
        keys.append(cell.geometry[h]['key'](z))
        queries.append(cell.geometry[h]['query'](z))
    k,q = torch.stack(keys,2),torch.stack(queries,2)
    v = cell.W_v(x).reshape(b,t,4,128)
    beta = ((cell.W_write(x).sigmoid()>cell.cfg.write_enable_threshold)
            & (ids!=0).unsqueeze(-1)).float()
    pop, logit = cell.W_pop(x).sigmoid(),cell.W_append(x)
    native = init_state(b,cell.cfg,x.device)
    mass = torch.zeros_like(native.strength)
    state = None
    reads = []
    for pos in range(t):
        scores = torch.einsum('bhsk,bhk->bhs',native.keys,q[:,pos])*(2/math.sqrt(64))
        weights = scores.masked_fill(~native.live,-1e9).softmax(-1)*native.strength*native.live
        mass += weights
        args = [a[:,pos:pos+1].contiguous() for a in (k,q,v,beta,pop,logit)]
        # Unknown bound avoids a new constexpr specialization at every token.
        # This is only a shadow-replay capacity hint, not a policy change.
        if state is not None:
            state.live_bound = None
        read,state,_,_ = tape_scan(*args,state,cell.cfg,cell.append_err_coef)
        native = state.native
        reads.append(read)
    torch.testing.assert_close(torch.cat(reads,1),expected,atol=2e-4,rtol=2e-4)
    totals = mass.sum(-1,keepdim=True)
    p = mass/totals.clamp_min(1e-12)
    js = {}
    for i in range(4):
        for j in range(i+1,4):
            a,c = p[:,i],p[:,j]
            m = (a+c)/2
            kl = lambda z:(z*(z.clamp_min(1e-12).log()-m.clamp_min(1e-12).log())).sum(-1)
            valid = (totals[:,i,0]>0)&(totals[:,j,0]>0)
            js[f'{i}-{j}'] = float(((kl(a)+kl(c))/2/math.log(2))[valid].mean()) if valid.any() else None
    return dict(slot_usage_js=js,slot_usage_mass=mass.sum(0).tolist(),
                slot_usage_status='shadow replay parity passed; normalized un-decayed read mass; JS bits')
