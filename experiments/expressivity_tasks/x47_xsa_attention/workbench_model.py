"""One small FP32 Transformer; the decision packet (opcode, edges, byte) leaves the device."""
from enum import IntEnum

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Op(IntEnum):
    EMIT = 0
    BASE = 1          # current-suffix reads in each admitted source
    BASE_RECALL = 2   # current-suffix live read; substitute addressed record for archive
    FOLLOW = 3        # append the returned edge byte to the archive locus, read child
    WRITE_EMIT = 4


# Fixed workspace shape; DATA is 256 bytes + structural END. Control and embeddings
# occupy disjoint lanes, so a prefix byte ID cannot become a branch vote.
W, D, H = 20, 288, 4
BOOK0, BOOK1 = 8, 9
LIVE, ARCH, CHILD, ROW, META = range(10, 15)
RL, RA, RC, RR, PLAN = range(15, 20)
FLAGS = slice(257, 263)
BYTE_ID = 264


def pack(tail, live, archive, child, flags, keys):
    if len(tail) > 8 or len(flags) != 6:
        raise ValueError('workspace capacity')
    x = np.zeros((W, D), dtype=np.float32)
    x[:8, BYTE_ID] = 256  # embedding padding symbol, not END or a corpus byte
    if tail:
        x[8-len(tail):8, BYTE_ID] = list(tail)
    for i, h in ((LIVE, live), (ARCH, archive), (CHILD, child)):
        if h is not None:
            x[i, :257] = h
    for i, key in keys.items():
        if len(key) > 16:
            raise ValueError('logical key too long for workspace')
        x[i, D-16:D-16+len(key)] = [(b+1)/256 for b in key]
    x[META, FLAGS] = flags
    return torch.from_numpy(x)


def positive_dist(x):
    h = x.clamp_min(0)
    s = h.sum(-1, keepdim=True)
    return h / s.clamp_min(1e-20), (s > 0).to(h.dtype)


class Workbench(nn.Module):
    def __init__(self):
        super().__init__()
        self.slot = nn.Parameter(torch.randn(W, 16) * .02)
        self.byte = nn.Embedding(257, 16, padding_idx=256)
        self.qkv = nn.Linear(D, 3*D, bias=False)
        self.out = nn.Linear(D, D, bias=False)
        self.ff = nn.Sequential(nn.LayerNorm(D), nn.Linear(D, 96), nn.GELU(), nn.Linear(96, D))
        self.local = nn.Linear(D, 256, bias=False)
        self.edge = nn.Linear(D, 256)  # correction to count logits
        self.writer = nn.Linear(256, 256, bias=False)
        self.op = nn.Linear(8, len(Op))
        # A typed reader attends to eight recent slots, two handles AND its source. All
        # other rows have a bidirectional view. The registers have no input payload.
        mask = torch.ones(W, W, dtype=torch.bool)
        for dst, src in zip((RL, RA, RC, RR, PLAN), (LIVE, ARCH, CHILD, ROW, META)):
            mask[dst] = False; mask[dst, :10] = True; mask[dst, src] = True
        self.register_buffer('mask', mask)
        with torch.no_grad():
            eye = torch.eye(D)
            self.qkv.weight.copy_(torch.cat((torch.zeros_like(eye), eye, eye)))
            self.out.weight.copy_(11 * eye)  # uniform 1/11 at birth, exact payload read
            for p in (self.ff[-1].weight, self.ff[-1].bias, self.local.weight,
                      self.edge.weight, self.edge.bias):
                p.zero_()
            self.writer.weight.copy_(torch.eye(256))
            # Flags: nonempty, base_done, eligible_record, speculate, recalled, child_done,
            # parent_has_byte_mass, child_has_byte_mass. EMIT is the default (score 0).
            self.op.weight.copy_(torch.tensor([
                [0,0,0,0,0,0,0,0], [3,-6,0,0,0,0,0,0], [3,-6,2,0,0,0,0,0],
                [0,4,0,4,-4,-4,4,0], [0,0,0,0,0,4,0,4]]))
            self.op.bias.copy_(torch.tensor([0,-1,-2,-11,-7]))

    def encode(self, packed: torch.Tensor, record: torch.Tensor | None):
        """Shared W1/W2 workbench; typed register layout and birth weights unchanged."""
        x = packed.clone()
        ids = packed[:8, BYTE_ID].long()
        x[:, -16:] += self.slot
        x[:8, -16:] += self.byte(ids)
        if record is not None:
            x[ROW, :256] = record  # actual device tensor; preserve dependence on previous writer
        q,k,v = self.qkv(x).view(W, 3, H, D//H).permute(1, 2, 0, 3).unbind(0)
        context = F.scaled_dot_product_attention(q[None], k[None], v[None], attn_mask=self.mask)
        z = x + self.out(context[0].transpose(0, 1).reshape(W, D))
        return z + self.ff(z)

    def forward(self, packed: torch.Tensor, record: torch.Tensor | None):
        z = self.encode(packed, record)
        flags = z[PLAN, FLAGS]
        evidence, present = positive_dist(z[[RL, RA, RC, RR], :256])
        # Do not let the child vote one step early. For recall, do not vote twice.
        use = flags[4].round().clamp(0, 1)
        a = (1-use)*evidence[1] + use*evidence[3]
        present_a = (1-use)*present[1] + use*present[3]
        local = self.local(z[PLAN]).softmax(0)
        prob = (local + evidence[0] + 4*a) / (1 + present[0] + 4*present_a)
        opcode = self.op(torch.cat((flags, present[1:3, 0]))).argmax()
        legal = packed[ARCH, :256] > 0
        edge_scores = (z[RA, :256].clamp_min(1e-20).log() + self.edge(z[RA])).masked_fill(~legal, -torch.inf)
        rank = edge_scores.argsort(descending=True, stable=True)[:2]
        rank = torch.where(legal[rank], rank, -1)
        value, _ = positive_dist(self.writer(evidence[2]))
        packet = torch.stack((opcode, rank[0], rank[1], prob.argmax()))
        return packet, prob, value
