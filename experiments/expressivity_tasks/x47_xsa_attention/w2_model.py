"""W2 specialization of the W1 workspace. No fixture labels, target time or world ID."""
from enum import IntEnum

import torch
from torch import nn

from . import workbench_model as w


class Op(IntEnum):
    EMIT, STORE_EMIT, PARENT, CHILDREN = range(4)


class BindingWorkbench(w.Workbench):
    def __init__(self):
        super().__init__()
        self.register_buffer('symbols', torch.arange(256))
        # A positional byte-embedding sum recognizes each marker on the device.
        # Left padding = 256. Do not inspect the value byte to detect a write.
        self.marker = nn.Parameter(torch.zeros(2, 8, 257))
        self.register_buffer('positions', torch.arange(8))
        self.register_buffer('threshold', torch.tensor([4., 3.]))
        self.register_buffer('suffix_len', torch.tensor(len(b'key=')))
        self.op = nn.Linear(5, len(Op))
        with torch.no_grad():
            for row, start, text in ((0, 2, b'unit='), (1, 4, b'key=')):
                for pos, byte in enumerate(text, start):
                    self.marker[row, pos, byte] = 1
            # Indicators: store marker, query marker, parent answered, children
            # answered, two legal byte edges. EMIT wins unless an op is enabled.
            self.op.weight.copy_(torch.tensor([[0,0,0,0,0], [2,0,-4,0,0],
                [0,2,-4,0,0], [0,0,2,-4,2]]))
            self.op.bias.copy_(torch.tensor([0,-1,-1,-3]))

    def forward(self, packed, record, gain, mixing):
        ids = packed[:8, w.BYTE_ID].long()
        marks = (self.marker[:, self.positions, ids].sum(-1) - self.threshold).clamp(0, 1)
        # All arms execute the same shapes, even on ordinary filler prefixes.
        x = packed.clone()
        x[w.LIVE, :256] = (ids[-1] == self.symbols).to(x.dtype)
        x[[w.CHILD, w.META], :257] = mixing @ packed[[w.CHILD, w.META], :257]
        z = self.encode(x, gain * record)
        parent_done, children_done = z[w.PLAN, w.FLAGS][:2].unbind()
        legal = packed[w.ARCH, :256] > 0
        score = (z[w.RA, :256].clamp_min(1e-20).log() + self.edge(z[w.RA])).masked_fill(~legal, -torch.inf)
        rank = score.argsort(descending=True, stable=True)[:2]
        rank = torch.where(legal[rank], rank, -1)
        flags = torch.stack((parent_done, children_done, legal.sum().eq(2).to(x.dtype)))
        op = self.op(torch.cat((marks, flags))).argmax()

        # The two operands are bound to the handles actually requested. The stored
        # row is the only older input, not a re-read of a session or archive byte.
        child_h = z[[w.RC, w.PLAN], :256].clamp_min(0)
        join_mass = child_h @ z[w.RR, :256].clamp_min(0)
        codes = packed[[w.BOOK0, w.BOOK1], w.BYTE_ID].long()
        basis = (codes[:, None] == self.symbols).to(x.dtype)
        joined, has_join = w.positive_dist(join_mass @ basis)
        parent, has_parent = w.positive_dist(z[w.RA, :256])
        evidence = torch.where(has_join.bool(), joined, parent)
        local = self.local(z[w.PLAN]).softmax(0)
        prob = (local + 4*evidence) / (1 + 4*torch.maximum(has_join, has_parent))
        value, _ = w.positive_dist(self.writer(z[w.RL, :256]))
        packet = torch.stack((op, self.suffix_len, rank[0], rank[1], prob.argmax()))
        return packet, prob, value
