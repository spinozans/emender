"""Two prefix-scoped records between incumbent K15 layers. One window per call."""
import math
from pathlib import Path
import subprocess

import torch
from torch import nn
import torch.nn.functional as F

from experiments.expressivity_tasks.x43_fork_resolved.xsa_binding_live import Session


class BookmarkSession(Session):
    def __init__(self, exe: Path):
        self.p = subprocess.Popen([exe, 'session', '32'], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE)


class BookmarkPort(nn.Module):
    def __init__(self, width, size=32):
        super().__init__()
        self.query = nn.Linear(width, size)
        self.slot_bias = nn.Linear(width, 2)
        self.action = nn.Linear(width + size, 4)
        self.key = nn.Linear(width + size, size)
        self.payload = nn.Linear(width + size, size)
        self.byte = nn.Embedding(256, size)
        self.inject = nn.Linear(size, width, bias=False)

    def forward(self, hidden, ids, session, forced=None, audit=False, stride=16):
        keys = hidden.new_zeros(2, self.byte.embedding_dim)
        values = torch.zeros_like(keys)
        # Immutable byte strings and occurrence starts are non-differentiable fields
        # of the same two rows as keys/values. Invalidated together on retirement.
        handles = [None, None]
        output, trace, log_probs, writes = [], [], [], []
        raw = bytes(ids[0].tolist())
        for i, start in enumerate(range(0, len(raw), stride)):
            stop = min(start + stride, len(raw))
            chunk = hidden[:, start:stop]
            q = self.query(chunk)
            scores = self.slot_bias(chunk) + q @ keys.T / math.sqrt(keys.size(1))
            attention = scores.float().softmax(-1).to(values.dtype)
            recalled = attention @ values
            output.append(chunk + self.inject(recalled))
            assert session.rpc('append ' + raw[start:stop].hex())['visible'] == stop
            if stop == len(raw):
                break
            h = chunk[0, -1]
            def choose(logits, forced_value):
                dist = torch.distributions.Categorical(logits=logits.float())
                idx = (forced_value if forced_value is not None else
                       int(dist.sample() if self.training else logits.argmax()))
                log_probs.append(dist.log_prob(logits.new_tensor(idx).long()))
                return idx
            slot = choose(scores[0, -1], None if forced is None else forced[i][0])
            prior = handles[slot]
            logits = self.action(torch.cat([h, values[slot]]))
            if prior is None:
                logits = logits.masked_fill(torch.arange(4, device=h.device) >= 2, -torch.inf)
            op = choose(logits, None if forced is None else forced[i][1])
            event = dict(stop=stop, slot=slot, op=op, calls=[])
            def read(cmd):
                ans = session.rpc(cmd)
                event['calls'].append((cmd, ans))
                return ans
            if op == 3:
                k, v = keys[slot]*0, values[slot]*0
                handles[slot] = None
            else:
                if op == 2:
                    pattern, pos = prior
                    offset = pos + len(pattern)
                    assert offset < stop
                    b = read(f'byte {offset}')['byte']
                    pattern = (pattern + bytes([b]))[-32:]
                else:
                    pattern = raw[max(0, stop-(3 if op == 0 else 12)):stop]
                ans = read(f'locate {pattern.hex()} 0 {stop} {stop}')
                hits = ans['positions']
                if audit:
                    assert hits == [j for j in range(stop-len(pattern)+1)
                                    if raw.startswith(pattern, j, stop)]
                pos = hits[0]
                offset = min(stop-1, pos+len(pattern))
                b = read(f'byte {offset}')['byte']
                if audit:
                    assert raw[offset] == b
                response = torch.cat([h, self.byte.weight[b]])
                k, v = self.key(response), self.payload(response).tanh()
                handles[slot] = (pattern, pos)
                writes.append(v)
            mask = (torch.arange(2, device=h.device) == slot)[:, None]
            keys = torch.where(mask, k, keys)
            values = torch.where(mask, v, values)
            trace.append(event)
        return torch.cat(output, dim=1), trace, log_probs, writes


class BookmarkK15(nn.Module):
    def __init__(self, backbone, exe):
        super().__init__()
        self.backbone = backbone
        self.port = BookmarkPort(backbone.embed.embedding_dim)
        self.exe = exe

    def forward(self, ids, *, doors=True, forced=None, audit=False):
        if not doors:
            return self.backbone.head(self.backbone.features(ids)), [], [], []
        if ids.shape[0] != 1 or not 1 <= ids.shape[1] <= 65536:
            raise ValueError('one bounded independent window required')
        h = self.backbone.embed(ids)
        for i, layer in enumerate(self.backbone.blocks):
            if i == len(self.backbone.blocks)//2:
                session = BookmarkSession(self.exe)
                try:
                    h, trace, probs, writes = self.port(h, ids, session, forced, audit)
                finally:
                    session.close()
            h = layer(h)
        return self.backbone.head(self.backbone.norm(h)), trace, probs, writes
