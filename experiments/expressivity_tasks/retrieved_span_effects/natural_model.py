"""Small fully trainable p50k token hosts. All real continuation targets are scored.

Two recurrent blocks; the lower block sees actual tokens only. Save its state at
one lookup per window and reuse a learned passage memory with different queries.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .triton_controls import pack
from .triton_loss import token_nll
from .triton_read import terminal, weights
from .triton_scan import pool, scan


@dataclass(frozen=True)
class Shape:
    vocab: int = 50281
    width: int = 128
    heads: int = 4
    key: int = 16
    time: int = 64
    prefix: int = 16
    hits: int = 4
    page: int = 32


ARMS = (
    "local",
    "g-state",
    "g-neutral",
    "g-effect",
    "e-state",
    "e-neutral",
    "e-effect",
    "mix-effect",
)
HOSTS = ("gdn2", "e97")
SEEDS = (5411, 5412, 5413)


def groups(x, h):
    b, t, _ = x.shape
    return x.view(b, t, h, -1).transpose(1, 2).reshape(b * h, t, -1).contiguous()


def batch_norm(x):
    return F.layer_norm(x, (x.shape[-1],))


class Block(nn.Module):
    def __init__(self, shape, family):
        super().__init__()
        self.shape = shape
        self.family = family
        d, h, k = shape.width, shape.heads, shape.key
        self.norm = nn.LayerNorm(d)
        self.controls = nn.Linear(d, 5 * k * h, bias=False)
        self.query = nn.Linear(d, k * h, bias=False)
        self.out = nn.Linear(k * h, d, bias=False)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, 2 * d),
            nn.GELU(approximate="tanh"),
            nn.Linear(2 * d, d),
        )

    def forward(self, x, count, *, cut=False):
        sh = self.shape
        b, t, _ = x.shape
        h = sh.heads
        k = sh.key
        z = self.norm(x)
        c = pack(groups(self.controls(z), h), k, k, family=self.family)
        q = groups(self.query(z), h)
        lens = count[:, None].expand(-1, h).reshape(-1, 1)
        seed = x.new_zeros((b * h, k, k), dtype=torch.float32)
        if cut:
            n = sh.prefix
            a, seed = scan(
                c[:, :n], q[:, :n], seed, lens.clamp_max(n), family=self.family
            )
            tail, _ = scan(
                c[:, n:], q[:, n:], seed, (lens - n).clamp_min(0), family=self.family
            )
            y = torch.cat((a, tail), 1)
        else:
            y, seed = scan(c, q, seed, lens, family=self.family)
        y = y.view(b, h, t, k).transpose(1, 2).reshape(b, t, h * k)
        x = x + self.out(y.to(z.dtype))
        return x + self.mlp(x), seed.view(b, h, k, k)


class TokenLM(nn.Module):
    def __init__(self, shape, host, arm):
        super().__init__()
        self.sh = sh = shape
        self.host = host
        self.arm = arm
        if (
            arm not in ARMS
            or host not in HOSTS
            or not (sh.heads % 2 == 0 and sh.time > sh.prefix >= 1)
        ):
            raise ValueError("unsupported model")
        d, h, k = sh.width, sh.heads, sh.key
        self.embedding = nn.Embedding(((sh.vocab + 127) // 128) * 128, d)
        self.lower = Block(sh, host)
        self.upper = Block(sh, host)
        self.address = nn.Linear(d, k, bias=False)
        self.page_key = nn.Linear(d, k, bias=False)
        self.left = nn.Linear(d, d, bias=False)
        self.control = nn.Linear(d, 5 * h * k, bias=False)
        self.read_query = nn.Linear(d, h * k, bias=False)
        self.align_l = nn.Parameter(torch.eye(k).repeat(h, 1, 1))
        self.align_r = nn.Parameter(torch.eye(k).repeat(h, 1, 1))
        self.gate = nn.Linear(d + 1, h)
        self.out = nn.Linear(h * k, d, bias=False)
        self.norm = nn.LayerNorm(d)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, ids, count, pages, lengths, votes, *, intervention="intact"):
        sh = self.sh
        h = sh.heads
        k = sh.key
        n = sh.prefix - 1
        b, t = ids.shape
        pts = sh.page
        p = sh.hits
        if (t,) + tuple(pages.shape) != (sh.time, b, p, pts):
            raise ValueError("wrong live/archive geometry")
        x, live = self.lower(self.embedding(ids.long()), count, cut=True)
        # A learned change of basis for each archive head. Every hit gets the same seed.
        seed = (self.align_l[None] @ live @ self.align_r[None]).float()
        support = (lengths >= 0).any(-1)
        total = votes.sum(-1).float().log1p()
        if self.arm == "local":
            memory = seed * support[:, None, None, None]
        else:
            token = self.embedding(pages.long())
            mask = (
                torch.arange(pts, device=ids.device)[None, None, :]
                < lengths[:, :, None]
            )
            key = self.page_key(
                (token * mask[:, :, :, None]).sum(2) / lengths.clamp_min(1)[:, :, None]
            )
            bias = votes.clamp_min(1).float().log()
            prob = weights(self.address(x[:, n]), key, bias, lengths, pts)
            # A shuffled value should never alter its key, score or target. Only use
            # this particular donor when all candidates are present and equal length.
            donor = ((lengths == lengths[:, :1]).all(1) & (lengths[:, 0] > 0))[
                :, None, None, None, None
            ]
            z = token + self.left(F.pad(token[:, :, :-1], (0, 0, 1, 0)))
            raw = self.control(z).view(b, p, pts, h, 5 * k).permute(0, 3, 1, 2, 4)
            parts = []
            specs = (
                [(0, h, "gdn2" if self.arm.startswith("g-") else "e97")]
                if self.arm != "mix-effect"
                else [(0, h // 2, "gdn2"), (h // 2, h, "e97")]
            )
            for lo, hi, family in specs:
                nh = hi - lo
                c = pack(
                    raw[:, lo:hi].reshape(b * nh, p, pts, 5 * k), k, k, family=family
                )
                s = seed[:, lo:hi].reshape(b * nh, k, k)
                ns = lengths[:, None, :].expand(-1, nh, -1).reshape(b * nh, p)
                pp = prob[:, None, :].expand(-1, nh, -1).reshape(b * nh, p)
                seeded = self.arm.endswith("effect")
                m = terminal(c, s, ns, family=family, seeded=seeded).view(
                    b, nh, p, k, k
                )
                if intervention == "wrong_value":
                    m = torch.where(donor, m.roll(1, 2), m)
                if intervention == "wrong_effect" and seeded:
                    blank = terminal(c, s, ns, family=family, seeded=False).view_as(m)
                    delta = m - blank
                    m = torch.where(donor, blank + delta.roll(1, 2), m)
                a = pool(m.reshape(b * nh, p, k, k), pp).view(b, nh, k, k)
                if self.arm.endswith("neutral"):
                    a = a + seed[:, lo:hi] * support[:, None, None, None]
                parts.append(a)
            memory = torch.cat(parts, 1) if len(parts) > 1 else parts[0]
        q = self.read_query(x[:, n:]).view(b, t - n, h, k).transpose(1, 2)
        r = (q @ memory).transpose(1, 2)
        mass = total[:, None, None].expand(-1, t - n, -1)
        g = self.gate(torch.cat((x[:, n:], mass), -1)).sigmoid()
        hint = self.out((r * g[:, :, :, None]).reshape(b, t - n, h * k))
        hint = hint * 0 if intervention == "drop" else hint
        y, _ = self.upper(x + F.pad(hint, (0, 0, n, 0)), count)
        return F.linear(self.norm(y), self.embedding.weight)

    def losses(self, window, count, pages, lengths, votes, *, intervention="intact"):
        # Only input IDs enter the model; labels and the last token never enter search.
        x = window[:, :-1]
        logits = self(x, count, pages, lengths, votes, intervention=intervention)
        keep = torch.arange(self.sh.time, device=x.device)[None, :] < count[:, None]
        target = torch.where(keep, window[:, 1:], -100)
        return token_nll(logits, target, self.sh.vocab)
