"""Small causal Transformer + current-weight whole-corpus cross-read on every real byte."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..corpus_kernel.token_lm import CorpusTokenLM
from ..corpus_kernel.plan import Layout
from ..corpus_windows.packet import WindowArchive, Inputs, Block
from ..corpus_windows.training import encode, read
from .routes import Reference, TOKENIZER, compact, tiles

ARMS = ("local", "count", "exact", "indexed")
SEEDS = (5751, 5752)


class Layer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ff = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)
        )

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = (
            self.qkv(self.norm(x))
            .reshape(b, t, 3, 4, d // 4)
            .permute(2, 0, 3, 1, 4)
            .unbind(0)
        )
        z = (
            F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=0.0)
            .transpose(1, 2)
            .reshape(b, t, d)
        )
        x = x + self.proj(z)
        return x + self.ff(x)


class LanguageLM(nn.Module):
    def __init__(self, seed, d=64, rank=32, leaf=128, time=64):
        super().__init__()
        torch.manual_seed(seed)
        self.head = CorpusTokenLM(
            256, d, rank, tokenizer=TOKENIZER, leaf=leaf, feature_seed=5750
        )
        self.pos = nn.Parameter(0.02 * torch.randn(time, d))
        self.layers = nn.Sequential(Layer(d), Layer(d), nn.LayerNorm(d))
        self.gate = nn.Linear(d, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -1.0)

    def host(self, x, inp):
        b = len(inp.lengths)
        t = len(self.pos)
        if len(x) != b * t or any(k != t for k in inp.lengths):
            raise ValueError("expected complete independent windows")
        return self.layers(x.reshape(b, t, -1) + self.pos).reshape_as(x)

    def logits(self, batch, arm, backend):
        if arm not in ARMS:
            raise ValueError("foreign arm")
        w = batch.memory if arm in ("indexed", "exact") else batch.local
        mode = "indexed" if arm == "indexed" else "dense"
        step = encode(w, self.head, mode, backend, live_transform=self.host)
        v = read(w, self.head, step, mode, w.blocks[0], step.x, step.q, value_only=True)
        return self.head.lm(step.x + self.head.out(v)), step.x

    def forward(self, batch, arm, *, backend):
        from ..corpus_windows.triton_nll import nll_rows

        logits, x = self.logits(batch, arm, backend)
        y = batch.targets
        loss = (
            nll_rows(logits, y)
            if backend == "triton"
            else F.cross_entropy(logits, y, reduction="none")
        )
        if arm == "count":
            g = self.gate(x)[:, 0]
            p = batch.table[batch.count_ids, y]
            loss = -torch.logaddexp(F.logsigmoid(-g) - loss, F.logsigmoid(g) + p.log())
        return loss


@dataclass
class Batch:
    memory: object
    local: object
    targets: torch.Tensor
    count_ids: torch.Tensor
    table: torch.Tensor
    exact_pairs: int
    hit_queries: int


class Pack:
    def __init__(self, raw, m, batch=16):
        self.ref = Reference(raw)
        self.a = WindowArchive(m.head, self.ref.index)
        self.dev = self.a.device
        self.table = torch.tensor(self.ref.probs, device=self.dev)
        self.b = batch
        self.t = len(m.pos)
        self.layout = self.a.layout
        self.empty = Layout(self.layout.version, 0, self.layout.leaf)
        b, t = self.b, self.t
        q = b * t
        self.lengths = (t,) * b
        self.spans = self.ints(np.c_[np.arange(b) * t, np.full(b, t)])
        self.locals = self.ints(
            (np.arange(b)[:, None] * t + np.arange(t - 1)).flatten()
        )
        s = np.repeat(np.arange(b) * (t - 1), t)
        self.lt = self.ints(tiles(np.arange(q), s, s + np.tile(np.arange(t), b)))
        self.localblock = Block(0, q, None, None, self.lt)

    def ints(self, x):
        return torch.as_tensor(x, dtype=torch.long, device=self.dev)

    def bind(self, windows):
        b, t = self.b, self.t
        if windows.shape != (b, t + 1) or windows.dtype != np.uint8:
            raise ValueError("expected complete byte windows")
        x = windows[:, :-1].copy()
        y = self.ints(windows[:, 1:].flatten())
        counts, hits = self.ref.lookup(x)
        desc = compact(hits, self.layout.rows, self.layout.leaf, self.a.rank)
        work = SimpleNamespace(
            **dict(zip(("exact", "edge", "nodes"), map(self.ints, desc), strict=True))
        )
        ins = Inputs(self.ints(x.ravel()), self.lengths, self.spans)

        def packet(layout, block):
            return SimpleNamespace(
                inputs=ins,
                targets=y,
                archive=self.a,
                layout=layout,
                allow_skip=True,
                local_rows=self.locals,
                blocks=(block,),
            )

        ep = hits[:, 1] - hits[:, 0]
        return Batch(
            packet(self.layout, Block(0, b * t, None, work, self.lt)),
            packet(self.empty, self.localblock),
            y,
            self.ints(counts.flatten()),
            self.table,
            int(ep.sum()),
            int((ep > 0).sum()),
        )

    def dataset(self, docs, shuffle):
        # Window t+1 begins on window t's last target. The final answer is never a model input to its own prediction.
        if docs.ndim != 2 or docs.dtype != np.uint8 or (docs.shape[1] - 1) % self.t:
            raise ValueError("invalid raw document length")
        windows = np.lib.stride_tricks.sliding_window_view(docs, self.t + 1, axis=1)[
            :, :: self.t, :
        ].reshape(-1, self.t + 1)
        if len(windows) % self.b:
            raise ValueError("incomplete registered batch")
        order = (
            np.arange(len(windows))
            if not shuffle
            else np.random.default_rng(5750).permutation(len(windows))
        )
        return [
            self.bind(windows[order[i : i + self.b]])
            for i in range(0, len(order), self.b)
        ]
