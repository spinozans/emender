"""SAL-1 model: one causal byte host, three registered corpus-law arms.

LOCAL is the host with causal local rows and a no-read row. SKELETON adds the
order-0 count source: corpus mass comes exactly from static suffix counts and
its value is the depth-weighted successor histogram mapped through the same
learned successor-vector head every arm uses. ORDER2 adds the learned
residual: current-weight keys/values over the whole archive, the deepest
matched suffix interval scored exactly per row (leaf-aligned, capped), and
the complement covered by order-2 moment nodes with the value tilt. All arms
share one parameter structure and paired starts; arm-specific parameters stay
dormant rather than absent.

Registered law details: the corpus prior is source-balanced (b =
exp(log_pi)/rows per corpus row; local rows carry prior 1 each; the no-read
row has no archive-size coupling). Count context is window-local under the
64-byte reset discipline: at window starts, depths beyond the observed prefix
miss and fall back to shallower classes. Cores larger than the cap fall back
entirely to moment nodes over the whole archive. Nothing bound in a batch
depends on weights; the bank is rebuilt inside each forward.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..corpus_kernel.index import EndpointIndex
from ..corpus_windows.packet import WindowArchive
from ..language_compression.lm import Layer
from ..language_compression.routes import TOKENIZER
from .ladder import MAX_DEPTH, _pack, CountIndex, cover, moment_statistics
from .settings import ARMS

ALPHA = 256
CORE_CAP = 256
LEAF = 128


class SalLM(nn.Module):
    """Host width 128, four blocks; reader width 64 for every arm."""

    def __init__(self, seed, host=128, width=64, time=64, rank=32):
        super().__init__()
        torch.manual_seed(seed)
        self.tokenizer = TOKENIZER
        self.vocab = ALPHA
        self.leaf = LEAF
        self.time = time
        self.width = width
        self.embed_host = nn.Embedding(ALPHA, host)
        self.pos = nn.Parameter(0.02 * torch.randn(time, host))
        self.blocks = nn.Sequential(
            Layer(host), Layer(host), Layer(host), Layer(host), nn.LayerNorm(host)
        )
        self.bridge = nn.Linear(host, width)
        self.embed = nn.Embedding(ALPHA, width)
        self.context = nn.Linear(width, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.out = nn.Linear(width, width)
        self.lm = nn.Linear(width, ALPHA)
        self.skip = nn.Linear(width, 1)
        self.register_buffer("omega", 0.05 * torch.randn(rank, width))
        self.log_pi = nn.Parameter(torch.tensor(-1.0))
        self.g = nn.Parameter(torch.linspace(-2.0, 2.0, MAX_DEPTH + 1))

    def states(self, ids, spans):
        """Archive/document encodings: byte embedding plus causal prefix mean."""
        e = self.embed(ids)
        n = spans.shape[0]
        lens = (spans[:, 1] - spans[:, 0]).tolist()
        if len(set(lens)) == 1:
            z = e.reshape(n, lens[0], -1).cumsum(1)
            mean = (
                z / torch.arange(1, lens[0] + 1, device=e.device)[None, :, None]
            ).reshape(e.shape)
        else:
            mean = torch.cat(
                [
                    seg.cumsum(0)
                    / torch.arange(1, len(seg) + 1, device=e.device)[:, None]
                    for seg in (
                        e[int(s) : int(s) + int(length)] for s, length in spans.tolist()
                    )
                ]
            )
        return (e + self.context(mean)).tanh()

    def live(self, ids, lengths):
        p = self.embed_host(ids).reshape(len(lengths), int(lengths[0]), -1)
        p = self.blocks(p + self.pos[None, : int(lengths[0])]).reshape(len(ids), -1)
        return self.bridge(p)

    def bank(self, archive):
        """Current-weight archive keys and values in bank order, rebuilt per call."""
        arch = self.states(archive.inputs.ids, archive.inputs.spans)
        return self.key(arch[archive.order]), self.value(self.embed(archive.successors))

    def node_partials(self, levels, t, owner, lv, idx, log_rows):
        """Moment-node (log mass, value) partials, grouped per tree level."""
        logs, vals, owners = [], [], []
        for level in sorted(set(lv.tolist())):
            sel = lv == level
            st = levels[level]
            i = idx[sel]
            n = st["n"][i]
            tt = t[owner[sel]]
            logm = (
                self.log_pi
                - log_rows
                + torch.log(n)
                + (tt * st["kbar"][i]).sum(-1)
                + 0.5
                * torch.einsum("pd,pde,pe->p", tt, st["S"][i] / n[:, None, None], tt)
            )
            val = st["vbar"][i] + torch.einsum(
                "ped,pd->pe", st["T"][i] / n[:, None, None], tt
            )
            logs.append(logm)
            vals.append(val)
            owners.append(owner[sel])
        if not logs:
            e = t.new_empty(0)
            return e, t.new_empty((0, self.width)), e.long()
        return torch.cat(logs), torch.cat(vals), torch.cat(owners)

    def forward(self, batch, arm, *, archive=None):
        if arm not in ARMS:
            raise ValueError("foreign arm")
        if arm == "order2" and archive is None:
            raise ValueError("order2 needs the archive")
        ids, y = batch["ids"], batch["targets"]
        rows = batch["rows"]
        log_rows = math.log(rows)
        x = self.live(ids, batch["lengths"])
        q = self.query(x)
        t = q * self.width**-0.5
        dev = x.device
        Q = len(x)
        logs, vals, owner = [], [], []

        lk = self.key(x[batch["local_rows"]])
        lv = self.value(self.embed(batch["local_succ"]))
        po, pr = batch["pair_owner"], batch["pair_row"]
        logs.append((lk[pr] * t[po]).sum(-1))
        vals.append(lv[pr])
        owner.append(po)

        logs.append(self.skip(x)[:, 0])
        vals.append(x.new_zeros((Q, self.width)))
        owner.append(torch.arange(Q, device=dev))

        if arm == "skeleton":
            w = torch.exp(self.g)
            den = (w[None, :] * batch["m"]).sum(-1)
            mass = torch.exp(self.log_pi - log_rows) * den
            e_vec = self.value(self.embed.weight)
            num = torch.einsum("d,qdv->qv", w, batch["h"] @ e_vec)
            logs.append(torch.log(mass))
            vals.append(num / den[:, None])
            owner.append(torch.arange(Q, device=dev))
        elif arm == "order2":
            k_bank, v_bank = self.bank(archive)
            levels = moment_statistics(k_bank, v_bank, LEAF)
            logm, val, own = self.node_partials(
                levels,
                t,
                batch["cover_owner"],
                batch["cover_lv"],
                batch["cover_idx"],
                log_rows,
            )
            logs.append(logm)
            vals.append(val)
            owner.append(own)
            core = batch["core"]
            cid = core.reshape(-1)
            ok = batch["core_valid"][:, None].expand_as(core).reshape(-1) & (cid >= 0)
            safe = cid.clamp(min=0)
            tt = t.repeat_interleave(core.shape[1], dim=0)
            clogm = (k_bank[safe] * tt).sum(-1) + self.log_pi - log_rows
            logs.append(clogm.masked_fill(~ok.reshape(-1), -float("inf")))
            vals.append(v_bank[safe])
            owner.append(torch.arange(Q, device=dev).repeat_interleave(core.shape[1]))

        logm = torch.cat(logs)
        val = torch.cat(vals)
        own = torch.cat(owner)
        peak = torch.full((Q,), -float("inf"), device=dev)
        peak.scatter_reduce_(0, own, logm, reduce="amax", include_self=True)
        wts = torch.exp(logm - peak[own])
        z = torch.zeros(Q, device=dev).index_add_(0, own, wts)
        acc = torch.zeros(Q, self.width, device=dev).index_add_(
            0, own, wts[:, None] * val
        )
        read = acc / z[:, None]
        logits = self.lm(x + self.out(read))
        return F.cross_entropy(logits, y, reduction="none")


class SalPack:
    """Static batch binding: counts, matched cores, covers, local pairs.

    Everything bound here is integer or frozen data; no learned tensor is
    stored across updates.
    """

    def __init__(
        self, build, m, batch=16, depth=MAX_DEPTH, cap=CORE_CAP, bank=True, counts=None
    ):
        a = np.asarray(build)
        if a.ndim != 2 or a.dtype != np.uint8 or a.shape[1] < 2 or not len(a):
            raise ValueError("rectangular uint8 archive")
        self.rows = a.shape[0] * (a.shape[1] - 1)
        self.leaves = self.rows // LEAF
        if bank and (self.rows % LEAF or self.leaves & (self.leaves - 1)):
            raise ValueError("complete dyadic tree geometry required")
        if counts is None:
            self.counts = CountIndex(a, depth, rowids=bank)
        else:
            if (
                counts.rows != self.rows
                or counts.depth != depth
                or bool(len(counts.levels[-1].rows)) != bank
            ):
                raise RuntimeError("prebuilt count table geometry mismatch")
            self.counts = counts
        if bank:
            self.index = EndpointIndex(
                [(f"build:{i}", p.tolist()) for i, p in enumerate(a)],
                tokenizer=TOKENIZER,
            )
            self.archive = WindowArchive(m, self.index)
        else:
            self.index = None
            self.archive = None
        self.bank = bank
        self.depth = depth
        self.cap = cap
        self.batch = batch
        self.time = m.time
        b, t = self.batch, self.time
        dev = m.embed_host.weight.device
        self.lengths = (t,) * b
        self.spans = torch.tensor(
            np.c_[np.arange(b) * t, np.full(b, t)], dtype=torch.long, device=dev
        )
        self.local_rows = torch.as_tensor(
            (np.arange(b)[:, None] * t + np.arange(t - 1)).flatten(),
            dtype=torch.long,
            device=dev,
        )
        tq = np.arange(t)
        counts = np.tile(tq, b)
        owner = np.repeat(np.arange(b * t), counts)
        row = np.concatenate([np.arange(k) for k in range(t)] * b) + np.repeat(
            np.arange(b) * (t - 1), t * (t - 1) // 2
        )
        self.pair_owner = torch.as_tensor(owner, dtype=torch.long, device=dev)
        self.pair_row = torch.as_tensor(row, dtype=torch.long, device=dev)
        self._covers = {}
        if bank:
            self._build_intervals(a)

    def _cover(self, lo_leaf, hi_leaf):
        key = (lo_leaf, hi_leaf)
        if key not in self._covers:
            self._covers[key] = cover(0, lo_leaf, self.leaves) + cover(
                hi_leaf, self.leaves, self.leaves
            )
        return self._covers[key]

    def _build_intervals(self, a):
        """Depth-`depth` matched intervals in bank order, per distinct code."""
        c, m = _pack(a, self.depth)
        flat = c.ravel()
        gids = (
            (np.arange(len(flat)) // m) * (a.shape[1] - 1)
            + (np.arange(len(flat)) % m)
            + self.depth
            - 1
        )
        bank = np.asarray(self.archive.index.inverse_order)[gids]
        order = np.lexsort((bank, flat))
        flat_sorted = flat[order]
        bank_sorted = bank[order]
        starts = np.r_[
            0, np.flatnonzero(flat_sorted[1:] != flat_sorted[:-1]) + 1, len(flat)
        ]
        self.core_codes = flat_sorted[starts[:-1]]
        self.core_lo = bank_sorted[starts[:-1]]
        self.core_hi = bank_sorted[starts[1:] - 1] + 1

    def bind(self, windows):
        b, t = self.batch, self.time
        if windows.shape != (b, t + 1) or windows.dtype != np.uint8:
            raise ValueError("expected complete byte windows")
        dev = self.local_rows.device
        x = windows[:, :-1].copy()
        ids = torch.as_tensor(x.ravel(), dtype=torch.long, device=dev)
        y = torch.as_tensor(windows[:, 1:].ravel(), dtype=torch.long, device=dev)
        _, _, m_ex, h_ex = self.counts.bind_stats(windows)
        Q = b * t
        if not self.bank:
            # Skeleton-only stages carry no core/cover machinery: those
            # tensors exist but stay empty, and the order-2 arm is refused
            # by construction (run spawns it only with a bank).
            return dict(
                ids=ids,
                targets=y,
                spans=self.spans,
                lengths=self.lengths,
                rows=self.rows,
                local_rows=self.local_rows,
                local_succ=ids[self.local_rows + 1],
                pair_owner=self.pair_owner,
                pair_row=self.pair_row,
                m=torch.as_tensor(m_ex.reshape(Q, -1), dtype=torch.float32, device=dev),
                h=torch.as_tensor(h_ex.reshape(Q, self.depth + 1, ALPHA), device=dev),
                core=torch.zeros((Q, 0), dtype=torch.long, device=dev),
                core_valid=torch.zeros(Q, dtype=torch.bool, device=dev),
                cover_owner=torch.zeros(0, dtype=torch.long, device=dev),
                cover_lv=torch.zeros(0, dtype=torch.long, device=dev),
                cover_idx=torch.zeros(0, dtype=torch.long, device=dev),
            )
        full = self._cover(0, 0)
        core = np.full((Q, self.cap), -1, np.int64)
        valid = np.zeros(Q, bool)
        cov = [full] * Q
        c, mc = _pack(windows, self.depth)
        flat = c.ravel()
        pos = np.minimum(
            np.searchsorted(self.core_codes, flat), len(self.core_codes) - 1
        )
        hit = self.core_codes[pos] == flat
        lo = np.where(hit, self.core_lo[pos], 0) // LEAF * LEAF
        hi = -(-np.where(hit, self.core_hi[pos], 0) // LEAF) * LEAF
        ok = hit & (hi - lo > 0) & (hi - lo <= self.cap)
        qidx = (
            (np.arange(len(flat)) // mc) * t
            + np.arange(len(flat)) % mc
            + self.depth
            - 1
        )
        for f in np.nonzero(ok)[0]:
            q = int(qidx[f])
            s, e = int(lo[f]), int(hi[f])
            core[q, : e - s] = np.arange(s, e)
            valid[q] = True
            cov[q] = self._cover(s // LEAF, e // LEAF)
        cover_owner, cover_lv, cover_idx = [], [], []
        for q, nodes in enumerate(cov):
            for lv, i in nodes:
                cover_owner.append(q)
                cover_lv.append(lv)
                cover_idx.append(i)
        return dict(
            ids=ids,
            targets=y,
            spans=self.spans,
            lengths=self.lengths,
            rows=self.rows,
            local_rows=self.local_rows,
            local_succ=ids[self.local_rows + 1],
            pair_owner=self.pair_owner,
            pair_row=self.pair_row,
            m=torch.as_tensor(m_ex.reshape(Q, -1), dtype=torch.float32, device=dev),
            h=torch.as_tensor(h_ex.reshape(Q, self.depth + 1, ALPHA), device=dev),
            core=torch.as_tensor(core, dtype=torch.long, device=dev),
            core_valid=torch.as_tensor(valid, device=dev),
            cover_owner=torch.as_tensor(cover_owner, dtype=torch.long, device=dev),
            cover_lv=torch.as_tensor(cover_lv, dtype=torch.long, device=dev),
            cover_idx=torch.as_tensor(cover_idx, dtype=torch.long, device=dev),
        )

    def dataset(self, docs, shuffle):
        if docs.ndim != 2 or docs.dtype != np.uint8 or (docs.shape[1] - 1) % self.time:
            raise ValueError("invalid raw document length")
        windows = np.lib.stride_tricks.sliding_window_view(docs, self.time + 1, axis=1)[
            :, :: self.time, :
        ].reshape(-1, self.time + 1)
        if len(windows) % self.batch:
            raise ValueError("incomplete registered batch")
        order = (
            np.arange(len(windows))
            if not shuffle
            else np.random.default_rng(5750).permutation(len(windows))
        )
        return [
            self.bind(windows[order[i : i + self.batch]])
            for i in range(0, len(order), self.batch)
        ]
