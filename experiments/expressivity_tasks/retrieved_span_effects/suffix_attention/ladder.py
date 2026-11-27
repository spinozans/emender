"""Reference laws for attention over a byte suffix index (XSA_ATTENTION_LAW.md).

Order 0 is the exact count skeleton: depth-only scores make the whole-corpus
law computable from suffix counts alone, exactly, at any archive size, with no
learned key anywhere. Order 2 is Gaussian moment closure over a complete
dyadic tree of fixed leaves, with the value tilt and per-node radius
certificates; the matched core is scored exactly per row and the complement
is covered by moment nodes.

Plain numpy + torch autograd only. Triton ports follow after these pass
against brute force. Nothing here touches natural data, and no learned bank is
ever reused across updates.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ALPHA = 256
MAX_DEPTH = 8  # one int64 code; deeper matches belong to the exact core


def _pack(a, depth):
    """int64 codes of the `depth` bytes ending at each valid row position."""
    n, w = a.shape
    m = w - depth  # window starts j in [0, w-1-depth]; row position i = j + depth
    if m <= 0:
        raise ValueError("archive narrower than code depth")
    c = a[:, 0:m].astype(np.int64)
    for k in range(1, depth):
        c = c * ALPHA + a[:, k : k + m]
    return c, m


@dataclass
class _Level:
    codes: np.ndarray  # sorted distinct depth-d context codes
    counts: np.ndarray  # occurrences per code
    row_starts: np.ndarray  # boundaries into `rows` per code
    rows: np.ndarray  # archive row ids grouped by code
    keys: np.ndarray  # sorted code_id*256 + successor byte
    key_counts: np.ndarray
    key_starts: np.ndarray  # boundaries into `keys` per code


def _level_arrays(a, d, rowids=True):
    """Per-depth sorted code table for one archive; row ids optional."""
    n, w = a.shape
    c, m = _pack(a, d)
    succ = a[:, d:].ravel().astype(np.int64)
    flat = c.ravel()
    codes, inv = np.unique(flat, return_inverse=True)
    counts = np.bincount(inv, minlength=len(codes))
    starts = np.concatenate(([0], np.cumsum(counts)))
    if rowids:
        grouped = np.argsort(inv, kind="stable")
        r_idx, j = grouped // m, grouped % m
        gids = r_idx * (w - 1) + j + d - 1
    else:
        gids = np.empty(0, np.int64)
    keys, key_counts = np.unique(inv * ALPHA + succ, return_counts=True)
    key_starts = np.searchsorted(keys, np.arange(len(codes) + 1) * ALPHA)
    return codes, counts, starts, gids, keys, key_counts, key_starts


def _json_put(path, obj):
    import json as _json

    with open(path, "w") as f:
        f.write(_json.dumps(obj, sort_keys=True) + "\n")


def _json_get(path):
    import json as _json

    with open(path) as f:
        return _json.loads(f.read())


# injected names used by CountIndex.save/load (kept local to avoid growing
# the module's public surface)
put_json = _json_put
get_json = _json_get


class CountIndex:
    """Static byte-suffix counts over a rectangular archive.

    A row is an eligible endpoint of one archive document row `a` (width w):
    position i in [1, w-1] has context a[:i] and real successor a[i]. The
    depth-d code of that row is the d bytes a[i-d:i]. The live query at
    target t uses the d bytes x[t-d:t]. Nothing here depends on weights.
    """

    def __init__(self, archive, depth=MAX_DEPTH, rowids=True):
        a = np.ascontiguousarray(archive)
        if a.ndim != 2 or a.dtype != np.uint8 or a.shape[1] < 2 or not len(a):
            raise ValueError("rectangular uint8 archive (documents x bytes)")
        if not 1 <= depth <= MAX_DEPTH:
            raise ValueError("code depth must fit one int64")
        n, w = a.shape
        self.archive, self.depth, self.rows = a, depth, n * (w - 1)
        self.root = np.bincount(a[:, 1:].ravel(), minlength=ALPHA).astype(np.int64)
        self.levels = [
            _Level(*_level_arrays(a, d, rowids)) for d in range(1, depth + 1)
        ]
        self._build_dense()

    def _build_dense(self):
        # Static dense successor histograms for shallow depths: bind-time work is
        # then a pure gather instead of a scatter over long posting runs.
        self.dense = {}
        for d in (1, 2):
            lev = self.levels[d - 1]
            if len(lev.codes) * ALPHA <= 2**26:
                table = np.bincount(
                    lev.keys,
                    weights=lev.key_counts,
                    minlength=len(lev.codes) * ALPHA,
                ).reshape(len(lev.codes), ALPHA)
                self.dense[d] = table.astype(np.int64)

    def save(self, path, meta=None):
        """Write every level plus a binding receipt to `path` (a directory).

        The cache binds to the archive through `meta` (at minimum its sha256);
        loading verifies the binding so a stale cache can never be used.
        """
        path = Path(path)
        if path.exists() and any(path.iterdir()):
            raise RuntimeError("cache directory not empty")
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        for d, lev in enumerate(self.levels, 1):
            np.savez(
                path / f"level-{d}.npz",
                codes=lev.codes,
                counts=lev.counts,
                starts=lev.row_starts,
                rows=lev.rows,
                keys=lev.keys,
                key_counts=lev.key_counts,
                key_starts=lev.key_starts,
            )
        put_json(
            path / "meta.json",
            dict(
                rows=self.rows,
                depth=self.depth,
                root=self.root.tolist(),
                levels=len(self.levels),
                **(meta or {}),
            ),
        )

    @classmethod
    def load(cls, path, meta=None, rowids=True):
        """Load a saved cache; refuse any binding mismatch.

        `meta` must contain the same keys written at save time (typically
        dict(archive_sha=...)); every value must match exactly.
        """
        path = Path(path)
        m = get_json(path / "meta.json")
        for k, v in (meta or {}).items():
            if m.get(k) != v:
                raise ValueError(f"cache binding mismatch on {k}")
        if not 1 <= m["depth"] <= MAX_DEPTH or m["levels"] != m["depth"]:
            raise ValueError("invalid cache geometry")
        self = cls.__new__(cls)
        self.archive = None
        self.depth = m["depth"]
        self.rows = m["rows"]
        self.root = np.asarray(m["root"], np.int64)
        self.levels = []
        for d in range(1, self.depth + 1):
            z = np.load(path / f"level-{d}.npz")
            lev = _Level(
                z["codes"],
                z["counts"],
                z["starts"],
                z["rows"],
                z["keys"],
                z["key_counts"],
                z["key_starts"],
            )
            if not rowids and len(lev.rows):
                raise ValueError("cache carries row ids")
            self.levels.append(lev)
        self._build_dense()
        return self

    def stats(self, x):
        """Exact-depth classes for every target of live windows/documents.

        Returns (counts, hists, m, h): for target position t in [1, w-1] and
        depth d in [0, depth], counts[t,d] is the number of archive rows
        matching at least depth d, hists[t,d,v] the number of those rows
        whose successor is v, and (m,h) the telescoped exact-depth classes
        m_d = c_d - c_{d+1}, h_d = hist_d - hist_{d+1}.
        """
        a = np.asarray(x)
        n, w = a.shape
        counts = np.zeros((n, w - 1, self.depth + 1), np.int64)
        hists = np.zeros((n, w - 1, self.depth + 1, ALPHA), np.int64)
        counts[:, :, 0] = self.rows
        hists[:, :, 0] = self.root
        for d in range(1, self.depth + 1):
            lev = self.levels[d - 1]
            c, m = _pack(a, d)
            flat = c.ravel()
            pos = np.minimum(np.searchsorted(lev.codes, flat), len(lev.codes) - 1)
            hit = lev.codes[pos] == flat
            cnt = np.where(hit, lev.counts[pos], 0).reshape(n, m)
            counts[:, d - 1 : w - 1, d] = cnt
            if hit.any():
                sel = np.nonzero(hit)[0]
                starts = lev.key_starts[pos[sel]]
                lens = lev.key_starts[pos[sel] + 1] - starts
                base = np.repeat(starts, lens)
                off = np.arange(int(lens.sum())) - np.repeat(
                    np.cumsum(lens) - lens, lens
                )
                kv = lev.keys[base + off]
                freq = lev.key_counts[base + off]
                qi = np.repeat(sel, lens)
                table = np.zeros((n * m, ALPHA), np.int64)
                np.add.at(table, (qi, kv % ALPHA), freq)
                hists[:, d - 1 : w - 1, d] = table.reshape(n, m, ALPHA)
        m = counts.copy()
        h = hists.copy()
        m[:, :, :-1] -= counts[:, :, 1:]
        h[:, :, :-1] -= hists[:, :, 1:]
        return counts, hists, m, h

    def bind_stats(self, x):
        """Fast per-batch version of `stats` for live reset windows.

        Shallow depths use the static dense tables (gather); deeper depths
        deduplicate the batch's codes and scatter only their posting runs.
        Returns the same four arrays as `stats`.
        """
        a = np.asarray(x)
        n, w = a.shape
        counts = np.zeros((n, w - 1, self.depth + 1), np.int64)
        hists = np.zeros((n, w - 1, self.depth + 1, ALPHA), np.float32)
        counts[:, :, 0] = self.rows
        hists[:, :, 0] = self.root
        for d in range(1, self.depth + 1):
            lev = self.levels[d - 1]
            c, m = _pack(a, d)
            flat = c.ravel()
            pos = np.minimum(np.searchsorted(lev.codes, flat), len(lev.codes) - 1)
            hit = lev.codes[pos] == flat
            counts[:, d - 1 : w - 1, d] = np.where(hit, lev.counts[pos], 0).reshape(
                n, m
            )
            if d in self.dense:
                h = self.dense[d][np.where(hit, pos, 0)].reshape(n, m, ALPHA)
                hists[:, d - 1 : w - 1, d] = np.where(
                    hit.reshape(n, m)[:, :, None], h, 0
                )
            else:
                ucodes, uinv = np.unique(flat[hit], return_inverse=True)
                upos = np.searchsorted(lev.codes, ucodes)
                if not (len(upos) == len(ucodes) and (lev.codes[upos] == ucodes).all()):
                    raise ValueError("unique depth code lookup failed")
                starts = lev.key_starts[upos]
                lens = lev.key_starts[upos + 1] - starts
                base = np.repeat(starts, lens)
                off = np.arange(int(lens.sum())) - np.repeat(
                    np.cumsum(lens) - lens, lens
                )
                kv = lev.keys[base + off]
                freq = lev.key_counts[base + off]
                table = np.bincount(
                    np.repeat(np.arange(len(ucodes)), lens) * ALPHA + kv % ALPHA,
                    weights=freq,
                    minlength=len(ucodes) * ALPHA,
                ).reshape(len(ucodes), ALPHA)
                h = np.zeros((len(flat), ALPHA), np.float32)
                h[hit] = table[uinv].astype(np.float32)
                hists[:, d - 1 : w - 1, d] = h.reshape(n, m, ALPHA)
        m_ex = counts.copy()
        h_ex = hists.copy()
        m_ex[:, :, :-1] -= counts[:, :, 1:]
        h_ex[:, :, :-1] -= hists[:, :, 1:]
        return counts, hists, m_ex, h_ex

    def matched_rows(self, row, t, d):
        """Archive row ids whose depth-d code equals x[row, t-d:t], or []."""
        if d < 1 or d > self.depth or t < d:
            return np.empty(0, np.int64)
        lev = self.levels[d - 1]
        if self.archive is None or not len(lev.rows):
            raise RuntimeError("row ids were not built for this index")
        code = _pack(self.archive[row : row + 1], d)[0].reshape(-1)[t - d]
        p = int(np.searchsorted(lev.codes, code))
        if p == len(lev.codes) or lev.codes[p] != code:
            return np.empty(0, np.int64)
        s = lev.row_starts[p]
        return lev.rows[s : s + lev.counts[p]]


class SkeletonSource(torch.nn.Module):
    """Order-0 corpus source: exact depth-scored attention over the counts.

    Its only learned parameters are one corpus log-prior (the per-row prior is
    b = exp(log_pi)/rows, so the whole corpus carries prior mass exp(log_pi))
    and the depth weights g(0..depth).
    """

    def __init__(self, rows, depth=MAX_DEPTH):
        super().__init__()
        self.rows, self.depth = rows, depth
        self.log_pi = torch.nn.Parameter(torch.tensor(-1.0))
        self.g = torch.nn.Parameter(torch.linspace(-2.0, 2.0, depth + 1))

    def forward(self, m, h):
        if m.ndim != 2 or h.ndim != 3 or m.shape[1] != self.depth + 1:
            raise ValueError("per-query exact-depth classes expected")
        w = torch.exp(self.g)
        b = torch.exp(self.log_pi) / self.rows
        z = b * torch.einsum("qd,d->q", m, w)
        u = b * torch.einsum("d,qdv->qv", w, h)
        return torch.log(z), u / z[:, None]


def cover(lo, hi, leaves):
    """Canonical aligned dyadic cover of [lo, hi) in leaf units.

    Nodes are (level, index) with level 0 a single leaf; a node of level s
    spans 2**s leaves starting at a multiple of 2**s.
    """
    if not 0 <= lo <= hi <= leaves or leaves < 1 or leaves & (leaves - 1):
        raise ValueError("aligned dyadic cover bounds")
    out = []
    p = lo
    while p < hi:
        k = 1
        while p % (2 * k) == 0 and p + 2 * k <= hi and 2 * k <= leaves:
            k *= 2
        out.append((int(np.log2(k)), p // k))
        p += k
    return out


def mix_sources(parts):
    """Joint law over sources: (log Z_s, read_s) -> (log Z, pooled read)."""
    logs = torch.stack([z for z, _ in parts])
    vals = torch.stack([r for _, r in parts])
    peak = logs.max()
    w = torch.exp(logs - peak)
    z = w.sum()
    return peak + torch.log(z), (w[:, None] * vals).sum(0) / z


def moment_statistics(k, v, leaf):
    """Complete dyadic tree of order-2 statistics over fixed leaves.

    Pure function of the current keys and values: per node the count, mean
    key, mean value, within-node key scatter S, value–key cross scatter T,
    and the exact leaf radius (merged conservatively upward). Parents merge
    by the parallel mean/covariance updates; every level is exact.
    """
    rows, d, dv = k.shape[0], k.shape[1], v.shape[1]
    if rows % leaf or rows // leaf & (rows // leaf - 1):
        raise ValueError("complete dyadic tree geometry required")
    L = rows // leaf
    kc = k.reshape(L, leaf, d)
    vc = v.reshape(L, leaf, dv)
    dev = kc - kc.mean(1, keepdim=True)
    devv = vc - vc.mean(1, keepdim=True)
    level = dict(
        n=torch.full((L,), float(leaf), dtype=k.dtype, device=k.device),
        kbar=kc.mean(1),
        vbar=vc.mean(1),
        S=torch.einsum("lid,lif->ldf", dev, dev),
        T=torch.einsum("lie,lif->lef", devv, dev),
        R=dev.norm(dim=2).amax(1),
    )
    levels = [level]
    while len(levels[-1]["n"]) > 1:
        cur = levels[-1]
        na, nb = cur["n"][0::2], cur["n"][1::2]
        n = na + nb
        ka, kb = cur["kbar"][0::2], cur["kbar"][1::2]
        kbar = (na[:, None] * ka + nb[:, None] * kb) / n[:, None]
        va, vb = cur["vbar"][0::2], cur["vbar"][1::2]
        vbar = (na[:, None] * va + nb[:, None] * vb) / n[:, None]
        dk = ka - kb
        dvv = va - vb
        coef = (na * nb / n)[:, None, None]
        levels.append(
            dict(
                n=n,
                kbar=kbar,
                vbar=vbar,
                S=cur["S"][0::2]
                + cur["S"][1::2]
                + coef * (dk[:, :, None] * dk[:, None, :]),
                T=cur["T"][0::2]
                + cur["T"][1::2]
                + coef * (dvv[:, :, None] * dk[:, None, :]),
                R=torch.maximum(
                    cur["R"][0::2] + (ka - kbar).norm(dim=1),
                    cur["R"][1::2] + (kb - kbar).norm(dim=1),
                ),
            )
        )
    return levels


class MomentTree(torch.nn.Module):
    """Order-2 Gaussian closure over a complete dyadic tree of fixed leaves.

    Statistics are exact, differentiable functions of the current keys and
    values. Parents merge by the parallel mean/covariance updates; the merged
    radius is a valid (conservative) upper bound. Rows must be a power-of-two
    multiple of the leaf size, in the bank's own row order (matched suffix
    intervals are contiguous in that order; cores are extended to leaf
    alignment so the complement is exactly coverable).
    """

    def __init__(self, rows, leaf, d, dv):
        super().__init__()
        if rows % leaf or rows // leaf & (rows // leaf - 1):
            raise ValueError("complete dyadic tree geometry required")
        self.rows, self.leaf, self.d, self.dv = rows, leaf, d, dv
        self.log_pi = torch.nn.Parameter(torch.tensor(-1.0))

    def statistics(self, k, v):
        if k.shape != (self.rows, self.d) or v.shape != (self.rows, self.dv):
            raise ValueError("bank keys/values shape mismatch")
        return moment_statistics(k, v, self.leaf)

    def _node(self, levels, node):
        lv, idx = node
        st = levels[lv]
        return (
            st["n"][idx],
            st["kbar"][idx],
            st["vbar"][idx],
            st["S"][idx] / st["n"][idx],
            st["T"][idx] / st["n"][idx],
        )

    def read(self, q, levels, nodes, core_k=None, core_v=None, extra=()):
        """One query's joint read over moment nodes, an exact core, and any
        already-priced local/no-read sources. Returns (log Z, read)."""
        t = q / float(self.d) ** 0.5
        b = self.log_pi - torch.log(torch.tensor(float(self.rows), dtype=q.dtype))
        logs = []
        vals = []
        for node in nodes:
            n, kbar, vbar, cov, cross = self._node(levels, node)
            lm = b + torch.log(n) + t @ kbar + 0.5 * t @ (cov @ t)
            logs.append(lm)
            vals.append(vbar + cross @ t)
        if core_k is not None and len(core_k):
            logs.extend(list(core_k @ t + b))
            vals.extend(list(core_v))
        if extra:
            for z, r in extra:
                logs.append(z)
                vals.append(r)
        stacked = torch.stack(logs)
        peak = stacked.max()
        w = torch.exp(stacked - peak)
        z = w.sum()
        read = 0
        for wi, r in zip(w.unbind(), vals, strict=True):
            read = read + wi * r
        return peak + torch.log(z), read / z

    def bounds(self, q, levels, node):
        """Rigorous two-sided bound on the node's log mass (XSA_ATTENTION_LAW
        section 4): log Z_u in log(b n) + t.kbar +/- R ||t||."""
        t = q / float(self.d) ** 0.5
        lv, idx = node
        st = levels[lv]
        b = self.log_pi - torch.log(torch.tensor(float(self.rows), dtype=q.dtype))
        centre = b + torch.log(st["n"][idx]) + t @ st["kbar"][idx]
        slack = st["R"][idx] * t.norm()
        return centre - slack, centre + slack


def dense(q, k, v, log_b):
    """Brute-force dense attention over explicit rows."""
    s = (k @ q) / float(k.shape[1]) ** 0.5 + log_b
    peak = s.max()
    w = torch.exp(s - peak)
    z = w.sum()
    return peak + torch.log(z), (w[:, None] * v).sum(0) / z
