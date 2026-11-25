"""PT: one frozen short-document ERROR packet, three isolated correction reads, diagnostics off their clocks."""

from dataclasses import dataclass
from itertools import permutations
from math import isfinite
from time import perf_counter
import numpy as np
import torch
from . import paired_correction as pc, correction_prefix as broad
from .endpoint import frozen_prefill
from .panels import SEEDS, T
from .surprise_fit import _labels

BRANCHES = ("MARG", "PAIR", "PREFIX")
ORDERS = tuple(permutations(BRANCHES))
ARMS = (*pc.ARMS, "PREFIX")
TV_TOL = 1e-12


def order(seed, rank):
    if type(seed) is not int or seed not in SEEDS or type(rank) is not int or rank < 0:
        raise ValueError("unknown PT seed or reserved rank")
    return ORDERS[(SEEDS.index(seed) + rank) % len(ORDERS)]


@dataclass(frozen=True, slots=True)
class _Rows:
    base: torch.Tensor
    active: torch.Tensor
    row: torch.Tensor
    symbol: torch.Tensor


def _marginals(sh):
    m = pc._marginal(sh.counts, sh.query_tape)
    coords = _Rows(
        sh.lp,
        torch.from_numpy(m.active),
        torch.from_numpy(m.row),
        torch.from_numpy(m.symbol),
    )
    return m, coords, torch.from_numpy(m.f0[:, None])


def _plus(x, f):
    return torch.cat((x, torch.from_numpy(f[:, None])), dim=1)


def _tilt(coords, x, a, b):
    return pc.eh._tilt(
        coords,
        x,
        pc.u.checked_vector(pc.DIMS, "PAIR" if a in ("PREFIX", "SWAP") else a, b),
    )


@torch.inference_mode()
def _chosen(sh, a, b):
    # Each arm starts with the same ERROR packet and exactly one coefficient vector.
    tic = perf_counter()
    m, coords, x = _marginals(sh)
    w = dict(m.work, marginal_feature_bytes=x.nbytes)
    clock = dict(marginal_s=perf_counter() - tic)
    obs, q = sh.shared.observed, sh.query_tape
    if a == "PAIR":
        tic = perf_counter()
        pairs = pc.ix.true_pairs(obs, q)
        clock["index_s"] = perf_counter() - tic
        tic = perf_counter()
        f, _, n = pc._conditional(m, sh.counts, pairs)
        x = _plus(x, f)
        w.update(
            true_nonmarginal_rows=n, **{"pair_" + k: v for k, v in pairs.work.items()}
        )
        clock["conditional_s"] = perf_counter() - tic
    elif a == "PREFIX":
        tic = perf_counter()
        rate, pw = broad.distribution(obs, q, sh.counts, m)
        clock["prefix_scan_s"] = perf_counter() - tic
        tic = perf_counter()
        x = _plus(x, m.alpha * (rate - m.v))
        w.update({"prefix_" + k: v for k, v in pw.items()})
        clock["conditional_s"] = perf_counter() - tic
    elif a != "MARG":
        raise ValueError("unknown PT chosen branch")
    w[a + "_feature_bytes"] = x.nbytes
    tic = perf_counter()
    lp = _tilt(coords, x, a, b)
    clock["response_s"] = perf_counter() - tic
    return lp, w, clock


@torch.inference_mode()
def _controls(sh, beta):
    # Independent rebuilds only for response controls and law/support comparisons. No second prefill.
    obs, q, tab = sh.shared.observed, sh.query_tape, sh.counts
    m, coords, x = _marginals(sh)
    true = pc.ix.true_pairs(obs, q)
    shuf = pc.ix.switched_pairs(obs, q)
    _, c, n = pc._conditional(m, tab, true)
    scol, v, sn = pc._conditional(m, tab, shuf)
    rate, pw = broad.distribution(obs, q, tab, m)
    tv = np.zeros(len(obs))
    np.add.at(tv, m.row, 0.5 * np.abs(c / m.h[m.row] - rate))
    if not np.isfinite(tv).all() or (tv > 1 + 1e-12).any():
        raise ValueError("invalid comparison between queried probability laws")
    x = _plus(x, scol)
    lp = pc._anchors(sh, beta["U"], beta["HISTORY"])
    lp["SHUF"] = _tilt(coords, x, "SHUF", beta["SHUF"])
    lp["SWAP"] = _tilt(coords, x, "SWAP", beta["PAIR"])
    diff = c - v
    w = dict(
        m.work,
        **{"pair_" + k: y for k, y in true.work.items()},
        **{"shuf_" + k: y for k, y in shuf.work.items()},
        **{"prefix_" + k: y for k, y in pw.items()},
        true_nonmarginal_rows=n,
        shuf_nonmarginal_rows=sn,
        changed_conditional_rows=len(np.unique(m.row[diff != 0])),
        changed_conditional_uses=int(np.maximum(diff, 0).sum()),
        prefix_different_law_rows=int(np.count_nonzero(tv > TV_TOL)),
        shuf_feature_bytes=x.nbytes,
    )
    return lp, w, float(tv.sum())


def _validate(obs, beta):
    if (
        type(obs) is not bytes
        or not 1 <= len(obs) < T
        or type(beta) is not dict
        or set(beta) != {"U", "J", "HISTORY", "ERROR", *pc.DIMS}
    ):
        raise ValueError("need one short observed PT prefix and seven fixed readouts")


@torch.inference_mode()
def _answers(head, obs, h, z, beta, seed, rank, writer, *, tic, pre_s):
    seq = order(seed, rank)
    sh = pc._shared_error(head, obs, h, z, beta["J"], beta["ERROR"], writer)
    s = dict(sh.seconds, shared_ERROR_s=perf_counter() - tic, prefill_s=pre_s)
    chosen = {}
    cw = {}
    for a in seq:
        start = perf_counter()
        chosen[a], cw[a], clock = _chosen(sh, a, beta["PAIR" if a == "PREFIX" else a])
        s[a + "_branch_s"] = perf_counter() - start
        s.update({a + "_" + k: v for k, v in clock.items()})
    start = perf_counter()
    lp, more, tvsum = _controls(sh, beta)
    lp.update(chosen)
    s["comparison_only_s"] = perf_counter() - start
    if any(v != more[k] for w in cw.values() for k, v in w.items() if k in more):
        raise ValueError("different correction support on chosen and comparison paths")
    return (
        lp,
        sh.shared.packet,
        dict(
            targets=len(obs),
            host_prefills=1,
            branch_order=list(seq),
            work=dict(sh.work, **more),
            chosen_work=cw,
            seconds=s,
            conditional_tv_sum=tvsum,
        ),
    )


@torch.inference_mode()
def answers(host, head, observed, beta, seed, rank, writer, *, width=256):
    _validate(observed, beta)
    order(seed, rank)
    tic = perf_counter()
    h, z, pre = frozen_prefill(host, observed, width)
    return _answers(head, observed, h, z, beta, seed, rank, writer, tic=tic, pre_s=pre)


@torch.inference_mode()
def predict(host, head, observed, jb, eb, arm, coefficient, writer, *, width=256):
    if type(observed) is not bytes or not 1 <= len(observed) < T or arm not in BRANCHES:
        raise ValueError("one short input, one chosen PT branch")
    h, z, _ = frozen_prefill(host, observed, width)
    sh = pc._shared_error(head, observed, h, z, jb, eb, writer)
    return _chosen(sh, arm, coefficient)[0]


@torch.inference_mode()
def document(host, head, word, beta, seed, rank, writer, *, width=256):
    if type(word) is not bytes or not 2 <= len(word) <= T:
        raise ValueError("one whole short PT document required")
    lp, p, row = answers(host, head, word[:-1], beta, seed, rank, writer, width=width)
    y, right = _labels(p, word[1:])
    idx = torch.arange(len(y))
    loss = {a: -float(lp[a][idx, y].sum()) for a in ARMS}
    if not all(isfinite(v) and v >= 0 for v in loss.values()):
        raise ValueError("nonfinite PT loss")
    return dict(row, nll_sums=loss, candidate_correct=int(right.sum()))
