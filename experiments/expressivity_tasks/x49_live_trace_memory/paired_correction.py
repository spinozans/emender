"""PC: read actual alternatives to the current bare-host guess at an earlier matching context.

All state is local to this call. MARG reads only frozen ERROR's marginals. PAIR also
queries the true sparse confusion edges. Neither needs the margin-preserving control.
"""

from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
from . import (
    correction_index as ix,
    error_history_index as old,
    error_history as eh,
    error_transfer_score as et,
    surprise as u,
)
from .endpoint import frozen_prefill
from .literal import check_window

DIMS = {"MARG": 1, "PAIR": 2, "SHUF": 2}
ANCHORS = ("B", "U", "J", "HISTORY", "ERROR")
ARMS = (*ANCHORS, *DIMS, "SWAP")


class Packet(eh.Packet):
    __slots__ = ()

    def features(self, arm):
        if arm in ("MARG", "PAIR") and self.x.shape[1] >= DIMS[arm]:
            return self.x[:, : DIMS[arm]]
        if arm in ("SHUF", "SWAP") and self.x.shape[1] == 3:
            return self.x[:, [0, 2]]
        raise ValueError("unavailable paired-correction features")


@dataclass(slots=True)
class _Law:
    # Sparse coordinate set: each active query and its possible alternatives from M.
    position: np.ndarray
    row: np.ndarray
    symbol: np.ndarray
    active: np.ndarray
    h: np.ndarray  # matching wrong-guess count for EVERY frontier
    alt: np.ndarray  # M_b excluding the current guessed byte, over returned coordinates
    total: np.ndarray  # total missed alternatives over returned coordinates
    alpha: np.ndarray
    v: np.ndarray
    f0: np.ndarray
    work: dict


def _marginal(tab, query):
    q = np.frombuffer(query, np.uint8)
    n = len(q)
    row = (tab.position >> 8).astype(np.int64)
    sym = (tab.position & 255).astype(np.int64)
    c = tab.counts.astype(np.int64)
    isq = sym == q[row]
    at = row[isq]
    h = np.zeros(n, dtype=np.int64)
    correct = h.copy()
    mq = h.copy()
    m = h.copy()
    h[at], correct[at], mq[at] = c[isq, 4], c[isq, 0] - c[isq, 3], c[isq, 3]
    np.add.at(m, row, c[:, 3])
    den = m - mq
    if np.any(correct < 0) or np.any(den < h):
        raise ValueError("impossible ERROR margins at the current guess")
    active = np.flatnonzero(h)
    keep = (h[row] > 0) & (isq | (c[:, 3] > 0))
    alt = np.where(isq[keep], 0, c[keep, 3])
    row, sym, isq = row[keep], sym[keep], isq[keep]
    total = den[row]
    v = alt / total
    alpha = h[row] / (1 + h[row] + correct[row])
    f0 = alpha * (v - isq)
    return _Law(
        tab.position[keep],
        row,
        sym,
        active,
        h,
        alt,
        total,
        alpha,
        v,
        f0,
        dict(
            marginal_entries=len(row),
            matched_query_rows=len(active),
            matched_mistake_uses=int(h.sum()),
            matched_correct_uses_on_active=int(correct[active].sum()),
            nonquery_alternative_uses_on_active=int(den[active].sum()),
        ),
    )


def _conditional(law, tab, pairs):
    if not all(
        np.array_equal(getattr(tab, k), getattr(pairs, k))
        for k in ("latest", "depth", "size")
    ):
        raise ValueError("corrections selected a different literal history")
    # Every retrieved edge is a genuine (or validly substituted) missed alternative.
    loc = np.searchsorted(law.position, pairs.position)
    if np.any(loc >= len(law.position)) or not np.array_equal(
        law.position[loc], pairs.position
    ):
        raise ValueError("correction outside marginal support")
    count = np.zeros(len(law.row), dtype=np.int64)
    count[loc] = pairs.count
    den = np.zeros_like(law.h)
    np.add.at(den, law.row, count)
    if (
        not np.array_equal(den, law.h)
        or np.any(count > law.alt)
        or len(law.active) != pairs.work["active_rows"]
    ):
        raise ValueError("correction changed a margin or introduced a diagonal edge")
    h = law.h[law.row]
    different = (
        count * law.total != h * law.alt
    )  # exact integer support, not FP comparison
    f1 = law.alpha * (count / h - law.v)
    return f1, count, len(np.unique(law.row[different]))


@torch.inference_mode()
def response(packet, arm, coefficients):
    b = u.checked_vector(DIMS, "PAIR" if arm == "SWAP" else arm, coefficients)
    return eh._tilt(packet, packet.features(arm), b)


def _queries(z):
    old_guess = old.guesses(
        z
    )  # earlier arrivals, one-position shift, verifies all host rows
    return old_guess, old_guess[1:] + bytes([int(z[-1].argmax())])


@torch.inference_mode()
def fields(head, observed, h, z, jb, eb, writer, *, mode, anchors=None):
    """One actual frozen ERROR response. Optional comparison anchors are (U,HISTORY) coefficients."""
    check_window(observed, 4096)
    if mode not in ("MARG", "PAIR", "panel") or (
        anchors is not None and (mode != "panel" or len(anchors) != 2)
    ):
        raise ValueError("unknown PC mode or comparison anchors on chosen path")
    start = perf_counter()
    base, w, s = et._shared(head, observed, h, z, jb, writer)
    tic = perf_counter()
    g, q = _queries(base.z)
    lit, nt = base.literal.numpy().tobytes(), base.neural.numpy().tobytes()
    w.update(
        guesses_written=len(observed) - 1,
        wrongs_written=sum(a != c for a, c in zip(g[1:], observed[1:], strict=True)),
        guess_tape_bytes=len(g),
        query_tape_bytes=len(q),
    )
    s["guess_tape_s"] = perf_counter() - tic
    tic = perf_counter()
    tab = old.true_counts(observed, lit, nt, g)
    eh._geometry(tab, base.packet)
    ew = {}
    ep = et._features(base, tab, None, ew)
    err = eh.response(ep, "ERROR", eb)
    s["error_fields_response_s"] = perf_counter() - tic
    s["shared_posthost_ERROR_s"] = perf_counter() - start
    w.update({"error_" + k: v for k, v in (tab.work | ew).items()})
    lp = {"ERROR": err}
    if anchors is not None:
        tic = perf_counter()
        ub, hb = anchors
        lp.update(
            B=base.packet.base.double(),
            U=u.response(base.packet, "U", ub),
            J=base.lp,
            HISTORY=eh.response(ep, "HISTORY", hb),
        )
        s["other_anchors_s"] = perf_counter() - tic
    tic = perf_counter()
    m = _marginal(tab, q)
    row = torch.from_numpy(m.row)
    sym = torch.from_numpy(m.symbol)
    active = torch.from_numpy(m.active)
    prob = (err[row, sym] - err.logsumexp(-1)[row]).exp()
    x = torch.from_numpy(m.f0[:, None])
    w.update(m.work, marginal_feature_bytes=x.nbytes)
    s["marginal_s"] = perf_counter() - tic
    if mode != "MARG":
        tic = perf_counter()
        true = ix.true_pairs(observed, q)
        s["true_pair_index_s"] = perf_counter() - tic
        tic = perf_counter()
        col, counts, n = _conditional(m, tab, true)
        x = torch.cat((x, torch.from_numpy(col[:, None])), dim=1)
        w.update(
            true_nonmarginal_rows=n,
            pair_feature_bytes=x.nbytes,
            **{"pair_" + k: v for k, v in true.work.items()},
        )
        s["true_pair_feature_s"] = perf_counter() - tic
    if mode == "panel":
        tic = perf_counter()
        sh = ix.switched_pairs(observed, q)
        s["shuf_index_s"] = perf_counter() - tic
        tic = perf_counter()
        col, repl, n = _conditional(m, tab, sh)
        diff = counts - repl
        x = torch.cat((x, torch.from_numpy(col[:, None])), dim=1)
        w.update(
            shuf_nonmarginal_rows=n,
            changed_conditional_rows=len(np.unique(m.row[diff != 0])),
            changed_conditional_uses=int(np.maximum(diff, 0).sum()),
            panel_feature_bytes=x.nbytes,
            **{"shuf_" + k: v for k, v in sh.work.items()},
        )
        s["shuf_feature_support_s"] = perf_counter() - tic
    return Packet(
        err,
        lp,
        base.packet.query,
        base.packet.candidate,
        active,
        row,
        sym,
        x,
        prob,
        w,
        s,
    )


@torch.inference_mode()
def predict(host, head, observed, jb, eb, arm, coefficients, writer, *, width=256):
    if arm not in ("MARG", "PAIR"):
        raise ValueError("choose one PC predictor without control work")
    check_window(observed, 4096)
    h, z, _ = frozen_prefill(host, observed, width)
    return response(
        fields(head, observed, h, z, jb, eb, writer, mode=arm), arm, coefficients
    )
