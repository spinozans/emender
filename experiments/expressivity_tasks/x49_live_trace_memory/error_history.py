"""EH: read one live error histogram and tilt the unchanged latest-match J scores.

No next target or externally supplied address enters a predictor. Host outputs and
packed counts are transient parts of one prefill, never a cross-document index.
"""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import numpy as np
import torch
from . import error_history_index as eh, joint_reader as joint, surprise as u
from .endpoint import frozen_prefill

DIMS = {"COUNT": 2, "HISTORY": 5, "ERROR": 6, "SHIFT": 6}
ARMS = ("B", "U", "J", "COUNT", "HISTORY", "ERROR", "SHIFT", "SWAP")


@dataclass(slots=True)
class Packet:
    base: torch.Tensor  # J; the new readout preserves its incoming log mass
    anchors: dict
    query: torch.Tensor  # latest-hit rows, including singletons
    candidate: torch.Tensor
    active: torch.Tensor  # older-history rows only
    row: torch.Tensor  # unique (row,symbol) pairs for sparse feature columns
    symbol: torch.Tensor
    x: torch.Tensor
    prob: torch.Tensor
    work: dict
    seconds: dict

    def features(self, arm):
        if arm in ("COUNT", "HISTORY", "ERROR") and self.x.shape[1] >= DIMS[arm]:
            return self.x[:, : DIMS[arm]]
        if arm in ("SHIFT", "SWAP") and self.x.shape[1] == 7:
            return torch.cat((self.x[:, :5], self.x[:, -1:]), dim=1)
        raise ValueError("unavailable error-history feature set")


def _geometry(table, ref):
    at = np.flatnonzero(table.latest >= 0)
    if not (
        np.array_equal(at, ref.query.numpy())
        and np.array_equal(table.latest[at], ref.latest.numpy())
        and np.array_equal(table.depth[at], ref.depth.numpy())
    ):
        raise ValueError("history did not select B's key and latest record")


def _columns(base, tab, other, work):
    """Sparse union includes wrong guesses of bytes never observed as followers."""
    if other is None:
        ix = tab.position
        counts = tab.counts
    else:
        if not all(
            np.array_equal(getattr(tab, k), getattr(other, k))
            for k in ("size", "depth", "latest")
        ):
            raise ValueError("sham changed old-key geometry")
        ix = np.union1d(tab.position, other.position)
        counts = np.zeros((len(ix), 6), np.uint32)
        counts[np.searchsorted(ix, tab.position), :5] = tab.counts
        counts[np.searchsorted(ix, other.position), 5] = other.counts[:, 0]
        work["changed_histogram_rows"] = len(
            np.unique(ix[counts[:, 4] != counts[:, 5]] >> 8)
        )
    row = torch.from_numpy((ix >> 8).astype(np.int64))
    sym = torch.from_numpy((ix & 255).astype(np.int64))
    active = torch.from_numpy(np.flatnonzero(tab.size))
    n = torch.from_numpy(tab.size.astype(np.float64))[row]
    r = torch.from_numpy(counts.astype(np.float64))
    logp = base[row, sym] - base.logsumexp(-1)[row]
    freq = r[:, 0] / n
    cols = [
        freq,
        freq * (u.quantize(-logp).double() / 64),
        r[:, 1] / (64 * n),
        r[:, 2] / (64 * n),
        r[:, 3] / n,
    ]
    if counts.shape[1] >= 5:  # four counters on ET's HISTORY-only path
        cols.append(-r[:, 4] / n)
    if other is not None:
        cols.append(-r[:, 5] / n)
    x = torch.stack(cols, dim=1)
    work["packed_feature_entries"] = len(row)
    work["fp64_feature_bytes"] = x.nbytes
    return active, row, sym, x, logp.exp()


@torch.inference_mode()
def fields(head, word, h, z, jb, writer, *, ub=None):
    """ub supplied for the complete research panel only; ERROR by itself does no control pass."""
    p, nt = u._fields_and_tape(head, word, h, z, controls=False)
    times = dict(p.seconds)
    work = dict(p.work)
    tic = perf_counter()
    lt, tw = writer(word)
    times["text_build_s"] = perf_counter() - tic
    work.update(tw)
    tic = perf_counter()
    base = joint._from_tape(p, lt, jb)
    times["J_response_s"] = perf_counter() - tic
    anchors = {"J": base}
    if ub is not None:
        tic = perf_counter()
        anchors.update(B=p.base.double(), U=u.response(p, "U", ub))
        times["anchor_BU_s"] = perf_counter() - tic
    tic = perf_counter()
    guess = eh.guesses(z)
    neural = nt.numpy().tobytes()
    literal = lt.numpy().tobytes()
    work.update(
        guess_tape_bytes=len(guess),
        guesses_written=len(word) - 1,
        wrong_guesses_written=sum(
            a != c for a, c in zip(guess[1:], word[1:], strict=True)
        ),
    )
    times["guess_and_tape_prep_s"] = perf_counter() - tic
    tic = perf_counter()
    tab = eh.true_counts(word, literal, neural, guess)
    _geometry(tab, p)
    work.update({"pool_" + k: v for k, v in tab.work.items()})
    times["true_pool_s"] = perf_counter() - tic
    other = None
    if ub is not None:
        tic = perf_counter()
        other, sw = eh.displaced_counts(word, neural, guess)
        work.update({"shift_" + k: v for k, v in (other.work | sw).items()})
        times["shift_pool_s"] = perf_counter() - tic
    tic = perf_counter()
    active, row, sym, x, prob = _columns(base, tab, other, work)
    times["feature_s"] = perf_counter() - tic
    return Packet(
        base, anchors, p.query, p.candidate, active, row, sym, x, prob, work, times
    )


@torch.inference_mode()
def response(packet, arm, coefficients):
    k = "ERROR" if arm == "SWAP" else arm
    beta = u.checked_vector(DIMS, k, coefficients)
    return _tilt(packet, packet.features(arm), beta)


def _tilt(packet, x, beta):
    """Apply validated sparse score columns without changing the incoming row log mass."""
    base = packet.base
    out = base.clone()
    at = packet.active
    if len(at):
        out[packet.row, packet.symbol] += x @ beta
        out[at] -= (out[at].logsumexp(-1) - base[at].logsumexp(-1))[:, None]
    if not out.isfinite().all():
        raise ValueError("nonfinite error-history response")
    return out


@torch.inference_mode()
def predict(host, head, word, jb, error_beta, writer, *, width=256):
    """Correct ERROR alone: one fresh input-only prefill, no displaced history or comparison arms."""
    h, z, _ = frozen_prefill(host, word, width)
    return response(fields(head, word, h, z, jb, writer), "ERROR", error_beta)
