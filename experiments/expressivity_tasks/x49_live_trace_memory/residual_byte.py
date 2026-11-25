"""RB: pay for the actual old computation at write time; return just a third byte.

The only callable predictors start from current observed bytes. A comparison can also gather
true vectors for B/U/J; that optional diagnostic is separate from the emitted record.
"""

from __future__ import annotations
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from time import perf_counter
import torch
from . import decode_record as dr, surprise as u, literal_mark as ls
from .decode_fit import Decoder, VALUE
from .endpoint import frozen_prefill
from .residual_codec import Book, Moments, encode

ARMS = ("B", "U", "J", "D", "G", "R", "S")


@dataclass(frozen=True, slots=True)
class _Record:
    scope: object  # identity of this one prefill, not a reference to host features
    text: torch.Tensor
    neural: torch.Tensor
    byte: torch.Tensor

    def __post_init__(self):
        ts = (self.text, self.neural, self.byte)
        if not (
            type(self.scope) is object
            and all(
                t.ndim == 1
                and t.shape == ts[0].shape
                and t.dtype == torch.uint8
                and t.device.type == "cpu"
                and t.is_contiguous()
                and t.untyped_storage().nbytes() == t.numel()
                and t[0].item() == 0
                for t in ts
            )
            and len({t.data_ptr() for t in ts}) == 3
        ):
            raise ValueError("RB requires three distinct complete byte tapes")


def _clip(x, name, work):
    if work is not None:
        work[name + "_clipped_components"] = int((x.abs() > 1).sum())
    return x.clamp(-1, 1)


def _emit(
    scope,
    head,
    w,
    h,
    z,
    ids,
    q,
    norm,
    coarse,
    axes,
    writer,
    times,
    *,
    full_at=None,
    fit=None,
):
    text, neural, work = dr._tapes(w, z, ids, q, norm, writer, times)
    tic = perf_counter()
    v = dr._teacher(head, h, ids)
    times["full_vector_write_s"] = perf_counter() - tic
    tic = perf_counter()
    d = _clip(coarse.raw(ids[1:], text[1:], neural[1:]), "D_write", work)
    e = v[1:] - d
    times["coarse_write_and_subtract_s"] = perf_counter() - tic
    tic = perf_counter()
    byte = torch.zeros(len(ids), dtype=torch.uint8)
    byte[1:] = encode(e, axes)
    times["encode_byte_s"] = perf_counter() - tic
    work.update(
        full_value_computations=len(ids),
        encoded_residuals=len(ids) - 1,
        residual_storage_bytes=byte.nbytes,
    )
    hist = None
    if fit is not None:
        tic = perf_counter()
        hist = fit.add(byte[1:], e)
        times["fit_statistics_s"] = perf_counter() - tic
        work["fit_writes"] = sum(hist)
    full = None
    if full_at is not None:
        tic = perf_counter()
        full = v[full_at]
        times["full_vector_read_s"] = perf_counter() - tic
    # v,d,e are temporary and are not reachable through the record. Even in a panel,
    # only the control's actually selected v rows leave this seam, via a separate return.
    return _Record(scope, text, neural, byte), work, hist, full


def _donors(w, text, neural, q, j, depth):
    """Index only arrived, sufficiently old triples. No old vector or residual code input."""
    ts = text.tolist()
    ns = neural.tolist()
    next_slot = 1
    index = {}
    ks = []
    for t, src, d in zip(q.tolist(), j.tolist(), depth.tolist(), strict=True):
        while next_slot <= t - 16:
            i = next_slot
            index.setdefault((w[i], ts[i], ns[i]), []).append(i)
            next_slot += 1
        bit = (t - src).bit_length() - 1
        low = max(max(0, t + 1 - 4096) + d, t - (2 ** (bit + 1) - 1))
        high = min(t - 16, t - 2**bit)
        if not low <= src <= high:
            raise ValueError("noncausal RB source")
        positions = index[(w[src], ts[src], ns[src])]
        nxt = bisect_right(positions, src)
        pick = (
            positions[nxt]
            if nxt < len(positions) and positions[nxt] <= high
            else positions[bisect_left(positions, low)]
        )
        if not low <= pick <= high:
            raise ValueError("bad equal-triple donor range")
        ks.append(pick)
    return torch.tensor(ks, dtype=torch.long), dict(
        sham_index_inserts=next_slot - 1, sham_index_triples=len(index)
    )


@torch.inference_mode()
def fit_observed(host, head, w, coarse, axes, fit, *, writer, width=256):
    dr._check(head, w)
    if type(coarse) is not Decoder or type(fit) is not Moments:
        raise ValueError("RB FIT requires DR table and residual statistics")
    h, z, sec = frozen_prefill(host, w, width)
    ids = torch.tensor(list(w), dtype=torch.long)
    times = dict(prefill_s=sec)
    _, work, hist, _ = _emit(
        object(),
        head,
        w,
        h,
        z,
        ids,
        ids[:0],
        z.new_empty(0, dtype=torch.float64),
        coarse,
        axes,
        writer,
        times,
        fit=fit,
    )
    return dict(
        targets=len(w), host_prefills=1, work=work, fit_code_counts=hist, seconds=times
    )


def _read(scope, head, w, h, cur, record, coarse, book, jb, times, work, *, full, ub):
    if record.scope is not scope or (full is None) != (ub is None):
        raise ValueError("foreign RB record/scope")
    j = cur.j
    c = cur.c
    tic = perf_counter()
    tc, nc, bc = record.text[j], record.neural[j], record.byte[j]
    times["bound_code_read_s"] = perf_counter() - tic
    tic = perf_counter()
    d = _clip(coarse.raw(c, tc, nc), "D_read", work)
    times["coarse_decode_s"] = perf_counter() - tic

    def response_code(b, a):
        tic = perf_counter()
        v = _clip(d + book.means[b.long()], a, work)
        times[a + "_residual_add_s"] = perf_counter() - tic
        return v

    val = response_code(bc, "R")
    hist = {}
    if full is None:
        vals = {"R": val}
    else:
        tic = perf_counter()
        g = _clip(d + book.global_mean, "G", work)
        times["G_residual_add_s"] = perf_counter() - tic
        tic = perf_counter()
        k, idx = _donors(
            w,
            record.text,
            record.neural,
            cur.q,
            j,
            (8 * cur.features[0][cur.q, 7]).long(),
        )
        s = record.byte[k]
        if not (
            torch.equal(cur.ids[k], c)
            and torch.equal(record.text[k], tc)
            and torch.equal(record.neural[k], nc)
        ):
            raise ValueError("sham changed the coarse record")
        times["sham_index_and_read_s"] = perf_counter() - tic
        vals = dict(B=full, D=d, G=g, R=val, S=response_code(s, "S"))
        tic = perf_counter()
        hist = {
            a: torch.bincount(b.long(), minlength=256).tolist()
            for a, b in (("true", bc), ("sham", s))
        }
        work.update(
            idx,
            full_value_reads=len(j),
            true_residual_reads=len(j),
            sham_residual_reads=len(k),
            sham_displaced=int((k != j).sum()),
            sham_code_changed=int((s != bc).sum()),
            true_unseen_code_reads=int((book.count[bc.long()] == 0).sum()),
            sham_unseen_code_reads=int((book.count[s.long()] == 0).sum()),
        )
        times["read_diagnostics_s"] = perf_counter() - tic
    scores = dr._responses(head, h, cur, tc, nc, vals, ub, jb, times)
    ss = {}
    if full is not None:
        work["mse_components"] = VALUE * len(cur.q)
        ss = {
            a: float((vals[a].double() - full.double()).square().sum()) for a in "DGRS"
        }
    return dr.Answers(scores, cur.q, c, times, work, ss), hist


@torch.inference_mode()
def _answers(host, head, w, coarse, axes, book, jb, ub, writer, width):
    dr._check(head, w)
    if type(coarse) is not Decoder or type(book) is not Book:
        raise ValueError("missing frozen RB tables")
    jb = u.checked_vector(ls.DIMS, "J", jb)
    if ub is not None:
        ub = u.checked_beta("U", ub)
    h, z, sec = frozen_prefill(host, w, width)
    times = dict(prefill_s=sec)
    cur, trips = dr._current(head, w, h, z, times)
    scope = object()
    record, work, _, full = _emit(
        scope,
        head,
        w,
        h,
        z,
        cur.ids,
        cur.q,
        cur.norm,
        coarse,
        axes,
        writer,
        times,
        full_at=cur.j if ub is not None else None,
    )
    work.update(trips)
    return _read(
        scope, head, w, h, cur, record, coarse, book, jb, times, work, full=full, ub=ub
    )


def compact_predict(host, head, w, coarse, axes, book, jb, *, writer, width=256):
    """R only. One real write pass; neither the full historical vectors nor sham enter the read phase."""
    return _answers(host, head, w, coarse, axes, book, jb, None, writer, width)[
        0
    ].scores["R"]


def panel_answers(host, head, w, coarse, axes, book, ub, jb, *, writer, width=256):
    if ub is None:
        raise ValueError("need frozen all-control panel")
    return _answers(host, head, w, coarse, axes, book, jb, ub, writer, width)


def document(
    host, head, word, *, coarse, axes, ub, jb, writer, fit=None, book=None, width=256
):
    if type(word) is not bytes or len(word) != 4097 or (fit is None) == (book is None):
        raise ValueError("one full window and exactly one RB phase")
    if fit is not None:
        return fit_observed(
            host, head, word[:-1], coarse, axes, fit, writer=writer, width=width
        )
    answers, hist = panel_answers(
        host, head, word[:-1], coarse, axes, book, ub, jb, writer=writer, width=width
    )
    return dict(**dr._score(answers, word[1:], ARMS), read_code_counts=hist)
