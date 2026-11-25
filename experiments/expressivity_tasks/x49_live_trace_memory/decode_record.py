"""X49-DR: compare actual, reconstructed and prototype payloads on the same causal address.

No corpus/lease access. The deployable path never builds the old vector or the sham.
"""

from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
import math
import torch
from . import surprise as u, literal_mark as ls
from .compact import evidence
from .decode_fit import Moments, Decoder, VALUE
from .endpoint import frozen_prefill
from .values import LiveValueHead

ARMS = ("B", "U", "J", "P", "D", "S")


def _check(head, w):
    if (
        type(head) is not LiveValueHead
        or head.training
        or type(w) is not bytes
        or not 1 <= len(w) <= 4096
    ):
        raise ValueError("one frozen B head and observed input required")


def _teacher(head, h, ids):
    return head.writer(torch.cat((h, head.byte(ids)), -1)).tanh()


def _tapes(w, z, ids, q, norm, writer, secs):
    tic = perf_counter()
    text, work = writer(w)
    work = dict(work)
    secs["text_build_s"] = perf_counter() - tic
    tic = perf_counter()
    neural, reused, extra = u._arrival_tape(z, ids, q, norm)
    secs["neural_build_s"] = perf_counter() - tic
    if (
        text.dtype != torch.uint8
        or text.shape != neural.shape
        or text[0].item() != 0
        or reused + extra != len(w) - 1
    ):
        raise ValueError("invalid arrival mark tapes")
    work.update(
        neural_storage_bytes=neural.nbytes,
        neural_arrivals_encoded=len(w) - 1,
        neural_norms_reused=reused,
        neural_norms_extra=extra,
        neural_marks_saturated=int((neural[1:] == 255).sum()),
    )
    return text, neural, work


@torch.inference_mode()
def fit_observed(host, head, observed, mom, *, writer, width=256):
    """No next byte or query mask; every eligible input contributes a teacher write."""
    _check(head, observed)
    if type(mom) is not Moments:
        raise ValueError("FIT requires write statistics")
    h, z, sec = frozen_prefill(host, observed, width)
    ids = torch.tensor(list(observed), dtype=torch.long)
    times = dict(prefill_s=sec)
    text, neural, work = _tapes(
        observed, z, ids, ids[:0], z.new_empty(0, dtype=torch.float64), writer, times
    )
    tic = perf_counter()
    values = _teacher(head, h, ids)
    times["full_vector_write_s"] = perf_counter() - tic
    tic = perf_counter()
    work.update(
        full_value_writes=len(ids), fit_writes=mom.add(observed, text, neural, values)
    )
    times["fit_statistics_s"] = perf_counter() - tic
    return dict(targets=len(observed), host_prefills=1, seconds=times, work=work)


@dataclass(slots=True)
class _Branch:
    base: torch.Tensor
    query: torch.Tensor
    candidate: torch.Tensor
    margin: torch.Tensor


def _trust(raw, q, c, now, old_text, old_neural, ub, jb):
    # The B-confidence field is NOT shared: raw already contains this branch's payload.
    v = u._candidate_margin(raw, q, c)
    bcode = u.quantize(u._softplus(-v)).double() / 64
    n = torch.stack((torch.ones_like(v), bcode, now), dim=1)
    p = _Branch(raw, q, c, v)
    j = u.copy_response(p, torch.cat((n, old_text, old_neural), dim=1), jb)
    if ub is None:
        return j, None
    return j, u.copy_response(p, torch.cat((n, old_neural), dim=1), ub)


@dataclass(slots=True)
class Answers:
    scores: dict
    query: torch.Tensor
    candidate: torch.Tensor
    seconds: dict
    work: dict
    squared_error_sums: dict


@torch.inference_mode()
def _answers(host, head, w, decoder, ub, jb, writer, width, *, panel):
    _check(head, w)
    if type(decoder) is not Decoder or (ub is None) == panel:
        raise ValueError("wrong decoder or panel coefficients")
    jb = u.checked_vector(ls.DIMS, "J", jb)
    if panel:
        ub = u.checked_beta("U", ub)
    h, z, secs = frozen_prefill(host, w, width)
    times = dict(prefill_s=secs)
    tic = perf_counter()
    ids = torch.tensor(list(w), dtype=torch.long)
    f, q, j, _, trips = evidence(w, 4096, device=h.device, rotate=False)
    if not ((j > 0) & (j <= q - 16)).all():
        raise ValueError("invalid selected value record")
    c = ids[j]
    code = head.byte(c)
    times["address_and_code_s"] = perf_counter() - tic
    tic = perf_counter()
    direct, _ = head.direct(h, z, f)
    times["direct_s"] = perf_counter() - tic
    tic = perf_counter()
    norm = z[q].double().logsumexp(-1)
    now = u.quantize(norm - z[q, c].double()).double() / 64
    times["current_host_field_s"] = perf_counter() - tic
    text, neural, work = _tapes(w, z, ids, q, norm, writer, times)
    work.update(trips)
    tic = perf_counter()
    tc, nc = text[j], neural[j]
    ts, ns = tc.double()[:, None] / 64, nc.double()[:, None] / 64
    times["true_note_reads_s"] = perf_counter() - tic
    values = {}
    work.update(full_value_writes=0, full_value_reads=0, compact_note_pairs_read=len(q))
    if panel:
        tic = perf_counter()
        written = _teacher(head, h, ids)
        times["full_vector_write_s"] = perf_counter() - tic
        tic = perf_counter()
        values["B"] = written[j]
        times["full_vector_read_s"] = perf_counter() - tic
        work.update(full_value_writes=len(ids), full_value_reads=len(j))
        tic = perf_counter()
        values["P"] = decoder.m[c]
        times["prototype_read_s"] = perf_counter() - tic
    tic = perf_counter()
    values["D"] = decoder.read(c, tc, nc)
    times["decode_true_s"] = perf_counter() - tic
    if panel:
        tic = perf_counter()
        k = u._rotation(w, q, j, (8 * f[0][q, 7]).long())
        if not torch.equal(ids[k], c):
            raise ValueError("wrong decoder donor byte")
        st, sn = text[k], neural[k]
        work.update(
            sham_positions_displaced=int((k != j).sum()),
            sham_pairs_changed=int(((st != tc) | (sn != nc)).sum()),
        )
        times["decoder_sham_search_read_s"] = perf_counter() - tic
        tic = perf_counter()
        values["S"] = decoder.read(c, st, sn)
        times["decode_sham_s"] = perf_counter() - tic
    scores = {}
    for name, val in values.items():
        tic = perf_counter()
        raw = head.answer(direct, h, f, q, code, val) if len(q) else direct
        times[name + "_head_s"] = perf_counter() - tic
        tic = perf_counter()
        joint, original = _trust(
            raw, q, c, now, ts, ns, ub if name == "B" else None, jb
        )
        if name == "B":
            scores.update(B=raw.double(), U=original, J=joint)
        else:
            scores[name] = joint
        times[name + "_trust_s"] = perf_counter() - tic
    # Teacher diagnostics only AFTER all the answers are available. Not computed by D alone.
    ss = {}
    if panel:
        ss = {
            a: float((values[a].double() - values["B"].double()).square().sum())
            for a in ("P", "D", "S")
        }
        work["mse_components"] = VALUE * len(q)
    return Answers(scores, q, c, times, work, ss)


def compact_predict(host, head, observed, decoder, jb, *, writer, width=256):
    """D only, from observed bytes. No historical value writer, teacher, U, prototype or sham."""
    return _answers(
        host, head, observed, decoder, None, jb, writer, width, panel=False
    ).scores["D"]


def panel_answers(host, head, observed, decoder, ub, jb, *, writer, width=256):
    return _answers(host, head, observed, decoder, ub, jb, writer, width, panel=True)


def score(a, targets):
    if (
        type(targets) is not bytes
        or set(a.scores) != set(ARMS)
        or any(v.shape != (len(targets), 256) for v in a.scores.values())
    ):
        raise ValueError("unaligned six-way score packet")
    y = torch.tensor(list(targets), dtype=torch.long)
    i = torch.arange(len(y))
    sums = {arm: float(-a.scores[arm][i, y].sum()) for arm in ARMS}
    if not all(math.isfinite(x) and x >= 0 for x in sums.values()):
        raise ValueError("invalid all-target loss")
    return dict(
        targets=len(y),
        host_prefills=1,
        work=a.work,
        seconds=a.seconds,
        nll_sums=sums,
        candidate_correct=int((y[a.query] == a.candidate).sum()),
        squared_error_sums=a.squared_error_sums,
    )


def document(host, head, word, *, fit=None, decoder=None, ub, jb, writer, width=256):
    if (
        type(word) is not bytes
        or len(word) != 4097
        or (fit is None) == (decoder is None)
    ):
        raise ValueError("one full document window and one DR phase required")
    if fit is not None:
        return fit_observed(host, head, word[:-1], fit, writer=writer, width=width)
    answers = panel_answers(
        host, head, word[:-1], decoder, ub, jb, writer=writer, width=width
    )
    return score(answers, word[1:])
