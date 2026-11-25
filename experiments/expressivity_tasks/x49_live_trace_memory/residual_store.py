"""Private, seed-attached RB write means, global control and sufficient statistics."""

from math import fsum
import numpy as np
import torch
from ..x48_muon_continuation.data import check
from .decode_store import _write, _load
from .residual_codec import Book, Moments
from .panels import SEEDS


def _path(d, s, name):
    check(
        type(s) is int and s in SEEDS and name in ("tables", "moments"),
        "invalid RB state path",
    )
    return d / f"rb-{name}-{s}.npz"


def store(d, s, m, b):
    return dict(
        tables_sha=_write(
            _path(d, s, "tables"),
            dict(
                means=b.means.numpy(),
                global_mean=b.global_mean.numpy(),
                count=b.count.numpy(),
            ),
        ),
        moments_sha=_write(_path(d, s, "moments"), dict(total=m.total, count=m.count)),
    )


def tables(d, s, state):
    shapes = dict(
        means=(np.float32, (256, 32)),
        global_mean=(np.float32, (32,)),
        count=(np.int64, (256,)),
    )
    z = _load(_path(d, s, "tables"), state["tables_sha"], shapes, name="RB")
    return Book(**{k: torch.from_numpy(v) for k, v in z.items()})


def moments(d, s, state):
    z = _load(
        _path(d, s, "moments"),
        state["moments_sha"],
        dict(total=(np.float64, (256, 32)), count=(np.int64, (256,))),
        name="RB",
    )
    m = Moments()
    m.total = z["total"]
    m.count = z["count"]
    return m


def check_state(d, s, state, logs):
    m = moments(d, s, state)
    book = tables(d, s, state)
    replay = m.solve()
    counts = np.sum([x["fit_code_counts"] for x in logs], axis=0, dtype=np.int64)
    check(
        int(counts.sum()) == sum(x["work"]["fit_writes"] for x in logs) > 0
        and np.array_equal(counts, m.count),
        "RB FIT occupancy mismatch",
    )
    check(
        all(
            torch.equal(getattr(book, k), getattr(replay, k))
            for k in ("means", "global_mean", "count")
        ),
        "codebook differs from pre-probe FIT moments",
    )
    return dict(
        tables_sha=state["tables_sha"],
        moments_sha=state["moments_sha"],
        fit_writes=int(counts.sum()),
        seen_codes=int((counts > 0).sum()),
        singleton_codes=int((counts == 1).sum()),
        max_cell_writes=int(counts.max()),
        residual_table_bytes=book.means.nbytes,
        global_control_bytes=book.global_mean.nbytes,
    )


def independent(d, s, seal):
    m = moments(d, s, seal)
    book = tables(d, s, seal)
    n = int(m.count.sum())
    check(
        n == seal["fit_writes"] and np.array_equal(m.count, book.count.numpy()),
        "wrong RB write denominator",
    )
    g = np.array([fsum(m.total[:, i]) / n for i in range(32)], dtype=np.longdouble)

    def cast_check(stored, expected, ops):
        # FP32 deployment rounding plus bounded FP64 summation/division error.
        # Each input residual has magnitude <=2. No tolerance estimated from PROBE.
        err = np.abs(stored.astype(np.longdouble) - expected)
        tol = (
            np.spacing(np.abs(stored)).astype(np.longdouble) / 2
            + ops * np.finfo(np.float64).eps * 2
        )
        check((err <= tol).all(), "RB independent mean/FP32 check failed")
        return float(err.max())

    worst = cast_check(book.global_mean.numpy(), g, 257)
    for i, ct in enumerate(m.count.tolist()):
        avg = m.total[i].astype(np.longdouble) / ct if ct else g
        worst = max(worst, cast_check(book.means[i].numpy(), avg, 1 if ct else 257))
    return dict(audited_codes=256, max_abs_deployed_mean_minus_independent=worst)
