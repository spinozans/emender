"""Private DR arrays. All filenames derived from seed. Verify shape/dtype/bytes before use."""

import os
import numpy as np
import torch
from ..x48_muon_continuation.data import check, sha
from .decode_fit import Decoder, Moments, RIDGE
from .panels import SEEDS


def _path(d, seed, which):
    check(
        type(seed) is int and seed in SEEDS and which in ("tables", "moments"),
        "invalid DR array identity",
    )
    return d / f"dr-{which}-{seed}.npz"


def _write(p, arrays):
    with p.open("xb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    return sha(p)


def store(d, seed, mom, dec):
    return dict(
        tables_sha=_write(_path(d, seed, "tables"), dec.arrays()),
        moments_sha=_write(_path(d, seed, "moments"), mom.arrays()),
    )


def _load(p, h, types):
    check(not p.is_symlink() and sha(p) == h, "changed DR array bytes: " + p.name)
    with np.load(p, allow_pickle=False) as z:
        check(sorted(z.files) == sorted(types), "wrong DR array set")
        a = {k: z[k] for k in types}
    for k, (dtype, shape) in types.items():
        x = a[k]
        check(
            x.shape == shape and x.dtype == dtype and np.isfinite(x).all(),
            "wrong DR array shape/dtype: " + k,
        )
    return a


def tables(d, seed, seal):
    shapes = dict(m=(256, 32), b=(256, 2), k=(256, 2, 32))
    a = _load(
        _path(d, seed, "tables"),
        seal["tables_sha"],
        {k: (np.float32, sh) for k, sh in shapes.items()},
    )
    return Decoder(*(torch.from_numpy(a[k]) for k in ("m", "b", "k")))


def moments(d, seed, seal):
    m = Moments()
    a = _load(
        _path(d, seed, "moments"),
        seal["moments_sha"],
        {k: (v.dtype, v.shape) for k, v in m.arrays().items()},
    )
    for k, v in a.items():
        setattr(m, k, v)
    return m


def check_state(d, seed, state, writes):
    m = moments(d, seed, state)
    dec = tables(d, seed, state)
    check(
        writes == int(m.count.sum()) > 0 and np.all(m.count >= 0),
        "wrong DR FIT write denominator",
    )
    absent = m.count == 0
    check(
        all(not arr[absent].any() for arr in m.arrays().values()),
        "nonempty statistics for absent FIT byte",
    )
    replay = m.solve()
    check(
        all(torch.equal(getattr(dec, k), getattr(replay, k)) for k in dec.arrays()),
        "decoder does not reproduce FIT solution",
    )
    return dict(
        tables_sha=state["tables_sha"],
        moments_sha=state["moments_sha"],
        fit_writes=writes,
        absent_cells=int(absent.sum()),
        singleton_cells=int((m.count == 1).sum()),
        seen_cells=int((~absent).sum()),
        max_cell_writes=int(m.count.max()),
        table_bytes=sum(x.nbytes for x in dec.arrays().values()),
    )


def independent(d, seed, record):
    """Extended-precision, unnormalized centered cross products and explicit 2x2 inverse.

    Check the deployed FP32 entries, including fallback/constant-column semantics. Allow
    final FP32 rounding and dimension-scaled FP64 centering/solve error, not fit error.
    """
    m = moments(d, seed, record)
    dec = tables(d, seed, record)
    n = m.count
    check(int(n.sum()) == record["fit_writes"], "unbound DR fit count")
    glob = m.v.astype(np.longdouble).sum(0) / int(n.sum())
    worst = 0.0
    f64 = np.finfo(np.float64).eps
    for c, nc in enumerate(n.tolist()):
        b, mv, K = (
            getattr(dec, k)[c].numpy().astype(np.longdouble) for k in ("b", "m", "k")
        )
        if nc == 0:
            np.testing.assert_allclose(
                mv, glob, atol=64 * f64, rtol=np.finfo(np.float32).eps / 2
            )
            check(not b.any() and not K.any(), "nonconstant absent DR cell")
            continue
        a, v, aa, av = (x[c].astype(np.longdouble) for x in (m.a, m.v, m.aa, m.av))
        np.testing.assert_array_equal(b.astype(np.float32), (a / nc).astype(np.float32))
        np.testing.assert_array_equal(
            mv.astype(np.float32), (v / nc).astype(np.float32)
        )
        M = nc * aa - np.outer(a, a)
        R = nc * av - np.outer(a, v)
        for j in range(2):
            if M[j, j] == 0:
                check(not K[j].any(), "constant feature has a slope")
                R[j] = 0
        M += np.eye(2, dtype=np.longdouble) * (nc**2 * RIDGE)
        det = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
        check(
            det > 0 and M[0, 0] > 0 and np.array_equal(M, M.T),
            "invalid regularized decoder covariance",
        )
        X = (
            np.stack((M[1, 1] * R[0] - M[0, 1] * R[1], M[0, 0] * R[1] - M[1, 0] * R[0]))
            / det
        )
        # Norm bound on residual from the short FP64 centering + SPD solve. Ridge
        # lower-bounds lambda_min; b<=255/64, mv<=1. Check each teacher component.
        xscale = np.abs(X).max(0)
        center_error = 64 * f64 * (4 + 32 * xscale) / RIDGE
        err = np.abs(X - K)
        ulp = np.spacing(np.abs(K).astype(np.float32)).astype(np.longdouble)
        check(
            (err <= ulp / 2 + center_error).all(),
            "DR independent ridge/FP32 audit failed",
        )
        worst = max(worst, float(err.max()))
    return dict(audited_cells=len(n), max_abs_deployed_slope_minus_extended_solve=worst)
