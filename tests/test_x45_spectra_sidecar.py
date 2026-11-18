"""The X45 spectra sidecar's guardrail tests (the supervisor's ruling).

The registered saturation triggers' statistic must be EXACT and UNCHANGED
by the sidecar: bit-equality of the sidecar's spectra against the
same-LAPACK single-process reference is asserted (the decisive guardrail);
PR values are checked against an independent svdvals reference to fp32
tolerance; the zero-state guard, the hard worker cap, the single-thread
worker environment (asserted inside the workers at attach time), chunked
handoff correctness above one chunk, and the box-load receipts are all
exercised.  CPU-only (the pool is CPU; the GPU gram is exercised by the
lanes' own receipts).
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from experiments.expressivity_tasks.x45_matrix_emender import (
    x45_spectra_sidecar as SPECTRA)


def _states(n=512, seed=3401):
    torch.manual_seed(seed)
    base = torch.randn(n, SPECTRA.DIM, 256)
    out = torch.empty_like(base)
    for k, d in enumerate((3.0, 1.0, 0.4, 0.03)):
        sl = slice(k * (n // 4), (k + 1) * (n // 4))
        out[sl] = base[sl] * torch.logspace(0, -d, SPECTRA.DIM).view(1, -1, 1)
    # a few exact-empty states: the zero-state guard
    out[:4] = 0.0
    return out


def test_pr_matches_svdvals_reference():
    s = _states()
    with SPECTRA.SpectraPool() as pool:
        pr = pool.pr_from_states(s)
    sv = torch.linalg.svdvals(s)
    ref = (sv.sum(-1) ** 2) / (sv ** 2).sum(-1).clamp(min=1e-30)
    assert float((pr - ref).abs().max()) < 5e-3
    # the zero-state guard: an empty state is maximally UNsaturated
    assert float(pr[:4].abs().max()) == 0.0
    assert float(pr[4:].min()) >= 1.0


def test_bit_equal_same_lapack_reference():
    # the decisive guardrail: the sidecar's spectra are BIT-IDENTICAL to the
    # same-LAPACK (numpy/OpenBLAS) single-process reference.  torch CPU links
    # a different LAPACK and differs at ulp scale; that is receipted in the
    # module's self_test MAGMA cross-backend check, not asserted here.
    s = _states(1024)
    gram = SPECTRA._gram(s)
    with SPECTRA.SpectraPool() as pool:
        lam = pool.eigvalsh(gram)
    ref = torch.from_numpy(np.linalg.eigvalsh(gram.numpy()))
    assert torch.equal(lam, ref)


def test_chunked_handoff_above_one_chunk():
    # more matrices than one shared-memory chunk: every row must land
    s = _states(64)
    big = s.repeat(400, 1, 1).contiguous()      # 25,600 > SPECTRA_CHUNK
    gram = SPECTRA._gram(big)
    with SPECTRA.SpectraPool(chunk=8192) as pool:
        lam = pool.eigvalsh(gram)
        pr = SPECTRA.participation_ratio(lam)
    ref = SPECTRA.participation_ratio(
        torch.from_numpy(np.linalg.eigvalsh(gram.numpy())))
    assert torch.equal(pr, ref)


def test_hard_worker_cap():
    with pytest.raises(AssertionError):
        SPECTRA.SpectraPool(workers=SPECTRA.SPECTRA_WORKERS + 1)


def test_worker_single_thread_env_receipt():
    # the attach-time assertion inside every worker is the receipt; a pool
    # that reaches map() has passed it in all workers.  This test additionally
    # checks the parent restores its own environment after construction.
    before = {k: os.environ.get(k) for k in SPECTRA._SINGLE_THREAD_ENV}
    with SPECTRA.SpectraPool(workers=4) as pool:
        pool.eigvalsh(SPECTRA._gram(_states(64)))
    after = {k: os.environ.get(k) for k in SPECTRA._SINGLE_THREAD_ENV}
    assert before == after


def test_box_load_receipt_shape():
    r = SPECTRA.box_load_receipt()
    assert {"uptime_s", "loadavg_1", "loadavg_5", "loadavg_15",
            "running", "total", "nproc"} <= set(r)
    assert r["nproc"] >= SPECTRA.SPECTRA_WORKERS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
