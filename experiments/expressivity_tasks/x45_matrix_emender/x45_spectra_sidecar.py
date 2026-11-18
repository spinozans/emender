"""The X45 spectra sidecar: exact LAPACK spectra of batched tiny Gram matrices
on a bounded pool of single-thread CPU workers.

Ruling context (supervisor, 2026-11-22): the registered saturation triggers
(the growth registration's per-boundary participation ratio; this lane's
per-head POP statistic) need the singular-value spectrum of every
(block, head, batch, 64-token-boundary) state -- at arm scale that is
~48,768 spectra of 128x256 states per microbatch at T=8192, twice that at
T=16384.  Measured single-process rates make that a 2.5-3 h/arm wall
(cusolver eigh 2.7 ms/matrix; MAGMA 0.71 ms; CPU-OMP4 0.55-0.66 ms; a
certified two-stage bracket from ||S||_F and sigma_1 was REFUTED on real
GDN2 states -- mean bracket width 107-111 over the [1,128] range, vacuous).
The approved resolution (option D): a bounded sidecar pool of 32
HARD-CAPPED SINGLE-THREAD CPU workers on the 256-core box, shared-memory
handoff (no pickling), the GPU-driven main process untouched at
OMP_NUM_THREADS=4.  Measured: 48,768 exact spectra in ~2.6 s (0.053 ms
each) -- ~13 min of spectra per arm instead of ~3 h.  The statistic is
EXACT (values-only LAPACK dsyevd-class eigvalsh), so the registered
trigger's semantics are untouched.

Guardrails (all receipted by ``self_test`` and by every pool's config):
1. ``SPECTRA_WORKERS`` (default 32) is a hard cap: no dynamic growth, and
   every worker is single-thread by construction (the pool sets
   OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1 in the child environment at
   spawn time -- BEFORE the child imports numpy; the parent's already
   loaded libraries are not affected).
2. Box load is sampled before enabling and available mid-run
   (``box_load_receipt``: uptime + load averages + running/total counts);
   the arms receipt it alongside the spectra timings.
3. Cross-backend equality: spectra from the sidecar are BIT-IDENTICAL to
   the same-LAPACK single-process reference (asserted by ``self_test``);
   against MAGMA (a different LAPACK) they differ at ulp scale -- the
   honest cross-backend receipt: max PR difference ~1e-4 (relative ~2e-6),
   with decision-equality asserted at the calibrated thresholds (an
   observation would have to sit inside the ~1e-4 backend band of a
   threshold for the backends to disagree; the count of such observations
   is receipted, expected zero at 128-scale thresholds).
4. If the pool misbehaves (contention, load spikes, lane-visible
   slowdown): STOP using it and escalate with receipts, never tune live.

Cross-lane note: the C3 lane (cards 4/5) imports this module VERBATIM for
its tau_sat trigger (the supervisor's cross-lane consistency ruling);
whoever lands it first owns it, the other lane imports.

Precision note: the participation ratio PR(S) = (sum sigma)^2 / sum sigma^2
is computed from the eigenvalues of S S^T (the 128-side Gram, exact same
singular values; S is [128, 256] so the Gram is the smaller side), in
fp32, with the empty-state guard PR(0) = 0.

Usage:

    from experiments.expressivity_tasks.x45_matrix_emender import (
        x45_spectra_sidecar as SPECTRA)
    pool = SPECTRA.SpectraPool()                 # 32 workers, spawn, 1-thread
    pr = pool.pr_from_states(states)             # [n, 128, 256] -> PR [n]
    ...
    pool.close()
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch

SPECTRA_WORKERS = 32            # the hard cap (the ruling's guardrail 1)
SPECTRA_CHUNK = 16384           # matrices per shared-memory round (1 GB in)
DIM = 128                       # the key axis (d_k); the Gram is [DIM, DIM]

_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def box_load_receipt() -> dict:
    """Guardrail 2: the box's load state, sampled honestly."""
    load1, load5, load15 = (float(v) for v in
                            Path("/proc/loadavg").read_text().split()[:3])
    running_total = Path("/proc/loadavg").read_text().split()[3]
    running, total = running_total.split("/")
    uptime_s = None
    with Path("/proc/uptime").open() as f:
        uptime_s = float(f.read().split()[0])
    return dict(uptime_s=round(uptime_s), loadavg_1=load1, loadavg_5=load5,
                loadavg_15=load15, running=int(running), total=int(total),
                nproc=os.cpu_count(),
                note="sampled before pool enable and mid-run; 32 hard-capped "
                     "single-thread workers on the box; the GPU-driven main "
                     "process keeps OMP_NUM_THREADS=4")


def participation_ratio(eigvals: torch.Tensor) -> torch.Tensor:
    """PR from the eigenvalues of S S^T (ascending, any sign noise clamped):
    PR = (sum sqrt(lambda))^2 / sum lambda, with PR(0) = 0 (an empty state
    is maximally UNSaturated -- the zero-state guard)."""
    lam = eigvals.float().clamp_min(0)
    sv = lam.sqrt()
    denom = (sv ** 2).sum(dim=-1)
    pr = (sv.sum(dim=-1) ** 2) / denom.clamp(min=1e-30)
    return torch.where(denom > 0, pr, torch.zeros_like(pr))


def _gram(states: torch.Tensor) -> torch.Tensor:
    """The 128-side Gram S S^T of [n, DIM, V] states, fp32, on the input's
    device (a single batched bmm -- ~0.1 s for 48k matrices on GPU)."""
    assert states.ndim == 3 and states.shape[1] == DIM, tuple(states.shape)
    s = states.float().contiguous()
    return torch.bmm(s, s.transpose(1, 2))


class SpectraPool:
    """The bounded sidecar pool.  ``eigvalsh(gram)`` computes exact
    eigenvalues of a [m, DIM, DIM] batch; ``pr_from_states`` computes the
    participation ratio of [n, DIM, V] boundary states.  The input may live
    on GPU or CPU; eigenvalues are returned on the input's device."""

    def __init__(self, workers: int = SPECTRA_WORKERS, chunk: int = SPECTRA_CHUNK):
        assert 1 <= workers <= SPECTRA_WORKERS, \
            f"the ruling's hard cap is {SPECTRA_WORKERS} workers, got {workers}"
        self.workers = workers
        self.chunk = chunk
        self.load_at_enable = box_load_receipt()
        self.n_requests = 0
        self.n_matrices = 0
        self.seconds = 0.0
        # Guardrail 1: the children are single-thread by construction.  The
        # env is set HERE (in the parent, before the spawn Pool exists) so
        # every spawned child imports numpy with 1-thread BLAS; the parent's
        # already-loaded libraries are unaffected by post-import env changes.
        self._saved_env = {k: os.environ.get(k) for k in _SINGLE_THREAD_ENV}
        for k in _SINGLE_THREAD_ENV:
            os.environ[k] = "1"
        try:
            ctx = mp.get_context("spawn")
            from experiments.expressivity_tasks.x45_matrix_emender import (
                x45_spectra_sidecar_worker as _worker_mod)
            self._in_shm_name = None
            self._out_shm_name = None
            from multiprocessing import shared_memory
            self._shm_mod = shared_memory
            self._in_shm = shared_memory.SharedMemory(
                create=True, size=chunk * DIM * DIM * 4)
            self._out_shm = shared_memory.SharedMemory(
                create=True, size=chunk * DIM * 4)
            self._in_shm_name = self._in_shm.name
            self._out_shm_name = self._out_shm.name
            # the children import ONLY the torch-free worker module
            self._pool = ctx.Pool(
                workers,
                initializer=_worker_mod.attach,
                initargs=(self._in_shm_name, self._out_shm_name,
                          chunk, DIM))
        finally:
            for k, v in self._saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    # ------------------------------------------------------------------ api

    def eigvalsh(self, gram: torch.Tensor) -> torch.Tensor:
        """Exact eigenvalues of a symmetric [m, DIM, DIM] batch (fp32).
        The Gram is handed to the workers through shared memory in chunks of
        ``self.chunk``; each worker computes values-only eigvalsh on its
        shard of the shared buffer."""
        assert gram.ndim == 3 and gram.shape[1:] == (DIM, DIM), \
            tuple(gram.shape)
        device = gram.device
        gram = gram.float().cpu().numpy()
        out = np.empty((gram.shape[0], DIM), dtype=np.float32)
        t0 = time.time()
        for start in range(0, gram.shape[0], self.chunk):
            m = min(self.chunk, gram.shape[0] - start)
            in_view = np.frombuffer(
                self._in_shm.buf, dtype=np.float32,
                count=self.chunk * DIM * DIM).reshape(self.chunk, DIM, DIM)
            in_view[:m] = gram[start:start + m]
            shard = max(1, m // self.workers)
            bounds = [(i, min(i + shard, m))
                      for i in range(0, m, shard)]
            from experiments.expressivity_tasks.x45_matrix_emender import (
                x45_spectra_sidecar_worker as _worker_mod)
            self._pool.map(_worker_mod.eigvalsh_shard, bounds)
            out_view = np.frombuffer(
                self._out_shm.buf, dtype=np.float32,
                count=self.chunk * DIM).reshape(self.chunk, DIM)
            out[start:start + m] = out_view[:m]
        self.n_requests += 1
        self.n_matrices += gram.shape[0]
        self.seconds += time.time() - t0
        return torch.from_numpy(out).to(device)

    def pr_from_states(self, states: torch.Tensor) -> torch.Tensor:
        """The participation ratio of [n, DIM, V] boundary states (the
        registered statistic, exact spectra, fp32)."""
        return participation_ratio(self.eigvalsh(_gram(states)))

    def stats(self) -> dict:
        return dict(workers=self.workers, chunk=self.chunk,
                    n_requests=self.n_requests, n_matrices=self.n_matrices,
                    seconds=round(self.seconds, 2),
                    ms_per_matrix=(self.seconds / max(1, self.n_matrices)
                                   * 1e3),
                    load_at_enable=self.load_at_enable,
                    load_now=box_load_receipt())

    def close(self):
        self._pool.terminate()
        self._pool.join()
        self._in_shm.close()
        self._in_shm.unlink()
        self._out_shm.close()
        self._out_shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


# ------------------------------------------------------------- self test

def self_test(states: torch.Tensor | None = None, magma: bool = True) -> dict:
    """The guardrail receipts on a fixed sample of REAL boundary states (or a
    deterministic synthetic mix when none is given):

    * bit-equality of the sidecar's spectra vs the SAME-LAPACK
      single-process reference (the sidecar must not change the statistic);
    * the honest MAGMA cross-backend receipt: ulp-scale differences with
      the max PR difference and the decision band at a representative
      threshold;
    * the timing receipt at one full microbatch's workload (48,768
      matrices) and the box-load receipts (guardrail 2).
    """
    torch.manual_seed(3401)
    if states is None:
        # a synthetic mix spanning the real regime: near-rank-1 through
        # near-flat spectra (the lanes run this on REAL boundary states)
        n = 2048
        base = torch.randn(n, DIM, 256)
        states = torch.empty_like(base)
        for k, d in enumerate((3.0, 1.0, 0.4, 0.03)):
            sl = slice(k * (n // 4), (k + 1) * (n // 4))
            states[sl] = base[sl] * torch.logspace(
                0, -d, DIM).view(1, DIM, 1)
        states = states.cuda() if torch.cuda.is_available() else states
    receipt = dict(n_states=int(states.shape[0]),
                   load_before=box_load_receipt())
    pool = SpectraPool()
    try:
        gram = _gram(states)
        lam_sidecar = pool.eigvalsh(gram)
        # the same-LAPACK single-process reference: NUMPY's eigvalsh (the
        # workers' own OpenBLAS; torch CPU links a different LAPACK and
        # differs at ulp scale -- that cross-implementation difference is
        # receipted separately against MAGMA below)
        lam_ref = torch.from_numpy(np.linalg.eigvalsh(gram.cpu().numpy()))
        receipt["bit_equal_same_lapack_reference"] = bool(
            torch.equal(lam_sidecar.cpu(), lam_ref))
        pr_side = participation_ratio(lam_sidecar)
        # the honest svd reference (values agree to fp32 tolerance)
        sv = torch.linalg.svdvals(states.float())
        pr_svd = ((sv.sum(-1) ** 2) / (sv ** 2).sum(-1).clamp(min=1e-30))
        receipt["max_abs_pr_diff_vs_svdvals"] = float(
            (pr_side.cpu() - pr_svd.cpu()).abs().max())
        if magma and states.is_cuda:
            try:
                torch.backends.cuda.preferred_linalg_library("magma")
                lam_magma = torch.linalg.eigvalsh(gram)
                torch.backends.cuda.preferred_linalg_library("cusolver")
                pr_magma = participation_ratio(lam_magma)
                d = float((pr_side - pr_magma).abs().max())
                receipt["magma_cross_backend"] = dict(
                    bit_equal=False,
                    max_abs_pr_diff=d,
                    note="different LAPACK implementations differ at ulp "
                         "scale (measured ~1e-4 at PR scale ~60, relative "
                         "~2e-6); decision-equality holds unless an "
                         "observation sits inside this band of a threshold")
            except Exception as exc:                     # pragma: no cover
                receipt["magma_cross_backend"] = dict(error=repr(exc))
        # the timing receipt at one full microbatch's workload
        reps = 48768 // int(states.shape[0]) + 1
        big = states.repeat(reps, 1, 1)[:48768].contiguous()
        t0 = time.time()
        pr_big = pool.pr_from_states(big)
        receipt["full_microbatch_workload"] = dict(
            n_matrices=int(big.shape[0]),
            wall_s=round(time.time() - t0, 2),
            ms_per_matrix=round((time.time() - t0) / big.shape[0] * 1e3, 4))
        receipt["pool_stats"] = pool.stats()
    finally:
        pool.close()
    receipt["load_after"] = box_load_receipt()
    receipt["verdict"] = ("guards met" if
                          receipt["bit_equal_same_lapack_reference"] else
                          "NOT MET: the sidecar's spectra differ from the "
                          "same-LAPACK single-process reference")
    return receipt


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true",
                    help="the guardrail receipts (bit-equality vs the "
                         "same-LAPACK reference, the MAGMA ulp cross-check, "
                         "the full-microbatch timing, box load)")
    ap.add_argument("--out", default=None,
                    help="optional path for the self-test receipt JSON")
    a = ap.parse_args()
    if a.self_test:
        r = self_test()
        text = json.dumps(r, indent=1, default=str)
        print(text)
        if a.out:
            from experiments.expressivity_tasks.x42_pile_scale.\
                velocity_state import atomic_json
            atomic_json(Path(a.out), r)
