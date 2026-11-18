"""The X45 spectra sidecar's WORKER module (torch-free by design).

The spawned pool children import ONLY this module (never torch, never the
lane machinery), with a single-thread BLAS environment set by the parent at
spawn time (the ruling's guardrail 1).  The workers attach to the pool's
shared-memory buffers once (``attach``, called from the pool's initializer)
and thereafter compute values-only ``numpy.linalg.eigvalsh`` on disjoint
shards of the shared Gram batch (``eigvalsh_shard``) -- no data pickling,
no return values (results are written into the shared output buffer).
"""
from __future__ import annotations

import os

# The guardrail receipt, visible in every worker: the parent set these to
# "1" before the spawn pool was created, so the child's numpy loads with a
# single-thread BLAS (asserted at attach time).
_SINGLE_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                      "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")

_state = {}


def attach(in_name: str, out_name: str, chunk: int, dim: int):
    import numpy as np
    from multiprocessing import shared_memory
    for k in _SINGLE_THREAD_ENV:
        assert os.environ.get(k) == "1", (
            f"guardrail violated in worker: {k}={os.environ.get(k)!r} "
            "(the sidecar's workers must be single-thread by construction)")
    shm_in = shared_memory.SharedMemory(name=in_name)
    shm_out = shared_memory.SharedMemory(name=out_name)
    _state["np"] = np
    _state["in"] = np.frombuffer(
        shm_in.buf, dtype=np.float32,
        count=chunk * dim * dim).reshape(chunk, dim, dim)
    _state["out"] = np.frombuffer(
        shm_out.buf, dtype=np.float32,
        count=chunk * dim).reshape(chunk, dim)


def eigvalsh_shard(bounds):
    """Values-only eigvalsh of rows [lo, hi) of the shared Gram batch,
    written into the shared eigenvalue buffer.  Returns the shard size
    (map-order bookkeeping only; no spectra cross process boundaries)."""
    lo, hi = bounds
    np = _state["np"]
    _state["out"][lo:hi] = np.linalg.eigvalsh(_state["in"][lo:hi])
    return hi - lo
