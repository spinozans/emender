"""Assemble new window kernels for SM89 offline, without touching a CUDA context."""

import hashlib
import json
import os
from pathlib import Path
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from .triton_prefix import _scan
from .triton_leaf import _leaf, _bscore, _bvalue, row_tile
from .triton_nll import _nll, _dnll


# Reported maximum dynamic shared memory per block on the metered RTX 6000 Ada.
# Compiling a cubin alone does not check the limit enforced at handle initialization.
METER_SHARED_LIMIT = 101376


def require_shared_fit(name, const, shared):
    if shared > METER_SHARED_LIMIT:
        raise RuntimeError(
            f"{name} {const} needs {shared} shared bytes; meter limit {METER_SHARED_LIMIT}"
        )


def check():
    if (
        os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or os.getenv("TRITON_INTERPRET") == "1"
    ):
        raise RuntimeError("offline only")
    out = []

    def compile(fn, const, nw=4, *, n=None, aligned=False):
        const = dict(const)
        if aligned and n == 1:
            const["N"] = 1
        sig = {
            k: "constexpr"
            if k in const
            else "i32"
            if k == "N"
            else "*i64"
            if k == "Spans" or (k == "Y" and fn in (_nll, _dnll))
            else "*fp32"
            for k in fn.arg_names
        }
        attrs = {
            (i,): [["tt.divisibility", 16]]
            for i, (k, t) in enumerate(sig.items())
            if aligned and (t.startswith("*") or (k == "N" and n % 16 == 0))
        }
        mod = triton.compile(
            ASTSource(fn, signature=sig, constexprs=const, attrs=attrs),
            target=GPUTarget("cuda", 89, 32),
            options={"num_warps": nw},
        )
        require_shared_fit(fn.__name__, const, mod.metadata.shared)
        blob = mod.asm["cubin"]
        out.append(
            dict(
                kernel=fn.__name__,
                constants=const,
                n_rows=n,
                aligned=aligned,
                shared_bytes=mod.metadata.shared,
                warps=nw,
                bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
            )
        )

    for d in (7, 32, 45):
        for rev in (False, True):
            compile(
                _scan,
                dict(D=d, BT=32, BF=min(32, triton.next_power_of_2(d)), REVERSE=rev),
            )
    # Include every original device leaf fixture, the full-step geometry, the failed specialization
    # (including its asymmetric BF/BV), and the largest supported leaf size.
    for n, r, w, leaf in (
        (1, 1, 1, 1),
        (16, 11, 7, 3),
        (16, 11, 12, 3),
        (79, 13, 47, 11),
        (257, 39, 65, 128),
        (205, 33, 9, 201),
        (205, 39, 65, 201),
        (16384, 32, 32, 128),
        (513, 39, 65, 256),
    ):
        for fn in (_leaf, _bscore, _bvalue):
            const = dict(
                R=r,
                W=w,
                L=leaf,
                BR=row_tile(leaf, backward=fn != _leaf),
                BF=max(16, min(32, triton.next_power_of_2(r))),
                BV=max(16, min(32, triton.next_power_of_2(w))),
            )
            for aligned in (False, True):
                compile(fn, const, n=n, aligned=aligned)
    for vocab in (12, 1024, 50281):
        bv = min(4096, triton.next_power_of_2(vocab))
        for fn in (_nll, _dnll):
            compile(fn, dict(V=vocab, BV=bv), 8 if bv >= 2048 else 4)
    return dict(
        rows=out,
        triton=triton.__version__,
        target="sm89",
        shared_limit=METER_SHARED_LIMIT,
        device_launches=0,
    )


if __name__ == "__main__":
    data = check()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    print("assembled", len(data["rows"]), "variants offline")
