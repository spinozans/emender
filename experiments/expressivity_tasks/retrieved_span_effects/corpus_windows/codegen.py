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
from .triton_leaf import _leaf, _bscore, _bvalue
from .triton_nll import _nll, _dnll


def check():
    if (
        os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or os.getenv("TRITON_INTERPRET") == "1"
    ):
        raise RuntimeError("offline only")
    out = []

    def compile(fn, const, nw=4):
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
        mod = triton.compile(
            ASTSource(fn, signature=sig, constexprs=const),
            target=GPUTarget("cuda", 89, 32),
            options={"num_warps": nw},
        )
        blob = mod.asm["cubin"]
        out.append(
            dict(
                kernel=fn.__name__,
                constants=const,
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
    for leaf, r, w in ((3, 11, 7), (11, 13, 47), (128, 32, 32), (201, 39, 65)):
        const = dict(
            R=r,
            W=w,
            L=leaf,
            BR=max(16, triton.next_power_of_2(leaf)),
            BF=max(16, min(32, triton.next_power_of_2(r))),
            BV=max(16, min(32, triton.next_power_of_2(w))),
        )
        for fn in (_leaf, _bscore, _bvalue):
            compile(fn, const)
    for vocab in (12, 1024, 50281):
        bv = min(4096, triton.next_power_of_2(vocab))
        for fn in (_nll, _dnll):
            compile(fn, dict(V=vocab, BV=bv), 8 if bv >= 2048 else 4)
    return dict(rows=out, triton=triton.__version__, target="sm89", device_launches=0)


if __name__ == "__main__":
    data = check()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(data, f, sort_keys=True, indent=2)
        f.write("\n")
    print("assembled", len(data["rows"]), "variants offline")
