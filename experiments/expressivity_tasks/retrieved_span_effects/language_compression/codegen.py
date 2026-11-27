"""Offline SM89 resource check; never initializes CUDA; compilation is not a device pass."""

import hashlib
import json
import os
import sys
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from ..corpus_windows.triton_mix import _mix, _dmix
from ..corpus_windows.triton_parent import _parent, _unparent
from ..corpus_windows.triton_leaf import _bvalue, row_tile
from ..corpus_windows.codegen import require_shared_fit


def check():
    if (
        os.environ.get("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or os.getenv("TRITON_INTERPRET")
    ):
        raise RuntimeError("offline only")
    result = []

    def one(fn, c):
        sig = {
            k: (
                "constexpr"
                if k in c
                else "*i64"
                if k in ("Ids", "Off")
                else "i32"
                if k == "N"
                else "*fp32"
            )
            for k in fn.arg_names
        }
        attrs = {
            (i,): [["tt.divisibility", 16]]
            for i, t in enumerate(sig.values())
            if t.startswith("*")
        }
        m = triton.compile(
            ASTSource(fn, signature=sig, constexprs=c, attrs=attrs),
            target=GPUTarget("cuda", 89, 32),
            options={"num_warps": 4},
        )
        require_shared_fit(fn.__name__, c, m.metadata.shared)
        result.append(
            dict(
                name=fn.__name__,
                constants=c,
                smem=m.metadata.shared,
                sha256=hashlib.sha256(m.asm["cubin"]).hexdigest(),
            )
        )

    for w in (7, 64, 73):
        for fn in (_mix, _dmix):
            one(fn, dict(W=w, BR=16, BV=min(64, triton.next_power_of_2(w))))
    for r, w in ((5, 8), (13, 73), (32, 64)):
        for fn in (_parent, _unparent):
            one(
                fn,
                dict(
                    R=r,
                    W=w,
                    BF=min(8, triton.next_power_of_2(r)),
                    BV=min(64, triton.next_power_of_2(w)),
                ),
            )
    one(_bvalue, dict(L=201, R=33, W=9, BR=row_tile(201, backward=True), BF=32, BV=16))
    return result


if __name__ == "__main__":
    r = check()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(r, f, indent=2, sort_keys=True)
        f.write("\n")
    print(len(r), "SM89 cases, max shared bytes", max(v["smem"] for v in r))
