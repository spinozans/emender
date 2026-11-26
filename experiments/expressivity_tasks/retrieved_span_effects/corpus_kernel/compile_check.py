"""Offline CUDA codegen/assembler check. No CUDA context, launch or timing claim."""

import hashlib
import json
import os
from pathlib import Path
import sys

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

from .attention import TILE
from .triton_frontier import _seek
from .triton_tiles import _forward, _backward


def check():
    if (
        os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("TRITON_INTERPRET") == "1"
        or os.getenv("GPU_LEASE_HELD")
    ):
        raise RuntimeError("offline compile only")
    records = []
    for dim, width in ((7, 11), (27, 71), (32, 128)):
        for mode in (0, 1, 2):
            const = dict(
                D=dim,
                W=width,
                BR=TILE,
                BD=triton.next_power_of_2(dim),
                BV=min(32, triton.next_power_of_2(width)),
                MODE=mode,
                SCALE=dim**-0.5,
            )
            for fn in (_forward, _backward):
                sig = {
                    k: (
                        "constexpr"
                        if k in const
                        else "i32"
                        if k in ("G", "N")
                        else "*i64"
                        if k == "Desc"
                        else "*fp32"
                    )
                    for k in fn.arg_names
                }
                mod = triton.compile(
                    ASTSource(fn, signature=sig, constexprs=const),
                    target=GPUTarget("cuda", 89, 32),
                    options={"num_warps": 4},
                )
                blob = mod.asm["cubin"]
                records.append(
                    dict(
                        name=fn.__name__,
                        mode=mode,
                        dim=dim,
                        width=width,
                        bytes=len(blob),
                        sha256=hashlib.sha256(blob).hexdigest(),
                    )
                )
    for all_depths in (False, True):
        const = dict(T=17, DEPTH=17 if all_depths else 2, IT=16, BQ=16, ALL=all_depths)
        sig = {
            k: (
                "constexpr"
                if k in const
                else "i32"
                if k in ("B", "N")
                else "*i32"
                if k == "Lengths"
                else "*i64"
            )
            for k in _seek.arg_names
        }
        mod = triton.compile(
            ASTSource(_seek, signature=sig, constexprs=const),
            target=GPUTarget("cuda", 89, 32),
            options={"num_warps": 4},
        )
        blob = mod.asm["cubin"]
        records.append(
            dict(
                name="_seek",
                all_depths=all_depths,
                bytes=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
            )
        )
    return dict(
        compiler=triton.__version__,
        target="cuda:sm89",
        kernels=records,
        device_launches=0,
    )


if __name__ == "__main__":
    obj = check()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    print("offline cubins:", len(obj["kernels"]))
