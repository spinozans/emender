"""Explicit local CPU build/load of the exact literal surprise writer; no implicit compilation or fallback."""

from __future__ import annotations
import ctypes as C
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter
import torch

SOURCE = Path(__file__).with_name("text_tape.cc")
CXX = Path("/usr/bin/g++")
BINARY = "text_tape.so"
FLAGS = (
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-shared",
    "-fno-fast-math",
    "-ffp-contract=off",
    "-Wall",
    "-Wextra",
    "-Werror",
)
KEYS = (
    "text_mark_storage_bytes",
    "text_arrivals_encoded",
    "text_order_zero_updates",
    "text_context_lookups",
    "text_context_updates",
    "text_seen_context_lookups",
    "text_distinct_contexts",
    "text_distinct_pairs",
    "text_marks_saturated",
)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def build(folder):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or os.environ.get("GPU_LEASE_HELD"):
        raise ValueError("CPU-only writer build required")
    folder = Path(folder)
    folder.mkdir(mode=0o700)
    lib = folder / BINARY
    src = digest(SOURCE)
    compiler = CXX.resolve(strict=True)
    compiler_sha = digest(compiler)
    version = subprocess.check_output([compiler, "--version"], text=True).splitlines()[
        0
    ]
    tic = perf_counter()
    subprocess.run([compiler, *FLAGS, str(SOURCE), "-o", str(lib)], check=True)
    elapsed = perf_counter() - tic
    if digest(SOURCE) != src or digest(compiler) != compiler_sha:
        raise ValueError("source or compiler changed during build")
    note = dict(
        abi=1,
        flags=list(FLAGS),
        source_sha256=src,
        compiler_sha256=compiler_sha,
        compiler=version,
        build_wall_s=elapsed,
        library_sha256=digest(lib),
        library_bytes=lib.stat().st_size,
    )
    with (folder / "build.json").open("x") as f:
        json.dump(note, f, sort_keys=True, allow_nan=False)
        f.write("\n")
    return note


def _load(folder, note):
    """Bind the actual local binary to the separately supplied build receipt once, not per byte."""
    folder = Path(folder)
    lib = folder / BINARY
    f = folder / "build.json"
    if (
        not lib.is_file()
        or lib.is_symlink()
        or not f.is_file()
        or f.is_symlink()
        or json.loads(f.read_bytes()) != note
        or note["abi"] != 1
        or note["flags"] != list(FLAGS)
        or note["source_sha256"] != digest(SOURCE)
        or note["library_sha256"] != digest(lib)
        or note["library_bytes"] != lib.stat().st_size
    ):
        raise ValueError("unbound or stale native writer")
    fn = C.CDLL(str(lib.resolve())).x49_text_tape_v1
    fn.argtypes = (C.c_char_p, C.c_uint, C.POINTER(C.c_ubyte), C.POINTER(C.c_uint32))
    fn.restype = C.c_int

    def writer(word):
        if type(word) is not bytes or not 1 <= len(word) <= 4096:
            raise ValueError("expected 1..4096 observed bytes")
        n = len(word)
        out = bytearray(n)
        stats = (C.c_uint32 * len(KEYS))()
        # Embedded NULs are data. Native uses the explicit byte length, never strlen.
        rc = fn(word, n, (C.c_ubyte * n).from_buffer(out), stats)
        if rc != 0:
            raise ValueError(f"native literal writer refused: {rc}")
        return torch.frombuffer(out, dtype=torch.uint8), dict(
            zip(KEYS, stats, strict=True)
        )

    return writer
