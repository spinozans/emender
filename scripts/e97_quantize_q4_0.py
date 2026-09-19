#!/usr/bin/env python3
"""Quantize an E97 F32 GGUF to GGML q4_0 for the llama.cpp CPU port.

Same method as scripts/e97_quantize_q8_0.py (port dir), extended to q4_0:
every 2-D block matmul weight becomes q4_0 blocks (fp16 scale + 16 packed
nibble bytes per 32-value block), exactly ggml-quants.c quantize_row_q4_0_ref:

  per block: amax = max |x| (first occurrence wins ties, `if (amax < fabsf(v))`)
             d    = max / -8          (signed max, so quants live in [0, 15])
             id   = d ? 1/d : 0
             q_j    = MIN(15, (int8)trunc(x_j * id + 8.5f))      j in  [0, 16)
             q_j+16 = MIN(15, (int8)trunc(x_{j+16} * id + 8.5f))
             qs[j]  = q_j | (q_{j+16} << 4)

Kept in F32 (identical policy to the q8_0 quantizer):
  - token_embd.weight (raw row lookups + tied lm_head matmul),
  - all 1-D tensors (attn/ffn/output norms, mixer_a_log, mixer_dt_bias).

All 2-D inputs (3840, 9600) are divisible by QK4_0=32.

The codec is verified in-process against the C reference
(quantize_row_q4_0_ref via scripts/e97_quantize_q4_refcheck.c built against
the pinned fork's static ggml) and by an independent numpy decode round-trip.

Re-qualification: same frozen 32-prompt panel, per-quant top-1 + KL(top-64)
against the f32 GGUF forward (the reference is the f32 GGUF, not the GPU),
gate from the q8 precedent (agent-GGUF gate: top-1 >= 0.97, KL <= 0.05).
"""

from __future__ import annotations

import argparse
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e97_gguf_io import GGUFWriter, GGML_F32, GGML_Q4_0  # noqa: E402

QK4_0 = 32
Q4_0_BLOCK = np.dtype([("d", "<f2"), ("qs", "u1", QK4_0 // 2)])  # 18 bytes, packed

# 2-D matmul weights inside a block (name suffix -> quantized)
QUANT_2D = re.compile(
    r"^blk\.\d+\.(mixer_qkv|mixer_a|mixer_erase|mixer_vw|mixer_g|mixer_o|ffn_gate|ffn_up|ffn_down)\.weight$"
)

FORK = Path("/mnt/nvme2n1/erikg/e97_systematic_posttraining/e97-llama-cpp-port-v1/llama.cpp-fork")


def read_gguf_tensor_index(path: Path):
    """Minimal bool-preserving GGUF v3 header parse (mmap; independent of the
    P1 reader on purpose). Returns (kvs, {name: (offset, nbytes, ne)})."""
    f = open(path, "rb")
    assert f.read(4) == b"GGUF"
    (version,) = struct.unpack("<I", f.read(4))
    assert version == 3
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

    def r_str():
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8")

    def r_value(t):
        if t == 8:
            return r_str()
        if t == 9:
            (et,) = struct.unpack("<I", f.read(4))
            (n,) = struct.unpack("<Q", f.read(8))
            return [r_value(et) for _ in range(n)]
        if t == 7:
            return bool(struct.unpack("<B", f.read(1))[0])
        fmt = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

    kvs = {}
    for _ in range(n_kv):
        key = r_str()
        (t,) = struct.unpack("<I", f.read(4))
        kvs[key] = r_value(t)

    infos = {}
    for _ in range(n_tensors):
        name = r_str()
        (n_dims,) = struct.unpack("<I", f.read(4))
        ne = list(struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims)))  # fastest first
        (dtype,) = struct.unpack("<I", f.read(4))
        (offset,) = struct.unpack("<Q", f.read(8))
        nbytes = int(np.prod(ne)) * {0: 4, 1: 18, 7: 1, 8: 34}[dtype]  # f32, q4_0, i8-ish, q8_0
        infos[name] = (offset, nbytes, ne)
    align = int(kvs.get("general.alignment", 32))
    data_start = (f.tell() + align - 1) // align * align
    f.close()
    mm = np.memmap(path, dtype="<f4", mode="r")
    return kvs, infos, mm, data_start


def quantize_q4_0(arr_f32: np.ndarray) -> bytes:
    """arr: [out, in] fp32, C-contiguous; quantize along the last (input) axis.

    Bit-exact replication of ggml-quants.c quantize_row_q4_0_ref (see module
    docstring): fp32 amax scan, signed max, d = max/-8, fp16 scale storage
    (round-half-even), nibble = MIN(15, (int8)trunc(x*id + 8.5)).
    """
    x = np.ascontiguousarray(arr_f32, dtype=np.float32)
    assert x.ndim == 2 and x.shape[1] % QK4_0 == 0, x.shape
    xb = x.reshape(-1, QK4_0)
    absx = np.abs(xb)
    # `if (amax < fabsf(v))` -> strictly-less: the FIRST absolute max wins
    idx = np.argmax(absx, axis=1)
    rows = np.arange(xb.shape[0])
    maxv = xb[rows, idx]                                        # signed value at amax
    d = (maxv / np.float32(-8.0)).astype(np.float32)
    inv = np.divide(np.float32(1.0), d, out=np.zeros_like(d), where=d != 0)
    q = (xb * inv[:, None] + np.float32(8.5)).astype(np.int8)   # (int8) cast == trunc
    q = np.minimum(q, 15).astype(np.uint8)
    lo, hi = q[:, :16], q[:, 16:]
    qs = (lo | (hi << 4)).astype(np.uint8)
    blk = np.empty(xb.shape[0], dtype=Q4_0_BLOCK)
    blk["d"] = d.astype(np.float16)                             # round-half-even
    blk["qs"] = qs
    return blk.tobytes()


def dequantize_q4_0(blob: bytes, n: int) -> np.ndarray:
    """Independent decode: ggml dequantize_row_q4_0 semantics."""
    blk = np.frombuffer(blob, dtype=Q4_0_BLOCK)
    d = blk["d"].astype(np.float32)[:, None]
    lo = (blk["qs"] & 0x0F).astype(np.int8) - 8
    hi = (blk["qs"] >> 4).astype(np.int8) - 8
    x = np.empty((len(blk), QK4_0), dtype=np.float32)
    x[:, :16] = lo
    x[:, 16:] = hi
    return (x * d).reshape(-1)[:n]


def verify_codec(fork: Path) -> None:
    """Two independent checks that quantize_q4_0 is ggml-exact:

    1. C reference: quantize_row_q4_0_ref from the pinned fork's static ggml
       quantizes random vectors; our numpy output must be byte-identical.
    2. numpy decode round-trip vs ggml dequantize_row_q4_0 semantics.
    """
    rng = np.random.default_rng(31337)
    n = 32 * 4096
    x = rng.standard_normal(n).astype(np.float32) * np.float32(0.05)
    mine = quantize_q4_0(x.reshape(128, n // 128))

    with tempfile.TemporaryDirectory() as td:
        refbin = Path(td) / "ref.bin"
        xs = Path(td) / "x.bin"
        xs.write_bytes(x.astype("<f4").tobytes())
        check_c = Path(__file__).resolve().parent / "e97_quantize_q4_refcheck.c"
        subprocess.run(
            ["g++", "-O2", "-std=c++17", "-fopenmp",
             "-I", str(FORK / "ggml" / "include"),
             "-I", str(FORK / "ggml" / "src"), "-I", str(FORK / "build-plain" / "ggml" / "src"),
             str(check_c), str(FORK / "build-plain" / "ggml" / "src" / "libggml.a"),
             str(FORK / "build-plain" / "ggml" / "src" / "libggml-cpu.a"),
             str(FORK / "build-plain" / "ggml" / "src" / "libggml-base.a"),
             "-lpthread", "-lm", "-ldl", "-fopenmp", "-o", str(Path(td) / "refcheck")],
            check=True, capture_output=True)
        subprocess.run([str(Path(td) / "refcheck"), str(xs), str(refbin)], check=True)
        ref = refbin.read_bytes()

    assert len(ref) == len(mine), (len(ref), len(mine))
    assert ref == mine, "numpy q4_0 codec is NOT byte-identical to quantize_row_q4_0_ref"
    print(f"  codec check 1: byte-identical to ggml quantize_row_q4_0_ref over {n} values ({len(ref)} bytes)")

    # decode round-trip: error bound = one half q4 step from the +8.5 trunc
    # rounding, PLUS one full extra step for the block element at x = -max
    # when max > 0 (MIN(15, ...) clamps its quant from 8 down to 7), PLUS the
    # fp16 scale-storage relative error (<= 2^-12 per element)
    y = dequantize_q4_0(mine, n)
    blk = np.frombuffer(ref, dtype=Q4_0_BLOCK)
    dr = blk["d"].astype(np.float32)[:, None]
    lo = (blk["qs"] & 0x0F).astype(np.int8) - 8
    hi = (blk["qs"] >> 4).astype(np.int8) - 8
    yr = np.empty((len(blk), QK4_0), dtype=np.float32)
    yr[:, :16] = lo * dr[:, 0][:, None]
    yr[:, 16:] = hi * dr[:, 0][:, None]
    assert np.array_equal(y, yr.reshape(-1)), "numpy decode disagrees with the C dequantize formula"
    err = np.abs(x - y)
    step = np.abs(np.repeat(blk["d"].astype(np.float32), QK4_0).reshape(-1))
    fp16_rel = np.float32(2.0 ** -12)
    assert np.all(err <= step * 1.5 + np.abs(x) * fp16_rel + np.float32(1e-12)), \
        "round-trip error exceeds the half-step + clamp + fp16-scale bound"
    print(f"  codec check 2: numpy decode round-trip matches dequantize_row_q4_0 (max err {err.max():.3e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-codec-check", action="store_true")
    args = ap.parse_args()

    if not args.skip_codec_check:
        print("verifying the q4_0 codec against ggml:")
        verify_codec(FORK)

    kvs, infos, mm, data_start = read_gguf_tensor_index(args.gguf)
    assert kvs.get("general.architecture") in ("emender_e97", "e97"), kvs.get("general.architecture")

    w = GGUFWriter(args.out)
    for key, value in kvs.items():
        if key == "general.file_type":
            continue  # re-emitted below as MOSTLY_Q4_0
        w.add_kv(key, value)
    w.add_kv("general.file_type", 2)  # LLAMA_FTYPE_MOSTLY_Q4_0 (except 1d/embd tensors)
    w.add_kv("general.quantization_note",
             "q4_0 block weights (ggml quantize_row_q4_0_ref, byte-verified); "
             "token_embd + 1-D tensors kept F32 (same policy as the q8_0 quant)")

    n_q4 = n_f32 = 0
    bytes_q4 = bytes_f32 = 0
    for name, (off, nbytes, ne) in infos.items():
        arr = np.asarray(mm[(data_start + off) // 4 : (data_start + off + nbytes) // 4]).view("<f4")
        logical = arr.reshape(tuple(reversed(ne)))  # torch [out, in]
        if QUANT_2D.match(name):
            blob = quantize_q4_0(logical)
            w.add_raw_tensor(name, list(ne), GGML_Q4_0, blob)
            n_q4 += 1
            bytes_q4 += len(blob)
        else:
            w.add_raw_tensor(name, list(ne), GGML_F32, logical.astype("<f4").tobytes())
            n_f32 += 1
            bytes_f32 += nbytes
        print(f"  {name}: ne={ne} -> {'q4_0' if QUANT_2D.match(name) else 'f32'}", flush=True)

    w.close()
    total = bytes_q4 + bytes_f32
    print(f"tensors: {n_q4} q4_0 ({bytes_q4/1e9:.2f} GB) + {n_f32} f32 ({bytes_f32/1e9:.2f} GB)")
    print(f"out: {args.out} ({args.out.stat().st_size/1e9:.2f} GB data {total/1e9:.2f} GB; "
          f"fp32 source was {args.gguf.stat().st_size/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
