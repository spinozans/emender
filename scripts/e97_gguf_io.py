"""Minimal self-contained GGUF v3 writer/reader for the E97 llama.cpp port.

No external dependencies (numpy + stdlib only). Supports exactly the subset
of GGUF needed by scripts/convert_e97_4b_to_gguf.py and its verifier:
scalar KVs (u32/i32/f32/bool/string), string arrays, and F32 tensors.

Layout reference: GGUF v3 spec (ggml-org/ggml docs/gguf.md). Tensor data is
stored row-major with dims listed in NE order (fastest-varying first), which
for a torch/np contiguous tensor [d0, d1, ...] means dims = [dN-1, ..., d0].
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STRING, T_ARRAY, T_U64, T_I64, T_F64 = range(13)
# GGML tensor data types (subset)
GGML_F32 = 0
GGML_Q8_0 = 8  # ggml enum ggml_type (fork pin 4fea119): fp16 scale + 32 int8 quants, 34 bytes/block


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


_SCALAR_FMT = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<B", T_U64: "<Q", T_I64: "<q", T_F64: "<d"}


def _encode_value(value):
    if isinstance(value, bool):
        return struct.pack("<I", T_BOOL) + struct.pack("<B", int(value))
    if isinstance(value, int):
        return struct.pack("<I", T_U32) + struct.pack("<I", value)
    if isinstance(value, float):
        return struct.pack("<I", T_F32) + struct.pack("<f", value)
    if isinstance(value, str):
        return struct.pack("<I", T_STRING) + _pack_str(value)
    if isinstance(value, list) and value and all(isinstance(v, str) for v in value):
        out = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_STRING) + struct.pack("<Q", len(value))
        return out + b"".join(_pack_str(v) for v in value)
    if isinstance(value, list) and value and all(isinstance(v, int) and not isinstance(v, bool) for v in value):
        out = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_U32) + struct.pack("<Q", len(value))
        return out + b"".join(struct.pack("<I", v) for v in value)
    if isinstance(value, list) and value and all(isinstance(v, float) for v in value):
        out = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_F32) + struct.pack("<Q", len(value))
        return out + b"".join(struct.pack("<f", v) for v in value)
    if isinstance(value, np.ndarray) and value.dtype == np.float32:
        out = struct.pack("<I", T_ARRAY) + struct.pack("<I", T_F32) + struct.pack("<Q", value.size)
        return out + value.astype("<f4", copy=False).tobytes()
    raise TypeError(f"unsupported GGUF value type: {type(value)}")


class GGUFWriter:
    """Accumulates KVs and F32 tensors, writes a v3 GGUF file on close()."""

    def __init__(self, path: Path, alignment: int = DEFAULT_ALIGNMENT):
        self.path = Path(path)
        self.alignment = alignment
        self.kvs: dict[str, bytes] = {}
        self.kv_order: list[str] = []
        self.tensors: list[tuple[str, np.ndarray]] = []  # fp32, logical torch shape
        # pre-packed tensors: (name, dims in NE order (fastest first), ggml_type, blob)
        self.raw_tensors: list[tuple[str, list[int], int, bytes]] = []

    def add_kv(self, key: str, value) -> None:
        if key in self.kvs:
            raise ValueError(f"duplicate kv {key}")
        self.kvs[key] = _encode_value(value)
        self.kv_order.append(key)

    def add_f32_tensor(self, name: str, array) -> None:
        """array: anything convertible to a contiguous fp32 ndarray (torch ok)."""
        if hasattr(array, "detach"):  # torch tensor
            array = array.detach().float().numpy()
        arr = np.ascontiguousarray(array, dtype=np.float32)
        self.tensors.append((name, arr))

    def add_raw_tensor(self, name: str, dims_ne: list[int], ggml_type: int, data: bytes) -> None:
        """Pre-packed tensor blob with an explicit GGML dtype (e.g. a quantized
        block layout). dims_ne is GGUF NE order (fastest-varying first), matching
        the on-disk layout of ``data``."""
        self.raw_tensors.append((name, list(dims_ne), ggml_type, data))

    def close(self) -> None:
        packed = [(name, list(arr.shape[::-1]), GGML_F32, arr.tobytes(order="C"))
                  for name, arr in self.tensors]
        packed += [(name, dims_ne, ggml_type, data)
                   for name, dims_ne, ggml_type, data in self.raw_tensors]
        header = GGUF_MAGIC + struct.pack("<IQQ", GGUF_VERSION, len(packed), len(self.kv_order))
        for key in self.kv_order:
            header += _pack_str(key) + self.kvs[key]
        # tensor infos
        offset = 0
        infos = b""
        blobs = []
        for name, dims, ggml_type, data in packed:
            assert isinstance(data, (bytes, bytearray))
            infos += _pack_str(name)
            infos += struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
            infos += struct.pack("<I", ggml_type)
            infos += struct.pack("<Q", offset)
            blobs.append(data)
            offset += len(data)
            # GGUF spec: each tensor's data offset must be aligned (ggml's loader
            # enforces this); pad after every tensor (fixed 2026-09-18, P2: the
            # unaligned file was rejected by gguf_init_from_file)
            pad = (-offset) % self.alignment
            if pad:
                blobs.append(b"\x00" * pad)
                offset += pad
        data_start = (len(header) + len(infos) + self.alignment - 1) // self.alignment * self.alignment
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(header)
            f.write(infos)
            f.write(b"\x00" * (data_start - len(header) - len(infos)))
            for blob in blobs:
                f.write(blob)


class GGUFReader:
    """Parses a GGUF v3 file written by GGUFWriter (KVs + F32 tensors)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._buf = self.path.read_bytes()
        self.kvs: dict[str, object] = {}
        self.tensor_infos: dict[str, dict] = {}
        self._parse()

    def _read_str(self, pos: int) -> tuple[str, int]:
        (n,) = struct.unpack_from("<Q", self._buf, pos)
        pos += 8
        return self._buf[pos : pos + n].decode("utf-8"), pos + n

    def _read_value(self, pos: int, vtype: int):
        if vtype == T_STRING:
            return self._read_str(pos)
        if vtype == T_BOOL:
            return bool(self._buf[pos]), pos + 1
        fmt = _SCALAR_FMT[vtype]
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self._buf, pos)[0], pos + size

    def _read_kv(self, pos: int) -> tuple[str, object, int]:
        key, pos = self._read_str(pos)
        (vtype,) = struct.unpack_from("<I", self._buf, pos)
        pos += 4
        if vtype == T_ARRAY:
            (etype,) = struct.unpack_from("<I", self._buf, pos)
            pos += 4
            (count,) = struct.unpack_from("<Q", self._buf, pos)
            pos += 8
            values = []
            for _ in range(count):
                v, pos = self._read_value(pos, etype)
                values.append(v)
            return key, values, pos
        v, pos = self._read_value(pos, vtype)
        return key, v, pos

    def _parse(self) -> None:
        if self._buf[:4] != GGUF_MAGIC:
            raise ValueError("not a GGUF file")
        version, tensor_count, kv_count = struct.unpack_from("<IQQ", self._buf, 4)
        if version != GGUF_VERSION:
            raise ValueError(f"unsupported GGUF version {version}")
        pos = 24
        for _ in range(kv_count):
            key, value, pos = self._read_kv(pos)
            self.kvs[key] = value
        for _ in range(tensor_count):
            name, pos = self._read_str(pos)
            (n_dims,) = struct.unpack_from("<I", self._buf, pos)
            pos += 4
            dims = []
            for _ in range(n_dims):
                (d,) = struct.unpack_from("<Q", self._buf, pos)
                pos += 8
                dims.append(d)
            (dtype,) = struct.unpack_from("<I", self._buf, pos)
            pos += 4
            (offset,) = struct.unpack_from("<Q", self._buf, pos)
            pos += 8
            self.tensor_infos[name] = {"dims": dims, "dtype": dtype, "offset": offset}
        # data start: first offset-aligned position after header
        kv_end = pos
        # data section start = align up
        first_offset = min((t["offset"] for t in self.tensor_infos.values()), default=0)
        self.data_start = kv_end + ((first_offset - kv_end) % 1 if first_offset < kv_end else 0)
        # align: recompute conservatively (tensor offsets are relative to aligned data start)
        align = int(self.kvs.get("general.alignment", DEFAULT_ALIGNMENT))
        self.data_start = (kv_end + align - 1) // align * align

    def get_f32_tensor(self, name: str) -> np.ndarray:
        info = self.tensor_infos[name]
        if info["dtype"] != GGML_F32:
            raise ValueError(f"{name}: expected F32")
        n = int(np.prod(info["dims"][::-1]))
        off = self.data_start + info["offset"]
        raw = np.frombuffer(self._buf, dtype="<f4", count=n, offset=off)
        return raw.reshape(info["dims"][::-1])  # logical torch shape

    def summary(self) -> dict:
        return {
            "path": str(self.path),
            "bytes": self.path.stat().st_size,
            "n_tensors": len(self.tensor_infos),
            "n_kv": len(self.kvs),
            "arch": self.kvs.get("general.architecture"),
        }

    def json_kvs(self) -> str:
        def plain(v):
            if isinstance(v, bytes):
                return v.decode("utf-8", errors="replace")
            return v

        return json.dumps({k: plain(v) for k, v in sorted(self.kvs.items())}, indent=2, sort_keys=True, default=str)
