"""Private, byte-bounded feature transport for a *frozen deterministic causal* host.

The expected binding and file digest come from the trusted caller, not from parsing the file.
No addresses, labels, controller state, executable objects or unbounded shape metadata. A file
contains one length bucket of document-bounded observations. Local retention is the owner's
responsibility. This is not a cache for a trainable encoder or a substitute for access control.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
from stat import S_ISREG
import struct
import tempfile

import numpy as np
import torch


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class Window:
    document: int
    start: int
    length: int
    input_sha: str

    @classmethod
    def observe(cls, document, start, data):
        if type(data) is not bytes or not data:
            raise ValueError("nonempty raw observed bytes required")
        return cls(document, start, len(data), sha256(data).hexdigest())

    def __post_init__(self):
        if (
            any(type(v) is not int or v < 0 for v in (self.document, self.start))
            or type(self.length) is not int
            or self.length < 1
            or not _digest(self.input_sha)
        ):
            raise ValueError("invalid window identity")


@dataclass(frozen=True)
class PacketBinding:
    # encoder_sha includes model recipe/weights, kernel, precision, mode, bridge and microbatching.
    encoder_sha: str
    source_sha: str
    split: str
    width: int
    rows: tuple[Window, ...]

    def __post_init__(self):
        if (
            not _digest(self.encoder_sha)
            or not _digest(self.source_sha)
            or self.split not in ("train", "dev", "test")
            or type(self.width) is not int
            or self.width < 1
            or type(self.rows) is not tuple
            or not 0 < len(self.rows) <= 4096
            or not all(type(r) is Window for r in self.rows)
            or len({(r.document, r.start) for r in self.rows}) != len(self.rows)
            or len({r.length for r in self.rows}) != 1
        ):
            raise ValueError("invalid or ragged frozen binding")

    @property
    def shape(self):
        return len(self.rows), self.rows[0].length, self.width + 256

    def layout(self, byte_limit):
        if type(byte_limit) is not int or byte_limit < 1:
            raise ValueError("invalid byte limit")
        header = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "ascii"
        )
        prefix = b"FZBYT001" + struct.pack(">I", len(header)) + header
        b, t, d = self.shape
        size = len(prefix) + 4 * b * t * d
        if size > byte_limit:
            raise ValueError("frozen packet exceeds budget")
        return prefix, size


def write_packet(path, binding, values, *, byte_limit):
    """No replace. Publish only a finite, complete, private FP32 packet in an owned directory."""
    path = Path(path)
    prefix, size = binding.layout(byte_limit)
    if (
        not isinstance(values, torch.Tensor)
        or values.device.type != "cpu"
        or values.dtype != torch.float32
        or values.requires_grad
        or tuple(values.shape) != binding.shape
        or not torch.isfinite(values).all().item()
    ):
        raise ValueError("expected finite detached causal CPU features")
    # Private snapshot; don't let the buffer change underneath a digest/publish pair.
    raw = values.contiguous().numpy().astype("<f4", copy=False).tobytes()
    if len(prefix) + len(raw) != size:
        raise ValueError("wrong packet size")
    h = sha256(prefix)
    h.update(raw)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".packet-")  # mode 0600
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(prefix)
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.link(tmp, path)  # atomic, FileExistsError on collision
        parent = os.open(path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        os.unlink(tmp)
    return h.hexdigest(), size


class FrozenPacket:
    def __init__(self, binding, values):
        self.binding = binding
        self._values = values

    @classmethod
    def load(cls, path, binding, expected_sha, *, byte_limit):
        prefix, size = binding.layout(byte_limit)
        if not _digest(expected_sha):
            raise ValueError("independent SHA256 required")
        # Use the same regular fd for size and contents; never map mutable caller storage into tensors.
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as f:
            st = os.fstat(f.fileno())
            if not S_ISREG(st.st_mode) or st.st_size != size:
                raise ValueError("incomplete or excessive packet")
            if f.read(len(prefix)) != prefix:
                raise ValueError("wrong frozen host/source/rows")
            raw = f.read(size - len(prefix))
            if len(raw) != size - len(prefix) or f.read(1):
                raise ValueError("incomplete or trailing packet")
        h = sha256(prefix)
        h.update(raw)
        if h.hexdigest() != expected_sha:
            raise ValueError("frozen packet digest mismatch")
        vals = torch.from_numpy(
            np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True).reshape(binding.shape)
        )
        if not vals.isfinite().all().item():
            raise ValueError("nonfinite stored feature")
        return cls(binding, vals)

    def encoder(self, sessions, *, row_indices=None):
        """Bind fresh origins and actual IDs, optionally selecting distinct rows of this same verified file."""
        rows = self.binding.rows
        if row_indices is not None:
            if (
                not isinstance(row_indices, (tuple, list))
                or not 0 < len(row_indices) <= len(rows)
                or any(type(i) is not int or not 0 <= i < len(rows) for i in row_indices)
                or len(set(row_indices)) != len(row_indices)
            ):
                raise ValueError("invalid packet row selection")
            row_indices = tuple(row_indices)
            rows = tuple(rows[i] for i in row_indices)
        ss = tuple(sessions)

        def encode(ids):
            if (
                len(ss) != len(rows)
                or len(set(s.id for s in ss)) != len(ss)
                or any(
                    (s.document, s.start) != (r.document, r.start) or s.visible or s.pending
                    for s, r in zip(ss, rows)
                )
                or ids.device.type != "cpu"
                or ids.dtype != torch.long
                or tuple(ids.shape) != (len(rows), rows[0].length)
                or not ((ids >= 0) & (ids < 256)).all().item()
            ):
                raise ValueError("wrong packet sessions or input shape")
            if any(
                sha256(row.tobytes()).hexdigest() != r.input_sha
                for row, r in zip(ids.to(torch.uint8).numpy(), rows, strict=True)
            ):
                raise ValueError("changed raw host observations")
            # Fresh owned outputs; caller cannot mutate the verified artifact or another consumer.
            values = (
                self._values.clone()
                if row_indices is None
                else self._values.index_select(0, torch.tensor(row_indices, device="cpu", dtype=torch.long))
            )
            return values.split((self.binding.width, 256), dim=-1)

        return encode
