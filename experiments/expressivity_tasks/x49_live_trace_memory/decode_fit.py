"""DR's per-follower FP64 fit of a 32-D teacher value. No target labels or hit selection."""

from dataclasses import dataclass
import numpy as np
import torch

RIDGE = 1.0 / 4096
VALUE = 32


@dataclass(frozen=True, slots=True)
class Decoder:
    """All installed tables are FP32 on CPU. No teacher writes or model activations."""

    m: torch.Tensor
    b: torch.Tensor
    k: torch.Tensor

    def __post_init__(self):
        for x, shape in (
            (self.m, (256, VALUE)),
            (self.b, (256, 2)),
            (self.k, (256, 2, VALUE)),
        ):
            if (
                x.shape != shape
                or x.dtype != torch.float32
                or x.device.type != "cpu"
                or not x.isfinite().all()
            ):
                raise ValueError("invalid frozen decoder table")
        if self.m.abs().max() > 1 or self.b.min() < 0 or self.b.max() > 255 / 64:
            raise ValueError("impossible cell means")

    def read(self, c, text, neural):
        return self.raw(c, text, neural).clamp(-1, 1)

    def raw(self, c, text, neural):
        # Discrete lookup, no teacher input. Scale and all arithmetic in FP32, in registered order.
        m = self.m[c]
        b = self.b[c]
        k = self.k[c]
        u0 = (text.float() / 64 - b[:, 0])[:, None]
        u1 = (neural.float() / 64 - b[:, 1])[:, None]
        return m + u0 * k[:, 0] + u1 * k[:, 1]

    def arrays(self):
        return {name: getattr(self, name).numpy() for name in ("m", "b", "k")}


class Moments:
    """Raw sums by observed follower. No persistent per-position information."""

    def __init__(self):
        self.count = np.zeros(256, np.int64)
        self.a = np.zeros((256, 2), np.float64)
        self.v = np.zeros((256, VALUE), np.float64)
        self.aa = np.zeros((256, 2, 2), np.float64)
        self.av = np.zeros((256, 2, VALUE), np.float64)

    def add(self, word, literal, neural, value):
        # word is observed inputs, never shifted targets. Exclude the unreadable first slot.
        if (
            type(word) is not bytes
            or not 1 <= len(word) <= 4096
            or any(
                x.shape != (len(word),)
                or x.device.type != "cpu"
                or x.dtype != torch.uint8
                or x[0].item() != 0
                for x in (literal, neural)
            )
            or value.shape != (len(word), VALUE)
            or value.dtype != torch.float32
            or value.device.type != "cpu"
            or not value.isfinite().all()
            or value.abs().max() > 1
        ):
            raise ValueError("unaligned teacher writes/arrival notes")
        c = np.frombuffer(word, dtype=np.uint8)[1:]
        if not len(c):
            return 0
        order = np.argsort(c, kind="stable")
        columns, first, counts = np.unique(
            c[order], return_index=True, return_counts=True
        )
        a = (
            np.stack((literal.numpy()[1:], neural.numpy()[1:]), axis=1)[order].astype(
                np.float64
            )
            / 64
        )
        v = value.detach().numpy()[1:][order].astype(np.float64)
        self.count[columns] += counts
        for dst, rows in (
            (self.a, a),
            (self.v, v),
            (self.aa, a[:, :, None] * a[:, None, :]),
            (self.av, a[:, :, None] * v[:, None, :]),
        ):
            dst[columns] += np.add.reduceat(rows, first, axis=0)
        return len(c)

    def arrays(self):
        return {name: getattr(self, name) for name in ("count", "a", "v", "aa", "av")}

    def solve(self):
        n = self.count
        if (
            n.sum() <= 0
            or n.min() < 0
            or not all(np.isfinite(x).all() for x in self.arrays().values())
        ):
            raise ValueError("cannot fit empty or invalid write statistics")
        seen = n > 0
        m = np.zeros_like(self.v)
        m[~seen] = self.v.sum(axis=0) / n.sum()
        m[seen] = self.v[seen] / n[seen, None]
        b = np.zeros_like(self.a)
        b[seen] = self.a[seen] / n[seen, None]
        s = self.aa[seen] / n[seen, None, None] - b[seen, :, None] * b[seen, None, :]
        s = (s + s.transpose(0, 2, 1)) * 0.5
        y = self.av[seen] / n[seen, None, None] - b[seen, :, None] * m[seen, None, :]
        # Constant code columns have identically zero centered products, including
        # with the teacher. Two rounded divisions must not invent a tiny slope.
        # Test the integer variance numerator, not a floating-point tolerance. At
        # this dose both uncentered sums in code units are exactly representable.
        for i, c in enumerate(np.flatnonzero(seen)):
            for j in range(2):
                if (
                    int(n[c]) * int(self.aa[c, j, j] * 4096)
                    == int(self.a[c, j] * 64) ** 2
                ):
                    s[i, j, :] = 0
                    s[i, :, j] = 0
                    y[i, j, :] = 0
        k = np.zeros_like(self.av)
        k[seen] = np.linalg.solve(s + RIDGE * np.eye(2), y)
        return Decoder(*(torch.from_numpy(x.astype(np.float32)) for x in (m, b, k)))
