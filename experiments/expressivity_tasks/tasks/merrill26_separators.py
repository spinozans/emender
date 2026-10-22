"""Merrill et al. 26a Defs 11/12; explicit Appendix-F sampling adaptations.

Graph indices are unary (zero is the empty word). Integer products never clip,
round, or overflow. Mod-97 final-entry classification is a secondary adaptation,
not the theorem-bearing integer-positivity task.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

TASKS = ('graph', 'integer', 'control', 'mod97')
VOCABS = {'graph': 8, 'integer': 4, 'control': 4, 'mod97': 97}
CLASSES = {'graph': 2, 'integer': 2, 'control': 2, 'mod97': 97}
SPLITS = {'train': (70000, 1, 100, 23001), 'validation': (20000, 1, 100, 23002),
          'test_id': (10000, 1, 100, 23003), 'test_ood1': (10000, 101, 200, 23004),
          'test_ood2': (10000, 201, 300, 23005)}


def reachable(source: int, edges: list[tuple[int, int]], target: int) -> int:
    """Def 11 single-pass oracle on sorted deterministic forward edges."""
    if any(i >= j for i, j in edges) or any(a[0] >= b[0] for a, b in zip(edges, edges[1:])):
        raise ValueError('edges must be deterministic and topologically sorted')
    current = source
    for i, j in edges:
        if current == target:
            return 1
        if current == i:
            current = j
    return int(current == target)


def bucket_edges(bucket: np.ndarray) -> list[tuple[int, int]]:
    edges = []
    for b in (0, 1):
        nodes = np.flatnonzero(bucket == b).tolist()
        edges.extend(zip(nodes, nodes[1:]))
    return sorted(edges)


def unary_graph(source: int, edges: list[tuple[int, int]], target: int) -> np.ndarray:
    # PAD=0, unary digit=1, SOURCE=2, EDGE=3, PAIR=4, TARGET=5, EOS=6, SOS=7.
    reachable(source, edges, target)  # Reject malformed inputs, not just bad labels.
    tokens = [7, 2] + [1] * source
    for i, j in edges:
        tokens.extend([3] + [1] * i + [4] + [1] * j)
    tokens.extend([5] + [1] * target + [6])
    return np.asarray(tokens, dtype=np.int64)


def matrix_product(matrices: np.ndarray, modulus: int | None = None) -> np.ndarray:
    """Row-major right multiplication; Python ints for the unbounded Z oracle."""
    if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not np.isin(matrices, [-1, 0, 1]).all():
        raise ValueError('expected N consecutive 3x3 matrices over {-1,0,1}')
    p = np.eye(3, dtype=object if modulus is None else np.int64)
    for a in matrices:
        p = p @ a
        if modulus is not None:
            p %= modulus
    return p


def determinant(a: np.ndarray) -> int:
    a = a.astype(np.int64)
    return int(a[0, 0] * (a[1, 1]*a[2, 2]-a[1, 2]*a[2, 1])
               - a[0, 1] * (a[1, 0]*a[2, 2]-a[1, 2]*a[2, 0])
               + a[0, 2] * (a[1, 0]*a[2, 1]-a[1, 1]*a[2, 0]))


def sample(task: str, n: int, rng: np.random.Generator, label: int | None = None):
    if n < 1 or task not in TASKS:
        raise ValueError('invalid task or length')
    if task == 'graph':
        bucket = rng.integers(0, 2, n + 1, dtype=np.int8)
        bucket[0] = 0
        bucket[-1] = 0 if label else 1
        y = reachable(0, bucket_edges(bucket), n)
        return bucket, y
    if task == 'control':
        # Exactly the X-series ParityTask(mode='final') stream: n bits, marker 2.
        bits = rng.integers(0, 2, n, dtype=np.int8)
        if label is not None:
            bits[-1] = (int(bits[:-1].sum()) % 2) ^ label
        return bits, int(bits.sum()) % 2
    while True:
        if task == 'integer':
            a = rng.choice(np.array([-1, 0, 1], dtype=np.int8), (n, 3, 3), p=[.45, .10, .45])
            y = int(matrix_product(a)[0, 0] > 0)
        else:
            # Appendix F says independent matrices, rejected until invertible.
            rows = []
            while len(rows) < n:
                m = rng.integers(-1, 2, (3, 3), dtype=np.int8)
                if determinant(m) % 97:
                    rows.append(m)
            a = np.stack(rows)
            y = int(matrix_product(a, 97)[0, 0])
        if label is None or y == label:
            return a.reshape(-1), y


def encode(task: str, payload: np.ndarray) -> np.ndarray:
    if task == 'graph':
        return unary_graph(0, bucket_edges(payload), len(payload) - 1)
    if task == 'control':
        return np.r_[payload, 2].astype(np.int64)
    # Def 12: ONLY 9N scalar tokens, no length/header/target leakage. PAD=3.
    return (payload.astype(np.int64) + 1)


@dataclass
class Dataset:
    task: str
    lengths: np.ndarray
    offsets: np.ndarray
    payload: np.ndarray
    labels: np.ndarray

    def __len__(self):
        return len(self.labels)

    def example(self, index: int):
        return encode(self.task, self.payload[self.offsets[index]:self.offsets[index + 1]]), int(self.labels[index])

    def save(self, path):
        with open(path, 'wb') as stream:
            np.savez_compressed(stream, lengths=self.lengths, offsets=self.offsets,
                                payload=self.payload, labels=self.labels)

    @classmethod
    def load(cls, task, path):
        with np.load(path, allow_pickle=False) as a:
            return cls(task, **{key: a[key] for key in ('lengths', 'offsets', 'payload', 'labels')})


def make_dataset(task: str, size: int, lo: int, hi: int, seed: int) -> Dataset:
    rng = np.random.default_rng(seed)
    lengths = rng.integers(lo, hi + 1, size, dtype=np.int32)
    desired = rng.permutation(np.arange(size) % 2) if task != 'mod97' else [None] * size
    offsets, payloads, labels = [0], [], []
    for n, label in zip(lengths, desired):
        payload, y = sample(task, int(n), rng, None if label is None else int(label))
        payloads.append(payload)
        offsets.append(offsets[-1] + len(payload))
        labels.append(y)
    return Dataset(task, lengths, np.asarray(offsets, dtype=np.int64),
                   np.concatenate(payloads), np.asarray(labels, dtype=np.int64))
