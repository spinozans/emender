"""EH: write guesses on arrival, read older counts from a live suffix key.

One full-window horizon; no address, future target or previous window is an input.
The newest eligible record at each key is pending, NOT in its counters. Neither
true nor control queries enumerate old predictions. These Python objects and
packed snapshots are experiment storage, not a compact serving implementation.
"""

from __future__ import annotations
from array import array
from dataclasses import dataclass, field
import numpy as np
import torch
from .literal import DEPTHS, MIN_AGE, check_window


@torch.inference_mode()
def guesses(z):
    if (
        z.ndim != 2
        or z.shape[1] != 256
        or not 1 <= len(z) <= 4096
        or z.device.type != "cpu"
        or z.dtype != torch.float32
        or not z.isfinite().all()
    ):
        raise ValueError("one finite aligned host forecast required")
    # No actual byte used in argmax; includes the old actual byte if the guess was correct.
    return b"\0" + z[:-1].argmax(dim=1).to(torch.uint8).numpy().tobytes()


def _check(word, *tapes):
    check_window(word, 4096)
    if not all(type(a) is bytes and len(a) == len(word) and not a[0] for a in tapes):
        raise ValueError("one aligned byte tape per field; slot zero is unused")


@dataclass(slots=True, frozen=True)
class _Donors:
    guess: bytes
    moved: bytes
    changed: bytes
    work: dict


def _donors(word, neural, actual_guess):
    """At a mistake only: *previous* mistake of this observed follower and exact neural code."""
    _check(word, neural, actual_guess)
    prior = {}
    alt = bytearray(actual_guess)
    moved = bytearray(len(word))
    changed = bytearray(len(word))
    errors = 0
    for j in range(1, len(word)):
        a = actual_guess[j]
        c = word[j]
        if a != c:
            errors += 1
            k = (c, neural[j])
            if k in prior:
                alt[j] = prior[k]
                moved[j] = 1
                changed[j] = int(alt[j] != a)
            prior[k] = a
    return _Donors(
        bytes(alt),
        bytes(moved),
        bytes(changed),
        dict(
            mistakes_written=errors,
            donor_slots_written=sum(moved),
            different_guesses_written=sum(changed),
            peak_donor_keys=len(prior),
        ),
    )


@dataclass(slots=True)
class _Key:
    pending: int
    size: int = 0
    mistakes: int = 0
    moved: int = 0
    changed: int = 0
    cells: dict = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class Packed:
    latest: np.ndarray
    depth: np.ndarray
    size: np.ndarray  # number of other occurrences at B's chosen key
    position: np.ndarray  # ascending 256*t+b; no repeated coordinate
    counts: np.ndarray  # true: C,L,N,M,W; comparison-only pass: V
    work: dict


def _scan(word, add, cols):
    """add folds the OLD pending record when a newer occurrence becomes eligible."""
    n = len(word)
    by = {}
    latest = np.full(n, -1, np.int16)
    depth = np.zeros(n, np.uint8)
    size = np.zeros(n, np.uint16)
    pos = array("I")
    nums = array("I")
    admitted = commits = active = events = errors = moved = changed = 0
    for t in range(n):
        j = t - MIN_AGE
        for d in DEPTHS:
            if j >= d:
                key = word[j - d : j]
                bucket = by.get(key)
                admitted += 1
                if bucket is None:
                    by[key] = _Key(j)
                else:
                    add(bucket, bucket.pending)
                    commits += 1
                    bucket.size += 1
                    bucket.pending = j
        for d in DEPTHS:
            if t + 1 < d:
                continue
            s = by.get(word[t + 1 - d : t + 1])
            if s is None:
                continue
            latest[t] = s.pending
            depth[t] = d
            size[t] = s.size
            if s.size:
                active += 1
                events += s.size
                errors += s.mistakes
                moved += s.moved
                changed += s.changed
                for b in sorted(s.cells):
                    pos.append(t * 256 + b)
                    if cols == 1:
                        nums.append(s.cells[b])
                    else:
                        nums.extend(s.cells[b])
            # Even a singleton at a deeper key wins; no fallback to a busier key.
            break
    ix = np.asarray(pos, dtype=np.uint32)
    v = np.asarray(nums, dtype=np.uint32).reshape(-1, cols)
    assert len(v) == len(ix) and commits + len(by) == admitted
    work = dict(
        key_admissions=admitted,
        committed_records=commits,
        peak_keys=len(by),
        peak_cells=sum(len(s.cells) for s in by.values()),
        hits=int((latest >= 0).sum()),
        active_rows=active,
        older_occurrence_uses=events,
        older_mistake_uses=errors,
        packed_cells=len(ix),
        packed_bytes=sum(a.nbytes for a in (latest, depth, size, ix, v)),
    )
    if cols == 1:
        work.update(older_displaced_uses=moved, older_changed_uses=changed)
    return Packed(latest, depth, size, ix, v, work)


def _cell(s, b):
    v = s.cells.get(b)
    if v is None:
        v = s.cells[b] = [0, 0, 0, 0, 0]
    return v


def true_counts(word, literal, neural, guess):
    _check(word, literal, neural, guess)

    def add(s, j):
        c = word[j]
        a = guess[j]
        v = _cell(s, c)
        v[0] += 1
        v[1] += literal[j]
        v[2] += neural[j]
        if a != c:
            v[3] += 1
            _cell(s, a)[4] += 1
            s.mistakes += 1

    return _scan(word, add, 5)


def displaced_counts(word, neural, guess):
    """Separate diagnostic pass. Standalone true counts never call it or build a donor map."""
    alt = _donors(word, neural, guess)

    def add(s, j):
        b = alt.guess[j]
        if b != word[j]:
            s.cells[b] = s.cells.get(b, 0) + 1
            s.mistakes += 1
            s.moved += alt.moved[j]
            s.changed += alt.changed[j]

    return _scan(word, add, 1), alt.work
