"""Exhaustive oracle on invented bytes, independent of the mutable suffix tables."""

import numpy as np
import pytest
import torch
from experiments.expressivity_tasks.x49_live_trace_memory import (
    error_history_index as ix,
)


def word():
    return b"".join(
        b"color: " + bytes([c]) + b" \0\xff\x1e key=xyz; " for c in b"bgrbbrrgbr"
    )


def oracle(w, literal, n, a):
    """Dense C,L,N,M,W,V by scanning the *observed prefix only* at each query."""
    alt = list(a)
    moved = np.zeros(len(w), bool)
    for j in range(1, len(w)):
        if a[j] != w[j]:
            ds = [
                k for k in range(1, j) if w[k] == w[j] and n[k] == n[j] and a[k] != w[k]
            ]
            if ds:
                alt[j] = a[ds[-1]]
                moved[j] = True
    h = np.zeros((len(w), 256, 6), np.int64)
    latest = []
    depth = []
    uses = sw = 0
    for t in range(len(w)):
        js = []
        d = 0
        for d in (8, 4, 2, 1):
            if t + 1 >= d:
                js = [
                    j
                    for j in range(d, t - 16 + 1)
                    if w[j - d : j] == w[t + 1 - d : t + 1]
                ]
            if js:
                break
        latest.append(js[-1] if js else -1)
        depth.append(d if js else 0)
        for j in js[:-1]:
            c = w[j]
            h[t, c, 0] += 1
            h[t, c, 1] += literal[j]
            h[t, c, 2] += n[j]
            if c != a[j]:
                h[t, c, 3] += 1
                h[t, a[j], 4] += 1
                h[t, alt[j], 5] += 1
                uses += int(moved[j])
                sw += int(a[j] != alt[j])
    return h, latest, depth, uses, sw


def dense(tab):
    n = len(tab.size)
    out = np.zeros((n, 256, tab.counts.shape[1]), np.int64)
    pos = tab.position
    assert np.all(pos[1:] > pos[:-1]) and (pos < n * 256).all()
    out[pos >> 8, pos & 255] = tab.counts
    return out


def test_all_keys_pending_records_and_sham_vs_independent_prefix_scans():
    w = word()
    literal = b"\0" + bytes(255 - i % 7 for i in range(len(w) - 1))
    n = b"\0" + bytes(i % 3 for i in range(len(w) - 1))
    a = b"\0" + bytes(w[j] if j % 4 == 2 else 200 + j % 5 for j in range(1, len(w)))
    h, js, ds, moved, changed = oracle(w, literal, n, a)
    t = ix.true_counts(w, literal, n, a)
    s, work = ix.displaced_counts(w, n, a)
    np.testing.assert_array_equal(dense(t), h[:, :, :5])
    np.testing.assert_array_equal(dense(s), h[:, :, 5:])
    np.testing.assert_array_equal(t.latest, js)
    np.testing.assert_array_equal(t.depth, ds)
    assert not np.array_equal(h[:, :, 4], h[:, :, 5]) and 0 < changed <= moved <= int(
        h[:, :, 3].sum()
    )
    for tab in (t, s):
        np.testing.assert_array_equal(tab.size, h[:, :, 0].sum(1))
        assert (
            tab.work["older_occurrence_uses"] == h[:, :, 0].sum()
            and tab.work["older_mistake_uses"]
            == h[:, :, 3].sum()
            == h[:, :, 4].sum()
            == h[:, :, 5].sum()
        )
    assert (s.work["older_displaced_uses"], s.work["older_changed_uses"]) == (
        moved,
        changed,
    ) and 0 < work["different_guesses_written"] <= work["donor_slots_written"] < work[
        "mistakes_written"
    ]
    for end in (1, 18, 27, 59, len(w) - 1):
        small = ix.true_counts(w[:end], literal[:end], n[:end], a[:end])
        sh, _ = ix.displaced_counts(w[:end], n[:end], a[:end])
        np.testing.assert_array_equal(dense(small), h[:end, :, :5])
        np.testing.assert_array_equal(dense(sh), h[:end, :, 5:])
        np.testing.assert_array_equal(small.latest, js[:end])


def test_open_zero_overlap_full_window_large_sums_and_no_singleton_fallback():
    w = bytes(4096)
    tape = b"\x00" + b"\xff" * 4095
    t = ix.true_counts(w, tape, tape, tape)
    v = dense(t)
    assert np.all(t.latest[:17] == -1) and (t.depth[18], t.latest[18], t.size[18]) == (
        2,
        2,
        0,
    )
    assert (t.depth[24], t.size[24], t.size[25]) == (8, 0, 1) and (
        t.latest[-1],
        t.size[-1],
    ) == (4079, 4071)
    np.testing.assert_array_equal(v[-1, 0], [4071, 255 * 4071, 255 * 4071, 4071, 0])
    assert v[-1, 255, 4] == 4071 and not np.any(v[24])
    for bad in (tape[:-1], b"Q" + tape[1:], bytearray(tape)):
        with pytest.raises(ValueError):
            ix.true_counts(w, bad, tape, tape)
    with pytest.raises(ValueError):
        ix.true_counts(w + b"z", tape, tape, tape)


def test_guess_written_one_row_late_first_argmax_including_correct_and_no_future():
    z = torch.zeros(7, 256, dtype=torch.float32)
    z[0, 255] = 3
    z[1, [4, 17]] = 3
    z[3, 0] = 8
    z[4, [127, 128]] = 9
    z[5, 33] = 4
    z[6, 16] = 15
    assert ix.guesses(z) == bytes([0, 255, 4, 0, 0, 127, 33])
    assert all(ix.guesses(z[:t]) == ix.guesses(z)[:t] for t in range(1, len(z)))
    with pytest.raises(ValueError):
        ix.guesses(z.double())
    z[0, 3] = float("nan")
    with pytest.raises(ValueError):
        ix.guesses(z)
