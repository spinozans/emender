"""No language data. Counterbalanced edit panels; distinct orbits across panels."""

from functools import cache
from hashlib import sha256
from itertools import combinations, product

import numpy as np

from ..corpus_kernel.index import EndpointIndex
from ..corpus_windows.packet import WindowArchive
from ..semantic_tail.editions import ALIAS, KEY, VALUE, K, TOKENIZER, canonical


def admit(phase, keys, rep, used):
    fixed = tuple(i for i in range(K) if i not in keys)
    for counter in range(100000):
        name = f"binding-edits-v1:{phase}:{','.join(map(str, keys))}:{rep}:{counter}"
        bg = tuple(i % K for i in sha256(name.encode("ascii")).digest()[: len(fixed)])
        if not 2 <= len(set(bg)) < len(fixed):
            continue
        rows = []
        for shift in range(K) if phase == "fit" else (0,):
            for assign in product(range(K), repeat=len(keys)):
                row = [None] * K
                for i, b in zip(fixed, bg, strict=True):
                    row[i] = (b + shift) % K
                for i, b in zip(keys, assign, strict=True):
                    row[i] = b
                rows.append(tuple(row))
        orbits = {canonical(p) for p in rows}
        if len(set(rows)) != len(rows) or orbits & used:
            continue
        used.update(orbits)
        return tuple(rows)
    raise RuntimeError("insufficient disjoint edit panels")


@cache
def fits():
    used = set()
    return tuple(
        p for i in range(K) for rep in range(2) for p in admit("fit", (i,), rep, used)
    )


@cache
def panels():
    used = {canonical(p) for p in fits()}
    return tuple(
        (keys, admit("probe", keys, 0, used)) for keys in combinations(range(K), 2)
    )


def order(seed):
    return tuple(map(int, np.random.default_rng(57100 + seed).permutation(len(fits()))))


def packet(m, p, corpus=True):
    if (
        len(p) != K
        or not all(type(x) is int and 0 <= x < K for x in p)
        or len(set(p)) == K
    ):
        raise ValueError("expected eight invented bindings including a repeat")
    idx = EndpointIndex(
        [(f"key:{i}", (KEY[i], VALUE[a])) for i, a in enumerate(p)], tokenizer=TOKENIZER
    )
    w = WindowArchive(m, idx).windows(
        [(f"cue:{i}", (0, ALIAS[i], VALUE[a])) for i, a in enumerate(p)],
        chunk_targets=5,
        min_match=1,
        lookback=None,
        corpus=corpus,
        local=True,
        allow_skip=True,
    )
    if (
        len(w.targets) != 16
        or len(w.local_rows) != 8
        or any(
            b.plan.exact
            or b.plan.edge
            or len(b.plan.nodes) != (b.stop - b.start if corpus else 0)
            for b in w.blocks
        )
    ):
        raise RuntimeError("not a no-hit edit fixture")
    return w
