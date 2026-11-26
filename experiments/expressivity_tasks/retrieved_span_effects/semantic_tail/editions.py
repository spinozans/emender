"""Balanced permutations of invented successor IDs. No corpus or tokenizer download."""

from functools import cache
from hashlib import sha256
from itertools import combinations, permutations
import json

import numpy as np

from ..corpus_kernel.index import EndpointIndex
from ..corpus_windows.packet import WindowArchive

K = 8
KEY = tuple(range(1, 9))
ALIAS = tuple(9 + (3 * i + 5) % K for i in range(K))
VALUE = tuple(range(17, 25))
VOCAB = 25
SEEDS = (5701, 5702, 5703)
ARMS = ("local", "dense", "indexed")
EPOCHS = 8


def digest(obj):
    return sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


TOKENIZER = digest(
    dict(version="semantic-tail-v1", prompt=0, key=KEY, alias=ALIAS, value=VALUE)
)


@cache
def canonical_forms():
    forms = ((0, *z) for z in permutations(range(1, K)))
    return tuple(
        sorted(forms, key=lambda p: sha256(b"semantic-tail-v1:" + bytes(p)).digest())
    )


def offset(p, delta):
    return tuple((v + delta) % K for v in p)


def canonical(p):
    return offset(p, -p[0])


def transpose(p, i, j):
    q = list(p)
    q[i], q[j] = q[j], q[i]
    return tuple(q)


def fit_editions():
    return tuple(offset(p, d) for p in canonical_forms()[:16] for d in range(K))


def probe_pairs():
    used = set(canonical_forms()[:16])
    for i, j in combinations(range(K), 2):
        for p in canonical_forms():
            q = transpose(p, i, j)
            other = canonical(q)
            if p not in used and other not in used and p != other:
                break
        else:
            raise RuntimeError("insufficient unused probe orbits")
        used.update((p, other))
        for d in range(K):
            yield offset(p, d), offset(q, d), (i, j)


def order(seed):
    n = len(fit_editions())
    rng = np.random.default_rng(57100 + seed)
    return tuple(int(i) for _ in range(EPOCHS) for i in rng.permutation(n))


def docs(p):
    if sorted(p) != list(range(K)):
        raise ValueError("expected a permutation of the eight answers")
    archive = [(f"key:{i}", (KEY[i], VALUE[v])) for i, v in enumerate(p)]
    live = [(f"cue:{i}", (0, ALIAS[i], VALUE[v])) for i, v in enumerate(p)]
    return archive, live


def packet(m, p, *, corpus):
    archive, live = docs(p)
    index = EndpointIndex(archive, tokenizer=TOKENIZER)
    w = WindowArchive(m, index).windows(
        live,
        corpus=corpus,
        local=True,
        allow_skip=True,
        min_match=1,
        lookback=None,
        chunk_targets=5,
    )
    # Fail closed if this stops testing the learned background. For the complete
    # nonempty cover with no hits each frontier has exactly one root node.
    if (
        len(index) != K
        or len(w.targets) != 2 * K
        or len(w.local_rows) != K
        or any(
            b.plan.exact
            or b.plan.edge
            or len(b.plan.nodes) != (b.stop - b.start if corpus else 0)
            for b in w.blocks
        )
    ):
        raise RuntimeError("unexpected no-hit packet geometry")
    return w
