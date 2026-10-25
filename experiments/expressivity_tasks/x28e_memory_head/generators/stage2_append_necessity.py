"""Stage 2 — append-necessity (fresh-key selective copy), the frozen data contract.

Binding: REGISTRATION.md training-curriculum stage 2 ("append-necessity
tasks (fresh-key recall / ``tasks/selective_copy.py`` family where no good
match exists): the append gate is load-bearing") and the frozen data
contract. The task MECHANICS are the reviewed
``experiments/expressivity_tasks/tasks/selective_copy.py`` family,
unchanged; this module binds the registered constants, the seed-fixed pools,
and the SLOT-OCCUPANCY RECEIPT INSTRUMENTATION the stage-2 gate consumes.

Why this family receipts append-necessity: each content token is presented
once at a random position and must be re-emitted at a later answer slot —
no later query re-presents the stored key, so each copy item is a FRESH
binding with no good match in the store: R-NOMATCH (or R-EMPTY) must APPEND
(or the head cannot hold the item). Growth of the store with item count is
the registered receipt that the append gate actually fires.

Word layout (the selective-copy family, verbatim from the reviewed
generator):

    [pad-interleaved content tokens] [GO] [answer slots ...]

  * vocabulary: 16 content tokens, ids 2..17; id 0 = pad; id 1 = GO;
    vocab_size = 18.
  * K content tokens at random positions in the first region; GO then
    follows; targets are the K content tokens in order at the answer slots
    (masked supervision only there).
  * registered train/validation shape: T=128, K=4 copied items.

SLOT-OCCUPANCY INSTRUMENTATION (registered receipt input): the occupancy
grid pools carry per-example item counts (n_items = 2, 4, 8, 16 at T=256);
the trainer measures PEAK live-slot occupancy per item count from the
routing traces (append allocations minus traced frees) and receipts the
store growing with item count. The grid pools are instrumentation-only,
never trained on.

Stage-2 receipt gate (trainer): >= 99% validation mean per-class recall AND
the receipted slot-occupancy growth (strictly increasing mean peak
occupancy across the registered item-count grid).

Determinism: pools are generated from ``np.random.default_rng(split_seed)``
in fixed example order; ``stage2_hash`` is SHA-256 over the canonical byte
serialization (inputs, targets, mask, item counts). Same seed -> byte-
identical pools (self-checked).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

# ---------------------------------------------------------------------------
# Registered constants (FROZEN at M2; any change requires a new registration).
# ---------------------------------------------------------------------------
N_CONTENT = 16          # content-token vocabulary (excluding pad/GO)
VOCAB_SIZE = 2 + N_CONTENT   # ids: 0 = pad, 1 = GO, 2..17 = content
PAD = 0
GO = 1


@dataclass(frozen=True)
class AppendNecessitySplits:
    """One registered pool: split name, sequence length, copied-item count,
    example count, seed."""
    split: str
    length: int
    n_to_copy: int
    n_examples: int
    seed: int


STAGE2_SPLITS: Dict[str, AppendNecessitySplits] = {
    "train": AppendNecessitySplits("train", 128, 4, 70_000, 28401),
    "validation": AppendNecessitySplits("validation", 128, 4, 20_000, 28402),
}
SPLIT_ORDER = ("train", "validation")

# Registered occupancy-instrumentation grid (never trained on): item-count
# sweep at T=256, 256 examples per item count, one fixed seed per grid row.
OCCUPANCY_GRID: List[AppendNecessitySplits] = [
    AppendNecessitySplits(f"occupancy_K{k}", 256, k, 256, 28403 + k)
    for k in (2, 4, 8, 16)
]


def generate_pool(length: int, n_to_copy: int, n_examples: int, seed: int):
    """Generate one registered pool. Returns (inputs, targets, mask,
    n_items, classes). ``n_items`` is the per-example item count (the
    slot-occupancy receipt instrumentation key; constant = n_to_copy here,
    carried per example so the trainer's receipt is per-example keyed)."""
    rng = np.random.default_rng(seed)
    K = n_to_copy
    if length < 2 * K + 2:
        raise ValueError(f"T={length} too small for {K} tokens to copy")
    inputs = np.zeros((n_examples, length), dtype=np.int64)
    targets = np.zeros((n_examples, length), dtype=np.int64)
    mask = np.zeros((n_examples, length), dtype=bool)
    first_half_len = length - K - 1     # GO + K answer slots reserved at end
    for b in range(n_examples):
        content_pos = rng.choice(first_half_len, size=K, replace=False)
        content_pos.sort()
        content_tokens = rng.integers(2, 2 + N_CONTENT, size=K)
        for cp, ct in zip(content_pos, content_tokens):
            inputs[b, cp] = ct
        inputs[b, first_half_len] = GO
        for i, ct in enumerate(content_tokens):
            slot = first_half_len + 1 + i
            inputs[b, slot] = PAD
            targets[b, slot] = ct
            mask[b, slot] = True
    n_items = np.full((n_examples,), K, dtype=np.int64)
    classes = np.arange(2, 2 + N_CONTENT, dtype=np.int64)   # 16 content classes
    return inputs, targets, mask, n_items, classes


def stage2_pools():
    """All registered stage-2 pools (train, validation, occupancy grid)."""
    pools = {name: generate_pool(s.length, s.n_to_copy, s.n_examples, s.seed)
             for name, s in ((k, STAGE2_SPLITS[k]) for k in SPLIT_ORDER)}
    for g in OCCUPANCY_GRID:
        pools[g.split] = generate_pool(g.length, g.n_to_copy, g.n_examples, g.seed)
    return pools


def stage2_hash(split: str, inputs: np.ndarray, targets: np.ndarray,
                mask: np.ndarray, n_items: np.ndarray) -> str:
    """Canonical data-identity hash for one stage-2 pool (freeze receipt)."""
    h = hashlib.sha256()
    h.update(f"stage2_append_necessity|{split}|{inputs.shape[0]}|"
             f"{inputs.shape[1]}|{int(n_items[0])}".encode())
    h.update(np.ascontiguousarray(inputs, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(targets, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(n_items, dtype=np.int64).tobytes())
    return h.hexdigest()


def class_counts(targets: np.ndarray, mask: np.ndarray, classes: np.ndarray):
    """Per-class supervised-position counts (the per-class-count receipt)."""
    supervised = targets[mask]
    return {int(c): int((supervised == c).sum()) for c in classes}


def _self_check() -> int:
    """Determinism self-check: every registered pool (incl. the occupancy
    grid) generated twice from its fixed seed must hash identically."""
    ok = True
    for s in list(STAGE2_SPLITS.values()) + OCCUPANCY_GRID:
        a = generate_pool(s.length, s.n_to_copy, s.n_examples, s.seed)
        b = generate_pool(s.length, s.n_to_copy, s.n_examples, s.seed)
        ha = stage2_hash(s.split, *a[:4])
        hb = stage2_hash(s.split, *b[:4])
        same = ha == hb and np.array_equal(a[0], b[0])
        print(f"stage2_append_necessity {s.split}: n={s.n_examples} "
              f"T={s.length} K={s.n_to_copy} hash={ha} deterministic={same}")
        ok &= bool(same)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
