"""Stage 1 — pure associative recall (MQAR family), the frozen data contract.

Binding: REGISTRATION.md training-curriculum stage 1 ("pure associative
recall (reviewed ``tasks/mqar_recall.py`` generator family)") and the frozen
data contract ("stage-1/stage-2 generator files with registered
vocabularies, sequence-length bands, per-class counts and split seeds
receipted at freeze"). The task MECHANICS are the reviewed
``experiments/expressivity_tasks/tasks/mqar_recall.py`` family, unchanged;
this module binds the registered constants and the seed-fixed pools.

Word layout (the MQAR family, verbatim from the reviewed generator):

    [ k1 v1 k2 v2 ... kP vP ][ q1 . q2 . ... qP . ]  (+ trailing pad)

  * content vocabulary: 64 distinct tokens, ids 1..64; id 0 = pad /
    query-gap slot (never a supervised target); vocab_size = 65.
  * keys are DISTINCT content tokens (sampled without replacement); values
    are i.i.d. content tokens; keys and values share the one content vocab.
  * P = max(2, min(64, T // 16)) pairs (T=128 -> 8 pairs; the T=256 length
    band -> 16 pairs: length-extrapolating storage demand).
  * supervision: masked cross-entropy AT the query positions only; the
    metric is mean per-class recall over the 64 content classes at those
    positions (multiclass balanced accuracy, the registration's stage-1/2
    metric).

Stage-1 receipt gate (trainer): >= 99% validation mean per-class recall.

Determinism: pools are generated from ``np.random.default_rng(split_seed)``
in fixed example order; ``stage1_hash`` is SHA-256 over the canonical
byte serialization (inputs, targets, mask, in that order) — the freeze's
data-identity receipt. Same seed -> byte-identical pools (self-checked).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict

import numpy as np

# ---------------------------------------------------------------------------
# Registered constants (FROZEN at M2; any change requires a new registration).
# ---------------------------------------------------------------------------
VOCAB_CONTENT = 64          # distinct content tokens (keys and values share it)
MIN_PAIRS = 2               # minimum stored pairs per word
QUERY_FRAC = 1.0            # every stored pair is queried
VOCAB_SIZE = VOCAB_CONTENT + 1   # ids: 0 = pad/query-gap, 1..64 = content
PAD = 0


@dataclass(frozen=True)
class MQARSplits:
    """One registered pool: split name, sequence length, example count, seed.

    Sequence-length band: train and validation at T=128 (8 pairs); a T=256
    length-band diagnostic pool (16 pairs, never trained on) receipts that
    the stage-1 generator's storage demand extrapolates with length.
    """
    split: str
    length: int
    n_examples: int
    seed: int


STAGE1_SPLITS: Dict[str, MQARSplits] = {
    "train": MQARSplits("train", 128, 70_000, 28301),
    "validation": MQARSplits("validation", 128, 20_000, 28302),
    "length_band_T256": MQARSplits("length_band_T256", 256, 10_000, 28303),
}
SPLIT_ORDER = ("train", "validation", "length_band_T256")


def n_pairs(length: int) -> int:
    """P = max(MIN_PAIRS, min(VOCAB_CONTENT, T // 16)) — the family's law."""
    return max(MIN_PAIRS, min(VOCAB_CONTENT, length // 16))


def generate_pool(length: int, n_examples: int, seed: int):
    """Generate one registered pool. Returns (inputs, targets, mask, classes).

    inputs/targets: [n_examples, T] int64; mask: [n_examples, T] bool (True
    only at the P query positions). Deterministic in (length, n_examples,
    seed) — byte-identical for the same seed.
    """
    rng = np.random.default_rng(seed)
    P = n_pairs(length)
    Q = max(1, int(round(P * QUERY_FRAC)))
    inputs = np.zeros((n_examples, length), dtype=np.int64)
    targets = np.zeros((n_examples, length), dtype=np.int64)
    mask = np.zeros((n_examples, length), dtype=bool)
    for b in range(n_examples):
        keys = rng.choice(VOCAB_CONTENT, size=P, replace=False) + 1   # distinct
        values = rng.integers(0, VOCAB_CONTENT, size=P) + 1           # i.i.d.
        for i in range(P):
            inputs[b, 2 * i] = keys[i]
            inputs[b, 2 * i + 1] = values[i]
        qpos = 2 * P
        qidx = rng.permutation(P)[:Q]
        for j, qi in enumerate(qidx):
            p = qpos + 2 * j
            if p >= length:
                break
            inputs[b, p] = keys[qi]
            targets[b, p] = values[qi]
            mask[b, p] = True
    classes = np.arange(1, VOCAB_CONTENT + 1, dtype=np.int64)   # 64 content classes
    return inputs, targets, mask, classes


def stage1_pools():
    """All registered stage-1 pools, in SPLIT_ORDER (deterministic)."""
    return {name: generate_pool(s.length, s.n_examples, s.seed)
            for name, s in ((k, STAGE1_SPLITS[k]) for k in SPLIT_ORDER)}


def stage1_hash(split: str, inputs: np.ndarray, targets: np.ndarray,
                mask: np.ndarray) -> str:
    """Canonical data-identity hash for one stage-1 pool (freeze receipt)."""
    h = hashlib.sha256()
    h.update(f"stage1_mqar|{split}|{inputs.shape[0]}|{inputs.shape[1]}|"
             f"{STAGE1_SPLITS[split].seed}".encode())
    h.update(np.ascontiguousarray(inputs, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(targets, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    return h.hexdigest()


def mean_per_class_recall(pred: np.ndarray, true: np.ndarray,
                          classes: np.ndarray) -> float:
    """The registered stage-1/2 metric: mean per-class recall (multiclass
    balanced accuracy) over ``classes`` (X26c discipline: fixed class set,
    never the observed-label union)."""
    recalls = []
    for c in classes:
        sel = true == c
        if sel.any():
            recalls.append(float((pred[sel] == c).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def class_counts(targets: np.ndarray, mask: np.ndarray, classes: np.ndarray):
    """Per-class supervised-position counts (the freeze's per-class-count
    receipt for this pool)."""
    supervised = targets[mask]
    return {int(c): int((supervised == c).sum()) for c in classes}


def _self_check() -> int:
    """Determinism self-check: every registered pool generated twice from its
    fixed seed must hash identically (and differ from a shifted-seed run)."""
    ok = True
    for name in SPLIT_ORDER:
        s = STAGE1_SPLITS[name]
        a = generate_pool(s.length, s.n_examples, s.seed)
        b = generate_pool(s.length, s.n_examples, s.seed)
        ha = stage1_hash(name, *a[:3])
        hb = stage1_hash(name, *b[:3])
        same = ha == hb and np.array_equal(a[0], b[0])
        # sanity: a different seed must NOT reproduce the same pool
        c = generate_pool(s.length, min(s.n_examples, 64), s.seed + 1)
        hc = stage1_hash(name, *c[:3])
        same &= (hc != ha)
        print(f"stage1_mqar {name}: n={s.n_examples} T={s.length} "
              f"seed={s.seed} hash={ha} deterministic={same}")
        ok &= bool(same)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
