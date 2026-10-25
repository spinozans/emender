"""Stage 3 — the a^k b^k ladder, the frozen data contract (the instrument).

Binding: REGISTRATION.md "The instrument: the a^k b^k ladder" + the frozen
data contract. The certificate context (X21T both-impossible theorems via
``StreamingExpressivity``; X28T constructive tape model
``tapeModel_decides_balanced`` / ``TapeMemory.tapeModel_no_finite_
abstraction``) is in REGISTRATION.md; this module is DATA ONLY.

TOKENIZATION (registered scalar-token convention, the
``merrill26_separators.py`` standard — explicit SOS/EOS markers, PAD fill):

    PAD = 0, A = 1, B = 2, SOS = 3, EOS = 4      (vocab_size = 5)

    word  = [SOS]  a/b token stream  [EOS]  [PAD ... right-fill to batch max]
    label = final-only binary decision at end-of-word (the EOS position);
            accept = 1, reject = 0. No per-step supervision, no teacher
            forcing; BCE on a two-logit margin reading the final READ
            output (X23E house supervision, G8).

CLASSES BALANCED FROM DAY ONE (X26c prior-drift lesson): every pool is
50/50 accept/reject with fixed marginals; every reported number includes
binary balanced accuracy, per-class recalls, the confusion matrix, class
counts, and the majority baseline per pool (``binary_report``).

REJECT SHAPES (the three X28T malformed shapes, equally weighted within
reject, sampled within the pool's depth band; degenerate single-depth bands
-- the fine sweep pools -- use the registered nearest-mismatch rule below):

    LEADING-B       b^k a^k        (a b before any a)
    COUNT-MISMATCH  a^j b^k, j != k (j, k both sampled in the band)
    TRAILING-A      a^k b^k a^j    (an a after the b-phase began; j in [1..k])

Degenerate-band rule (sweep pools, band [k, k]): count-mismatch alternates
j = k+1 / j = max(k-1, 1) across its examples (the nearest possible mismatch
at that depth; deterministic, receipted in the pool metadata).

For a pool of n_reject rejects, the three shapes are weighted as equally as
integers allow, in the fixed order above (e.g. 256 -> 85/86/85; 128 ->
43/42/43); per-shape counts are receipted in the pool metadata.

GRADED DEPTH BINS (registered): train pool k in [1..128] (in-range); OOD
bins ID [1..128], 2x [129..256], 4x [257..512], 9x [1025..1152] of the
training-top k=128 — each an immutable 512-word balanced pool, never trained
on. Fine per-k sweep points (bounded tape arms only, window [B-128, B+128],
every 16 slots) are 256-word balanced pools, one per k, seed-fixed as a
function of k (so all arms share identical sweep pools at shared k).

Determinism: pools are generated from ``np.random.default_rng(pool_seed)``
in fixed example order (accepts and rejects interleaved 50/50);
``ladder_hash`` is SHA-256 over the canonical byte serialization (inputs,
labels, lengths). Same seed -> byte-identical pools (self-checked).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Registered tokenization (scalar-token convention; FROZEN at M2).
# ---------------------------------------------------------------------------
PAD, A, B, SOS, EOS = 0, 1, 2, 3, 4
LADDER_VOCAB = 5
ACCEPT, REJECT = 1, 0
SHAPE_NAMES = ("leading_b", "count_mismatch", "trailing_a")


@dataclass(frozen=True)
class LadderSplits:
    """One registered pool: name, inclusive depth band (k_lo, k_hi), balanced
    example count, seed."""
    split: str
    k_lo: int
    k_hi: int
    n_examples: int
    seed: int


STAGE3_SPLITS: Dict[str, LadderSplits] = {
    "train": LadderSplits("train", 1, 128, 70_000, 28501),
    "validation": LadderSplits("validation", 1, 128, 2_000, 28502),
    "bin_id": LadderSplits("bin_id", 1, 128, 512, 28511),
    "bin_2x": LadderSplits("bin_2x", 129, 256, 512, 28512),
    "bin_4x": LadderSplits("bin_4x", 257, 512, 512, 28513),
    "bin_9x": LadderSplits("bin_9x", 1025, 1152, 512, 28514),
}
TRAIN_SPLIT = "train"
VALIDATION_SPLIT = "validation"          # in-band [1..128], never trained on;
                                         # the OOD bins are test-only (the
                                         # selected checkpoint is the only one
                                         # ever evaluated on them)
BIN_ORDER = ("bin_id", "bin_2x", "bin_4x", "bin_9x")

# Fine per-k sweep pools: seed is a FIXED function of k (all arms share the
# same sweep pool at the same k; immutable, never trained on).
SWEEP_SEED_BASE = 28_600
SWEEP_POOL_WORDS = 256          # 128 accept + 128 reject per sweep point


def shape_counts(n_reject: int) -> Tuple[int, int, int]:
    """The three malformed shapes equally weighted within reject (as equal
    as integers allow, fixed order SHAPE_NAMES; receipted in pool metadata)."""
    base, rem = divmod(n_reject, 3)
    return base + (1 if rem > 0 else 0), base + (1 if rem > 1 else 0), base


def _word_tokens(shape: str, k: int, j: int) -> List[int]:
    if shape == "balanced":
        return [A] * k + [B] * k
    if shape == "leading_b":
        return [B] * k + [A] * k
    if shape == "count_mismatch":
        return [A] * j + [B] * k
    if shape == "trailing_a":
        return [A] * k + [B] * k + [A] * j
    raise ValueError(f"unknown ladder shape {shape}")


def generate_pool(k_lo: int, k_hi: int, n_examples: int, seed: int):
    """Generate one balanced ladder pool over depth band [k_lo, k_hi].

    Returns ``(inputs [n, L_max] int64 PAD-right-filled, labels [n] int64,
    lengths [n] int64 (incl. SOS/EOS), meta)`` where meta receipts the
    per-shape counts and the band. Words alternate accept/reject (fixed
    interleaved order, 50/50 fixed marginals); within reject the three
    malformed shapes fire in the registered equal weighting.
    """
    if n_examples % 2:
        raise ValueError("ladder pools are balanced 50/50: even n_examples")
    rng = np.random.default_rng(seed)
    band = max(k_hi - k_lo + 1, 1)
    n_reject = n_examples // 2
    counts = shape_counts(n_reject)
    words: List[List[int]] = []
    labels: List[int] = []
    depths: List[int] = []            # per-word depth k (F7: per-k occupancy receipts)
    # fixed example order: accept/reject interleaved; the reject-shape
    # sequence cycles through the registered equal weighting
    shapes: List[str] = sum(([n * [s] for n, s in zip(counts, SHAPE_NAMES)]), [])
    assert len(shapes) == n_reject
    for i in range(n_examples):
        k = int(rng.integers(k_lo, k_hi + 1))
        depths.append(k)
        if i % 2 == 0:                      # accept: a^k b^k
            words.append(_word_tokens("balanced", k, k))
            labels.append(ACCEPT)
        else:                               # reject: next registered shape
            s = shapes[i // 2]
            if s == "leading_b":
                words.append(_word_tokens(s, k, k))
            elif s == "count_mismatch":
                if k_lo == k_hi:                 # degenerate band (sweep pool)
                    n_mis = (i // 2) % 2
                    j = k + 1 if (n_mis == 0 or k <= 1) else k - 1
                else:
                    j = k
                    while j == k:               # j != k, both within the band
                        j = int(rng.integers(k_lo, k_hi + 1))
                words.append(_word_tokens(s, k, j))
            else:                           # trailing_a: j in [1..k]
                j = int(rng.integers(1, k + 1))
                words.append(_word_tokens(s, k, j))
            labels.append(REJECT)
    lengths = np.array([len(w) + 2 for w in words], dtype=np.int64)
    l_max = int(lengths.max())
    inputs = np.full((n_examples, l_max), PAD, dtype=np.int64)
    for i, w in enumerate(words):
        row = [SOS] + w + [EOS]
        inputs[i, :len(row)] = row
    meta = {
        "band": [k_lo, k_hi], "n_examples": n_examples, "seed": seed,
        "n_accept": n_examples // 2, "n_reject": n_reject,
        "reject_shape_counts": dict(zip(SHAPE_NAMES, counts)),
        "majority_baseline": 0.5,           # balanced pools (X26c receipt)
        # F7 (M2-review correction): per-word depths — occupancy receipts are
        # keyed by depth k (returned metadata; not part of ladder_hash).
        "depths": depths,
        # F8 (M2-review correction): the count-mismatch rule, receipted in the
        # returned metadata (returned metadata; not part of ladder_hash).
        "count_mismatch_rule": (
            "degenerate band (k_lo == k_hi): j = k+1 / max(k-1, 1) alternating "
            "across count-mismatch examples; j != k always; k = 1 uses j = k+1"
            if k_lo == k_hi else
            "j resampled uniformly in [k_lo, k_hi] until j != k (always j != k)"),
    }
    return inputs, np.array(labels, dtype=np.int64), lengths, meta


def ladder_train_pool():
    s = STAGE3_SPLITS[TRAIN_SPLIT]
    return generate_pool(s.k_lo, s.k_hi, s.n_examples, s.seed)


def ladder_validation_pool():
    s = STAGE3_SPLITS[VALIDATION_SPLIT]
    return generate_pool(s.k_lo, s.k_hi, s.n_examples, s.seed)


def ladder_bin_pool(bin_name: str):
    s = STAGE3_SPLITS[bin_name]
    return generate_pool(s.k_lo, s.k_hi, s.n_examples, s.seed)


def ladder_sweep_pool(k: int):
    """One fine-sweep pool at depth k (256 balanced words; seed = 28600 + k)."""
    return generate_pool(k, k, SWEEP_POOL_WORDS, SWEEP_SEED_BASE + k)


def ladder_hash(split: str, inputs: np.ndarray, labels: np.ndarray,
                lengths: np.ndarray) -> str:
    """Canonical data-identity hash for one ladder pool (freeze receipt).
    PAD fill beyond each word's EOS is excluded (padding is not data)."""
    h = hashlib.sha256()
    h.update(f"stage3_ladder|{split}|{inputs.shape[0]}".encode())
    for i in range(inputs.shape[0]):
        row = np.ascontiguousarray(inputs[i, :int(lengths[i])], dtype=np.int64)
        h.update(row.tobytes())
    h.update(np.ascontiguousarray(labels, dtype=np.int64).tobytes())
    return h.hexdigest()


def binary_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Registered stage-3 reporting (X26c discipline): binary balanced
    accuracy, per-class recalls, confusion matrix, class counts and the
    majority baseline — one report per pool/sweep point, exact counts."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = int(y_true.shape[0])
    acc = float((y_true == y_pred).mean()) if n else 0.0
    recalls = {}
    confusion = {}
    for c, name in ((ACCEPT, "accept"), (REJECT, "reject")):
        sel = y_true == c
        recalls[name] = float((y_pred[sel] == c).mean()) if sel.any() else 0.0
        confusion[name] = {
            "pred_accept": int(((y_pred == ACCEPT) & sel).sum()),
            "pred_reject": int(((y_pred == REJECT) & sel).sum()),
        }
    majority = max(int((y_true == ACCEPT).sum()), int((y_true == REJECT).sum()))
    return {
        "n": n,
        "balanced_accuracy": float(0.5 * (recalls["accept"] + recalls["reject"])),
        "raw_accuracy": acc,
        "per_class_recall": recalls,
        "confusion": confusion,
        "class_counts": {"accept": int((y_true == ACCEPT).sum()),
                         "reject": int((y_true == REJECT).sum())},
        "majority_baseline": (majority / n) if n else 0.0,
    }


def _self_check() -> int:
    """Determinism self-check: every registered pool (train, four bins, sweep
    points spanning the bounded arms' registered windows) generated twice
    from its fixed seed must hash identically."""
    ok = True
    makes = {"train": ladder_train_pool, "validation": ladder_validation_pool}
    makes.update({name: (lambda name=name: ladder_bin_pool(name)) for name in BIN_ORDER})
    makes.update({f"sweep_k{k}": (lambda k=k: ladder_sweep_pool(k))
                  for k in (608, 736, 864)})
    for name, make in makes.items():
        pool = make()
        again = make()
        ha = ladder_hash(name, *pool[:3])
        hb = ladder_hash(name, *again[:3])
        same = ha == hb and np.array_equal(pool[0], again[0])
        print(f"stage3_ladder {name}: n={pool[0].shape[0]} "
              f"band={pool[3]['band']} shapes={pool[3]['reject_shape_counts']} "
              f"hash={ha} deterministic={same}")
        ok &= bool(same)
    rep = binary_report(np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]))
    assert abs(rep["balanced_accuracy"] - 0.75) < 1e-12   # metric self-test
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_check())
