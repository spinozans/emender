"""Freeze-time determinism check — ALL registered pools (M2-review
correction F8).

Coverage (the review's "all pools" obligation): the union of sweep points
derived from ALL THREE bounded configs' fine-sweep windows (17 points per
window: 608..864, 2816..3072, 11648..11904 = 51 sweep pools) PLUS every
other registered pool (3 stage-1 splits, 2 stage-2 splits, the 4-row
stage-2 occupancy grid, stage-3 train/validation and the four OOD bins =
15) — 66 pools total. Every pool is generated twice from its fixed seed and
receipt hash-equal data (same seed -> byte-identical).

Seed sensitivity is COMPUTED BY ACTUAL COMPARISON (never hardcoded True):
for the shifted-seed sanity pools the shifted hash is compared against the
registered pool's hash and the boolean difference is receipted.

The degenerate-band sweep mismatch rule (adjudication (e) + F8) is receipted
from the returned pool metadata AND verified by construction: for the
registered sweep pools every count-mismatch word has j != k (j = k+1 /
max(k-1, 1) alternating; k = 1 uses j = k+1), checked from the generated
words.

The receipt is BYTE-STABLE (it is manifest-covered): no wall time or any
run-varying field is written. Zero optimizer updates; no CUDA required
(generators are pure numpy).

Usage: python generators/check_determinism.py
Exit code 0 iff every pool is deterministic, every seed-sensitivity
comparison shows a real difference, and the mismatch rule holds.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # x28e_memory_head/

from generators.stage1_mqar import (  # noqa: E402
    SPLIT_ORDER as S1_ORDER, STAGE1_SPLITS, generate_pool as gen1,
    stage1_hash, class_counts as counts1,
)
from generators.stage2_append_necessity import (  # noqa: E402
    SPLIT_ORDER as S2_ORDER, STAGE2_SPLITS, OCCUPANCY_GRID,
    generate_pool as gen2, stage2_hash, class_counts as counts2,
)
from generators.stage3_ladder import (  # noqa: E402
    A, B, BIN_ORDER, STAGE3_SPLITS, ladder_bin_pool, ladder_hash,
    ladder_sweep_pool, ladder_train_pool, ladder_validation_pool,
    binary_report, SWEEP_SEED_BASE,
)

RECEIPTS = Path(__file__).resolve().parent / "receipts"
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
BOUNDED_TAPE_ARMS = ("tape_B0", "tape_B0x4", "tape_B0x16")


def sweep_points() -> list:
    """F8: the union of fine-sweep points derived from ALL THREE bounded
    configs' registered windows (never a hardcoded list)."""
    points: dict = {}
    for arm in BOUNDED_TAPE_ARMS:
        cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
        window = cfg["fine_sweep"]["window"]
        pts = cfg["fine_sweep"]["points"]
        assert pts == list(range(window[0], window[1] + 1,
                                 cfg["fine_sweep"]["step_slots"])), (
            f"{arm}: fine_sweep points do not match its window")
        for k in pts:
            points.setdefault(k, []).append(arm)
    return sorted(points)


def check_pool(name: str, make, hash_of) -> dict:
    first = make()
    second = make()
    h1 = hash_of(name, first)
    h2 = hash_of(name, second)
    identical = bool(h1 == h2 and np.array_equal(first[0], second[0]))
    # NOTE: wall_seconds is deliberately NOT written into the receipt - this
    # receipt is manifest-covered, so its bytes must be reproducible; timing
    # is a diagnostic printed to stdout only.
    print(f"{name}: hash={h1} deterministic={identical}")
    return {"pool": name, "hash": h1, "deterministic": identical}


def mismatch_rule_check(k: int) -> dict:
    """F8: verify the degenerate-band mismatch rule BY CONSTRUCTION on the
    sweep pool at depth k: every count-mismatch word a^j b^k has j != k,
    j in {k+1, max(k-1, 1)}, alternating across its examples, and k = 1
    uses j = k+1 (j never 0)."""
    inputs, labels, lengths, meta = ladder_sweep_pool(k)
    js = []
    for i in range(inputs.shape[0]):
        if labels[i] != 0:
            continue
        body = inputs[i, 1:int(lengths[i]) - 1]        # strip SOS/EOS
        n_a = int((body == A).sum())
        n_b = int((body == B).sum())
        first_b = int(np.argmax(body == B)) if n_b else len(body)
        if n_b == 0 or body[0] == B:
            continue                                 # leading_b (b^k a^k)
        if bool((body[first_b + 1:] == A).any()):
            continue                                 # trailing_a (a after b)
        if n_a != n_b:                               # count_mismatch a^j b^k
            js.append((n_a, n_b))
    ok_j = all(j != kk for j, kk in js)
    ok_values = all(
        j in (kk + 1, max(kk - 1, 1)) for j, kk in js)
    alternating = all(
        js[i][0] != js[i + 1][0] for i in range(len(js) - 1)) if js else True
    k1_uses_plus1 = True
    if k == 1:
        k1_uses_plus1 = all(j == 2 for j, _ in js)
    # Alternation is only REQUIRED when both neighbors are valid bands:
    # at k = 1, j = 0 is excluded (j never 0), the band is single-valued
    # (j = 2 always), and no alternation is possible - by the rule's own
    # design. Demanding alternation at k = 1 made the rule fail its own
    # degenerate case (parent fix, pre-M2).
    alternation_required = k >= 2
    return {"k": k, "n_count_mismatch": len(js),
            "j_never_equals_k": bool(ok_j),
            "j_in_k_plus_minus_1": bool(ok_values),
            "j_alternates": bool(alternating),
            "k1_uses_k_plus_1": bool(k1_uses_plus1),
            "alternation_required": bool(alternation_required),
            "rule": meta["count_mismatch_rule"],
            "pass": bool(ok_j and ok_values and k1_uses_plus1
                          and (alternating or not alternation_required))}


def main() -> int:
    points = sweep_points()
    rows = []
    # stage 1
    for name in S1_ORDER:
        s = STAGE1_SPLITS[name]

        def make(s=s, name=name):
            pool = gen1(s.length, s.n_examples, s.seed)
            return pool[:3] + (name,)

        def hash_of(name, made):     # made = (inputs, targets, mask, name)
            return stage1_hash(made[3], *made[:3])
        rows.append(check_pool(f"stage1_mqar.{name}", make, hash_of))
    # stage 2 (+ occupancy grid)
    for s in list(STAGE2_SPLITS.values()) + list(OCCUPANCY_GRID):
        def make(s=s):
            pool = gen2(s.length, s.n_to_copy, s.n_examples, s.seed)
            return pool[:4]

        def hash_of(name, made):
            return stage2_hash(name.rsplit(".", 1)[-1], *made[:4])
        rows.append(check_pool(f"stage2_append_necessity.{s.split}", make, hash_of))
    # stage 3 (train, validation, bins)
    rows.append(check_pool("stage3_ladder.train",
                           ladder_train_pool,
                           lambda n, p: ladder_hash("train", *p[:3])))
    rows.append(check_pool("stage3_ladder.validation",
                           ladder_validation_pool,
                           lambda n, p: ladder_hash("validation", *p[:3])))
    for name in BIN_ORDER:
        rows.append(check_pool(f"stage3_ladder.{name}",
                               lambda name=name: ladder_bin_pool(name),
                               lambda n, p: ladder_hash(n.rsplit(".", 1)[-1], *p[:3])))
    # stage 3 sweep pools: the FULL union of the three bounded windows (F8)
    for k in points:
        rows.append(check_pool(f"stage3_ladder.sweep_k{k}",
                               lambda k=k: ladder_sweep_pool(k),
                               lambda n, p: ladder_hash(f"sweep_k{k}", *p[:3])))

    # seed-sensitivity sanity, COMPUTED BY ACTUAL COMPARISON (F8: never
    # hardcoded True): a shifted seed must change the hash.
    shifted = []
    reg = gen1(STAGE1_SPLITS["train"].length,
               STAGE1_SPLITS["train"].n_examples,
               STAGE1_SPLITS["train"].seed)
    p = gen1(STAGE1_SPLITS["train"].length,
             STAGE1_SPLITS["train"].n_examples,
             STAGE1_SPLITS["train"].seed + 1)
    h_reg = stage1_hash("train", *reg[:3])
    h_shift = stage1_hash("train", *p[:3])
    shifted.append({"pool": "stage1_mqar.train_seed+1",
                    "hash": h_shift,
                    "registered_hash": h_reg,
                    "differs_from_registered": bool(h_shift != h_reg)})
    q_reg = gen2(STAGE2_SPLITS["train"].length,
                 STAGE2_SPLITS["train"].n_to_copy,
                 STAGE2_SPLITS["train"].n_examples,
                 STAGE2_SPLITS["train"].seed)
    q = gen2(STAGE2_SPLITS["train"].length,
             STAGE2_SPLITS["train"].n_to_copy,
             STAGE2_SPLITS["train"].n_examples,
             STAGE2_SPLITS["train"].seed + 1)
    h_reg2 = stage2_hash("train", *q_reg[:4])
    h_shift2 = stage2_hash("train", *q[:4])
    shifted.append({"pool": "stage2_append_necessity.train_seed+1",
                    "hash": h_shift2,
                    "registered_hash": h_reg2,
                    "differs_from_registered": bool(h_shift2 != h_reg2)})

    # degenerate-band mismatch rule: receipted metadata + by-construction
    # verification on every sweep point (F8 / adjudication (e))
    mismatch_rules = [mismatch_rule_check(k) for k in points]

    # per-class-count receipts (the frozen data contract's count receipts)
    s1v = gen1(STAGE1_SPLITS["validation"].length,
               STAGE1_SPLITS["validation"].n_examples,
               STAGE1_SPLITS["validation"].seed)
    s2v = gen2(STAGE2_SPLITS["validation"].length,
               STAGE2_SPLITS["validation"].n_to_copy,
               STAGE2_SPLITS["validation"].n_examples,
               STAGE2_SPLITS["validation"].seed)
    train = ladder_train_pool()
    bins = {name: ladder_bin_pool(name) for name in BIN_ORDER}
    counts_receipt = {
        "stage1_validation_class_counts": counts1(s1v[1], s1v[2], s1v[3]),
        "stage2_validation_class_counts": counts2(s2v[1], s2v[2], s2v[4]),
        "stage3_train_meta": {
            **train[3], "depths": f"[{min(train[3]['depths'])}.."
                                  f"{max(train[3]['depths'])}, "
                                  f"{len(train[3]['depths'])} words]"},
        "stage3_bins_meta": {
            k: {**v[3], "depths": f"[{min(v[3]['depths'])}.."
                                   f"{max(v[3]['depths'])}, "
                                   f"{len(v[3]['depths'])} words]"}
            for k, v in bins.items()},
        "stage3_bin_binary_report_zero_knowledge": binary_report(
            train[1][:512], train[1][:512]),   # identity probe: report sanity
    }

    receipt = {
        "check": "generator determinism (frozen data contract, M2 freeze; "
                 "M2-review correction F8: ALL pools)",
        "rule": "same seed -> byte-identical pools (generate twice, hash-equal)",
        "coverage": "union of the three bounded configs' fine-sweep windows "
                    f"({len(points)} sweep points: "
                    f"{points[0]}..{points[len(points)//3-1]}, "
                    f"{points[len(points)//3]}..{points[2*len(points)//3-1]}, "
                    f"{points[2*len(points)//3]}..{points[-1]}) plus every "
                    "other registered pool (stage-1 splits, stage-2 splits + "
                    "occupancy grid, stage-3 train/validation and the four "
                    "OOD bins)",
        "sweep_points": points,
        "n_pools_checked": len(rows),
        "all_deterministic": all(r["deterministic"] for r in rows),
        "pools": rows,
        "seed_sensitivity": shifted,
        "seed_sensitivity_rule": "computed by actual hash comparison against "
                                 "the registered pool (never hardcoded)",
        "sweep_mismatch_rule_receipts": mismatch_rules,
        "sweep_mismatch_rule": mismatch_rules[0]["rule"] if mismatch_rules else None,
        "class_count_receipts": counts_receipt,
        "sweep_seed_rule": f"seed(k) = {SWEEP_SEED_BASE} + k",
        "zero_updates": True,
    }
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / "determinism_check.json").write_text(
        json.dumps(receipt, indent=2, default=int) + "\n")
    ok = (receipt["all_deterministic"]
          and all(s["differs_from_registered"] for s in shifted)
          and all(m["pass"] for m in mismatch_rules))
    print(f"all_deterministic={receipt['all_deterministic']} "
          f"({len(rows)} pools; {len(points)} sweep points from the three "
          f"bounded windows); mismatch_rule_pass="
          f"{all(m['pass'] for m in mismatch_rules)}; "
          f"seed_sensitivity={all(s['differs_from_registered'] for s in shifted)}")
    print(f"receipt -> {RECEIPTS / 'determinism_check.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
