"""Receipt-only arithmetic and task-implied checks. Does not load a model."""

import json
from math import isclose, log
from pathlib import Path
import sys

from .editions import (
    ARMS,
    K,
    SEEDS,
    TOKENIZER,
    digest,
    fit_editions,
    order,
    probe_pairs,
)


def check(r):
    if (
        r["tokenizer"] != TOKENIZER
        or r["fit_editions_sha256"] != digest(fit_editions())
        or r["probe_pairs_sha256"] != digest(tuple(probe_pairs()))
        or r["cuda_launches"] != 0
    ):
        raise ValueError("foreign fixture or device")
    n = r["counts_per_fit"]
    assert (
        n["edited_answers"],
        n["changed_bindings"],
        n["unchanged_bindings"],
        n["exact_refined_pairs"],
        n["all_shifted_targets"],
    ) == (1792, 448, 1344, 0, 7168)
    assert len(r["fits"]) == len(r["metrics"]) == len(SEEDS) * len(ARMS)

    def eq(x, y):
        if not isclose(x, y, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(f"inconsistent receipt: {x} != {y}")

    for s in SEEDS:
        names = [f"{s}.{a}" for a in ARMS]
        fits = [r["fits"][k] for k in names]
        assert (
            len({v["initial"] for v in fits})
            == len({v["trainable_scalars"] for v in fits})
            == 1
            and {v["order_sha256"] for v in fits} == {digest(order(s))}
            and all(v["targets"] == 16384 and v["updates"] == 1024 for v in fits)
        )
        local = r["metrics"][names[0]]
        assert (
            local["changed.old_and_new_correct"]
            == local["changed.top1_changed"]
            == local["unchanged.top1_changed"]
            == local["share.corpus"]
            == 0
        )
        assert (
            local["edited.answer_acc"] <= 1 / K + 1e-12
            and local["edited.answer_nll"] >= log(K) - 1e-8
        )
        eq(local["changed.gain_vs_stale"], 0)
        eq(local["old.answer_nll"], local["edited.answer_nll"])
        for arm, k in zip(ARMS, names, strict=True):
            v = r["metrics"][k]
            # Model source shares are FP32; all loss/count identities below accumulate in FP64.
            assert (
                abs(sum(v["share." + t] for t in ("corpus", "local", "no_read")) - 1)
                < 2e-6
            )
            for ed in ("old", "edited"):
                eq(v[ed + ".all_nll"], (v[ed + ".cue_nll"] + v[ed + ".answer_nll"]) / 2)
                assert v[ed + ".cue_nll"] >= log(K) - 1e-8
            eq(
                v["edited.answer_nll"],
                (0.25 * v["changed.edited_nll"] + 0.75 * v["unchanged.edited_nll"]),
            )
            for mask, num in [("changed", 448), ("unchanged", 1344)]:
                eq(
                    v[mask + ".gain_vs_stale"],
                    v[mask + ".stale_nll"] - v[mask + ".edited_nll"],
                )
                count = v[mask + ".old_and_new_correct"] * num
                eq(count, round(count))
            if arm == "local":
                continue
            row = r["primary"][arm]["by_seed"][str(s)]
            gain = local["edited.answer_nll"] - v["edited.answer_nll"]
            eq(row["gain_vs_local"], gain)
            eq(row["changed_old_and_new_correct"], v["changed.old_and_new_correct"])
            eq(row["changed_gain_vs_stale"], v["changed.gain_vs_stale"])
            assert row["passes"] == (
                gain > 0
                and v["changed.old_and_new_correct"] > 0.5
                and v["changed.gain_vs_stale"] > 0
            )
            for key in (
                "other_law.token_tv",
                "other_law.row_tv",
                "address.intended_mass",
                "address.intended_top1",
            ):
                assert 0 <= v[key] <= 1 + 1e-9
    for key in ("dense", "indexed"):
        row = r["primary"][key]
        assert row["passes"] == all(z["passes"] for z in row["by_seed"].values())


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("do not disable audit assertions")
    check(json.loads(Path(sys.argv[1]).read_text()))
    print("receipt arithmetic, pairing, balance and primary verified")
