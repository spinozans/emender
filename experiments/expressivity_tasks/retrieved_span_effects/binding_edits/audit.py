"""Check the published arithmetic and the two independent mean-odds identities (no weights)."""

import json
from math import isclose, log
from pathlib import Path
import sys

from ..semantic_tail.editions import ARMS, K, SEEDS, digest
from .worlds import fits, order, panels


def check(d):
    assert (
        d["fit_maps_sha256"] == digest(fits())
        and d["panel_maps_sha256"] == digest(panels())
        and d["additional_gpu_seconds"] == 0
    )
    assert d["counts_per_fit"] == dict(
        panels=28,
        editions=1792,
        active_answers=3584,
        directed_single_edits=25088,
        directed_same_bag_swaps=3136,
        exact_refined_pairs=0,
    )
    assert len(d["replacement_fits"]) == 9 and len(d["metrics"]) == 18
    for s in SEEDS:
        rows = [d["replacement_fits"][f"{s}.{a}"] for a in ARMS]
        assert (
            len({r["initial"] for r in rows}) == 1
            and {r["order_sha256"] for r in rows} == {digest(order(s))}
            and all(r["targets"] == 16384 and r["updates"] == 1024 for r in rows)
        )
    for key, r in d["metrics"].items():
        c, s, a = key.split(".")
        base = d["metrics"][f"{c}.{s}.local"]["answer_nll"]
        assert r["clamped_acc"] <= 1 / K + 1e-12 and r["clamped_nll"] >= log(K) - 1e-8
        w, mass, kappa = (
            r["target_joint_mass"],
            r["corpus_share"],
            r["decoder_contrast_mean"],
        )
        for edit, n, expected in (
            ("single", 25088, w * kappa),
            ("swap", 3136, (K * w - mass) * kappa / (K - 1)),
        ):
            odds = r[f"{edit}.delta_odds"]
            assert abs(odds - expected) <= r["max_odds_identity_error"] + 1e-10
            # Every ordered intervention also occurs in reverse, on the identical fixed background.
            assert isclose(
                odds, 2 * r[f"{edit}.gain_vs_stale"], abs_tol=1e-10, rel_tol=1e-10
            )
            for k in ("both_correct", "both_with_negative_decoder_contrast"):
                count = r[f"{edit}.{k}"] * n
                assert abs(count - round(count)) < 1e-8
        if a == "local":
            assert (
                r["answer_nll"] >= log(K) - 1e-8
                and r["answer_acc"] <= 1 / K + 1e-12
                and w == mass == 0
            )
            assert (
                r["single.both_correct"]
                == r["swap.both_correct"]
                == r["single.gain_vs_stale"]
                == r["swap.gain_vs_stale"]
                == 0
            )
        else:
            primary = d["primary"][c + "." + a]
            i = SEEDS.index(int(s))
            gain = base - r["answer_nll"]
            assert isclose(gain, primary["gains_vs_local"][i], abs_tol=1e-12)
            assert primary["per_seed"][i] == (
                gain > 0
                and r["single.both_correct"] > 0.5
                and r["single.gain_vs_stale"] > 0
                and r["swap.both_correct"] > 0.5
            )
            assert primary["passes"] == all(primary["per_seed"])


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError("do not disable audit assertions")
    check(json.loads(Path(sys.argv[1]).read_text()))
    print("closed edit counts, source masses, decoder contrasts and decisions verified")
