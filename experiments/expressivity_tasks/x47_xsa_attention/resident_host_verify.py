"""S6 offline identity, lifecycle and resident memory plateau audit."""

import argparse, json
from pathlib import Path
from .serving_verify import HERE, ledger, load, require

ROOT = HERE / "receipts/memory-program-s6"


def audit(root=ROOT):
    p = load(HERE / "receipts/memory-program-s5/verified.json")
    prior = p["cumulative_x47_upper"]
    require(prior == 681, "bad S5 charge")
    aa, good = ledger(root, 2968)
    paid = sum(a["upper"] for a in aa)
    require(len(good) == 1 and paid <= 120, "incomplete or over budget")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    ident = load(r / "identity.json")
    old = load(HERE / "receipts/memory-program-s5/attempt1/identity.json")
    require(
        ident["ref"] == ref == (r / "ref").read_text().strip()
        and ident["profile"] == "resident-churn"
        and ident["memory_readers_opened"] == 0
        and all(
            ident[k] == v
            for k, v in old.items()
            if k not in ("ref", "owner", "server_pid", "generation_base", "generated_length")
        )
        and all(
            ident["generation_base"][k] == v for k, v in old["generation_base"].items() if k != "model_load_s"
        ),
        "bad model or workload identity",
    )
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require("72 passed" in q and "failed" not in q and "skipped" not in q, "bad CPU gate")
    o = load(r / "oracle.json")
    c = load(r / "churn.json")
    rs = c["waves"]
    s = load(r / "setup.json")
    require(
        len(rs) == 40
        and len(o) == len(set(x["sha"] for x in o)) == 16
        and all(x["wave"] == i and x["signature"] == i for i, x in enumerate(o)),
        "oracle signatures",
    )
    require(
        all(
            x["emitted"] == 53 and x["admissions"] == 17 and x["erased"] and len(x["sha"]) == 64
            for x in rs + o
        ),
        "partial churn",
    )
    require(
        all(
            x["wave"] == i and x["signature"] == i % 16 and x["sha"] == o[i % 16]["sha"]
            for i, x in enumerate(rs)
        ),
        "changed byte/frontier",
    )
    require(
        set(s["buckets"]) == {"1", "16"}
        and s["preparation_streams"] == 1
        and s["pool_bytes"] == sum(v["private_pool_reserved"] for v in s["buckets"].values()) <= 64 << 20
        and s["tensor_bytes"] == 7523840 * 32 <= 256 << 20,
        "preparation budget",
    )
    counts = c["counts"]
    require(
        counts["arena"]
        == dict(
            prefill_batches=80,
            prefill_tokens=17 * 32 * 40,
            step_batches=160,
            step_tokens=36 * 40,
            released=680,
        )
        and counts["dispatch"] == dict(prefill=80, eager_width_miss=40, retirements=80),
        "wrong workload",
    )
    for b, n in [("1", 40), ("16", 80)]:
        require(
            counts["buckets"][b] == dict(real_replays=n, neutral_retire_replays=80),
            "wrong real/neutral replay",
        )
    ranges = {
        k: [min(x["post_release"][k] for x in rs[4:]), max(x["post_release"][k] for x in rs[4:])]
        for k in ("allocated", "reserved", "rss_kib")
    }
    passed = (
        ranges["allocated"][1] - ranges["allocated"][0] <= 2 * 2**20
        and ranges["reserved"][1] - ranges["reserved"][0] <= 32 * 2**20
    )
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prior + paid,
        remaining_s=1800 - prior - paid,
        reference_bytes=848,
        churn_bytes=2120,
        resident_admissions=680,
        postrelease_ranges=ranges,
        counts=counts,
        setup=s,
        after_close=c["after_close"],
        after_purge=c["after_purge"],
        passes=passed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(p.parse_args().root), indent=2, allow_nan=False))
