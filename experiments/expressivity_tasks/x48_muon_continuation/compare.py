"""Compare only completed, source- and exposure-matched TRAIN trajectories."""

from hashlib import sha256
from statistics import fmean

from .data import OUT, B, T, STEPS, check, load, sha, stream_id, write_json
from .train import BRACKET, STREAM_ID, elect, metrics


def compare(root):
    charge = load(root / "panel/charges.json")
    check(
        set(charge) == {"program", "direct"} and all(0 < v <= 10800 for v in charge.values()), "wrong charges"
    )
    check(stream_id(load(root / "data/identity.json")) == STREAM_ID, "wrong data identity")
    meta = root / "data/rows.jsonl"
    check(sha(meta) == load(root / "data/identity.json")["rows_sha"], "changed document metadata")
    import json

    rows = [json.loads(line)["window_sha"] for line in meta.read_text().splitlines()]
    batches = [sha256("".join(rows[i : i + B]).encode()).hexdigest() for i in range(0, len(rows), B)]
    check(len(batches) == STEPS, "missing batches")
    result = {}
    main = {}
    for arm in ("direct", "program"):
        d = root / arm
        check(
            load(d / "charge.json")["upper_card_s"] == charge[arm] and load(d / "charge.json")["rc"] == 0,
            "incomplete lease",
        )
        c = {k: metrics(d / k, 1, BRACKET) for k in ("L", "H")}
        winner, scores = elect(c)
        check(load(d / "election.json") == dict(winner=winner, train_scores=scores), "unregistered election")
        main[arm] = m = metrics(d / "main", BRACKET + 1, STEPS)
        for k, rr in [*c.items(), ("main", m)]:
            orig = load(d / k / "origin.json")
            expected_label = winner if k == "main" else k
            check(
                orig["binding"]["stream"] == STREAM_ID
                and orig["binding"]["ref"] == (root / "gated-ref").read_text().strip()
                and orig["binding"]["arm"] == arm
                and orig["binding"]["candidate"] == expected_label,
                "cross-arm or source mismatch",
            )
            check(
                all(
                    r["batch_sha"] == batches[r["step"] - 1]
                    and r["candidate"] == expected_label
                    and r["arm"] == arm
                    for r in rr
                ),
                "unpaired exposure",
            )
        check(
            load(d / "main/origin.json")["from_checkpoint"] == load(d / winner / "latest.json")["sha256"]
            and load(d / "main/origin.json")["binding"] == load(d / winner / "origin.json")["binding"],
            "not the selected transition",
        )
        ck = load(d / "main/latest.json")
        check(
            ck["step"] == STEPS
            and sha(d / "main" / ck["file"]) == load(d / "complete.json")["final_checkpoint"] == ck["sha256"],
            "missing endpoint weights",
        )
        comps = [load(d / k / "complete.json") for k in ("L", "H", "main")]
        allrows = c["L"] + c["H"] + m
        check(
            sum(s["optimizer_steps"] for s in comps) == len(allrows) == STEPS + BRACKET,
            "missing physical updates",
        )
        result[arm] = dict(
            winner=winner,
            bracket_nll=scores,
            unique_targets=STEPS * B * T,
            physical_training_targets=len(allrows) * B * T,
            optimizer_steps_in_selected_trajectory=STEPS,
            extra_bracket_steps=BRACKET,
            upper_card_s=charge[arm],
            charged_unique_targets_per_s=STEPS * B * T / charge[arm],
            selected_trajectory_mean_preupdate_nll=fmean(r["preupdate_nll"] for r in c[winner] + m),
            main_plus_native_core_s=sum(s["main_core_s"] + s["children_core_s"] for s in comps),
            measured_segment_wall_s=sum(s["wall_s"] for s in comps),
            peak_allocated_bytes=max(s["peak_allocated_bytes"] for s in comps),
            physical_keys=sum(
                r["work"].get("program_keys", 0) + r["work"].get("direct_keys", 0) for r in allrows
            ),
            physical_enters=sum(r["work"].get("program", 0) for r in allrows),
        )
    contrasts = {}
    for n, label in ((STEPS - BRACKET, "post_election"), (256, "last_256")):
        pair = {
            arm: {k: fmean(r[k] for r in m[-n:]) for k in ("preupdate_nll", "priced_nll")}
            for arm, m in main.items()
        }
        contrasts[label] = dict(
            targets=n * B * T,
            means=pair,
            direct_minus_program={k: pair["direct"][k] - v for k, v in pair["program"].items()},
        )
    return dict(
        scope="pre-update TRAIN predictive loss, not held-out",
        data=STREAM_ID,
        arms=result,
        contrasts=contrasts,
        held_out_targets=0,
    )


if __name__ == "__main__":
    r = compare(OUT)
    write_json(OUT / "panel/comparison.json", r)
    print(r)
