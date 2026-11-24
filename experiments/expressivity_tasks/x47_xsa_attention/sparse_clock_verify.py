"""Offline T2 check, including independently regenerated gate-coordinate receipts."""

from pathlib import Path
from hashlib import sha256, blake2b
from struct import pack
from math import isfinite, isclose, log
import argparse, json
from .serving_verify import HERE, ledger, load, require

ROOT = HERE / "receipts/memory-program-t2"


def entered(rows):
    out = []
    h = sha256()
    for i in rows:
        v = [
            j
            for j in range(128)
            if int.from_bytes(
                blake2b(pack(">QQQ", 7823, i, j + 1), digest_size=8, person=b"byte-entry-v1").digest()[:3],
                "big",
            )
            < 2**20
        ]
        b = repr(v).encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
        out.append(v)
    return out, h.hexdigest()


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s12/verified.json")["cumulative_x47_upper"]
    require(prev == 1153, "prior charge")
    aa, good = ledger(root, 9216, work_key="total_targets", actor_key="trainer_pid")
    paid = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < paid <= 155, "no successful in-budget batch")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    i = load(r / "identity.json")
    t1 = load(HERE / "receipts/memory-program-t1/attempt1/identity.json")
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "92 passed" in q
        and not {"skipped", "failed"}.intersection(q.split())
        and i["ref"] == ref == (r / "ref").read_text().strip()
        and all(i[k] == v for k, v in t1.items() if k not in ("owner", "ref", "trainer_pid"))
        and i["gate_seed"] == 7823
        and i["program_seed"] == 7809
        and i["entry_parameters"] == 257
        and isclose(i["entry_initial_bias"], -log(15), abs_tol=1e-6),
        "identity/gate",
    )
    tot = 0
    by = {}
    proof = {}
    ratios = {}
    for n, rows in [(4, list(range(4, 8))), (32, list(range(32)))]:
        a, h = entered(rows)
        ne = sum(map(len, a))
        tg = n * 128
        eq = load(r / f"b{n}-equivalence.json")
        g = eq["gradients"]
        l = eq["logical"]
        num = eq["numeric"]
        require(
            l["events"]
            and l["train_rows"] == rows
            and l["entry_positions"] == h
            and l["entries_per_doc"] == list(map(len, a))
            and l["window_sha"] == [i["data"]["window129_sha"][j] for j in rows],
            "different routing or TRAIN windows",
        )
        require(
            {k: v["count"] for k, v in num.items()}
            == dict(byte_logp=tg * 256, gate_logp=tg, joint_logp=tg, nll=1, objective=1, conditional_logp=ne)
            and all(isfinite(v["max_abs"]) for v in num.values()),
            "missing numeric check",
        )
        counts = {
            k: sum(v["n"] for p, v in g.items() if p.startswith(k + "/")) for k in ("host", "gate", "program")
        }
        require(
            counts["host"] == 149855670
            and counts["gate"] == 257
            and counts["program"] > 0
            and all(
                0 <= v["error"] <= 2e-4 + 2e-4 * v["ref_max"] and isfinite(v["ref_max"]) for v in g.values()
            ),
            "bad gradient coverage/error",
        )
        proof[n] = dict(
            numeric=num,
            checked=counts,
            max_gradient_error=max(v["error"] for v in g.values()),
            unused=eq["unused"],
        )
        pair = {}
        for schedule in ("index", "compact"):
            name = f"b{n}-{schedule}"
            x = load(r / (name + ".json"))
            w = x["work"]
            tm = x["times"]
            nb = len(set(j for seq in a for j in seq)) if schedule == "index" else max(map(len, a))
            require(
                x["logical"] == l
                and x["optimizer_steps"] == 0
                and x["mode"] == "fresh_sample"
                and x["targets"] == tg == w["observed_targets"]
                and w["enter"] == x["ops"]["EMIT"] == ne
                and w["base"] == tg - ne
                and w["program_batches"] == nb
                and nb <= w["controller_rounds"]
                and w["append_bytes"] == n * 129
                and w["append_calls"] <= (n * 129)
                and (schedule != "index" or w["append_calls"] == n * 129),
                "changed clock",
            )
            require(
                x["host"] == dict(forward_batches=n // 4, forward_tokens=tg)
                and tm["forward_ns"] + tm["backward_ns"] == tm["gradient_wall_ns"]
                and isclose(x["targets_s"], tg * 1e9 / tm["gradient_wall_ns"])
                and x["initial"] <= x["post_rollout"] <= x["peak_allocated"] <= x["peak_reserved"]
                and all(x["gradients"][k]["used_elements"] == v for k, v in counts.items()),
                "bad costing",
            )
            pair[schedule] = x
            by[name] = {
                k: x[k]
                for k in (
                    "work",
                    "ops",
                    "physical_dispatches",
                    "logical_keys",
                    "targets_s",
                    "times",
                    "post_rollout",
                    "peak_allocated",
                    "peak_reserved",
                    "nll",
                    "objective",
                    "rss_kib",
                    "cpu_core_s",
                    "gradients",
                )
            }
            tot += tg
        x, y = pair["compact"], pair["index"]
        require(x["ops"] == y["ops"] and x["logical_keys"] == y["logical_keys"], "logical program drift")
        ratios[n] = dict(
            wall=x["times"]["gradient_wall_ns"] / y["times"]["gradient_wall_ns"],
            peak=x["peak_allocated"] / y["peak_allocated"],
        )
    require(tot == 9216, "missing targets")
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        total_gradient_targets=tot,
        optimizer_steps=0,
        proof=proof,
        metrics=by,
        compact_over_index=ratios,
        promote_clock=ratios[32]["wall"] <= 0.80 and ratios[32]["peak"] <= 1.10,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
