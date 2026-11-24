"""S9 offline: distinguish numerical veto (zero timed bytes) from a conformant service run."""

import argparse, json
from pathlib import Path
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, load, ledger, require, pct, fields

ROOT = HERE / "receipts/memory-program-s9"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-t1/verified.json")["cumulative_x47_upper"]
    require(prev == 912, "wrong prior")
    aa, good = ledger(root, 1632)
    paid = sum(x["upper"] for x in aa)
    require(0 < paid <= 170 and len(good) <= 1, "overbudget/double run")
    r = good[0] if good else root / aa[-1]["attempt"]
    idn = load(r / "identity.json")
    ref = (root / "gated-ref").read_text().strip()
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        idn["ref"] == ref == (r / "ref").read_text().strip()
        and "81 passed" in q
        and "failed" not in q
        and "skipped" not in q,
        "wrong gate or source",
    )
    s8 = load(HERE / "receipts/memory-program-s8/attempt1/identity.json")
    require(
        all(
            idn[k] == s8[k]
            for k in (
                "controller_training_base_sha",
                "checkpoint",
                "policy_sha",
                "mount",
                "position_exe_sha",
                "branch_exe_sha",
                "route_seed",
                "limits",
                "data",
                "device",
                "torch",
            )
        )
        and all(
            idn["generation_base"][k] == v
            for k, v in s8["generation_base"].items()
            if k not in ("max_context", "model_load_s")
        )
        and (
            idn["generation_base"]["max_context"],
            idn["generated_length"],
            idn["prompt_bytes"],
            len(set(idn["prompt_sha"])),
            idn["chunk_sha"],
        )
        == (2048, 8, 512, 32, load(HERE / "receipts/memory-program-t1/attempt1/identity.json")["chunk_sha"]),
        "wrong identities",
    )
    if not (r / "device-parity.json").exists():
        # Post-run disposition for the actual build failure, never treat a missing oracle as success.
        msg = "shared memory, Required: 122880, Hardware limit: 101376."
        text = (r / "console").read_text()
        a = aa[-1]
        ini = fields(r / "initial")
        last = fields(r / "last")
        obs = [json.loads(s) for s in (r / "survival.jsonl").read_text().splitlines()]
        end = load(r / "progress.json")
        require(
            not good
            and a["rc"] == 1
            and msg in text
            and "recompute_w_u_fwd_gdn2_kernel" in text
            and "triton.runtime.errors.OutOfResources:" in text
            and not list(r.glob("*-b[0-9]*.json"))
            and not (r / "setup.json").exists()
            and end["phase"] == "checking"
            and end["total_emitted"] == 0
            and all(idn["owner"][k] == v for k, v in ini.items() if k != "heartbeat")
            and all(
                (x["owner"], x["ppid"], x["starttime"], x["progress"]) == (a["pid"], 1, a["start"], end)
                for x in obs
            )
            and int(ini["heartbeat"]) < max(x["heartbeat"] for x in obs) <= int(last["heartbeat"]),
            "unclassified failure or incomplete witness",
        )
        return dict(
            ref=ref,
            attempts=aa,
            card_s_upper=paid,
            cumulative_x47_upper=prev + paid,
            remaining_s=1800 - prev - paid,
            disposition="native FP32 chunk kernel cannot launch",
            shared_required_bytes=122880,
            shared_limit_bytes=101376,
            completed_oracle_sets=0,
            total_emitted=0,
            qualified=False,
            promote=False,
        )
    dev = load(r / "device-parity.json")
    require(
        [c["frontier"] for c in dev["cases"]] == [63, 65, 128, 1025, 2047, 2048, 129, 64, 1026, 66, 65, 513]
        and dev["zeroed_and_reused"],
        "missing lifecycle checks",
    )
    worst = {}
    bad = []
    for c in dev["cases"]:
        require(
            {k: v["elements"] for k, v in c["errors"].items()}
            == dict(
                stock_full_hidden=640,
                stock_full_logp=256,
                split_hidden=640,
                split_logp=256,
                state_recurrent=27 * 2 * 128 * 256,
                state_q=27 * 256 * 4,
                state_k=27 * 256 * 4,
                state_v=27 * 512 * 4,
            ),
            "wrong state geometry",
        )
        for k, v in c["errors"].items():
            if not v["ok"]:
                bad.append((c["tag"], c["frontier"], k, v["max_abs"]))
            if v["max_abs"] is not None:
                worst[k] = max(worst.get(k, 0), v["max_abs"])
        if not c["argmax"]:
            bad.append((c["tag"], c["frontier"], "argmax", None))
    require(dev["qualified"] == (not bad) and len(dev["failures"]) == len(bad), "invalid disposition")
    result = dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        max_errors=worst,
        failed_checks=bad,
        qualified=dev["qualified"],
    )
    if not good:
        require(
            bad
            and aa[-1]["rc"] != 0
            and not list(r.glob("*-b[0-9]*.json"))
            and not (r / "setup.json").exists(),
            "not a pre-timing numerical veto",
        )
        obs = [json.loads(x) for x in (r / "survival.jsonl").read_text().splitlines()]
        a = aa[-1]
        ini = fields(r / "initial")
        require(
            all(
                (
                    x["owner"],
                    x["starttime"],
                    x["ppid"],
                    x["progress"]["server_pid"],
                    x["progress"]["total_emitted"],
                )
                == (a["pid"], a["start"], 1, idn["server_pid"], 0)
                for x in obs
            )
            and all(idn["owner"][k] == v for k, v in ini.items() if k != "heartbeat")
            and max(x["heartbeat"] for x in obs) > int(ini["heartbeat"])
            and max(x["progress"].get("oracle_vectors", 0) for x in obs) > 0,
            "missing live veto witness",
        )
        return dict(**result, total_emitted=0, promote=False)
    require(not bad, "timed failed host")
    summ = {}
    mismatch = []
    ratios = {}
    total = 0
    for b in (1, 16):
        for m in ("direct", "fused-greedy", "fused-sample"):
            refrows = []
            for h in ("full", "resident"):
                rr = []
                arm = f"{h}-{m}"
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{arm}.json")
                    check_cell(x, b, 8, 512, block, arm)
                    rr.append(x)
                    total += x["emitted"]
                    host = x["host"]
                    require(
                        host["encoded_tokens"] == b * (sum(range(512, 520)) if h == "full" else 519)
                        and x["policy_device"] == ("cuda:0" if m == "direct" else "cpu"),
                        "wrong path",
                    )
                    if h == "full":
                        refrows.append(x)
                    else:
                        require(
                            host["all_released_zero"]
                            and host["stats"]
                            == dict(
                                prefill_batches=1,
                                prefill_tokens=512 * b,
                                step_batches=7,
                                step_tokens=7 * b,
                                released=b,
                            ),
                            "bad cache",
                        )
                        diff = [
                            i
                            for i in range(b)
                            if any(x[k][i] != refrows[block][k][i] for k in ("generated_sha", "event_sha"))
                        ]
                        if diff:
                            mismatch.append(dict(cell=x["cell"], documents=diff))
                val = summarize(rr)
                val.update(
                    first_median_ms=pct(sum((x["first_ns"] for x in rr), []), 0.5),
                    after_first_median_ms=pct(
                        [ms for x in rr for row in x["interbyte_ns"] for ms in row[1:]], 0.5
                    ),
                    max_server_rss_kib=max(x["host_rss_kib"] for x in rr),
                )
                summ[f"b{b}/{arm}"] = val
            match = not any(x["cell"].endswith(f"-b{b}-resident-{m}") for x in mismatch)
            ratios[f"b{b}/{m}"] = (
                summ[f"b{b}/resident-{m}"]["bytes_per_s"] / summ[f"b{b}/full-{m}"]["bytes_per_s"]
                if match
                else None
            )
    require(total == 1632, "missing bytes")
    return dict(
        **result,
        total_emitted=total,
        metrics=summ,
        mismatches=mismatch,
        ratios=ratios,
        promote=not mismatch and min(ratios.values()) >= 1.2,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
