"""Offline S10 audit for segmented recurrent prompt ingestion. Not stock FP32 chunk."""

import argparse, json
from pathlib import Path
from .cached_host_verify import check_cell, summarize
from .serving_verify import HERE, load, ledger, require, pct, fields

ROOT = HERE / "receipts/memory-program-s10"


def audit(root=ROOT):
    prev = load(HERE / "receipts/memory-program-s9/verified.json")["cumulative_x47_upper"]
    require(prev == 927, "prior charge")
    aa, good = ledger(root, 1632)
    paid = sum(x["upper"] for x in aa)
    require(len(good) <= 1 and 0 < paid <= 175, "attempt count/cap")
    r = good[0] if good else root / aa[-1]["attempt"]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    y = load(HERE / "receipts/memory-program-s9/attempt1/identity.json")
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "82 passed" in q
        and not {"skipped", "failed"}.intersection(q.split())
        and ref == (r / "ref").read_text().strip() == idn["ref"]
        and all(
            idn[k] == v for k, v in y.items() if k not in ("generation_base", "ref", "owner", "server_pid")
        )
        and all(
            idn["generation_base"][k] == v for k, v in y["generation_base"].items() if k != "model_load_s"
        ),
        "identity/source/gate",
    )
    dev = load(r / "device-parity.json")
    series = [
        "2048-token-stream",
        "129-native-singletons",
        "130-native-singletons",
        "ragged-65",
        "ragged-513",
        "reordered-66",
        "reordered-514",
    ]
    require(
        dev["old_slot_scrubbed"] and dev["peer_unchanged"] and [x["tag"] for x in dev["checks"]] == series,
        "incomplete stream oracle",
    )
    worst = {}
    bad = []
    for c in dev["checks"]:
        require(
            {k: v["elements"] for k, v in c["fields"].items()}
            == dict(
                recurrent=27 * 2 * 128 * 256,
                q=27 * 256 * 4,
                k=27 * 256 * 4,
                v=27 * 512 * 4,
                hidden=640,
                base=256,
            ),
            "wrong state coverage",
        )
        for k, v in c["fields"].items():
            if not v["ok"]:
                bad.append([c["tag"], k])
            if v["max_abs"] is not None:
                worst[k] = max(worst.get(k, 0.0), v["max_abs"])
        if not c["argmax"]:
            bad.append([c["tag"], "argmax"])
    require(dev["failed"] == bad and dev["qualified"] == (not bad), "wrong conformance disposition")
    out = dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=prev + paid,
        remaining_s=1800 - prev - paid,
        failed_checks=bad,
        qualified=not bad,
        max_errors=worst,
        oracle_frontiers=series,
    )
    if not good:
        require(
            bad and aa[-1]["rc"] and not (r / "setup.json").exists() and not list(r.glob("*-b[0-9]*.json")),
            "not conformance veto",
        )
        obs = [json.loads(z) for z in (r / "survival.jsonl").read_text().splitlines()]
        a = aa[-1]
        ini = fields(r / "initial")
        require(
            all(
                (
                    x["owner"],
                    x["ppid"],
                    x["starttime"],
                    x["progress"]["server_pid"],
                    x["progress"]["total_emitted"],
                )
                == (a["pid"], 1, a["start"], idn["server_pid"], 0)
                for x in obs
            )
            and all(idn["owner"][k] == v for k, v in ini.items() if k != "heartbeat")
            and max(x["heartbeat"] for x in obs) > int(ini["heartbeat"])
            and max(x["progress"].get("observed_prefix_bytes", 0) for x in obs) >= 128,
            "unwitnessed oracle",
        )
        return dict(**out, total_emitted=0, promote=False)
    s = load(r / "setup.json")
    require(
        not bad and s["tensor_bytes"] == 240762880 and 0 < s["private_bytes"] <= 64 << 20, "bad resident pool"
    )
    ms = {}
    mis = []
    ratios = {}
    total = 0
    for b in (1, 16):
        for m in ("direct", "fused-greedy", "fused-sample"):
            ys = []
            matched = True
            for kind in ("serial", "segment"):
                arm = f"{kind}-{m}"
                rr = []
                serial = kind == "serial"
                pf = 1 if serial else 512
                steps = 519 - pf
                for block in (0, 1):
                    x = load(r / f"{block}-b{b}-{arm}.json")
                    rr.append(x)
                    check_cell(x, b, 8, 512, block, arm)
                    total += x["emitted"]
                    h = x["host"]
                    require(
                        h["all_released_zero"]
                        and h["tensor_bytes"] == s["tensor_bytes"]
                        and h["encoded_tokens"] == 519 * b
                        and h["stats"]
                        == dict(
                            prefill_batches=1,
                            prefill_tokens=pf * b,
                            step_batches=steps,
                            step_tokens=steps * b,
                            released=b,
                        )
                        and h["native_prompt_pieces"] == dict(calls=1 if serial else 8, tokens=pf * b)
                        and h["buckets"][str(b)] == dict(real_replays=steps, neutral_retire_replays=1)
                        and x["policy_device"] == ("cuda:0" if m == "direct" else "cpu"),
                        "changed work or placement",
                    )
                    if serial:
                        ys.append(x)
                    else:
                        dif = [
                            i
                            for i in range(b)
                            if any(x[k][i] != ys[block][k][i] for k in ("generated_sha", "event_sha"))
                        ]
                        if dif:
                            mis.append(dict(cell=x["cell"], docs=dif))
                            matched = False
                v = summarize(rr)
                v.update(
                    first_median_ms=pct(sum((x["first_ns"] for x in rr), []), 0.5),
                    after_first_median_ms=pct(
                        [ns for x in rr for row in x["interbyte_ns"] for ns in row[1:]], 0.5
                    ),
                    max_server_rss_kib=max(x["host_rss_kib"] for x in rr),
                )
                ms[f"b{b}/{arm}"] = v
            ratios[f"b{b}/{m}"] = (
                ms[f"b{b}/segment-{m}"]["bytes_per_s"] / ms[f"b{b}/serial-{m}"]["bytes_per_s"]
                if matched
                else None
            )
    require(total == 1632, "wrong complete bytes")
    return dict(
        **out,
        total_emitted=total,
        metrics=ms,
        mismatches=mis,
        ratios=ratios,
        promote=not mis and min(ratios.values()) >= 1.2,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
