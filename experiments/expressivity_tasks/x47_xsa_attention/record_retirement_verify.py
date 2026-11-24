"""Public audit: separately count proven reachability, actual current RSS, peak RSS and wall."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean

from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, load, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-s18"
ORDER = ("ordinary-A", "retire-A", "retire-B", "ordinary-B")
GATES = 180
BYTES = 16119767 * 8


def oracle():
    src = HERE / "receipts/b2/samples.jsonl"
    reg = HERE.parent / "x43_fork_resolved/receipts/p100_query_verification/registration.json"
    mounts = HERE / "b1_mounts.json"
    hh = [sha256(p.read_bytes()).hexdigest() for p in (reg, src, mounts)]
    require(
        hh
        == [
            "5ea29f13388c3a3b3fc402cb54f41be51c61fdef664707a7f1d6b215e2b0bef1",
            "520c669473b714ad00e92b396d064663efa176d9312f7d8ff65da0d1f153aaba",
            "eabfebf0b7b02b3f0cfdd8d213efd7cacafcca378b792474d4ac6ab4e0573c93",
        ],
        "wrong historical bindings",
    )
    with src.open() as f:
        r = next(
            z
            for z in map(json.loads, f)
            if z["bank"] == "P100" and z["ids"] == [2, 6] and z["mode"] == "original"
        )
    data = repr([sorted(a) for a in r["positions"]]).encode()
    h = sha256(len(data).to_bytes(8, "little") + data).hexdigest()
    idn = load(HERE / "receipts/memory-program-s17/attempt1/identity.json")
    return dict(
        mount=idn["mount"],
        branch_exe_sha=idn["branch_exe_sha"],
        position_exe_sha=idn["position_exe_sha"],
        registration_sha=hh[0],
        samples_sha=hh[1],
        mounts_sha=hh[2],
        pattern_hashes=r["pattern_hashes"],
        table_bytes=BYTES,
    ), h


def sample_ok(v, n):
    require(
        0 < v["VmRSS"] <= v["VmHWM"]
        and 0 <= v["RssAnon"] <= v["VmRSS"]
        and 0 <= v["RssFile"] <= v["VmRSS"]
        and v["registered_fds"] == 0
        and v["fds"] >= 3
        and set(v["alive"]) == {"descriptor", "starts", "encoder"}
        and all(len(rows) == n and all(type(b) is bool for b in rows) for rows in v["alive"].values())
        and len(v["collections"]) == len(v["gc_counts"]) == 3,
        "invalid post-close process/object sample",
    )


def check_cell(root, name, ref):
    path = root / name
    c = load(path / "cell.json")
    ex = load(path / "exit.json")
    drv = load(root / "driver.json")
    retire = name.startswith("retire")
    prod, coords = oracle()
    require(
        c["ref"] == ref == drv["ref"]
        and c["name"] == name
        and c["inputs"] == prod
        and c["pid"] != drv["pid"] == c["parent"]
        and c["strict"]
        and not c["warn_only"]
        and c["threads"] == 4
        and c["visible"] == ""
        and not c["cuda_initialized"]
        and not c["cuda_available"]
        and c["gc_enabled"]
        and c["waves"] == 6
        and c["handles_at_first_gc"] == 9
        and c["explicit_collections_in_primary"] == ex["rc"] == c["card_s"] == 0
        and c["stale_refusals"] == (9 if retire else 0)
        and ex["process_wall_ns"] >= c["lifecycle_ns"] >= c["wall_ns"] > 0
        and c["core_s"] > 0
        and c["gc_threshold"][0] > 0
        and len(c["collected"]) == 2
        and min(c["collected"]) >= 0
        and load(path / "progress.json") == dict(pid=c["pid"], phase="released", wave=6, completed=6),
        "unbound workload or cell",
    )
    sample_ok(c["before"], 0)
    waves = []
    logical = []
    actors = {c["pid"]}
    stamps = [c["before"]["clock_ns"]]
    high = [c["before"]["VmHWM"]]
    for n in range(1, 7):
        u = load(path / f"wave-{n}.json")
        p = u["position"]
        v = u["post_close"]
        s = p["work"]
        e = u["evidence"]
        z = wire(u["wire"], False, prod)
        sample_ok(v, n)
        stamps.extend([u["start_ns"], u["start_ns"] + u["whole_ns"], v["clock_ns"]])
        high.append(v["VmHWM"])
        ids = [p["pid"], u["wire"]["pid"]]
        require(
            u["number"] == n
            and u["retain_handles"] == (n > 3)
            and p["rc"] == 0
            and p["fd_closed"]
            and len(set(ids)) == 2
            and not actors.intersection(ids)
            and p["start_ns"] > 0
            and u["position_peak_kib"] > 0
            and u["whole_ns"] == u["open_through_check_ns"] + u["query_ns"] + u["close_ns"]
            and min(u[k] for k in ("open_through_check_ns", "query_ns", "close_ns")) > 0,
            "unreaped/reused or mispriced wave",
        )
        actors.update(ids)
        require(
            e["positions"] == coords
            and valid_sha(e["evidence"])
            and e["spans"] == e["occurrence_keys"] == e["contents"] == 3
            and e["certified_refusals"] == s["certified_overcap"] == s["worker_batches"] == 1
            and s["worker_keys"] == 2
            and s["verified_spans"] == 3
            and s.get("native_overcap", 0) == 0
            and 0 < e["following"] <= 3
            and s["preads"] == e["contents"] + e["following"]
            and e["independent_preads"] == 6
            and e["values"] == 8 * s["preads"]
            and e["bytes"] == s["bytes"]
            and 96 < e["bytes"] <= 144
            and s["lf_rows"] >= 3
            and s["lf_steps"] > 0
            and s["ranks"] > 0
            and e["spans"] + e["certified_refusals"] == 4
            and u["row_frames"]["frames"] == z["r_frames"] == u["grant"]["batches"]
            and u["row_frames"]["items"] == z["r_rows"] == u["grant"]["rows"]
            and valid_sha(u["row_frames"]["sha"])
            and 0 < u["grant"]["cached_before_clear"] <= 256,
            "incomplete positive service or copy/coordinate check",
        )
        logical.append(
            dict(
                evidence=e,
                row_traffic_sha=u["row_frames"]["sha"],
                r_rows=z["r_rows"],
                r_frames=z["r_frames"],
                wire_bytes=z["wire_bytes"],
                grant=u["grant"],
                native={k: x for k, x in s.items() if not k.endswith("_ns")},
            )
        )
        waves.append(u)
    held, final = c["held_after_gc"], c["all_dropped_after_gc"]
    for v in (held, final):
        sample_ok(v, 6)
        stamps.append(v["clock_ns"])
        high.append(v["VmHWM"])
    require(
        len(actors) == 13
        and all(l == logical[0] for l in logical)
        and stamps == sorted(stamps)
        and high == sorted(high)
        and c["wall_ns"] >= waves[-1]["post_close"]["clock_ns"] - waves[0]["start_ns"]
        and all(not any(b) for b in final["alive"].values()),
        "divergence or retained object after final release",
    )
    posts = [x["post_close"] for x in waves]
    immediate = all(not any(b) for v in posts for b in v["alive"].values())
    held_expected = all(
        b == ([False] * 6 if retire else [False] * 3 + [True] * 3) for b in held["alive"].values()
    )
    metrics = dict(
        wall_s=c["wall_ns"] / 1e9,
        lifecycle_s=c["lifecycle_ns"] / 1e9,
        interpreter_s=ex["process_wall_ns"] / 1e9,
        core_s=c["core_s"],
        open_s=sum(u["open_through_check_ns"] for u in waves) / 1e9,
        query_s=sum(u["query_ns"] for u in waves) / 1e9,
        close_s=sum(u["close_ns"] for u in waves) / 1e9,
        initial_mib=c["before"]["VmRSS"] / 1024,
        post_close_mib=[v["VmRSS"] / 1024 for v in posts],
        held_gc_mib=held["VmRSS"] / 1024,
        final_gc_mib=final["VmRSS"] / 1024,
        peak_main_mib=max(high) / 1024,
        peak_m20_mib=max(x["wire"]["peak_kib"] for x in waves) / 1024,
        peak_pos_mib=max(x["position_peak_kib"] for x in waves) / 1024,
        tables_alive=[sum(v["alive"]["starts"]) for v in posts],
        encoders_alive=[sum(v["alive"]["encoder"]) for v in posts],
        descriptors_alive=[sum(v["alive"]["descriptor"]) for v in posts],
        tables_after_gc=[sum(v["alive"]["starts"]) for v in (held, final)],
        fds=[v["fds"] for v in [c["before"], *posts, held, final]],
        stale_refusals=c["stale_refusals"],
        immediate_drop=immediate,
        retained_test=held_expected,
        primary_collections=[
            j - i for i, j in zip(c["before"]["collections"], posts[-1]["collections"], strict=True)
        ],
    )
    return logical[0], metrics, actors, c


def audit(root=ROOT):
    prior = load(HERE / "receipts/memory-program-s17/verified.json")
    ref = (root / "gated-ref").read_text().strip()
    d = load(root / "driver.json")
    end = load(root / "complete.json")
    require(
        ref == (root / "source-ref").read_text().strip() == d["ref"] == end["ref"]
        and prior["functional_gate"]
        and prior["cumulative_x47_upper"] == 1429
        and d["order"] == list(ORDER)
        and d["pid"] == end["pid"]
        and d["card_s"] == end["card_s"] == 0
        and end["waves"] == 24
        and 0 < end["elapsed_ns"] <= 360e9
        and f"{GATES} passed in" in (root / "qualification.txt").read_text().splitlines()[-1],
        "unbound diagnostic",
    )
    metrics = {}
    logical = None
    actors = {d["pid"]}
    epochs = [d["start_epoch_ns"]]
    th = None
    for name in ORDER:
        ev, m, ids, c = check_cell(root, name, ref)
        require(
            not actors.intersection(ids)
            and (logical is None or logical == ev)
            and (th is None or th == c["gc_threshold"]),
            "different exposure/collector or re-used actor",
        )
        logical, th = ev, c["gc_threshold"]
        actors.update(ids)
        metrics[name] = m
        epochs.extend([c["start_epoch_ns"], c["end_epoch_ns"]])
    require(
        epochs + [end["end_epoch_ns"]] == sorted(epochs + [end["end_epoch_ns"]]) and len(actors) == 53,
        "overlapping/missing lifetimes",
    )
    contrasts = {}
    for rep in "AB":
        a, b = (metrics[f"{label}-{rep}"] for label in ("ordinary", "retire"))
        contrasts[rep] = dict(
            held_gc_rss_saved_bytes=int(round((a["held_gc_mib"] - b["held_gc_mib"]) * 2**20)),
            wall_ratio=b["wall_s"] / a["wall_s"],
        )
    wall = mean(metrics[k]["wall_s"] for k in ORDER if k.startswith("retire")) / mean(
        metrics[k]["wall_s"] for k in ORDER if k.startswith("ordinary")
    )
    reachable = all(
        m["retained_test"] and (not k.startswith("retire") or m["immediate_drop"]) for k, m in metrics.items()
    )
    return dict(
        ref=ref,
        card_s=0,
        cumulative_x47_upper=1429,
        remaining_s=371,
        diagnostic_s=end["elapsed_ns"] / 1e9,
        generated_bytes=0,
        gradient_targets=0,
        updates=0,
        waves=24,
        m20_exits=24,
        positional_exits=24,
        verified_spans=24 * logical["evidence"]["spans"],
        value_preads=24 * logical["native"]["preads"],
        lf_steps=24 * logical["native"]["lf_steps"],
        per_wave=logical,
        gc_threshold=th,
        metrics=metrics,
        contrasts=contrasts,
        mean_wall_ratio=wall,
        positive_service_gate=True,
        retirement_gate=reachable,
        synchronous_footprint_recommended=reachable
        and wall <= 1.25
        and all(z["held_gc_rss_saved_bytes"] >= 2 * BYTES for z in contrasts.values()),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
