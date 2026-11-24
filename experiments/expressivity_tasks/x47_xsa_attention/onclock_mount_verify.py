"""Receipt-only S19 audit. All client clocks start at independently stamped actual offers."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

from .demand_tape_verify import expected, scores
from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, fields, ledger, load, pct, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-s19"
S17 = HERE / "receipts/memory-program-s17/attempt1"
ORDER = dict(
    replicated=["fresh-A", "image-A", "image-B", "fresh-B"],
    retained=["image-A", "fresh-A", "fresh-B", "image-B"],
)
PREFIX = dict(
    replicated="3e58e3def94fdc78676b0fe68f4f33a0c7268281a2ced0573b19cf27e8c98c8c",
    retained="71508975e871e7970626d6a4e722014b9b0230d35aefcd64f422e5acc29008db",
)
OPS = ["byte_edges", "occurrences", "contents", "following"]
GATES = 208


def sealed(p, digest):
    require(sha256(p.read_bytes()).hexdigest() == digest, "changed prior seal: " + p.name)
    return load(p)


def prior_prefix(size):
    return sealed(S17 / f"{size}-closed-eager.json", PREFIX[size])["arrival_prefix"]


def untimed(x):
    return {k: v for k, v in x.items() if k != "ns" and not k.endswith("_ns")}


def signature(w):
    """Compare complete physical R request/response transcript as well as the chosen logical work."""
    return dict(
        tap=untimed(w["tap"]),
        cache=w["cache"],
        work=untimed(w["work"]),
        routing=untimed(w["routing"]),
        planner=w["planner"],
        provider={k: untimed(v) for k, v in w["provider"].items()},
        position=untimed(w["position"]["work"]),
        host=w["host"],
        dispatch=w["dispatch"],
        buckets={str(k): v for k, v in w["buckets"].items()},
        private=[
            (p["row"], p["observed_sha"], untimed(p["work"]), p["exit"] is not None) for p in w["private"]
        ],
    )


def latencies(x, waves):
    require(
        len(waves) == 4
        and x["origin"] < x["schedule_start"] <= x["schedule_ready"] <= waves[0]["start"]
        and waves[-1]["after_release"]["clock_ns"] <= x["close_start"] <= x["schedule_closed"] <= x["end"],
        "setup/close outside offered clock",
    )
    first, inter, queued, box, jitter, sojourn = [], [], [], [], [], []
    puts, takes, gaps = [], [], []
    setup = x["schedule_ready"] - x["schedule_start"]
    for j, w in enumerate(waves):
        rs = w["requests"]
        ix = list(range(j * 8, j * 8 + 8))
        ts = w["ticks"]
        require(
            [r["row"] for r in rs] == ix
            and [t["rows"] for t in ts] == [ix] * 3 + [ix[1:]] * 9
            and w["start"] < w["ready"] <= w["stale_checked"] <= rs[0]["opening"]
            and all(t["start"] < t["ack"] for t in ts)
            and all(a["ack"] <= b["start"] for a, b in zip(ts, ts[1:]))
            and all(a["admitted"] <= b["opening"] for a, b in zip(rs, rs[1:]))
            and rs[-1]["admitted"] <= ts[0]["start"]
            and ts[-1]["ack"] <= rs[-1]["released"] <= w["retired_at"]
            and rs[0]["released"] <= ts[3]["start"]
            and w["retired_at"] <= w["after_release"]["clock_ns"],
            "wrong live wave, release or timing",
        )
        if j:
            prev = waves[j - 1]["after_release"]["clock_ns"]
            require(prev <= rs[0]["taken"] <= w["start"], "successor borrowed too early")
            gaps.append((w["start"] - waves[j - 1]["retired_at"]) / 1e6)
        setup += w["ready"] - w["start"]
        for r in rs:
            i, ack = r["row"], r["acks"]
            require(
                r["scheduled"]
                == x["origin"] + j * 60_000_000
                <= r["offered"]
                <= r["taken"]
                <= (w["start"] if j else x["schedule_start"])
                and r["opening"] <= r["admitted"] <= ts[0]["start"] < ack[0]
                and ack == [t["ack"] for t in ts if i in t["rows"]]
                and len(ack) == (3 if i % 8 == 0 else 12)
                and r["status"] == ("canceled" if i % 8 == 0 else "finished")
                and ack[-1] <= r["released"] <= w["retired_at"],
                "unbound offer, admission or response",
            )
            first.append(ack[0] - r["offered"])
            inter.extend(b - a for a, b in zip(ack, ack[1:]))
            queued.append(r["opening"] - r["offered"])
            box.append(r["taken"] - r["offered"])
            jitter.append(r["offered"] - r["scheduled"])
            sojourn.append(r["released"] - r["offered"])
            puts.append(r["offered"])
            takes.append(r["taken"])
    peak = max(sum(a <= t < b for a, b in zip(puts, takes, strict=True)) for t in puts)
    require(
        puts == sorted(puts) and takes == sorted(takes) and peak == x["peak_waiting"] and 0 < peak <= 32,
        "lost/backdated queue entry",
    )
    life = x["end"] - puts[0]
    require(
        0 < setup < life and len(first) == 32 and len(inter) == 316,
        "incomplete on-clock workload",
    )

    def ms(v):
        return dict(p50=pct(v, 0.50), p95=pct(v, 0.95), max=pct(v, 1))

    return dict(
        lifecycle_s=life / 1e9,
        bytes_per_lifecycle_s=348e9 / life,
        charged_setup_s=setup / 1e9,
        first_grant_wait_ms=ms(first[:8]),
        first_ms=ms(first),
        inter_ms=ms(inter),
        queue_ms=ms(queued),
        mailbox_ms=ms(box),
        jitter_ms=ms(jitter),
        sojourn_ms=ms(sojourn),
        peak_waiting=peak,
        offered_during_mount=sum(
            any(
                a <= t < b
                for a, b in [
                    (x["schedule_start"], x["schedule_ready"]),
                    *((w["start"], w["ready"]) for w in waves),
                ]
            )
            for t in puts
        ),
        grant_gaps_ms=gaps,
        wave_setup_s=[(w["ready"] - w["start"]) / 1e9 for w in waves],
        wave_verified_domain_s=[w["mount"]["open_ns"] / 1e9 for w in waves],
        wave_close_s=[(w["retired_at"] - w["requests"][-1]["released"]) / 1e9 for w in waves],
        final_image_scope_close_s=(x["schedule_closed"] - x["close_start"]) / 1e9,
    )


def routing_counts(w, ids, full, plain):
    d, work = Counter(w["routing"]), w["work"]
    keys = sum(full[i]["program_keys"] for i in ids)
    if plain:
        require(
            not d
            and work["transfer_batches"] == 12
            and work["transfer_bytes"] == 87 * 2048
            and work["blocking_transfer_ns"] > 0
            and w["planner"]["instructions"] == 87 + sum(full[i]["nonemit"] for i in ids),
            "unexpected gate/direct path or incomplete unconditional predictions",
        )
        return dict(program_predictions=87, program_keys=keys), [None] * len(ids)
    ent = sum(full[i]["enter"] for i in ids)
    draws = [expected(i, full[i]["length"]) for i in ids]
    prog_batches = sum(any(len(g) > t and g[t] for g in draws) for t in range(12))
    direct_batches = sum(any(len(g) > t and not g[t] for g in draws) for t in range(12))
    require(
        d["predictions"] == 87
        and d["enter"] == ent
        and d["direct"] == 87 - ent
        and d["transfer_batches"] == d["gate_batches"] == 12
        and d["transfer_bytes"] == 87 * 2048
        and d["blocking_transfer_ns"] > 0
        and d["direct_batches"] == direct_batches
        and d["program_batches"] == prog_batches
        and d["selected"] == d["direct"]
        and d["logical_probes"] == sum(full[i]["probes"] for i in ids)
        and d["logical_probes"] >= d["submitted_keys"] >= d["hist_calls"] > 0
        and d["program_keys"] == keys
        and d["nonemit"] == sum(full[i]["nonemit"] for i in ids),
        "different chosen conditional work",
    )
    return dict(enter=ent, direct=87 - ent, direct_probes=d["logical_probes"], program_keys=keys), draws


def wave_work(w, j, full, image, idn, actors, *, deferred=False, plain=False):
    n, p, m, c, a = w["native"], w["position"], w["mount"], w["cache"], w["tap"]
    ids = list(range(8 * j, 8 * j + 8))
    born = p["pid"] is not None
    ww = m["wire"]
    if plain and ww["rpc"] == {"S": {}, "R": {}}:
        require(
            ww["rc"] == 0
            and ww["peak_kib"] > 0
            and ww["handshake"] == ww["bytes"] == 16
            and (ww["binary"], ww["mount"]) == (idn["branch_exe_sha"], idn["mount"])
            and a == {"sha": sha256(b"").hexdigest()}
            and not any(c.values()),
            "invalid idle archive transport",
        )
        z = dict(
            s_frames=0,
            s_literal_bytes=0,
            r_frames=0,
            r_rows=0,
            wire_bytes=16,
            rpc_s=0.0,
            m20_peak_mib=ww["peak_kib"] / 1024,
        )
    else:
        z = wire(ww, False, idn)
    require(
        n["documents"] == 16_119_767
        and n["starts_bytes"] == 128_958_136
        and 0 < n["root"] <= n["nodes"] == 21_707_038
        and n["workspace_ns"] > 0
        and ((n["attest_ns"] == 0) if image else (n["attest_ns"] > 0))
        and w["start"] <= n["start_ns"] < w["ready"] < n["reaped_ns"] <= w["retired_at"]
        and 0 < m["open_ns"] <= w["ready"] - w["start"]
        and m["final_close_ns"] > 0
        and m["grants"] == 1
        and n["pid"] == m["wire"]["pid"]
        and set(c) == {"cached_before_clear", "rows", "batches"}
        and (0 < c["cached_before_clear"] <= min(256, c["rows"]) or (plain and not any(c.values())))
        and valid_sha(a["sha"])
        and {k: v for k, v in a.items() if k != "sha"} == m["wire"]["rpc"]["R"]
        and a.get("items", 0) == c["rows"]
        and a.get("frames", 0) == c["batches"]
        and w["slots_zero"]
        and w["stale_cancels"] == 1
        and w["stale_readers"] == (OPS if j else [])
        and p["fd_closed"]
        and (
            p["rc"] == 0 and p["start_ns"] > 0 and p["peak_kib"] > 0
            if born
            else deferred and all(p[k] is None for k in ("rc", "start_ns", "peak_kib", "birth"))
        ),
        "unbound source, native transcript or teardown",
    )
    mem = w["after_release"]
    require(
        0 < mem["VmRSS"] <= mem["VmHWM"] and mem["registered_fds"] == (5 if image else 0),
        "unexpected descriptor FD/RSS post-wave",
    )
    work, h = w["work"], w["host"]
    counts, draws = routing_counts(w, ids, full, plain)
    require(
        h["released"] == work["closed_bindings"] == work["admissions"] == 8
        and h["prefill_tokens"] == work["prefill_tokens"] == 256
        and h["step_tokens"] == work["step_tokens"] == 79
        and sum(int(k) * v.get("real_replays", 0) for k, v in w["buckets"].items()) == 79
        and w["dispatch"].get("eager_width_miss", 0) == 0
        and w["dispatch"]["cover_replays"] == 36
        and work["acknowledged"] == 87
        and work["batches"] == 12,
        "different logical or physical frontier work",
    )
    native = [n["pid"]] + ([p["pid"]] if born else [])
    priv = Counter()
    for i, q, g in zip(ids, w["private"], draws, strict=True):
        f = full[i]
        v, ex = Counter(q["work"]), q["exit"]
        calls = v["branch_rpcs"] + v["locate_rpcs"]
        require(
            q["row"] == i
            and (plain or (f["enter"] == sum(g) and f["entry_sha"] == sha256(g).hexdigest()))
            and q["matched"]
            and q["scrub"]
            and q["observed_sha"] == f["observed_sha"]
            and v["observations"] == 1 + f["length"]
            and v["observed_bytes"] == 32 + f["length"]
            and v["observed_bytes"] == v["native_bytes"] + v["unsynced_at_close"]
            and v["unsynced_at_close"] > 0
            and v["native_starts"] == int(ex is not None) == int(calls > 0)
            and int(ex is not None) <= v["native_appends"] <= calls
            and (32 <= v["native_bytes"] < v["observed_bytes"] if ex else not v["native_bytes"]),
            "wrong private binding or append debt",
        )
        if ex is not None:
            require(
                ex["rc"] == 0 and ex["peak_kib"] > 0 and v["open_ns"] > 0 and v["append_ns"] > 0,
                "private worker did not finish",
            )
            native.append(ex["pid"])
        priv.update(v)
    require(
        len(set(native)) == len(native)
        and not set(native) & actors
        and (bool(priv["native_starts"]) == bool(w["peak_private"]))
        and 0 <= w["peak_private"] <= priv["native_starts"],
        "reused/unbound native actor",
    )
    actors.update(native)
    return dict(
        **z,
        private_starts=priv["native_starts"],
        native_appends=priv["native_appends"],
        native_bytes=priv["native_bytes"],
        observed_bytes=priv["observed_bytes"],
        unsynced_end=priv["unsynced_at_close"],
        **counts,
        lf_steps=p["work"].get("lf_steps", 0),
        position_peak_mib=p["peak_kib"] / 1024 if born else None,
        rss_mib=mem["VmRSS"] / 1024,
        full_attest_s=n["attest_ns"] / 1e9,
        launch_s=n["workspace_ns"] / 1e9,
        position_open_s=p["start_ns"] / 1e9 if born else None,
        sig=sha256(json.dumps(signature(w), sort_keys=True).encode()).hexdigest(),
    )


def ratio_panel(cells):
    a, b = ([cells[f"{kind}-{rep}"] for rep in "AB"] for kind in ("fresh", "image"))

    def ratio(f, reduce=mean):
        return reduce(f(m) for m in b) / reduce(f(m) for m in a)

    out = dict(
        lifecycle=ratio(lambda m: m["lifecycle_s"]),
        first_p95=ratio(lambda m: m["first_ms"]["p95"]),
        setup=ratio(lambda m: m["charged_setup_s"]),
        cold_first_p95=ratio(lambda m: m["first_grant_wait_ms"]["p95"]),
        inter_p95=ratio(lambda m: m["inter_ms"]["p95"]),
        m20_hwm=ratio(lambda m: m["m20_peak_mib"], max),
        pos_hwm=ratio(lambda m: m["position_peak_mib"], max),
        device_peak=ratio(lambda m: m["peak_allocated_mib"], max),
        max_image_rss_drift_mib=max(m["rss_drift_mib"] for m in b),
    )
    limits = dict(
        lifecycle=0.85,
        first_p95=0.90,
        setup=0.65,
        cold_first_p95=1.30,
        inter_p95=1.25,
        m20_hwm=1.20,
        pos_hwm=1.20,
        device_peak=1.05,
        max_image_rss_drift_mib=64.0,
    )
    failed = [k for k, v in limits.items() if not 0 <= out[k] <= v]
    return dict(**out, failed=failed, passed=not failed)


def audit(root=ROOT):
    prior = sealed(
        HERE / "receipts/memory-program-t13/verified.json",
        "e05ec80992e93b4ed919030fc0f2493585423726d2b801fed2e1f575b1982c4a",
    )
    ref = (root / "gated-ref").read_text().strip()
    aa, good = ledger(root, 2784)
    cost = sum(a["upper"] for a in aa)
    require(
        prior["cumulative_x47_upper"] == 1429
        and prior["first_pair_parity"]
        and len(aa) == len(good) == 1
        and 0 < cost <= 150
        and aa[0]["attempt"] == "attempt1",
        "unbound prior or S19 charge",
    )
    r = good[0]
    idn = load(r / "identity.json")
    s17 = sealed(
        S17 / "identity.json",
        "3957788db0c43aa70e493daf052d5a1de9ae0d54f32581f9b4506f2131c03b95",
    )
    require(
        set(s17) == set(idn)
        and all(
            idn[k] == v
            for k, v in s17.items()
            if k not in ("ref", "owner", "server_pid", "generation_base", "order", "profile")
        )
        and all(
            idn["generation_base"][k] == v for k, v in s17["generation_base"].items() if k != "model_load_s"
        )
        and ref == idn["ref"] == (root / "source-ref").read_text().strip() == (r / "ref").read_text().strip()
        and (root / "protocol-ref").read_text().strip() == "a50d250e4dec26af56b8ec694a2a16226a7087d3"
        and idn["profile"] == "on-clock whole-archive grants"
        and idn["order"] == ORDER
        and idn["size_order"] == list(ORDER)
        and load(r / "allocator.json")["expandable_segments"]
        and re.fullmatch(
            rf"=+ {GATES} passed in [\d.]+s =+",
            (root / "qualification.txt").read_text().splitlines()[-1],
        )
        and int(fields(r / "released")["released_epoch"]) <= 1795531800,
        "wrong source, gate, host or predictor",
    )
    build = load(r / "construction.json")
    old_build = load(S17 / "construction.json")
    tr = load(r / "transfer.json")
    require(
        all(
            build[k] == old_build[k]
            for k in (
                "parameters_verified",
                "parameter_bytes",
                "distinct_storages",
                "rules",
                "tensors",
            )
        )
        and build["build_verify_ns"] > 0
        and build["small_to_cpu_ns"] > 0
        and tr["wide_drop_and_small_to_gpu_ns"] > 0,
        "incomplete host build",
    )
    actors = {idn["server_pid"], aa[0]["pid"]}
    clocks = []
    metrics = {}
    decisions = {}
    arena = {}
    errors = {}
    for size, order in ORDER.items():
        full = prior_prefix(size)
        setup = load(r / f"{size}-setup.json")
        end = load(r / f"{size}-closed.json")
        old = load(S17 / f"{size}-setup.json")
        require(
            all(setup[k] == old[k] for k in ("parameters", "tensor_bytes"))
            and sorted(setup["buckets"]) == sorted(old["buckets"])
            and 0 < setup["private_bytes"] <= (256 if size == "replicated" else 64) << 20
            and end["slots_freed"]
            and end["graphs_freed"]
            and end["lifecycle_ns"] > setup["prepare_ns"] > 0,
            "incomplete resident setup",
        )
        arena[size] = dict(
            prepare_s=setup["prepare_ns"] / 1e9,
            four_trials_and_arena_s=end["lifecycle_ns"] / 1e9,
            allocated_bytes=setup["tensor_bytes"],
            private_bytes=setup["private_bytes"],
        )
        results = {}
        first_sig = None
        for label in order:
            name = f"{size}-{label}"
            x = load(r / f"{name}.json")
            ws = [load(r / f"{name}-w{j}.json") for j in range(4)]
            image = label.startswith("image")
            im = x["image"]
            require(
                x["name"] == name
                and x["full"] == full
                and x["owner"] == idn["server_pid"] != x["sender"]
                and x["final_refusals"] == OPS
                and x["initial"]["registered_fds"] == x["final"]["registered_fds"] == 0
                and x["initial"]["clock_ns"] <= x["origin"] < x["end"] <= x["final"]["clock_ns"]
                and x["peak_reserved"] >= x["peak_allocated"] > 0
                and x["lifecycle_main_core_s"] > 0
                and x["lifecycle_child_core_s"] > 0,
                "changed S17 evidence or trial resources",
            )
            if image:
                require(
                    im["closed"]
                    and im["full_verifications"] == 1
                    and im["workers_started"] == im["workers_reaped"] == 4
                    and im["checks"] >= 4
                    and im["starts_bytes"] == 128958136
                    and im["offsets_checked"] == 16119767
                    and 0 < im["open_ns"] <= x["schedule_ready"] - x["schedule_start"]
                    and 0 < im["close_ns"] <= x["schedule_closed"] - x["close_start"],
                    "image not charged, rebound or closed",
                )
            else:
                require(im is None, "ordinary had retained descriptor")
            m = latencies(x, ws)
            z = [wave_work(w, j, full, image, idn, actors) for j, w in enumerate(ws)]
            sig = [v["sig"] for v in z]
            if first_sig is None:
                first_sig = sig
            else:
                require(sig == first_sig, "different per-wave requests, rows or work")
                proof = load(r / f"{name}-parity.json")
                scores(proof, 348, sum(v["enter"] for v in z))
                errors[name] = proof
            stats = {
                k: sum(v[k] for v in z)
                for k in (
                    "private_starts",
                    "native_appends",
                    "native_bytes",
                    "observed_bytes",
                    "unsynced_end",
                    "enter",
                    "direct",
                    "direct_probes",
                    "program_keys",
                    "lf_steps",
                    "r_frames",
                    "r_rows",
                    "wire_bytes",
                )
            }
            vmem = [v["rss_mib"] for v in z]
            m.update(
                **stats,
                waves=z,
                rss_drift_mib=max(abs(v - vmem[0]) for v in vmem[1:]),
                after_close_rss_mib=x["final"]["VmRSS"] / 1024,
                main_hwm_mib=x["final"]["VmHWM"] / 1024,
                m20_peak_mib=max(v["m20_peak_mib"] for v in z),
                position_peak_mib=max(v["position_peak_mib"] for v in z),
                peak_allocated_mib=x["peak_allocated"] / 2**20,
                peak_reserved_mib=x["peak_reserved"] / 2**20,
                main_core_s=x["lifecycle_main_core_s"],
                child_core_s=x["lifecycle_child_core_s"],
                image_build_s=im["open_ns"] / 1e9 if im else 0.0,
            )
            clocks.extend((x["initial"]["clock_ns"], x["final"]["clock_ns"]))
            results[label] = metrics[name] = m
        decisions[size] = ratio_panel(results)
    cross = load(r / "cross-size.json")
    scores(cross, 348, metrics["retained-image-A"]["enter"])
    done = load(r / "complete.json")
    require(
        done["steps"] == 0
        and done["server_pid"] == idn["server_pid"]
        and done["model"] == idn["model"]
        and clocks == sorted(clocks)
        and done["lifecycle_ns"] >= clocks[-1] - clocks[0]
        and done["total_emitted"] == 2784 == sum(m["enter"] + m["direct"] for m in metrics.values()),
        "incomplete generation",
    )
    return dict(
        ref=ref,
        attempts=aa,
        total_emitted=2784,
        completed=224,
        canceled=32,
        prompt_bytes=8 * 1024,
        host_advance_positions=8 * 316,
        host_updates=0,
        updates=0,
        gradient_targets=0,
        card_s_upper=cost,
        cumulative_x47_upper=1429 + cost,
        remaining_s=371 - cost,
        semantic_gate=True,
        onclock_image_recommended=all(v["passed"] for v in decisions.values()),
        ratios=decisions,
        metrics=metrics,
        arena=arena,
        cross_size=cross,
        errors=errors,
        full_main_s=done["lifecycle_ns"] / 1e9,
        load_and_predictor_s=build["prepare_head_host_ns"] / 1e9,
        wide_build_s=build["build_verify_ns"] / 1e9,
        transfer_to_cpu_s=build["small_to_cpu_ns"] / 1e9,
        drop_and_restore_s=tr["wide_drop_and_small_to_gpu_ns"] / 1e9,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
