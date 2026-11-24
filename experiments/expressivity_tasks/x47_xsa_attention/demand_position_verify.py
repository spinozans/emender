"""S20: independently place native births in admission or the first real issuer's synchronous call."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import re
from statistics import mean

from .demand_tape_verify import scores
from .frozen_packet_verify import valid_sha
from .onclock_mount_verify import OPS, latencies, prior_prefix, sealed, signature, wave_work
from .serving_verify import HERE, fields, ledger, load, pct, require

ROOT = HERE / "receipts/memory-program-s20"
S19 = HERE / "receipts/memory-program-s19"
S12 = HERE / "receipts/memory-program-s12/attempt1"
S12_SHA = (
    "744940c2f9d4782489fdb16726caee7e050b005bbd1cbb22e4a6ebf843595220",
    "821f8f74cfa074d5cb1c24ccbde058c459ed2efc283d15c99f55c962c4d679f8",
)
ORDER = {
    "entry": ["eager-A", "demand-A", "demand-B", "eager-B"],
    "program": ["demand-A", "eager-A", "eager-B", "demand-B"],
}
GATES = 232
CALL = (
    "chosen",
    "keys",
    "cap",
    "sha",
    "ranks",
    "lf_steps",
    "lf_rows",
    "verified_spans",
    "exposed_spans",
    "overcap",
    "issuers",
    "tick",
)


def same_reads(w):
    return dict(**signature(w), actual_locates=[{k: c[k] for k in CALL} for c in w["native_reads"]])


def check_prefix(router, full):
    if router == "entry":
        require(full == prior_prefix("retained"), "S17 entry path or observed prefixes differ")
        return
    require(router == "program" and len(full) == 32, "unknown router or incomplete program output")
    for b, digest in enumerate(S12_SHA):
        x = sealed(S12 / f"{b}-b16-oracle-fused-sample.json", digest)
        ix = list(range(16 * b, 16 * b + 16))
        require(x["train_rows"] == ix and len(x["prefix_digests"]) == 16, "unbound S12 oracle rows")
        for i, prior in zip(ix, x["prefix_digests"], strict=True):
            v = full[i]
            require(
                {k: v[k] for k in ("length", "event_sha", "byte_sha")} == prior
                and prior["length"] == (3 if i % 8 == 0 else 12)
                and valid_sha(v["observed_sha"])
                and valid_sha(v["evidence_sha"])
                and 0 <= v["registers"] <= v["nonemit"] <= 5 * v["length"]
                and v["program_keys"] >= 0,
                "unconditional byte/choice/result prefix differs from S12",
            )


def activation(w, deferred):
    p = w["position"]
    born = p["pid"] is not None
    birth = p["birth"]
    cs = w["native_reads"]
    work = Counter(p["work"])
    require(born == (not deferred or bool(cs)), "wrong first-use start count")
    native_ns = []
    prior = None
    for c in cs:
        i = c["tick"]
        t = w["ticks"][i]
        issuers = c["issuers"]
        require(
            0 <= i < len(w["ticks"])
            and issuers == sorted(set(issuers))
            and issuers
            and set(issuers) <= set(t["rows"])
            and t["start"]
            <= c["lookup_begin"]
            <= c["before"]
            <= c["begin"]
            < c["end"]
            <= c["lookup_end"]
            <= t["ack"]
            and (prior is None or prior <= c["before"])
            and valid_sha(c["sha"])
            and 0 < c["keys"] <= c["chosen"] <= 16
            and c["cap"] >= 0
            and c["ranks"] >= 0
            and c["lf_steps"] >= 0
            and 0 <= c["exposed_spans"] <= c["verified_spans"] <= c["lf_rows"] <= 8 * c["keys"]
            and 0 <= c["overcap"] <= c["keys"],
            "native request without its chosen issuer, tick or row evidence",
        )
        prior = c["end"]
        native_ns.append(t["ack"] - t["start"])
    require(
        len(cs) == work["worker_batches"]
        and all(
            sum(c[a] for c in cs) == work[b]
            for a, b in (
                ("keys", "worker_keys"),
                ("ranks", "ranks"),
                ("lf_steps", "lf_steps"),
                ("lf_rows", "lf_rows"),
                ("exposed_spans", "verified_spans"),
                ("overcap", "native_overcap"),
            )
        ),
        "incomplete raw position work",
    )
    if born:
        require(
            p["rc"] == 0
            and p["peak_kib"] > 0
            and birth["handshake_bytes"] > 0
            and birth["arena_bytes"] > 0
            and 0 < p["start_ns"] <= birth["ready"] - birth["start"],
            "failed native startup/exit",
        )
        if deferred:
            require(
                birth["reason"] == "unresolved occurrences"
                and cs[0]["before"] <= birth["start"] < birth["ready"] <= cs[0]["begin"],
                "launch not contained in first actual unresolved batch",
            )
        else:
            require(
                birth["reason"] == "admission"
                and w["start"] <= birth["start"] < birth["ready"] <= w["ready"],
                "not eager admission",
            )
    else:
        require(
            all(p[k] is None for k in ("pid", "rc", "start_ns", "birth", "peak_kib")) and not cs,
            "imagined native receipt",
        )
    return dict(
        locator_started=int(born),
        native_batches=len(cs),
        native_keys=work["worker_keys"],
        ranks=work["ranks"],
        lf_rows=work["lf_rows"],
        verified_spans=work["verified_spans"],
        certified=work["certified_overcap"],
        native_overcap=work["native_overcap"],
        birth_ms=(birth["ready"] - birth["start"]) / 1e6 if born else None,
        first_native_ms=(cs[0]["end"] - cs[0]["before"]) / 1e6 if cs else None,
        max_native_tick_ms=max(native_ns) / 1e6 if cs else None,
    )


def price(cells, inter):
    ratios = {}
    fails = []
    for router, order in ORDER.items():
        sparse = router == "entry"
        a, b = ([cells[f"{router}-{v}-{k}"] for k in "AB"] for v in ("eager", "demand"))

        def r(f, red=mean):
            return red(f(m) for m in b) / red(f(m) for m in a)

        d = dict(
            starts=r(lambda m: m["position_starts"]),
            lifecycle=r(lambda m: m["lifecycle_s"]),
            first_p95=r(lambda m: m["first_ms"]["p95"]),
            cold_first_p95=r(lambda m: m["first_grant_wait_ms"]["p95"]),
            pooled_inter_p95=pct(inter[router, "demand"], 0.95) / pct(inter[router, "eager"], 0.95),
            device_peak=r(lambda m: m["peak_allocated_mib"], max),
        )
        lim = dict(lifecycle=0.95 if sparse else 1.10, first_p95=0.95 if sparse else 1.10, device_peak=1.05)
        if sparse:
            lim.update(starts=0.5, cold_first_p95=1.10)
        else:
            d.update(
                pos_hwm=r(lambda m: m["position_peak_mib"], max),
                native_tick_max=r(lambda m: m["max_native_tick_ms"]),
            )
            lim.update(pos_hwm=1.20, native_tick_max=1.25, pooled_inter_p95=1.20)
        failed = [k for k, limit in lim.items() if not 0 <= d[k] <= limit]
        ratios[router] = dict(**d, failed=failed, passed=not failed)
        fails.extend((router, k) for k in failed)
    return ratios, not fails


def audit(root=ROOT):
    prior = sealed(S19 / "verified.json", "369064a20f2326618ee90bc52548922e6fbde5bac73428e708705f936dae1b39")
    aa, good = ledger(root, 2784)
    paid = sum(a["upper"] for a in aa)
    ref = (root / "gated-ref").read_text().strip()
    require(
        prior["cumulative_x47_upper"] == 1489
        and prior["semantic_gate"]
        and not prior["onclock_image_recommended"]
        and len(good) == len(aa) == 1
        and 0 < paid <= 160,
        "unbound prior or charge",
    )
    r = good[0]
    idn = load(r / "identity.json")
    old = load(S19 / "attempt1/identity.json")
    end = load(r / "complete.json")
    require(
        (root / "protocol-ref").read_text().strip() == "accec6181aad44ed70edce4c674c4c3d3976736c"
        and ref == idn["ref"] == (root / "source-ref").read_text().strip() == (r / "ref").read_text().strip()
        and re.fullmatch(
            rf"=+ {GATES} passed in [\d.]+s =+", (root / "qualification.txt").read_text().splitlines()[-1]
        )
        and int(fields(r / "released")["released_epoch"]) <= 1795531800
        and aa[0]["attempt"] == "attempt1"
        and idn["profile"] == "synchronous first-use archive position"
        and idn["order"] == ORDER
        and idn["router_order"] == list(ORDER)
        and idn["size_order"] == ["retained"]
        and all(
            idn[k] == v
            for k, v in old.items()
            if k not in ("profile", "owner", "server_pid", "ref", "order", "size_order", "generation_base")
        )
        and all(
            idn["generation_base"][k] == v for k, v in old["generation_base"].items() if k != "model_load_s"
        )
        and load(r / "allocator.json")["expandable_segments"]
        and end["model"] == idn["model"]
        and end["steps"] == 0
        and end["total_emitted"] == 2784
        and end["server_pid"] == idn["server_pid"],
        "wrong predictor/host/ledger",
    )
    boot = load(r / "retained-setup.json")
    shut = load(r / "retained-closed.json")
    loaded = load(r / "loaded.json")
    require(
        boot["tensor_bytes"] == 240762880
        and 0 < boot["private_bytes"] <= 64 << 20
        and set(boot["buckets"]) == {"1", "4", "8", "16"}
        and loaded["parameters"] == boot["parameters"] == 149855670
        and loaded["prepare_host_predictor_ns"] > 0
        and shut["slots_freed"]
        and shut["graphs_freed"]
        and shut["lifecycle_ns"] > boot["prepare_ns"] > 0,
        "wrong resident host construction",
    )
    actors = {idn["server_pid"], aa[0]["pid"]}
    clocks = []
    metrics = {}
    inter = {}
    errors = {}
    for router, order in ORDER.items():
        first = None
        plain = router == "program"
        for label in order:
            name = f"{router}-{label}"
            deferred = label.startswith("demand")
            x = load(r / f"{name}.json")
            ws = [load(r / f"{name}-w{j}.json") for j in range(4)]
            check_prefix(router, x["full"])
            require(
                x["name"] == name
                and x["owner"] == idn["server_pid"] != x["sender"]
                and x["image"] is None
                and x["final_refusals"] == OPS
                and x["initial"]["registered_fds"] == x["final"]["registered_fds"] == 0
                and x["initial"]["clock_ns"] <= x["origin"] < x["end"] <= x["final"]["clock_ns"]
                and x["peak_reserved"] >= x["peak_allocated"] > 0
                and x["lifecycle_child_core_s"] > 0
                and x["lifecycle_main_core_s"] > 0,
                "unreleased or wrong trial",
            )
            m = latencies(x, ws)
            z = []
            sig = []
            for j, w in enumerate(ws):
                v = wave_work(w, j, x["full"], False, idn, actors, deferred=deferred, plain=plain)
                v.update(activation(w, deferred))
                z.append(v)
                sig.append(sha256(json.dumps(same_reads(w), sort_keys=True).encode()).hexdigest())
                for req in w["requests"]:
                    inter.setdefault((router, label.split("-")[0]), []).extend(
                        b - a for a, b in zip(req["acks"], req["acks"][1:])
                    )
            if first is None:
                first = (sig, x["full"])
            else:
                require((sig, x["full"]) == first, "different native dispatch, trajectory or logical work")
                proof = load(r / f"{name}-parity.json")
                if plain:
                    require(
                        proof["passed"]
                        and proof["positions"] == 348
                        and proof["wrong_evidence"] == 0
                        and set(proof["numeric"]) == {"byte_logp", "joint_logp"}
                        and all(
                            v["violations"] == 0 and 0 <= v["max_abs"] < float("inf") and v["elements"] == n
                            for k, n in [("byte_logp", 89088), ("joint_logp", 348)]
                            for v in [proof["numeric"][k]]
                        ),
                        "incomplete unconditional full scores",
                    )
                else:
                    scores(proof, 348, sum(v["enter"] for v in z))
                errors[name] = proof
            totals = {
                k: sum(v[k] for v in z)
                for k in (
                    "native_batches",
                    "native_keys",
                    "lf_steps",
                    "lf_rows",
                    "verified_spans",
                    "ranks",
                    "certified",
                    "private_starts",
                    "native_appends",
                    "native_bytes",
                    "observed_bytes",
                    "program_keys",
                    "wire_bytes",
                    "r_frames",
                    "r_rows",
                )
            }
            require(
                not plain or (totals["lf_steps"] > 0 and totals["verified_spans"] > 0),
                "positive natural locate never exercised",
            )
            m.update(
                **totals,
                position_starts=sum(v["locator_started"] for v in z),
                position_peak_mib=max(
                    (v["position_peak_mib"] for v in z if v["locator_started"]), default=None
                ),
                max_native_tick_ms=max(
                    (v["max_native_tick_ms"] for v in z if v["native_batches"]), default=None
                ),
                m20_peak_mib=max(v["m20_peak_mib"] for v in z),
                postwave_rss_mib=[v["rss_mib"] for v in z],
                final_rss_mib=x["final"]["VmRSS"] / 1024,
                main_hwm_mib=x["final"]["VmHWM"] / 1024,
                peak_allocated_mib=x["peak_allocated"] / 2**20,
                peak_reserved_mib=x["peak_reserved"] / 2**20,
                main_core_s=x["lifecycle_main_core_s"],
                child_core_s=x["lifecycle_child_core_s"],
                waves=z,
            )
            if not plain:
                m.update(enter=sum(v["enter"] for v in z), direct=sum(v["direct"] for v in z))
            metrics[name] = m
            clocks.extend((x["initial"]["clock_ns"], x["final"]["clock_ns"]))
    require(
        clocks == sorted(clocks)
        and end["lifecycle_ns"] >= clocks[-1] - clocks[0]
        and all(len(v) == 632 for v in inter.values()),
        "missing intervals or reordered trial",
    )
    ratios, recommend = price(metrics, inter)
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=paid,
        cumulative_x47_upper=1489 + paid,
        remaining_s=311 - paid,
        semantic_gate=True,
        start_count_gate=True,
        demand_position_recommended=recommend,
        ratios=ratios,
        total_emitted=2784,
        completed=224,
        canceled=32,
        prompt_bytes=8192,
        host_advance_positions=2528,
        gradient_targets=0,
        updates=0,
        full_main_s=end["lifecycle_ns"] / 1e9,
        host_predictor_load_s=loaded["prepare_host_predictor_ns"] / 1e9,
        arena_prepare_s=boot["prepare_ns"] / 1e9,
        metrics=metrics,
        score_comparisons=errors,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
