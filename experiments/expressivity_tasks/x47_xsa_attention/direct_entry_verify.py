"""Public T11 receipt audit. Reconstruct the initialized host-only gate schedule, never any addresses."""

from collections import Counter
from hashlib import blake2b, sha256
import argparse, json, math, struct
from pathlib import Path
from statistics import mean
from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, load, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-t11"
ORDER = ("index-A", "queue-A", "queue-B", "index-B")


def digest(rows):
    h = sha256()
    for row in rows:
        b = repr(row).encode()
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()


def schedule(family):
    # 24-bit indexed uniform; initialized gate independent of the text, probability 1/16.
    entries = []
    for i in range(32):
        row = []
        for j in range(128):
            bits = blake2b(
                struct.pack(">QQQ", 7823, i, j + 1), digest_size=8, person=b"byte-entry-v1"
            ).digest()
            if int.from_bytes(bits[:3], "big") < 1 << 20:
                row.append(j)
        entries.append(row)
    pos = [0] * 32
    w = {k: Counter() for k in ("program", "direct")}
    held = 0
    while min(pos) < 128:
        on = []
        off = []
        for i, t in enumerate(pos):
            if t < 128:
                (on if t in entries[i] else off).append(i)
        if off:
            w["direct"][str(len(off))] += 1
            for i in off:
                pos[i] += 1
        if on:
            if family == "index" or len(on) >= 8 or not off:
                w["program"][str(len(on))] += 1
                for i in on:
                    pos[i] += 1
            else:
                held += len(on)
    return entries, w, held


def audit(root=ROOT):
    parent = load(HERE / "receipts/memory-program-s16/verified.json")
    h = load(HERE / "receipts/memory-program-s14/head.json")
    base = HERE / "receipts/memory-program-t4/attempt1"
    prod = load(base / "identity.json")
    p = load(base / "retained-packet.json")
    t2 = HERE / "receipts/memory-program-t2/attempt1"
    gi = load(t2 / "identity.json")
    hist = load(t2 / "b32-index.json")["logical"]
    ref = (root / "gated-ref").read_text().strip()
    dr = load(root / "driver.json")
    end = load(root / "complete.json")
    require(
        ref == (root / "source-ref").read_text().strip() == dr["ref"]
        and (root / "protocol-ref").read_text().strip() == "e97148a0bf86cfc4cbb16e7de87e4bbfc22c423e"
        and " 156 passed in " in (root / "qualification.txt").read_text().splitlines()[-1]
        and parent["cumulative_x47_upper"] == 1386
        and parent["remaining_s"] == 414
        and parent["exact_descriptor_gate"]
        and tuple(dr["order"]) == ORDER
        and dr["pid"] == end["pid"]
        and dr["card_s"] == end["card_s"] == end["updates"] == 0
        and 0 < end["elapsed_ns"] <= 360e9
        and end["targets"] == 16384
        and end["distinct"] == 4096,
        "incomplete T11 driver",
    )
    expected = dict(
        head=h["after"],
        policy=prod["policy_sha"],
        checkpoint=prod["checkpoint"],
        features=p["payload_sha"],
        packet=p["sha256"],
        mount=prod["mount"],
        binary=prod["branch_exe_sha"],
        input128_sha=prod["input128_sha"],
        split="train",
        gate=gi["entry_initial"],
        gate_seed=7823,
        program_seed=7809,
        window129_sha=hist["window_sha"],
        gate_floor=8,
        key_price=0.001,
        instruction_price=0.0001,
        remaining_baseline=2.0,
    )
    seen = {dr["pid"]}
    stamps = [dr["start_epoch_ns"]]
    cells = {}
    metrics = {}
    logical = mask = None
    diffs = {}
    for name in ORDER:
        fam = name.split("-")[0]
        entries, widths, held = schedule(fam)
        path = root / name
        c = load(path / "cell.json")
        ex = load(path / "exit.json")
        pair = load(path / "comparison.json")
        cells[name] = c
        require(
            c["ref"] == ref
            and c["identity"] == expected
            and c["pid"] not in seen
            and c["parent"] == dr["pid"]
            and c["name"] == name
            and c["torch"] == prod["torch"]
            and c["visible"] == ""
            and not c["cuda_available"]
            and not c["cuda_initialized"]
            and c["strict"]
            and not c["warn_only"]
            and c["threads"] == 4
            and c["targets"] == 4096
            and c["updates"] == ex["rc"] == 0
            and 0 < c["ipc_bytes"] <= 8 << 20
            and c["lifecycle_core_s"] >= c["core_s"] > 0
            and c["main_peak_kib"] > 0
            and c["forward_ns"] > c["feature_ns"] > 0
            and c["backward_ns"] > 0
            and c["forward_ns"] + c["backward_ns"] == c["wall_ns"]
            and ex["process_wall_ns"] >= c["lifecycle_ns"] >= c["setup_ns"] + c["wall_ns"] > c["wall_ns"]
            and load(path / "progress.json") == dict(pid=c["pid"], phase="released", targets=4096),
            "incomplete CPU learner",
        )
        stamps.extend([c["start_epoch_ns"], c["end_epoch_ns"]])
        seen.add(c["pid"])
        actors = c["worker_exits"]
        peaks = c["worker_peak_kib"]
        ww = c["wire"]
        wc = wire(ww, True, prod)
        require(
            len(actors) == len(peaks) == 33
            and all(x > 0 for x in peaks)
            and ww["pid"] not in seen
            and len(c["end_rss_kib"]) == 35,
            "missing mounts",
        )
        seen.add(ww["pid"])
        for pid, rc in actors:
            require(rc == 0 and pid not in seen, "overlap/unreaped worker")
            seen.add(pid)
        l = c["logical"]
        z = c["demand"]
        dw = z["trust"]
        work = z["clock"]
        docs = l["docs"]
        require(
            l["entry_positions"] == digest(entries) == hist["entry_positions"]
            and [len(e) for e in entries] == hist["entries_per_doc"]
            and len(docs) == 32
            and all(
                d["row"] == i
                and d["entries"] == len(entries[i])
                and d["direct"] == 128 - len(entries[i])
                and d["entry_sha"] == digest(entries[i])
                and all(
                    valid_sha(d[k])
                    for k in (
                        "program_events",
                        "register_bindings",
                        "direct_evidence",
                        "price_sha",
                        "argmax",
                    )
                )
                for i, d in enumerate(docs)
            )
            and l["program"] == l["opcodes"]["EMIT"] == 248
            and sum(l["opcodes"].values()) - 248 == l["nonemit"]
            and l["direct"] == 3848
            and l["direct_keys"] == dw["logical_probes"] >= l["direct"]
            and 0 < l["program_keys"]
            and dw["logical_probes"]
            == sum(dw.get(k, 0) for k in ("selected", "unavailable", "end_only", "absent"))
            and 0 < dw["hist_calls"] <= dw["submitted_keys"] <= dw["logical_probes"]
            and dw["submitted_keys"] >= wc["s_frames"]
            and wc["r_rows"] == z["branch_rows"] + z["direct_rows"]
            and wc["r_frames"] >= z["branch_batches"] + z["direct_batches"] > 0
            and l["local_increments"] == 20160
            and l["append_calls"] == l["append_bytes"] == z["live"]["appends"] == 4128
            and z["widths"] == widths
            and work.get("held_entry_ticks", 0) == held
            and work["scored"] == 4096
            and all(
                l[k] == work[k]
                for k in (
                    "direct",
                    "program",
                    "direct_keys",
                    "program_keys",
                    "nonemit",
                    "local_increments",
                    "append_calls",
                    "append_bytes",
                )
            )
            and work["controller_rounds"] >= sum(widths["program"].values())
            and valid_sha(z["dispatch_sha"])
            and z["dispatches"] == sum(x["batches"] for x in z["provider"].values())
            and 0 < sum(x["keys"] for x in z["provider"].values()) <= l["program_keys"],
            "changed gate/batching, observation, work price or native demand",
        )
        require(
            pair["same_logical"] and (logical is None or logical == l),
            "different ordered evidence or argmax",
        )
        logical = l
        ps = c["parameters"]
        require(mask is None or mask == ps, "parameter/adjoint mask mismatch")
        mask = ps
        cnt = Counter()
        shapes = dict(
            byte_logp=[4096, 256],
            gate_logp=[4096],
            conditional_logp=[248],
            joint_logp=[4096],
            trust=[3848, 4],
            losses=[4096],
            future=[4096],
            mean_nll=[],
            objective=[],
        )
        for k, s in ps.items():
            require(math.prod(s["shape"]) == s["elements"] > 0, "parameter shape")
            cnt[k.partition(".")[0]] += s["elements"]
            if s["used"]:
                shapes["grad:" + k] = s["shape"]
        m = c["snapshots"]
        require(
            cnt == {"entry": 257, "program": 288987, "direct": 17604}
            and all(0 < c["nonzero"][k] <= cnt[k] for k in cnt)
            and shapes == {k: a["shape"] for k, a in m.items()}
            and all(valid_sha(a["sha"]) for a in m.values()),
            "missing adjoint or full scores",
        )
        prevs = {
            "index-A": set(),
            "queue-A": {"index-A"},
            "queue-B": {"index-A", "queue-A"},
            "index-B": {"index-A"},
        }[name]
        require(set(pair["checks"]) == prevs, "incomplete numeric comparisons")
        for before, results in pair["checks"].items():
            exact = fam == before.split("-")[0]
            prior = cells[before]["snapshots"]
            require(set(results) == set(m), "comparison lost tensor")
            for k, d in results.items():
                require(
                    d["ref_sha"] == prior[k]["sha"]
                    and d["actual_sha"] == m[k]["sha"]
                    and d["elements"] == math.prod(shapes[k])
                    and 0 <= d["unequal_bits"] <= d["elements"]
                    and (d["unequal_bits"] == 0) == (d["ref_sha"] == d["actual_sha"])
                    and math.isfinite(d["max_abs"])
                    and d["max_abs"] >= 0
                    and d["violations"] == 0
                    and 0 <= d["max_fraction"] <= 1
                    and (not exact or d["unequal_bits"] == 0),
                    "numeric gate: " + k,
                )
            diffs[before + "->" + name] = dict(
                elements=sum(d["elements"] for d in results.values()),
                unequal_bits=sum(d["unequal_bits"] for d in results.values()),
                score_max_abs=max(d["max_abs"] for k, d in results.items() if not k.startswith("grad:")),
                grad_max_abs=max(d["max_abs"] for k, d in results.items() if k.startswith("grad:")),
                max_fraction=max(d["max_fraction"] for d in results.values()),
            )
        metrics[name] = dict(
            program_batches=sum(widths["program"].values()),
            direct_batches=sum(widths["direct"].values()),
            controller_rounds=work["controller_rounds"],
            held_entry_ticks=held,
            program_keys=l["program_keys"],
            direct_probes=l["direct_keys"],
            direct_hist_calls=dw["hist_calls"],
            dispatches=z["dispatches"],
            lf_rows=z["archive"].get("lf_rows", 0),
            lf_steps=z["archive"].get("lf_steps", 0),
            **wc,
            position_peak_mib=peaks[0] / 1024,
            live_summed_peaks_mib=sum(peaks[1:]) / 1024,
            main_peak_mib=c["main_peak_kib"] / 1024,
            forward_s=c["forward_ns"] / 1e9,
            backward_s=c["backward_ns"] / 1e9,
            wall_s=c["wall_ns"] / 1e9,
            gradient_targets_s=4096e9 / c["wall_ns"],
            setup_s=c["setup_ns"] / 1e9,
            core_s=c["core_s"],
            lifecycle_core_s=c["lifecycle_core_s"],
            lifecycle_s=c["lifecycle_ns"] / 1e9,
            subprocess_s=ex["process_wall_ns"] / 1e9,
        )
    require(
        stamps + [end["end_epoch_ns"]] == sorted(stamps + [end["end_epoch_ns"]]), "unordered process clocks"
    )

    def ratio(k, fn):
        return fn(m[k] for n, m in metrics.items() if n.startswith("queue")) / fn(
            m[k] for n, m in metrics.items() if n.startswith("index")
        )

    ratios = dict(
        wall=ratio("wall_s", mean),
        main_peak=ratio("main_peak_mib", max),
        m20_peak=ratio("m20_peak_mib", max),
        position_peak=ratio("position_peak_mib", max),
    )
    ok = (
        metrics["queue-A"]["program_batches"] < metrics["index-A"]["program_batches"]
        and ratios["wall"] <= 0.85
        and all(v <= 1.20 for k, v in ratios.items() if k != "wall")
    )
    return dict(
        ref=ref,
        gradient_targets=end["targets"],
        distinct_train_positions=4096,
        updates=0,
        host_gradients=False,
        generated_bytes=0,
        card_s=0,
        cumulative_x47_upper=1386,
        remaining_s=414,
        elapsed_s=end["elapsed_ns"] / 1e9,
        enter_per_cell=248,
        direct_per_cell=3848,
        comparisons=diffs,
        metrics=metrics,
        ratios=ratios,
        equivalence_gate=True,
        queued_clock_recommended=ok,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
