"""Receipt-only: compare complete two-update transitions, logical evidence and actual physical service."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from statistics import mean
from struct import pack

from .frozen_packet_verify import valid_sha
from .serving_verify import HERE, load, require
from .terminal_branch_verify import wire

ROOT = HERE / "receipts/memory-program-t12"
ORDER = ("separate-A", "pair-A", "pair-B", "separate-B")
PURE = {"op.weight", "prior", "transitions.weight", "pointer.weight", "pointer.bias"}
STEP = [sha256(pack("<f", float(i))).hexdigest() for i in range(3)]


def sig(x):
    return sha256(json.dumps(x, separators=(",", ":"), sort_keys=True, allow_nan=False).encode()).hexdigest()


def historical_inputs():
    r = HERE / "receipts/memory-program-t4/attempt1"
    m = load(r / "retained-packet.json")
    p = load(r / "identity.json")
    d = p["data"]
    b = m["binding"]
    return dict(
        initial=p["policy_sha"],
        checkpoint=p["checkpoint"],
        packet=m["sha256"],
        features=m["payload_sha"],
        encoder=b["encoder_sha"],
        source=b["source_sha"],
        manifests={k: d[k] for k in ("manifest", "shard", "panel")},
        inputs=p["input128_sha"],
        observed129_sha=d["window129_sha"],
        origins=[[x["document"], x["start"]] for x in b["rows"]],
        mount=p["mount"],
        branch_exe_sha=p["branch_exe_sha"],
        position_exe_sha=p["position_exe_sha"],
        split="train",
        seed=7809,
    )


def accounted_eager_debt(work):
    """Counter addition drops zero entries. Never infer sync from an absent field alone.

    Each closed eager tape appends only its missing observed suffix. Nonnegative per-tape
    debts sum to observed minus acknowledged bytes; both positive totals are mandatory.
    """
    seen, applied = work.get("observed_bytes"), work.get("native_bytes")
    reported = work.get("unsynced_at_close", 0)
    return (
        type(seen) is int
        and type(applied) is int
        and type(reported) is int
        and 0 < seen == applied
        and reported == seen - applied == 0
    )


def transition(b, prev):
    a = b["update"]
    used = []
    gr = a["grads"]
    params = a["params"]
    mom = a["moments"]
    p0 = b["before_params"]
    m0 = b["before_moments"]
    require(
        params.keys() == gr.keys() == mom.keys() == p0.keys() == m0.keys()
        and len(gr) == 35
        and PURE < set(gr)
        and all(valid_sha(v) for v in [*params.values(), *p0.values(), b["before"], a["after"]])
        and a["after"] != b["before"]
        and a["writer_norm"] > 0
        and a["preclip_norm"] >= a["writer_norm"]
        and params["writer.weight"] != p0["writer.weight"]
        and valid_sha(gr["writer.weight"]),
        "missing actual value update",
    )
    if prev is None:
        require(not any(m0.values()), "nonempty initial optimizer")
    else:
        require(
            (b["before"], p0, m0) == (prev["after"], prev["params"], prev["moments"]),
            "optimizer or values reset between disjoint batches",
        )
    for k, g in gr.items():
        if g is None:
            require(params[k] == p0[k] and mom[k] == m0[k], "updated unused tensor")
        else:
            used.append(k)
            require(
                valid_sha(g)
                and k not in PURE
                and set(mom[k]) == {"step", "exp_avg", "exp_avg_sq"}
                and all(valid_sha(v) for v in mom[k].values())
                and mom[k]["step"] == STEP[1 + (STEP.index(m0[k]["step"]) if m0[k] else 0)],
                "bad adjoint or moment transition",
            )
    return len(used)


def check_cell(root, name, ref):
    path = root / name
    meta = load(path / "identity.json")
    cell = load(path / "cell.json")
    ex = load(path / "exit.json")
    d = load(root / "driver.json")
    ident = historical_inputs()
    pair = name.startswith("pair")
    ps = [cell["pid"]]
    expected = load(HERE / "receipts/memory-program-t4/attempt1/identity.json")
    require(
        meta["ref"] == ref == cell["ref"] == d["ref"]
        and meta["pid"] == ps[0] != d["pid"] == meta["parent"]
        and meta["name"] == name == cell["name"]
        and meta["inputs"] == ident
        and meta["strict"]
        and not meta["warn_only"]
        and meta["visible"] == ""
        and meta["threads"] == 4
        and meta["torch"] == expected["torch"]
        and len(set(map(tuple, ident["origins"]))) == 32
        and len(set(ident["inputs"])) == 32
        and cell["scored"] == 8192
        and cell["backward_targets"] == 4096
        and cell["updates"] == 2
        and not cell["cuda_initialized"]
        and not cell["cuda_available"]
        and ex["rc"] == cell["card_s"] == 0
        and ex["wall_ns"] >= cell["wall_ns"] > 0
        and cell["core_s"] > 0
        and cell["main_peak_kib"] > 0
        and load(path / "progress.json")
        == dict(pid=ps[0], step=1, phase="released", scored_in_role=2048, updates=2),
        "unbound/nonterminal fixed-control cell",
    )
    compare = []
    batchmetrics = []
    clocks = Counter()
    lf = Counter()
    live = Counter()
    archive = Counter()
    stage = Counter()
    peak = Counter()
    pm = px = 0.0
    lp = 0.0
    wirebytes = 0
    prev = None
    for n in range(2):
        b = load(path / f"batch-{n}.json")
        ix = list(range(n * 16, (n + 1) * 16))
        phys = b["physical"]
        st = b["stage"]
        p = b["position"]
        s = b["selectors"]
        v = b["values"]
        require(
            b["step"] == n
            and b["ix"] == ix
            and b["authority"] == sig(dict(identity=ident, rows=ix, selector=ident["initial"], ref=ref))
            and valid_sha(b["feature_sha"])
            and b["selector_after"] == ident["initial"]
            and (n or b["before"] == ident["initial"])
            and len(s["rows"]) == len(v["rows"]) == 16
            and s["work"] == v["work"]
            and s["work"]["targets"] == s["work"]["EMIT"] == 2048
            and s["work"]["logical_keys"] > 0
            and s["work"]["old_recalls"] > 0
            and s["work"]["SAVE"] > 0
            and s["work"]["instructions"] > 2048
            and all(
                valid_sha(z["nll_sha"])
                and z["nll"] > 0
                and all(
                    set(r) == {"choices", "events", "joint", "bytes", "argmax"}
                    and all(map(valid_sha, r.values()))
                    for r in z["rows"]
                )
                for z in (s, v)
            )
            and all(
                (a["choices"], a["events"]) == (c["choices"], c["events"])
                for a, c in zip(s["rows"], v["rows"], strict=True)
            ),
            "different executed control or incomplete evidence",
        )
        different = sum(a["bytes"] != c["bytes"] for a, c in zip(s["rows"], v["rows"], strict=True))
        require(
            (s == v if n == 0 else different > 0) and len(phys) == (1 if pair else 2),
            "wrong source/value contrast",
        )
        used = transition(b, prev)
        prev = b["update"]
        bound = sig(
            {
                k: b[k]
                for k in (
                    "authority",
                    "ix",
                    "feature_sha",
                    "before",
                    "before_params",
                    "before_moments",
                    "selectors",
                    "values",
                    "update",
                    "selector_after",
                )
            }
        )
        nominal = None
        for m in phys:
            if pair:
                require(
                    not m["encoded"] and st["encoders"]["issue"] == st["encoders"]["consume"],
                    "shared raw reader encoded values",
                )
            encoded = st["encoders"]["consume"] if pair else m["encoded"]
            require(
                0 < encoded["calls"] <= encoded["fragments"] <= encoded["text_bytes"]
                and encoded["components"] == 48 * encoded["fragments"],
                "incomplete independent value encoding",
            )
            w = m["live_work"]
            le = m["live_exits"]
            ps.extend(e["pid"] for e in le)
            require(
                m["observed129_sha"] == ident["observed129_sha"][ix[0] : ix[-1] + 1]
                and len(le) == w["native_starts"] == 16
                and all(e["rc"] == 0 and e["peak_kib"] > 0 for e in le)
                and w["native_appends"]
                == w["native_bytes"]
                == w["observations"]
                == w["observed_bytes"]
                == 16 * 129
                and accounted_eager_debt(w)
                and m["physical"]["frames"] > 0
                and s["work"]["logical_keys"] >= m["physical"]["keys"] > 0
                and m["physical"]["rows"] > 0
                and valid_sha(m["requests"]),
                "missing native reads/observations/exits",
            )
            a = dict(
                physical=m["physical"],
                encoding_per_role=encoded,
                live_per_role={k: x for k, x in w.items() if not k.endswith("_ns")},
                actual_request_stream=m["requests"],
            )
            require(nominal is None or nominal == a, "separate roles did not reissue the same reads")
            nominal = a
            live.update(a["live_per_role"])
            lp = max(lp, max(e["peak_kib"] for e in le) / 1024)
        if pair:
            require(
                st is not None
                and st["frames_still_staged"] == 0
                and all(st["total"][k] == nominal["physical"][k] for k in ("frames", "keys", "rows"))
                and 0 < st["total"]["text_bytes"]
                and all(
                    0 < st["peak_per_frontier"][k] <= lim
                    for k, lim in dict(frames=512, keys=1024, rows=32768, text_bytes=8 << 20).items()
                ),
                "missing/oversize frontier handoff",
            )
            stage.update(st["total"])
            peak |= Counter(st["peak_per_frontier"])
        else:
            require(st is None, "unexpected shared oracle")
        z = wire(b["wire"], False, ident)
        x = p["work"]
        ps += [p["pid"], b["wire"]["pid"]]
        require(
            p["rc"] == 0
            and p["fd_closed"]
            and p["start_ns"] > 0
            and b["position_peak_kib"] > 0
            and x.get("worker_keys", 0) >= x.get("worker_batches", 0) >= 0
            and x.get("ranks", 0) >= 0
            and b["actual_rowhash"]["frames"] == z["r_frames"]
            and valid_sha(b["actual_rowhash"]["sha"])
            and 0 < b["clocks"]["backward_ns"] < b["clocks"]["complete_ns"]
            and b["clocks"]["close_ns"] > 0
            and b["clocks"]["setup_ns"] > 0,
            "incomplete real immutable service",
        )
        logical_archive = {}
        for k, val in x.items():
            if not k.endswith("_ns"):
                require(val % len(phys) == 0, "unmatched repeated native work")
                logical_archive[k] = val // len(phys)
        compare.append((bound, nominal, logical_archive))
        archive.update({k: y for k, y in x.items() if not k.endswith("_ns")})
        lf.update({k: y for k, y in x.items() if k.endswith("_ns")})
        pm = max(pm, z["m20_peak_mib"])
        px = max(px, b["position_peak_kib"] / 1024)
        wirebytes += z["wire_bytes"]
        clocks.update(b["clocks"])
        batchmetrics.append(
            dict(
                rows=[ix[0], ix[-1]],
                source_nll=s["nll"],
                value_nll=v["nll"],
                different_value_documents=different,
                old_recalls=s["work"]["old_recalls"],
                routes=s["work"]["instructions"],
                logical_keys=s["work"]["logical_keys"],
                used_gradients=used,
                encoding_per_role=nominal["encoding_per_role"],
                physical_reads_per_role=nominal["physical"],
                writer_norm=prev["writer_norm"],
                value_after=prev["after"],
                signature=bound,
            )
        )
    require(
        len(ps) == len(set(ps)) == (37 if pair else 69) and cell["wall_ns"] >= clocks["complete_ns"],
        "reused/overlapping actors or invalid clock",
    )
    metrics = dict(
        wall_s=cell["wall_ns"] / 1e9,
        interpreter_s=ex["wall_ns"] / 1e9,
        core_s=cell["core_s"],
        main_peak_mib=cell["main_peak_kib"] / 1024,
        m20_peak_mib=pm,
        positional_peak_mib=px,
        largest_live_peak_mib=lp,
        archive=dict(archive),
        live=dict(live),
        stage_total=dict(stage),
        max_one_frontier=dict(peak),
        m20_wire_bytes=wirebytes,
        native_locate_s=lf["locate_ns"] / 1e9,
        phases_s={k: v / 1e9 for k, v in clocks.items()},
        batches=batchmetrics,
    )
    return compare, metrics, set(ps), [meta["start_epoch_ns"], cell["end_epoch_ns"]]


def audit(root=ROOT):
    ref = (root / "gated-ref").read_text().strip()
    drv = load(root / "driver.json")
    end = load(root / "complete.json")
    p = load(HERE / "receipts/memory-program-s18/verified.json")
    require(
        ref == (root / "source-ref").read_text().strip() == drv["ref"] == end["ref"]
        and drv["pid"] == end["pid"]
        and end["updates"] == 8
        and drv["order"] == list(ORDER)
        and drv["card_s"] == end["card_s"] == 0
        and end["wall_ns"] <= 480e9
        and p["retirement_gate"]
        and p["cumulative_x47_upper"] == 1429
        and "196 passed in" in (root / "qualification.txt").read_text().splitlines()[-1],
        "bad fixed-control budget or gate",
    )
    first = None
    m = {}
    allp = {drv["pid"]}
    epochs = [drv["start_epoch_ns"]]
    for name in ORDER:
        c, m[name], ids, ts = check_cell(root, name, ref)
        require(
            (first is None or first == c) and not (allp & ids),
            "different logical read, full score vector, adjoint or optimizer transition",
        )
        first = c
        allp.update(ids)
        epochs.extend(ts)
    epochs.append(end["end_epoch_ns"])
    require(epochs == sorted(epochs) and len(allp) == 213, "incomplete cell lifetimes")
    pair = {}
    green = True
    for rep in "AB":
        a = m["separate-" + rep]
        b = m["pair-" + rep]
        steps = a["archive"].get("lf_steps", 0)
        z = dict(
            lf=b["archive"].get("lf_steps", 0) / steps if steps else None,
            opens=b["live"]["native_starts"] / a["live"]["native_starts"],
            appends=b["live"]["native_appends"] / a["live"]["native_appends"],
            wall=b["wall_s"] / a["wall_s"],
        )
        pair[rep] = z
        green &= steps > 0 and max(z[k] for k in ("lf", "opens", "appends")) <= 0.55

    def ratio(k, op):
        return op(m["pair-" + r][k] for r in "AB") / op(m["separate-" + r][k] for r in "AB")

    wall = ratio("wall_s", mean)
    ram = {
        k: max(m["pair-" + r][k] / m["separate-" + r][k] for r in "AB")
        for k in ("main_peak_mib", "m20_peak_mib", "positional_peak_mib")
    }
    return dict(
        ref=ref,
        elapsed_s=end["wall_ns"] / 1e9,
        card_s=0,
        cumulative_x47_upper=1429,
        remaining_s=371,
        distinct_train_positions=4096,
        selector_forward_targets=16384,
        value_gradient_targets=16384,
        value_updates=8,
        selector_updates=0,
        host_calls=0,
        m20_exits=8,
        positional_exits=8,
        private_exits=192,
        actor_processes=len(allp),
        complete_transition_signatures=[c[0] for c in first],
        metrics=m,
        contrasts=pair,
        worst_peak_ratio=ram,
        mean_wall_ratio=wall,
        equivalence_gate=True,
        paired_control_recommended=green and wall <= 1.10 and max(ram.values()) <= 1.25,
        wall_speedup_ge_1p10=1 / wall >= 1.10,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    print(json.dumps(audit(ap.parse_args().root), indent=2, allow_nan=False))
