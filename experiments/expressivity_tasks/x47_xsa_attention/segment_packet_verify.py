"""Audit complete T6 observed-frontier production, extraction rates and actual post-release updates."""

import argparse, hashlib, json, math, struct
from collections import Counter
from pathlib import Path
from .frozen_packet_verify import encoded, valid_sha
from .serving_verify import HERE, load, ledger, fields, require

ROOT = HERE / "receipts/memory-program-t6"


def audit(root=ROOT, packets=None):
    prev = load(HERE / "receipts/memory-program-t5/verified.json")
    require(prev["cumulative_x47_upper"] == 1284 and prev["qualify_strict_cpu"], "wrong prior")
    aa, good = ledger(root, 32768, work_key="total_frontiers", actor_key="producer_pid")
    cost = sum(x["upper"] for x in aa)
    require(len(good) == 1 and 0 < cost <= 115, "no budgeted producer")
    r = good[0]
    idn = load(r / "identity.json")
    ref = (root / "gated-ref").read_text().strip()
    t4 = HERE / "receipts/memory-program-t4/attempt1"
    old = load(t4 / "identity.json")
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    require(
        "103 passed" in q
        and not {"failed", "skipped"}.intersection(q.split())
        and idn["ref"] == ref == (r / "ref").read_text().strip()
        and all(
            idn[k] == old[k]
            for k in (
                "data",
                "host",
                "policy_sha",
                "checkpoint",
                "position_exe_sha",
                "branch_exe_sha",
                "device",
                "input128_sha",
                "microbatch",
                "precision",
                "artifact_cap",
                "seed",
                "torch",
                "limits",
            )
        ),
        "wrong identity",
    )
    release = int(fields(r / "released")["released_epoch"]) * 10**9
    cpu = root / "cpu"
    done = load(cpu / "complete.json")
    order = ["retained-serial", "retained-segment", "replicated-segment", "replicated-serial"]
    require(
        done["order"] == order and done["targets"] == 16384 and done["steps"] == 4,
        "partial post-release workload",
    )
    ids = [load(cpu / z / "identity.json") for z in order]
    ts = sum(([i["start_epoch_ns"], i["end_epoch_ns"]] for i in ids), [])
    require(
        len({i["pid"] for i in ids}) == 4
        and ts == sorted(set(ts))
        and ts[0] >= release
        and ts[-1] <= done["end_epoch_ns"]
        and all(
            i["ref"] == ref
            and i["visible"] == ""
            and not i["cuda_available"]
            and not i["cuda_initialized"]
            and i["threads"] == 4
            and i["strict"]
            and not i["warn_only"]
            for i in ids
        ),
        "not independent strict learners after release",
    )
    metrics = {}
    prices = {}
    learn = {}
    eligible = True
    learning_ok = True
    for size, n in [("retained", 149855670), ("replicated", 2221786584)]:
        a = load(r / (size + "-build.json"))
        o = load(r / (size + "-oracle.json"))
        p4 = load(t4 / (size + "-packet.json"))
        cm = o["comparisons"]
        require(
            a["count"] == n
            and (
                n == 149855670
                or a["proof"]["parameters_verified"] == n
                and a["proof"]["distinct_storages"] == 597
            )
            and o["payloads"]["serial"] == p4["payload_sha"]
            and all(
                v["violations"] == v["argmax_changes"] == 0
                and v["max_fraction"] <= 1
                and v["elements"] == v["positions"] * 512
                and 0 <= v["unequal_bits"] <= v["elements"]
                for v in cm.values()
            )
            and {k: v["positions"] for k, v in cm.items()} == {"full": 4096, "31": 16 * 31, "65": 16 * 65},
            "missing frontier qualification",
        )
        arms = {}
        trials = {}
        st = {}
        for method in ("serial", "segment"):
            key = f"{size}-{method}"
            arms[method] = []
            for j in (1, 2):
                x = load(r / f"{key}-{j}.json")
                spec = x["model_spec"]
                b = x["binding"]
                h = encoded(b)
                head = b"FZBYT001" + struct.pack(">I", len(h)) + h
                w = x["work"]
                require(
                    x["cell"] == f"{key}-{j}"
                    and x["method"] == method
                    and x["scrubbed"]
                    and x["shape"] == [32, 128, 512]
                    and x["payload_sha"] == o["payloads"][method]
                    and valid_sha(x["sha256"])
                    and x["bytes"] == len(head) + 8388608 <= 9 << 20
                    and b["rows"] == p4["binding"]["rows"]
                    and b["source_sha"] == p4["binding"]["source_sha"]
                    and b["split"] == "train"
                    and b["width"] == 256
                    and b["encoder_sha"] == hashlib.sha256(encoded(spec)).hexdigest()
                    and spec["code"] == ref
                    and spec["native"] == idn["host"]
                    and spec["parameters"] == n
                    and spec["weights"] == idn["host"]["checkpoint"],
                    "unbound or incomplete packet",
                )
                require(
                    x["extract_ns"] == x["features_ns"] + x["publish_hash_ns"] > 0
                    and x["lifecycle_ns"] == x["prepare_ns"] + x["extract_ns"] + x["check_and_close_ns"]
                    and x["peak_reserved"]
                    >= x["peak_allocated"]
                    >= x["prepared_allocated"]
                    >= x["starting_allocated"],
                    "bad cost",
                )
                if method == "serial":
                    require(
                        w
                        == dict(
                            prefill_batches=2,
                            prefill_tokens=32,
                            step_batches=254,
                            step_tokens=4064,
                            released=32,
                        )
                        and x["traffic"]["real_replays"] == 254
                        and x["graph_private"] > 0
                        and x["scratch_bytes"] == (240762880 if size == "retained" else 962953216)
                        and spec["graph_width"] == 16,
                        "not serial",
                    )
                else:
                    require(
                        w == dict(pieces=4, observed=4096, released_rows=32)
                        and x["graph_private"] == 0
                        and x["scratch_bytes"]
                        == 16 * 27 * (2 * 128 * 256 + 1024 * 4) * 4 * (1 if size == "retained" else 4)
                        and x["traffic"]["gather_bytes"] > 0
                        and x["traffic"]["scatter_bytes"] > 0
                        and spec["piece"] == 64,
                        "not short native pieces",
                    )
                if packets is not None:
                    with (packets / f"{key}-{j}.f32").open("rb") as f:
                        require(f.read(len(head)) == head, "wrong header")
                        raw = f.read(8388609)
                        require(
                            len(raw) == 8388608
                            and hashlib.sha256(raw).hexdigest() == x["payload_sha"]
                            and hashlib.sha256(head + raw).hexdigest() == x["sha256"],
                            "wrong private payload",
                        )
                arms[method].append(x)
            require(arms[method][0]["sha256"] == arms[method][1]["sha256"], "nonidentical repeat")
            trials[method] = [
                {
                    k: v[k]
                    for k in (
                        "extract_ns",
                        "prepare_ns",
                        "lifecycle_ns",
                        "features_ns",
                        "peak_allocated",
                        "peak_reserved",
                        "scratch_bytes",
                        "graph_private",
                    )
                }
                for v in arms[method]
            ]
            rec = load(cpu / key / "identity.json")
            step = load(cpu / key / "step.json")
            c = step["checks"]
            require(
                rec["input_file"] == arms[method][0]["sha256"]
                and c["input_sha"] == o["payloads"][method]
                and step["mode"] == "fresh_sample"
                and step["targets"] == 4096
                and step["optimizer_steps"] == 1
                and not step["coupled"]
                and c["before"] == idn["policy_sha"] != c["after"]
                and len(c["gradients"]) == 35
                and c["gradient_elements"] > 0
                and step["activity"]["opcodes"]["EMIT"] == 4096
                and c["byte_vectors"] == rec["tensors"]["score:byte_vectors"]["sha"]
                and c["joint_scores"] == rec["tensors"]["score:joint_scores"]["sha"],
                "bad post-release step",
            )
            st[method] = dict(rec=rec, step=step)
        sr, sg = [arms[k] for k in ("serial", "segment")]
        ratio = sum(x["extract_ns"] for x in sg) / sum(x["extract_ns"] for x in sr)
        peak = max(x["peak_allocated"] for x in sg) / max(x["peak_allocated"] for x in sr)
        eligible &= ratio <= 0.80 and peak <= 1.35
        prices[size] = dict(
            wall=ratio,
            peak=peak,
            lifecycle=sum(x["lifecycle_ns"] for x in sg) / sum(x["lifecycle_ns"] for x in sr),
        )
        metrics[size] = dict(oracle=cm, times=trials)
        x, y = st["serial"]["step"], st["segment"]["step"]
        c, d = x["checks"], y["checks"]
        require(
            c == load(HERE / f"receipts/memory-program-t5/{size}-strict-A/step.json")["checks"],
            "strict reference mismatch",
        )
        logical = (
            all(c[k] == d[k] for k in ("events", "per_document_events", "dispatch"))
            and x["activity"] == y["activity"]
            and x["dispatches"] == y["dispatches"]
        )
        u, v = [st[k]["rec"]["tensors"] for k in ("serial", "segment")]
        diff = load(cpu / (size + "-differences.json"))
        require(
            diff["serial_only"] == sorted(set(u) - set(v))
            and diff["segment_only"] == sorted(set(v) - set(u))
            and diff["tensors"].keys() == u.keys() & v.keys(),
            "incomplete comparison",
        )
        viol = Counter()
        unequal = Counter()
        mx = Counter()
        checked = Counter()
        for k, z in diff["tensors"].items():
            group = k.split(":")[0]
            require(
                z["elements"] == math.prod(u[k]["shape"])
                and u[k]["shape"] == v[k]["shape"]
                and (z["unequal"] == 0) == (u[k]["sha"] == v[k]["sha"])
                and 0 <= z["violations"] <= z["unequal"] <= z["elements"]
                and math.isfinite(z["max_abs"])
                and z["max_abs"] >= 0,
                "inconsistent difference",
            )
            checked[group] += z["elements"]
            viol[group] += z["violations"]
            unequal[group] += z["unequal"]
            mx[group] = max(mx[group], z["max_abs"])
        masks = (
            not diff["serial_only"]
            and not diff["segment_only"]
            and all((d["gradients"][k] is None) == (v is None) for k, v in c["gradients"].items())
        )
        require(
            checked["score"] == 4096 * 257 + 2 and checked["param"] == 288987, "incomplete numerical proof"
        )
        ok = bool(logical and masks and not sum(viol.values()))
        learning_ok &= ok
        learn[size] = dict(
            agrees=ok,
            logical=logical,
            masks=masks,
            checked=checked,
            violations=viol,
            unequal_bits=unequal,
            max_abs=mx,
            serial_nll=x["nll"],
            segment_nll=y["nll"],
            serial_cpu_s=x["times"]["wall_to_update_ns"] / 1e9,
            segment_cpu_s=y["times"]["wall_to_update_ns"] / 1e9,
        )
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=cost,
        cumulative_x47_upper=1284 + cost,
        remaining_s=516 - cost,
        published_frontiers=32768,
        distinct_target_positions=4096,
        standalone_targets=16384,
        disposable_updates=4,
        host_updates=0,
        private_files_checked=packets is not None,
        metrics=metrics,
        segment_over_serial=prices,
        learning=learn,
        price_gate=bool(eligible),
        learning_gate=bool(learning_ok),
        promote_encoder=bool(eligible and learning_ok),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--packets", type=Path)
    args = ap.parse_args()
    print(json.dumps(audit(args.root, args.packets), indent=2, allow_nan=False))
