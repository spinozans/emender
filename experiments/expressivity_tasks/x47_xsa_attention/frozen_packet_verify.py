"""Check T3 after release without raw bytes, checkpoints, CUDA or private feature storage."""

import argparse, hashlib, json, math, struct
from pathlib import Path
from .serving_verify import HERE, ledger, fields, load, require

ROOT = HERE / "receipts/memory-program-t3"


def encoded(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("ascii")


def valid_sha(s):
    return isinstance(s, str) and len(s) == 64 and all(c in "0123456789abcdef" for c in s)


def audit(root=ROOT, packets=None, *, stream=False):
    prior = "t3" if stream else "s13"
    prev = load(HERE / f"receipts/memory-program-{prior}/verified.json")["cumulative_x47_upper"]
    require(prev == (1227 if stream else 1213), "prior charge")
    aa, good = ledger(root, 8192, work_key="total_targets", actor_key="trainer_pid")
    cost = sum(a["upper"] for a in aa)
    # A failed prefix oracle is a terminal disposition, never successful packet/update parity.
    if not good and not stream:
        require(len(aa) == 1 and aa[0]["rc"] == 1 and 0 < cost <= 145, "unclassified producer failure")
        r = root / aa[0]["attempt"]
        ref = (root / "gated-ref").read_text().strip()
        i = load(r / "identity.json")
        q = (root / "qualification.txt").read_text().splitlines()[-1]
        pre = load(r / "retained-prefix.json")
        p = load(r / "progress.json")
        o = [json.loads(s) for s in (r / "survival.jsonl").read_text().splitlines()]
        init = fields(r / "initial")
        last = fields(r / "last")
        released = fields(r / "released")
        req = "Mismatched elements: 2 / 129024"
        text = (r / "console").read_text()
        require(
            "97 passed" in q
            and ref == i["ref"] == (r / "ref").read_text().strip()
            and pre["elements"] == 4 * 63 * 512
            and pre["violations"] == 2
            and math.isfinite(pre["max_error"])
            and req in text
            and "torch.testing.assert_close(actual, short" in text
            and p["total_targets"] == 0
            and p["cell"] == "retained"
            and not list(r.glob("*-packet.json"))
            and not list(r.glob("*-coupled.json"))
            and not (root / "cpu-learner").exists(),
            "not a prefix-gate stop",
        )
        require(
            o
            and all(
                (v["owner"], v["starttime"], v["ppid"], v["progress"]["trainer_pid"])
                == (aa[0]["pid"], aa[0]["start"], 1, i["trainer_pid"])
                for v in o
            )
            and int(init["heartbeat"])
            < max(v["heartbeat"] for v in o)
            <= int(last["heartbeat"])
            <= int(released["released_epoch"]),
            "missing live renewed failure witness",
        )
        return dict(
            ref=ref,
            attempts=aa,
            card_s_upper=cost,
            cumulative_x47_upper=prev + cost,
            remaining_s=1800 - prev - cost,
            qualified=False,
            disposition="retained BF16 chunk short-prefix oracle failed; no packet or learner",
            prefix=pre,
            scored_targets=0,
            independent_one_steps=0,
            standalone_cpu_targets=0,
            host_updates=0,
        )
    require(len(good) == 1 and 0 < cost <= (160 if stream else 145), "no budgeted producer")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    end = fields(r / "released")
    i = load(r / "identity.json")
    orig = load(HERE / "receipts/memory-program-t1/attempt1/identity.json")
    q = (root / "qualification.txt").read_text().splitlines()[-1]
    changing = {"owner", "ref", "trainer_pid"}
    if stream:
        changing.update(("microbatch", "precision"))
        require(
            i["microbatch"] == 16
            and i["precision"] == "FP32 eval recurrence / normalized byte log probabilities",
            "not the served host",
        )
    require(
        ("99 passed" if stream else "97 passed") in q
        and not {"failed", "skipped"}.intersection(q.split())
        and ref == i["ref"] == (r / "ref").read_text().strip()
        and all(i[k] == v for k, v in orig.items() if k not in changing)
        and i["frozen_host"]
        and i["opt"] == dict(lr=0.002, weight_decay=0.01, foreach=False)
        and i["clip"] == 1.0
        and i["artifact_cap"] == 9 << 20,
        "identity or CPU gate",
    )
    cpu = root / "cpu-learner"
    done = load(cpu / "complete.json")
    require(
        done["learner_pid"] != i["trainer_pid"]
        and done["ref"] == ref
        and done["visible"] == ""
        and not done["cuda_initialized"]
        and not done["cuda_available"]
        and done["threads"] == 4
        and done["producer_release_epoch"] == int(end["released_epoch"])
        and done["end_epoch_ns"] > done["start_epoch_ns"] >= int(end["released_epoch"]) * 10**9
        and done["total_targets"] == 8192
        and done["optimizer_steps"] == 2,
        "no independent post-release learner",
    )
    data = i["data"]
    source = hashlib.sha256(encoded({k: data[k] for k in ("manifest", "panel", "shard")})).hexdigest()
    expected_rows = [
        dict(document=z["doc"], start=z["start"], length=128, input_sha=h)
        for z, h in zip(data["documents"], i["input128_sha"], strict=True)
    ]
    require(
        len(expected_rows) == 32
        and len(set(i["input128_sha"])) == 32
        and all(map(valid_sha, i["input128_sha"])),
        "wrong input panel",
    )
    metrics = {}
    for name, count in [("retained", 149855670), ("replicated", 2221786584)]:
        p = load(r / (name + "-packet.json"))
        b = p["binding"]
        spec = p["model_spec"]
        desc = hashlib.sha256(encoded(spec)).hexdigest()
        prefix = b"FZBYT001" + struct.pack(">I", len(encoded(b))) + encoded(b)
        require(
            b == dict(encoder_sha=desc, source_sha=source, split="train", width=256, rows=expected_rows)
            and spec["host"] == name
            and spec["parameters"] == count
            and spec["weights"] == i["host"]["checkpoint"]
            and spec["native"] == i["host"]
            and (
                (spec["graph_width"] == 16 and spec["context"] == 128 and "chunk" not in spec)
                if stream
                else spec["chunk"] == i["chunk_sha"]
            )
            and spec["code"] == ref
            and p["payload_bytes"] == 32 * 128 * 512 * 4
            and p["bytes"] == len(prefix) + p["payload_bytes"] <= 9 << 20
            and valid_sha(p["sha256"])
            and valid_sha(p["payload_sha"]),
            "wrong binding/size",
        )
        proof = load(r / (name + "-build.json"))
        pre = load(r / (name + "-prefix.json"))
        if stream:
            arena = load(r / (name + "-arena.json"))
            w = p["work"]
            expected = {"stock-" + str(j): 4 * 512 for j in (1, 31, 63)}
            expected["streaming-segment-128"] = 16 * 512
            require(
                0 < arena["private_bytes"] <= 256 << 20
                and arena["tensor_bytes"] == (240762880 if name == "retained" else 962953216)
                and pre["argmax"]
                and pre["scrubbed"]
                and pre["short_exact"] == p["prefix_exact"] == 16 * 63 * 512
                and pre["elements"] == expected
                and sum(expected.values()) == p["prefix_elements"]
                and set(pre["max_tolerance_fraction"]) == set(expected)
                and all(0 <= v <= 1 for v in pre["max_tolerance_fraction"].values())
                and max(pre["max_errors"].values()) == p["prefix_max_error"]
                and w["prefill_tokens"] == w["released"] == 32
                and w["prefill_batches"] == 2
                and w["step_tokens"] == 4064
                and w["step_batches"] == 254,
                "missing serving-frontier proof",
            )
        else:
            require(
                p["prefix_elements"] == pre["elements"] == 4 * 63 * 512
                and pre["violations"] == 0
                and math.isfinite(pre["max_error"])
                and pre["max_error"] == p["prefix_max_error"],
                "missing chunk prefix proof",
            )
        require(
            proof["parameters"] == count
            and proof["shape_check"] == p["shape_check"]
            and p["host_gradients"] == 0
            and p["extract_ns"] == p["encode_copy_ns"] + p["hash_publish_ns"]
            and 0 < p["peak_allocated"] <= p["peak_reserved"],
            "missing host proof",
        )
        if name == "replicated":
            require(
                p["shape_check"]["parameters_verified"] == count
                and p["shape_check"]["distinct_storages"] == p["shape_check"]["tensors"] == 597,
                "not the wide dense shape",
            )
        else:
            require(p["shape_check"] is None, "original host not expected")
        loadrec = load(cpu / (name + "-read.json"))
        require(
            loadrec["expected_sha"] == p["sha256"]
            and loadrec["bytes"] == p["bytes"]
            and loadrec["load_ns"] > 0
            and loadrec["equal_update"],
            "missing post-release verified load",
        )
        a = load(r / (name + "-coupled.json"))
        z = load(cpu / (name + "-offline.json"))
        c = a["checks"]
        require(
            c == z["checks"]
            and c["input_sha"] == p["payload_sha"]
            and c["before"] == i["policy_sha"]
            and c["after"] != c["before"]
            and a["activity"] == z["activity"]
            and a["dispatches"] == z["dispatches"]
            and len(c["per_document_events"]) == 32
            and all(map(valid_sha, c["per_document_events"]))
            and all(
                valid_sha(c[k])
                for k in (
                    "byte_vectors",
                    "joint_scores",
                    "events",
                    "dispatch",
                    "nll",
                    "objective",
                    "before",
                    "after",
                    "optimizer",
                )
            )
            and len(c["gradients"]) == 35
            and all(valid_sha(v) or v is None for v in c["gradients"].values())
            and 0 < c["gradient_elements"] <= 288987
            and math.isfinite(c["grad_norm"])
            and c["grad_norm"] > 0,
            "nonidentical or missing update proof",
        )
        require(
            a["coupled"]
            and not z["coupled"]
            and a["peak_reserved"] >= a["peak_allocated"] > 0 == z["peak_allocated"] == z["peak_reserved"]
            and all(a[k] == z[k] for k in ("mode", "optimizer_steps", "targets", "nll", "objective"))
            and a["mode"] == "fresh_sample"
            and a["optimizer_steps"] == 1
            and a["targets"] == 4096
            and a["activity"]["opcodes"]["EMIT"] == 4096,
            "partial step",
        )
        for arm in (a, z):
            t = arm["times"]
            require(
                t["wall_to_update_ns"]
                == sum(
                    t[k] for k in ("forward_ns", "backward_ns", "gradient_attestation_ns", "clip_update_ns")
                )
                and arm["lifetime_ns"] >= t["setup_ns"] + t["wall_to_update_ns"]
                and math.isclose(arm["targets_s"], 4096e9 / t["wall_to_update_ns"]),
                "step clock",
            )
        if packets is not None:
            with (packets / (name + ".f32")).open("rb") as f:
                require(f.read(len(prefix)) == prefix, "wrong private file header")
                raw = f.read(p["payload_bytes"] + 1)
                require(
                    len(raw) == p["payload_bytes"]
                    and hashlib.sha256(raw).hexdigest() == p["payload_sha"]
                    and hashlib.sha256(prefix + raw).hexdigest() == p["sha256"],
                    "wrong private file contents",
                )

        def report(x):
            return {
                k: x[k]
                for k in (
                    "targets_s",
                    "times",
                    "peak_allocated",
                    "peak_reserved",
                    "lifetime_ns",
                    "cpu_core_s",
                    "rss_kib",
                    "child_rss_kib",
                )
            }

        metrics[name] = dict(
            parameters=count,
            feature_file_bytes=p["bytes"],
            extract_s=p["extract_ns"] / 1e9,
            encode_copy_s=p["encode_copy_ns"] / 1e9,
            hash_publish_s=p["hash_publish_ns"] / 1e9,
            load_s=loadrec["load_ns"] / 1e9,
            build_s=p["build_ns"] / 1e9,
            prefix_max_error=p["prefix_max_error"],
            extract_peak_allocated=p["peak_allocated"],
            extract_peak_reserved=p["peak_reserved"],
            coupled=report(a),
            offline=report(z),
            gradient_elements=c["gradient_elements"],
            unused=[k for k, v in c["gradients"].items() if v is None],
            after=c["after"],
            train_nll=z["nll"],
            work=z["activity"],
            logical_dispatches=z["dispatches"],
            coupled_archive=a["archive"],
            offline_archive=z["archive"],
        )
    return dict(
        ref=ref,
        attempts=aa,
        card_s_upper=cost,
        cumulative_x47_upper=prev + cost,
        remaining_s=1800 - prev - cost,
        scored_targets=16384,
        independent_one_steps=4,
        standalone_cpu_targets=8192,
        host_updates=0,
        metrics=metrics,
        private_files_checked=packets is not None,
        qualified=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--packets", type=Path)
    args = ap.parse_args()
    print(json.dumps(audit(args.root, args.packets), indent=2, allow_nan=False))
