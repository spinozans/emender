"""T4 offline accounting, including its failed *post-release* exact-update gate."""

import argparse, hashlib, json, struct
from pathlib import Path
from .frozen_packet_verify import audit as common, encoded, valid_sha
from .serving_verify import HERE, ledger, load, require

ROOT = HERE / "receipts/memory-program-t4"


def audit(root=ROOT, packets=None):
    cpu = root / "cpu-learner"
    if (cpu / "complete.json").exists():
        return common(root, packets, stream=True)
    prev = load(HERE / "receipts/memory-program-t3/verified.json")["cumulative_x47_upper"]
    attempts, good = ledger(root, 8192, work_key="total_targets", actor_key="trainer_pid")
    charge = sum(a["upper"] for a in attempts)
    require(prev == 1227 and len(good) == 1 and 0 < charge <= 160, "missing released producer")
    r = good[0]
    ref = (root / "gated-ref").read_text().strip()
    idn = load(r / "identity.json")
    old = load(HERE / "receipts/memory-program-t1/attempt1/identity.json")
    require(
        ref == idn["ref"] == (r / "ref").read_text().strip()
        and "99 passed" in (root / "qualification.txt").read_text().splitlines()[-1]
        and idn["frozen_host"]
        and idn["microbatch"] == 16
        and idn["precision"] == "FP32 eval recurrence / normalized byte log probabilities"
        and all(
            idn[k] == v
            for k, v in old.items()
            if k not in ("ref", "owner", "trainer_pid", "precision", "microbatch")
        ),
        "frozen source",
    )
    pr = load(cpu / "progress.json")
    require(
        pr["cell"] == "retained"
        and pr["phase"] == "updated"
        and pr["total_targets"] == 4096
        and pr["learner_pid"] != idn["trainer_pid"]
        and not (cpu / "replicated-offline.json").exists()
        and "AssertionError: post-release update mismatch" in (root / "post-release-learner.log").read_text(),
        "not the observed update stop",
    )
    rows = [
        dict(document=z["doc"], start=z["start"], length=128, input_sha=h)
        for z, h in zip(idn["data"]["documents"], idn["input128_sha"], strict=True)
    ]
    source = hashlib.sha256(encoded({k: idn["data"][k] for k in ("manifest", "panel", "shard")})).hexdigest()
    stages = {}
    for name, n in [("retained", 149855670), ("replicated", 2221786584)]:
        p = load(r / (name + "-packet.json"))
        oracle = load(r / (name + "-prefix.json"))
        arena = load(r / (name + "-arena.json"))
        a = load(r / (name + "-coupled.json"))
        c = a["checks"]
        s = p["model_spec"]
        header = p["binding"]
        meta = encoded(header)
        head = b"FZBYT001" + struct.pack(">I", len(meta)) + meta
        w = p["work"]
        expected = {"stock-" + str(j): 2048 for j in (1, 31, 63)}
        expected["streaming-segment-128"] = 8192
        require(
            s["host"] == name
            and s["parameters"] == n
            and s["weights"] == idn["host"]["checkpoint"]
            and s["native"] == idn["host"]
            and s["context"] == 128
            and s["graph_width"] == 16
            and s["code"] == ref
            and "chunk" not in s
            and header
            == dict(
                encoder_sha=hashlib.sha256(encoded(s)).hexdigest(),
                source_sha=source,
                split="train",
                width=256,
                rows=rows,
            )
            and p["prefix_exact"] == oracle["short_exact"] == 16 * 63 * 512
            and oracle["elements"] == expected
            and oracle["argmax"]
            and oracle["scrubbed"]
            and set(oracle["max_tolerance_fraction"]) == set(expected)
            and all(0 <= v <= 1 for v in oracle["max_tolerance_fraction"].values())
            and p["prefix_max_error"] == max(oracle["max_errors"].values()),
            "missing causal oracle/binding",
        )
        require(
            w == dict(prefill_batches=2, prefill_tokens=32, step_batches=254, step_tokens=4064, released=32)
            and p["payload_bytes"] == 8388608
            and p["bytes"] == len(head) + p["payload_bytes"] <= 9 << 20
            and p["host_gradients"] == 0
            and p["payload_sha"] == c["input_sha"]
            and valid_sha(p["sha256"])
            and p["extract_ns"] == p["encode_copy_ns"] + p["hash_publish_ns"]
            and 0 < p["peak_allocated"] <= p["peak_reserved"],
            "unstaged producer",
        )
        require(
            a["coupled"]
            and a["targets"] == a["activity"]["opcodes"]["EMIT"] == 4096
            and a["mode"] == "fresh_sample"
            and a["optimizer_steps"] == 1
            and c["before"] == idn["policy_sha"] != c["after"]
            and len(c["gradients"]) == 35
            and 0 < c["gradient_elements"] <= 288987,
            "no coupled update",
        )
        require(
            n == 149855670
            or (
                p["shape_check"]["parameters_verified"] == n
                and p["shape_check"]["tensors"] == p["shape_check"]["distinct_storages"] == 597
            ),
            "wrong dense shape",
        )
        if packets is not None:
            with (packets / (name + ".f32")).open("rb") as f:
                require(f.read(len(head)) == head, "bad private header")
                raw = f.read(p["payload_bytes"] + 1)
                require(
                    len(raw) == p["payload_bytes"]
                    and hashlib.sha256(raw).hexdigest() == p["payload_sha"]
                    and hashlib.sha256(head + raw).hexdigest() == p["sha256"],
                    "private file mismatch",
                )
        stages[name] = dict(
            oracle=oracle,
            extract_s=p["extract_ns"] / 1e9,
            payload_bytes=p["payload_bytes"],
            file_bytes=p["bytes"],
            coupled_s=a["times"]["wall_to_update_ns"] / 1e9,
            coupled_nll=a["nll"],
            coupled_native_s=a["archive"]["validated_ns"] / 1e9,
            arena_bytes=arena["tensor_bytes"],
            gpu_peak=p["peak_allocated"],
            sha=p["sha256"],
        )
    a = load(r / "retained-coupled.json")
    z = load(cpu / "retained-offline.json")
    c = a["checks"]
    d = z["checks"]
    delta = [k for k in c if c[k] != d[k]]
    changed = [k for k in c["gradients"] if c["gradients"][k] != d["gradients"][k]]
    require(
        c.keys() == d.keys()
        and set(delta) == {"after", "gradients", "optimizer"}
        and c["gradients"].keys() == d["gradients"].keys()
        and set(changed) == {"embed.weight", "query.weight"}
        and all(valid_sha(d["gradients"][k]) and valid_sha(c["gradients"][k]) for k in changed)
        and d["after"] != d["before"]
        and z["mode"] == "fresh_sample"
        and not z["coupled"]
        and z["targets"] == 4096
        and z["optimizer_steps"] == 1
        and all(a[k] == z[k] for k in ("activity", "nll", "objective", "dispatches")),
        "different or unverified post-release failure",
    )
    return dict(
        ref=ref,
        qualified=False,
        disposition="identical forward predictions and logical events; exact first-adjoint/update gate failed on retained CPU learner",
        attempts=attempts,
        card_s_upper=charge,
        cumulative_x47_upper=prev + charge,
        remaining_s=1800 - prev - charge,
        scored_targets=12288,
        optimizer_steps=3,
        host_updates=0,
        post_release_scored=4096,
        post_release_update_equivalent=False,
        replicated_offline_attempted=False,
        gradient_hash_mismatches=changed,
        same_gradient_tensors=35 - len(changed),
        cpu_wall_to_update_s=z["times"]["wall_to_update_ns"] / 1e9,
        cpu_core_s=z["cpu_core_s"],
        private_files_checked=packets is not None,
        stages=stages,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--packets", type=Path)
    args = ap.parse_args()
    print(json.dumps(audit(args.root, args.packets), indent=2, allow_nan=False))
