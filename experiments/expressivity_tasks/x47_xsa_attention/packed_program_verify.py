"""Offline accounting and consistency checks for P2. Never launch a sampler or read raw windows."""

from __future__ import annotations
import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE / "receipts/memory-program-p2"
CELLS = ("reference-b4", "replay-b4", "packed-b4", "packed-b32")


def require(ok, why):
    if not ok:
        raise ValueError(why)


def load(p):
    return json.loads(p.read_text())


def fields(p):
    return dict(kv.split("=", 1) for kv in p.read_text().splitlines())


def digest(p):
    return sha256(p.read_bytes()).hexdigest()


def audit(root=ROOT, checkpoints=None):
    receipts = HERE / "receipts"
    p1 = load(receipts / "memory-program-p1/verified.json")
    previous = p1["cumulative_x47_upper"]
    require(previous == 114, "prior budget")
    p1id = load(receipts / "memory-program-p1/attempt1/identity.json")
    w3 = receipts / "w3"
    manifest = load(w3 / "train-pages.json")
    panel = load(w3 / "panels.json")["panels"]["train"]
    require(
        manifest["stage"] == "train"
        and digest(w3 / "panels.json") == manifest["panel_sha"] == p1id["data"]["panel"],
        "W3 train identity",
    )
    gate = (root / "qualification.txt").read_text()
    require("47 passed" in gate and "skipped" not in gate and "failed" not in gate, "failed CPU gates")
    ref = (root / "gated-ref").read_text().strip()
    protocol = (root / "protocol-ref").read_text().strip()
    require(len(ref) == len(protocol) == 40, "refs")
    charged = 0
    attempts = []
    good = []
    for folder in sorted(root.glob("attempt*/")):
        a, b, c = (fields(folder / n) for n in ("initial", "last", "released"))
        info = load(folder / "launch.json")
        pid = info["pid"]
        start = info["start"]
        require(
            all(a[k] == b[k] for k in ("pid", "host", "starttime", "created", "gpu", "adopted")),
            "owner changed",
        )
        require(
            (a["host"], a["pid"], a["starttime"], a["adopted"]) == ("lambda01", str(pid), start, "1")
            and a["gpu"] in ("0", "1"),
            "bad owner",
        )
        require(fields_proc(folder / "detachment") == {"Pid": str(pid), "PPid": "1"}, "not detached")
        used = 1 + int(c["released_epoch"]) - int(a["created"])
        require(used > 0, "bad lease duration")
        charged += used
        attempts.append(
            dict(attempt=folder.name, owner=pid, start=start, gpu=a["gpu"], upper=used, rc=c["rc"])
        )
        if c["rc"] != "0":
            continue
        after = fields(folder / "after")
        require(all(after[k] == b[k] for k in b if k != "heartbeat"), "post-training owner changed")
        require(
            int(a["heartbeat"]) < int(after["heartbeat"]) <= int(b["heartbeat"]) <= int(c["released_epoch"]),
            "heartbeat/release ordering",
        )
        obs = [json.loads(s) for s in (folder / "survival.jsonl").read_text().splitlines()]
        require(len(obs) > 1, "missing observed progress")
        require(
            all((o["owner"], o["starttime"], o["ppid"]) == (pid, start, 1) for o in obs),
            "unverified PPid/starttime",
        )
        require(max(o["heartbeat"] for o in obs) > int(a["heartbeat"]), "no witnessed renewal")
        end = load(folder / "progress.json")
        require(
            end["cell"] == "packed-b32"
            and end["phase"] == "updated"
            and end["targets_scored"] == 4096
            and end["optimizer_steps"] == 1,
            "incomplete run",
        )
        require(
            any(o["progress"] == end for o in obs) and (folder / "watch-ack").is_file(),
            "missing watched update",
        )
        require(
            all(o["progress"]["trainer_pid"] == end["trainer_pid"] != pid for o in obs),
            "trainer identity drift",
        )
        steps = [o["progress"]["targets_scored"] for o in obs if o["progress"]["cell"] == "packed-b32"]
        require(0 <= min(steps) < max(steps) == 4096, "missing advancing B32 work")
        require((folder / "ref").read_text().strip() == ref, "wrong source")
        good.append(folder)
    require(0 < charged <= 320 and len(good) == 1, "incomplete, duplicated or over-budget run")
    run = good[0]
    initial = fields(run / "initial")
    fx = load(run / "device-parity.json")
    nat = load(run / "natural-parity.json")
    require(
        fx["ref"] == ref
        and fx["checks"]["factor_logp"] >= 70
        and fx["delayed_grad_norm"] > 0
        and fx["max_errors"]["later_loss_to_old_write"] < 2e-4,
        "missing CUDA conformance",
    )
    require(
        all(isfinite(x) for x in nat["max_errors"].values())
        and nat["scalar_counts"]["byte_logp"] == 512 * 256
        and nat["scalar_counts"]["joint_logp"] == 512,
        "incomplete scores",
    )
    require(
        len(nat["gradient_errors"]) == nat["checks"]["parameter_grad"] > 15
        and all(x["error"] <= 0.0002 + 0.0002 * x["reference_max"] for x in nat["gradient_errors"].values()),
        "preclip adjoint gate",
    )
    reports = {}
    rates = {}
    samples = {}
    init = None
    for name in CELLS:
        r = load(run / (name + ".json"))
        ident = load(run / (name + "-identity.json"))
        reports[name] = r
        n = 32 if name == "packed-b32" else 4
        sampled = name != "replay-b4"
        d = ident["data"]
        require(
            (ident["ref"], r["ref"], r["cell"], r["mode"], r["optimizer_steps"], r["targets"])
            == (ref, ref, name, "fresh_sample" if sampled else "fixed_trace", int(sampled), 128 * n),
            "mode/exposure mismatch",
        )
        require(
            ident["owner"]["pid"] == initial["pid"]
            and ident["owner"]["starttime"] == initial["starttime"]
            and ident["trainer_pid"] == fx["trainer_pid"]
            and str(ident["trainer_pid"]) != initial["pid"],
            "invalid paid trainer",
        )
        require(
            all(
                ident[k] == p1id[k]
                for k in (
                    "base_sha",
                    "initial_policy",
                    "mount",
                    "branch_exe_sha",
                    "position_and_live_exe_sha",
                    "init_seed",
                    "limits",
                )
            ),
            "changed P1 model/caps/mount",
        )
        require(ident["sampling_seed"] == (7412 if sampled else None), "unexpected sampling or replay RNG")
        require(
            d["panel"] == manifest["panel_sha"]
            and d["shard"] == manifest["shards"][0]["sha256"]
            and d["manifest"] == p1id["data"]["manifest"],
            "changed dataset",
        )
        ix = list(range(32)) if n == 32 else list(range(4, 8))
        require(d["train_rows"] == ix and d["documents"] == [panel[i] for i in ix], "wrong train rows")
        require(len(d["window129_sha"]) == len(set(d["window129_sha"])) == n, "incomplete window digests")
        if name == "reference-b4":
            init = d["window129_sha"]
        require(
            (d["window129_sha"][4:8] if n == 32 else d["window129_sha"]) == init, "changed matched B4 windows"
        )
        require(
            len(r["docs_nll"]) == n and isclose(sum(r["docs_nll"]) / n, r["train_nll"], abs_tol=3e-6),
            "incomplete nll",
        )
        for x in r["preclip_gradient_norms"].values():
            require(x is None or isfinite(x), "nonfinite gradient")
        require(isfinite(r["gradient_norm_before_clip"]) and r["gradient_norm_before_clip"] > 0, "bad norm")
        require(
            (r["final_policy"] != ident["initial_policy"]) == sampled
            and (r["checkpoint_sha"] is not None) == sampled,
            "update invariant",
        )
        a = r["activity"]
        require(
            a["opcodes"]["EMIT"] == 128 * n
            and all(sum(a["outcomes"][op].values()) == v for op, v in a["opcodes"].items()),
            "lost opcode",
        )
        require(
            a["opcodes"].get("LOOKUP", 0) == sum(sum(s.values()) for s in a["lookups"].values()),
            "lost lookup",
        )
        require(
            r["provider_dispatches"] == sum(v["batches"] for v in r["provider"].values()),
            "provider call count",
        )
        tm = r["times"]
        require(
            tm["batch_ns"]
            == sum(tm[k] for k in ("forward_ns", "backward_ns", "gradient_snapshot_ns", "clip_update_ns")),
            "batch clock mismatch",
        )
        require(r["peak_allocated"] >= r["resident_bytes"] and tm["batch_ns"] > 0, "missing timing/memory")
        if sampled:
            sec = tm["batch_ns"] / 1e9
            rates[name] = r["targets"] / sec
            samples[name] = dict(
                targets=r["targets"],
                seconds=sec,
                targets_per_s=rates[name],
                projected_1048576_s=1048576 / rates[name],
            )
            if checkpoints:
                verify_checkpoint(checkpoints / (name + ".pt"), r, ident)
    refrow = reports["reference-b4"]
    replay = reports["replay-b4"]
    require(replay["activity"] == refrow["activity"], "changed executed activity")
    require(
        replay["emitted_event_digest"] == refrow["emitted_event_digest"] == nat["exact"]["events"]["sha256"],
        "natural events changed",
    )
    require(
        nat["exact"]["events"]["count"] == 512
        and nat["exact"]["dispatch"]["count"]
        == refrow["provider_dispatches"]
        == replay["provider_dispatches"],
        "request trace mismatch",
    )
    require(
        refrow["registers_observed"] == replay["registers_observed"] <= nat["exact"]["pages"]["count"],
        "register trace mismatch",
    )
    require(
        all(refrow[k] == replay[k] for k in ("m20_pages", "m20_batches", "live")), "different native work"
    )
    best = max(("packed-b4", "packed-b32"), key=rates.get)
    remaining = 1800 - previous - charged
    need = 1048576 / rates[best]
    return dict(
        attempts=attempts,
        card_s_upper=charged,
        cumulative_x47_upper=previous + charged,
        remaining_conservative_s=remaining,
        fresh=samples,
        ref_to_packed_b4=rates["packed-b4"] / rates["reference-b4"],
        packed_b32_to_packed_b4=rates["packed-b32"] / rates["packed-b4"],
        best_packed=best,
        best_vs_historical_p1=rates[best] / p1["train_targets_per_second"],
        one_arm_required_rate=1048576 / remaining,
        one_arm_fits_at_best_measured_rate=need <= remaining,
        b4_fresh_exact_event_trace=refrow["emitted_event_digest"]
        == reports["packed-b4"]["emitted_event_digest"],
        parity=nat["max_errors"],
        qualifier="one batch per geometry, teacher forcing through update; one-arm fit is necessary, not sufficient. No dev, confirmation, steady-state, or serial decode estimate.",
    )


def fields_proc(p):
    return dict((s.split(":", 1)[0], s.split(":", 1)[1].strip()) for s in p.read_text().splitlines())


def verify_checkpoint(path, r, ident):
    import torch
    from ndm.memory_program import MemoryPolicy
    from .workbench_witness import digest_state

    require(digest(path) == r["checkpoint_sha"], "checkpoint bytes")
    ck = torch.load(path, map_location="cpu", weights_only=True)
    require(
        (ck["step"], ck["cell"], ck["targets"], ck["data"], ck["base_sha"], ck["ref"])
        == (1, r["cell"], r["targets"], ident["data"], ident["base_sha"], ident["ref"]),
        "checkpoint manifest",
    )
    require(
        ck["optimizer"]["state"] and all(v["step"].item() == 1 for v in ck["optimizer"]["state"].values()),
        "missing AdamW update",
    )
    model = MemoryPolicy(256)
    model.load_state_dict(ck["model"])
    require(digest_state(model) == ck["policy_sha"] == r["final_policy"], "checkpoint state")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--checkpoints", type=Path)
    args = ap.parse_args()
    print(json.dumps(audit(args.root, args.checkpoints), indent=2, allow_nan=False))
