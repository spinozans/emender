"""Offline receipt consistency, exposure and lease/scale arithmetic. Does not rerun the sampled policy."""

import argparse
from hashlib import sha256
import json
from math import isclose, isfinite
from pathlib import Path

ROOT = Path(__file__).with_name("receipts") / "memory-program-p1"


def audit(checkpoint=None):
    def load(p):
        return json.loads(p.read_text())

    def fields(p):
        return dict(kv.split("=", 1) for kv in p.read_text().splitlines())

    def require(ok):
        if not ok:
            raise ValueError("inconsistent profile receipts")

    w3 = ROOT.parent / "w3"
    charged = 0
    good = []
    for folder in sorted(ROOT.glob("attempt*/")):
        initial, last, end = (fields(folder / n) for n in ("initial", "last", "released"))
        launch = load(folder / "launch.json")
        require(all(initial[k] == last[k] for k in ("gpu", "pid", "starttime", "host", "created", "adopted")))
        require(
            (initial["host"], initial["adopted"], initial["pid"], initial["starttime"])
            == ("lambda01", "1", str(launch["pid"]), launch["start"])
        )
        require(
            initial["gpu"] in ("0", "1") and "PPid:\t1" in (folder / "detachment").read_text().splitlines()
        )
        charge = 1 + int(end["released_epoch"]) - int(initial["created"])
        require(charge > 0)
        charged += charge
        if end["rc"] == "0":
            good.append(folder)
            after = fields(folder / "after")
            require(all(after[k] == last[k] for k in last if k != "heartbeat"))
            require(
                int(initial["heartbeat"])
                < int(after["heartbeat"])
                <= int(last["heartbeat"])
                <= int(end["released_epoch"])
            )
            obs = [json.loads(line) for line in (folder / "survival.jsonl").read_text().splitlines()]
            require(
                all(
                    (str(o["owner"]), o["starttime"], o["ppid"]) == (initial["pid"], initial["starttime"], 1)
                    for o in obs
                )
            )
            steps = [o["progress"]["targets_scored"] for o in obs]
            require(
                max(steps) > min(steps)
                and max(steps) == 512
                and max(o["heartbeat"] for o in obs) > int(initial["heartbeat"])
            )
            require(int(last["heartbeat"]) > int(initial["heartbeat"]))
    require(0 < charged <= 200 and len(good) == 1)
    run = good[0]
    report = load(run / "profile.json")
    identity = load(run / "identity.json")
    require(
        report["ref"]
        == identity["ref"]
        == (run / "ref").read_text().strip()
        == (ROOT / "gated-ref").read_text().strip()
    )
    require(
        "41 passed" in (ROOT / "qualification.txt").read_text()
        and "skipped" not in (ROOT / "qualification.txt").read_text()
    )
    require(
        identity["owner"]["pid"] == fields(run / "initial")["pid"]
        and identity["trainer_pid"] != int(identity["owner"]["pid"])
    )
    require(
        load(run / "progress.json")["optimizer_steps"] == report["optimizer_steps"] == 1
        and report["targets"] == 512
    )
    require(
        identity["data"]["documents"] == load(w3 / "panels.json")["panels"]["train"][4:8]
        and identity["data"]["train_rows"] == [4, 5, 6, 7]
    )
    manifest = load(w3 / "train-pages.json")
    require(
        identity["data"]["panel"]
        == manifest["panel_sha"]
        == sha256((w3 / "panels.json").read_bytes()).hexdigest()
    )
    require(
        (identity["mount"], identity["branch_exe_sha"], identity["data"]["shard"])
        == (manifest["mount"], manifest["exe_sha"], manifest["shards"][0]["sha256"])
    )
    require(isclose(sum(report["docs_nll"]) / 4, report["sampled_train_nll"], abs_tol=2e-6))
    a = report["activity"]
    require(
        a["opcodes"]["EMIT"] == 512
        and all(a["opcodes"][k] == sum(v.values()) for k, v in a["outcomes"].items())
    )
    require(a["opcodes"]["LOOKUP"] == sum(sum(v.values()) for v in a["lookups"].values()))
    require(
        len(a["recall_lags"]) == a["nonempty_prior_byte_recalls"]
        and all(0 < lag < 128 for lag in a["recall_lags"])
    )
    require(all(isfinite(x) for x in report["clipped_norms"].values() if x is not None))
    require(report["gradient_norm_before_clip"] > 0)
    require(report["final_policy"] != identity["initial_policy"])
    tm = report["times"]
    sec = tm["batch_ns"] / 1e9
    require(tm["batch_ns"] == sum(tm[k] for k in ("forward_ns", "backward_ns", "clip_update_ns")))
    if checkpoint:
        import torch
        from ndm.memory_program import MemoryPolicy
        from .workbench_witness import digest_state

        require(sha256(checkpoint.read_bytes()).hexdigest() == report["checkpoint_sha"])
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(
            ck["step"] == 1
            and ck["targets"] == 512
            and ck["data"] == identity["data"]
            and ck["base_sha"] == identity["base_sha"]
            and ck["ref"] == report["ref"]
        )
        model = MemoryPolicy(256)
        model.load_state_dict(ck["model"])
        require(digest_state(model) == ck["policy_sha"] == report["final_policy"])
    total = load(w3 / "verified.json")["x47_upper"] + charged
    remaining = 1800 - total
    require(remaining > 0)
    projected = sec / 512 * 1048576
    require(isclose(projected, report["projected_seconds_one_arm_1048576"]))
    return dict(
        card_s_upper=charged,
        cumulative_x47_upper=total,
        remaining_conservative_s=remaining,
        train_targets_per_second=512 / sec,
        projected_one_arm_s=projected,
        required_rate_one_arm=1048576 / remaining,
        cost_gate_passed=projected <= remaining,
        qualifier="one measured training batch, not a steady-state timing or a quality estimate",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    print(json.dumps(audit(parser.parse_args().checkpoint), indent=2, allow_nan=False))
