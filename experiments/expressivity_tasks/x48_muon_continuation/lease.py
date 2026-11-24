"""Separate X48 authorization, detached driver and independent broker/progress witness."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from ..x47_xsa_attention.w3_launch import fields, procstat
from .data import OUT, ROOT, check, load, sha, write_json
from .qualification import proof

MODULE = "experiments.expressivity_tasks.x48_muon_continuation.lease"
PROTOCOL = "0f0f257cbc3c33f6cb1e4fd8dc1d3d1cd69055d9"
SECS = 10800


def authorize(root):
    p = root / "operator-authorization.json"
    check(
        p.is_file(), "X48 launch blocked: missing explicit separate card-hour and post-window authorization"
    )
    a = load(p)
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    check(
        a.get("protocol") == PROTOCOL
        and a.get("gated_ref") == ref
        and a.get("train_only") is True
        and a.get("after_1450") is True
        and a.get("arm_card_s") == SECS
        and a.get("total_card_s") == 2 * SECS
        and isinstance(a.get("operator_note"), str)
        and a["operator_note"].strip(),
        "different or incomplete operator scope",
    )
    check(
        (root / "protocol-ref").read_text().strip() == PROTOCOL
        and (root / "gated-ref").read_text().strip() == ref == (root / "source-ref").read_text().strip(),
        "unqualified source",
    )
    q = load(root / "qualification.json")
    lines = (root / "qualification.txt").read_text().splitlines()
    check(
        q["ref"] == ref and q["sha256"] == sha(root / "qualification.txt") and q["passed"] == proof(lines),
        "failed CPU qualification",
    )
    subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL, ref], cwd=ROOT, check=True)
    subprocess.run(
        [
            "git",
            "diff",
            "--quiet",
            "HEAD",
            "--",
            "ndm",
            "experiments/expressivity_tasks",
            "scripts/gpu_lease.sh",
        ],
        cwd=ROOT,
        check=True,
    )
    check(platform.node() == "lambda01", "local experiment only")
    return a


def launch(cmd, log, ident):
    with log.open("xb") as f:
        p = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=ROOT,
        )
    info = dict(pid=p.pid, start=procstat(p.pid)[19])
    write_json(ident, info)
    return info


def descends(pid, owner):
    while pid > 1:
        st = procstat(pid)
        if not st or st[0] == "Z":
            return False
        if pid == owner:
            return True
        pid = int(st[1])
    return False


def watch(folder):
    info = load(folder / "launch.json")
    owner = info["pid"]
    last = -1
    live_verified = False
    while not (folder / "released").exists():
        st = procstat(owner)
        if not st or st[19] != info["start"] or st[0] == "Z":
            check(
                (folder / "released").exists(),
                "owner exited before release (or refused before acquire); inspect console, no retry",
            )
            break
        if (folder / "initial").exists() and (folder / "progress.json").exists():
            init = fields(folder / "initial")
            gp = ROOT / ".wg/gpu_leases" / init["gpu"]
            prog = load(folder / "progress.json")
            if gp.exists() and prog["physical_updates"] > last:
                now = fields(gp)
                check(
                    (now["pid"], now["starttime"], st[1], now["adopted"], now["gpu"])
                    == (str(owner), info["start"], "1", "1", init["gpu"]),
                    "foreign owner",
                )
                if descends(prog["pid"], owner):
                    advanced = last >= 0 and prog["physical_updates"] > last
                    last = prog["physical_updates"]
                    live_verified |= advanced and int(now["heartbeat"]) > int(init["heartbeat"])
                    obs = dict(
                        pid=owner,
                        start=st[19],
                        ppid=1,
                        heartbeat=now["heartbeat"],
                        progress=prog,
                        trainer_start=procstat(prog["pid"])[19],
                    )
                    with (folder / "survival.jsonl").open("a") as f:
                        f.write(json.dumps(obs) + "\n")
                    if live_verified and not (folder / "live.json").exists():
                        write_json(folder / "live.json", obs)
                        print("renewed detached owner and advancing actual updates:", obs, flush=True)
        if (
            live_verified
            and last == 1152
            and (folder / "complete.json").exists()
            and not (folder / "watch-ack").exists()
        ):
            (folder / "watch-ack").write_text(
                "1152 actual optimizer updates witnessed under renewed detached lease\n"
            )
        time.sleep(0.5)
    end = fields(folder / "released")
    initial = fields(folder / "initial")
    charged = int(end["released_epoch"]) - int(initial["created"]) + 1
    write_json(
        folder / "charge.json", dict(upper_card_s=charged, rc=int(end["rc"]), gpu=initial["gpu"], pid=owner)
    )
    check(
        0 < charged <= SECS and int(end["rc"]) == 0 and (folder / "watch-ack").is_file(),
        "arm partial/over cap; no next lease",
    )
    return charged


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("check", "launch", "supervise", "one"))
    p.add_argument("--arm", choices=("direct", "program"))
    args = p.parse_args()
    os.umask(0o077)
    os.chdir(ROOT)
    authorize(OUT)
    if args.command == "check":
        return
    check(not os.environ.get("GPU_LEASE_HELD"), "nested lease")
    py = [str(ROOT / ".venv/bin/python"), "-m", MODULE]
    if args.command == "launch":
        (OUT / "panel").mkdir()
        print(launch(py + ["supervise"], OUT / "panel/console", OUT / "panel/launch.json"))
        return
    if args.command == "one":
        arm = args.arm
        check(
            arm is not None and (arm == "direct" or load(OUT / "direct/charge.json")["rc"] == 0),
            "missing first successful arm",
        )
        folder = OUT / arm
        folder.mkdir()
        launch(
            ["bash", str(Path(__file__).with_name("leased.sh")), arm],
            folder / "console",
            folder / "launch.json",
        )
        return
    # The supervisor is NOT the broker owner. Each short-lived 'one' parent exits
    # so the per-arm detached shell (and its lease keeper) have PPid 1.
    charged = {}
    for arm in ("direct", "program"):
        subprocess.run(py + ["one", "--arm", arm], check=True)
        charged[arm] = watch(OUT / arm)
    check(sum(charged.values()) <= 2 * SECS, "panel budget exceeded")
    write_json(OUT / "panel/charges.json", charged)
    subprocess.run(
        [sys.executable, "-m", "experiments.expressivity_tasks.x48_muon_continuation.compare"],
        check=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
    )


if __name__ == "__main__":
    main()
