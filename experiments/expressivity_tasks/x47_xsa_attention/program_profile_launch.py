"""Detached program-profile broker owner and independent receipt watcher (watch never acquires a card)."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from .w3_launch import fields, procstat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["launch", "watch"])
    parser.add_argument("attempt", type=int)
    kinds = parser.add_mutually_exclusive_group()
    profiles = {
        "packed": ("p2", None),
        "serving": ("s1", 17280),
        "cohort": ("s2", 13056),
        "cache": ("s3", 13056),
        "large": ("s4", 8704),
        "graph": ("s5", 8704),
        "resident": ("s6", 2968),
        "wide": ("s7", 8704),
        "placement": ("s8", 6528),
        "vjp": ("t1", 9216),
        "long-prefill": ("s9", 1632),
        "segments": ("s10", 1632),
        "width-cover": ("s11", 5280),
        "arrivals": ("s12", 2160),
        "sparse-clock": ("t2", 9216),
        "open-cohort": ("s13", 1392),
        "frozen-packets": ("t3", 8192),
        "stream-packets": ("t4", 8192),
        "segment-packets": ("t6", 32768),
        "direct-arrivals": ("s14", 3552),
        "demand-tape": ("s17", 3552),
        "onclock-mount": ("s19", 2784),
        "demand-position": ("s20", 2784),
    }
    for name in profiles:
        kinds.add_argument("--" + name, action="store_const", const=name, dest="profile")
    args = parser.parse_args()
    kind, expected = profiles.get(args.profile, ("p1", None))
    folder = Path(f"output/memory-program-{kind}") / f"attempt{args.attempt}"
    if args.action == "launch":
        folder.mkdir(exist_ok=False)
        with (folder / "console").open("xb") as stream:
            p = subprocess.Popen(
                ["bash", str(Path(__file__).with_name("program_profile_leased.sh")), str(args.attempt), kind],
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        info = dict(pid=p.pid, start=procstat(p.pid)[19])
        (folder / "launch.json").write_text(json.dumps(info))
        print(info)
        return
    info = json.loads((folder / "launch.json").read_text())
    pid = info["pid"]
    training = kind in ("t1", "t2", "t3", "t4")
    key, unit = (
        ("total_targets", "gradient targets")
        if training
        else (
            ("total_frontiers", "published frontiers") if kind == "t6" else ("total_emitted", "emitted bytes")
        )
    )
    prev = None
    while not (folder / "released").exists():
        st = procstat(pid)
        if not st or st[19] != info["start"] or st[0] == "Z":
            if (folder / "released").exists():
                break
            raise RuntimeError(
                "owner died without release (or early refusal); inspect console, do not restart"
            )
        if (folder / "progress.json").exists() and (folder / "initial").exists():
            initial = fields(folder / "initial")
            f = Path(".wg/gpu_leases") / initial["gpu"]
            prog = json.loads((folder / "progress.json").read_text())
            if f.exists() and prev != prog:
                now = fields(f)
                if (now["pid"], now["starttime"], st[1]) != (str(pid), info["start"], "1"):
                    raise ValueError("changed live owner")
                obs = dict(
                    owner=pid,
                    starttime=st[19],
                    ppid=int(st[1]),
                    heartbeat=int(now["heartbeat"]),
                    progress=prog,
                )
                with (folder / "survival.jsonl").open("a") as out:
                    out.write(json.dumps(obs) + "\n")
                if (
                    kind == "p2"
                    and prog.get("cell") == "packed-b32"
                    and prog["phase"] == "updated"
                    and prog["targets_scored"] == 4096
                    and int(now["heartbeat"]) > int(initial["heartbeat"])
                ):
                    (folder / "watch-ack").write_text("updated B32 under renewed live lease\n")
                if (
                    expected is not None
                    and prog["phase"] == "complete"
                    and prog[key] == expected
                    and int(now["heartbeat"]) > int(initial["heartbeat"])
                ):
                    (folder / "watch-ack").write_text(
                        f"observed all {expected} {unit} under renewed live lease\n"
                    )
                print(obs, flush=True)
                prev = prog
        time.sleep(0.25)
    released = fields(folder / "released")
    print(released, flush=True)
    raise SystemExit(int(released["rc"]))


if __name__ == "__main__":
    main()
