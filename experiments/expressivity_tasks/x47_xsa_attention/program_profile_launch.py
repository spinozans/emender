"""P1 detached broker owner and independent receipt watcher (watch never acquires a card)."""

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
    args = parser.parse_args()
    folder = Path("output/memory-program-p1") / f"attempt{args.attempt}"
    if args.action == "launch":
        folder.mkdir(exist_ok=False)
        with (folder / "console").open("xb") as stream:
            p = subprocess.Popen(
                ["bash", str(Path(__file__).with_name("program_profile_leased.sh")), str(args.attempt)],
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
                print(obs, flush=True)
                prev = prog
        time.sleep(0.25)
    released = fields(folder / "released")
    print(released, flush=True)
    raise SystemExit(int(released["rc"]))


if __name__ == "__main__":
    main()
