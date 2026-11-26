"""Single-use detached owner + independent durable observer. No retry or relaunch."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "output/retrieved-span-effects/triton-window-1"
PY = ROOT / ".venv/bin/python"
MOD = "experiments.expressivity_tasks.retrieved_span_effects.window"
JOBS = ("meter", "natural-meter", "panel")


def need(cond, why):
    if not cond:
        raise RuntimeError(why)


def get(p):
    return json.loads(p.read_text())


def put(p, obj):
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False) + "\n")
    tmp.replace(p)


def fields(p):
    return dict(s.split("=", 1) for s in p.read_text().splitlines())


def proc(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    except FileNotFoundError:
        return []


def descendant(pid, owner):
    while pid > 1:
        st = proc(pid)
        if not st or st[0] == "Z":
            return False
        if pid == owner:
            return True
        pid = int(st[1])
    return False


def authority(job):
    w = get(BASE / "window.json")
    now = int(time.time())
    need(
        os.uname().nodename == "lambda01"
        and job in JOBS
        and os.getenv("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD"),
        "local CPU admission only",
    )
    need(
        w["deadline_epoch"] == 1795712280
        and w["max_cards_at_once"] == 1
        and w["aggregate_card_seconds_upper"] == 43200
        and w["visible_allowlist"] == "0,1"
        and w["remote"] is w["auto_retry"] is False
        and now < w["deadline_epoch"] - 20,
        "invalid or expired window",
    )
    need(shutil.disk_usage(BASE).free >= 20 * 1024**3, "need 20 GiB free")
    spent = 0
    for prior in JOBS[: JOBS.index(job)]:
        d = BASE / prior
        c = get(d / "charge.json")
        need(
            c["rc"] == 0
            and c["release_verified"]
            and c["renewed_progress"]
            and c["upper_card_s"] > 0,
            "prior lease did not close cleanly",
        )
        spent += c["upper_card_s"]
        if prior == "meter":
            need(
                get(d / "complete.json")["pass_clock"], "no complete-step acceleration"
            )
    if job != "meter":
        from .natural_panel import preflight

        preflight(job)
    src = HERE.relative_to(ROOT)
    ref = (BASE / (job + "-source-ref")).read_text().strip()
    need(
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        == ref
        and not subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(src), "scripts/gpu_lease.sh"],
            cwd=ROOT,
        ),
        "unbound source",
    )
    cap = min(
        43200 - spent, w["deadline_epoch"] - now, 900 if job != "panel" else 43200
    )
    need(cap > 20, "exhausted capacity")
    return cap, w["deadline_epoch"]


def spawn(job):
    cap, end = authority(job)
    d = BASE / job
    d.mkdir()
    put(d / "admission.json", dict(cap=cap, deadline=end))
    os.umask(0o077)
    with (d / "console").open("xb") as f:
        p = subprocess.Popen(
            ["bash", str(HERE / "window_leased.sh"), job],
            cwd=ROOT,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    put(d / "owner.json", dict(pid=p.pid, start=proc(p.pid)[19]))
    with (d / "watch.log").open("xb") as f:
        watcher = subprocess.Popen(
            [PY, "-m", MOD, "watch", job],
            cwd=ROOT,
            start_new_session=True,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    print(f"detached {job}: owner={p.pid} observer={watcher.pid}", flush=True)


def prelease(job, pid, start):
    cap, end = authority(job)
    d = BASE / job
    need(
        get(d / "owner.json") == dict(pid=int(pid), start=start)
        and proc(pid)[1] == "1"
        and proc(pid)[19] == start,
        "wrong owner",
    )
    print(min(cap, get(d / "admission.json")["cap"]), end)


def watch(job):
    d = BASE / job
    owner = get(d / "owner.json")
    pid = owner["pid"]
    last = -1
    proven = False
    while not (d / "released").exists():
        st = proc(pid)
        if not st or st[0] == "Z" or st[19] != owner["start"]:
            if (d / "released").exists():
                break
            raise RuntimeError("owner exited without positive release; inspect console")
        if (d / "progress.json").is_file() and (d / "initial").is_file():
            ini = fields(d / "initial")
            file = ROOT / ".wg/gpu_leases" / ini["gpu"]
            prog = get(d / "progress.json")
            if file.exists() and prog["units"] > last:
                now = fields(file)
                need(
                    st[1] == "1"
                    and (now["pid"], now["starttime"], now["adopted"])
                    == (str(pid), owner["start"], "1")
                    and now["gpu"] == ini["gpu"],
                    "foreign owner",
                )
                if descendant(prog["pid"], pid):
                    proof = dict(
                        owner=owner,
                        ppid=1,
                        gpu=ini["gpu"],
                        heartbeat=int(now["heartbeat"]),
                        worker_start=proc(prog["pid"])[19],
                        progress=prog,
                    )
                    with (d / "survival.jsonl").open("a") as f:
                        f.write(json.dumps(proof) + "\n")
                    proven |= last >= 0 and int(now["heartbeat"]) > int(
                        ini["heartbeat"]
                    )
                    last = prog["units"]
                    if proven and not (d / "live.json").exists():
                        put(d / "live.json", proof)
                        print("RENEWED AND ADVANCING", proof, flush=True)
            # All model hashes must exist before any natural PROBE pass begins.
            if (
                proven
                and job == "panel"
                and (d / "frozen.json").exists()
                and not (d / "freeze-ack.json").exists()
            ):
                from .natural_panel import witness_freeze

                witness_freeze(d)
            if (
                proven
                and (d / "complete.json").exists()
                and last == get(d / "complete.json")["units"]
                and not (d / "watch-ack").exists()
            ):
                (d / "watch-ack").write_text(str(last) + "\n")
        time.sleep(0.12)
    end = fields(d / "released")
    ini = fields(d / "initial")
    charged = int(end["released_epoch"]) - int(ini["created"]) + 1
    term = fields(d / "last")
    need(
        (ini["pid"], ini["starttime"], ini["gpu"])
        == (term["pid"], term["starttime"], term["gpu"]),
        "release identity mismatch",
    )
    gp = ROOT / ".wg/gpu_leases" / ini["gpu"]
    released = not gp.is_file() or fields(gp)["pid"] != str(pid)
    a = get(d / "admission.json")
    rc = int(end["rc"])
    put(
        d / "charge.json",
        dict(
            upper_card_s=charged,
            rc=rc,
            release_verified=released,
            renewed_progress=proven,
        ),
    )
    need(
        0 < charged <= a["cap"]
        and int(end["released_epoch"]) <= a["deadline"]
        and rc == 0
        and released
        and proven
        and (d / "watch-ack").exists(),
        "incomplete or failed job; partial retained",
    )
    print("RELEASE VERIFIED", charged, "card-seconds upper", flush=True)


def monitor(job):
    # Managed notifier may disappear; neither the detached owner nor its witness depends on it.
    d = BASE / job
    offset = 0
    while True:
        for name in ["watch-failure", "charge.json"]:
            if (d / name).exists():
                print(name, (d / name).read_text(), flush=True)
                need(
                    name == "charge.json"
                    and get(d / name)["rc"] == 0
                    and get(d / name)["release_verified"]
                    and get(d / name)["renewed_progress"],
                    "inspect failed closure",
                )
                return
        if (d / "watch.log").exists():
            with (d / "watch.log").open() as f:
                f.seek(offset)
                text = f.read()
                offset = f.tell()
            if text:
                print(text, end="", flush=True)
        time.sleep(0.5)


if __name__ == "__main__":
    action, job, *rest = sys.argv[1:]
    need(job in JOBS, "unknown job")
    if action == "watch":
        try:
            watch(job)
        except BaseException as e:
            (BASE / job / "watch-failure").write_text(repr(e))
            raise
    else:
        {"spawn": spawn, "prelease": prelease, "monitor": monitor}[action](job, *rest)
