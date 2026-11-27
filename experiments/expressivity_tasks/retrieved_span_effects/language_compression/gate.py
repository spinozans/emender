"""No-wait detached single-use owner, independent progress/freeze witness and closed charge."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from ..window import ROOT, PY, fields, proc, descendant
from .settings import (
    HERE,
    BASE,
    DATA,
    JOB,
    CAP,
    END,
    REG,
    ARMS,
    SEEDS,
    sha,
    qualified,
    cpu,
    need,
    put,
    get,
)

MOD = __package__ + ".gate"


def order(s):
    return np.random.default_rng(s).permutation(1024).astype("<i4")


def order_sha(s):
    return hashlib.sha256(order(s).tobytes()).hexdigest()


def capacity(now, free):
    need(
        now < END - CAP - 10 and free >= 20 * 1024**3,
        "insufficient wall window or disk; no lease",
    )
    return CAP, END


def admission():
    cpu()
    ref = qualified()
    cap, end = capacity(int(time.time()), shutil.disk_usage(BASE).free)
    r = get(DATA / "ready.json")
    need(
        not (DATA / "probe-opened").exists()
        and r
        == dict(
            rows=sha(DATA / "rows.json"),
            fit=sha(DATA / "fit.json"),
            build=sha(DATA / "build.json"),
        ),
        "missing seals or previously opened PROBE",
    )
    return dict(
        registered=REG,
        qualified=ref,
        cap=cap,
        deadline=end,
        rows_sha=r["rows"],
        build_seal=r["build"],
        fit_seal=r["fit"],
        allowlist="0,1",
        automatic_retry=False,
    )


def spawn():
    a = admission()
    JOB.mkdir(mode=0o700)
    put(JOB / "admission.json", a)

    def start(args, name):
        with (JOB / name).open("xb") as f:
            return subprocess.Popen(
                args,
                cwd=ROOT,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    p = start(["bash", str(HERE / "leased.sh")], "console")
    put(JOB / "owner.json", dict(pid=p.pid, start=proc(p.pid)[19]))
    obs = start([PY, "-m", MOD, "watch"], "watch.log")
    print("detached owner", p.pid, "observer", obs.pid, flush=True)


def witness_freeze():
    f = get(JOB / "frozen.json")
    a = get(JOB / "admission.json")
    need(
        not (DATA / "probe-opened").exists()
        and f["rows_sha"] == a["rows_sha"] == sha(DATA / "rows.json")
        and f["qualified"] == a["qualified"]
        and f["fit_sha"] == sha(DATA / "fit.npy")
        and f["build_sha"] == sha(DATA / "build.npy")
        and set(f["fits"]) == {f"{s}.{arm}" for s in SEEDS for arm in ARMS},
        "invalid freeze",
    )
    for s in SEEDS:
        init = set()
        for arm in ARMS:
            k = f"{s}.{arm}"
            p = JOB / (k + ".json")
            z = get(p)
            need(
                sha(p) == f["fits"][k]
                and sha(JOB / (k + ".pt")) == z["weights_sha"]
                and (z["seed"], z["arm"], z["updates"], z["targets"], z["order_sha"])
                == (s, arm, 1024, 1048576, order_sha(s)),
                "missing/mismatched trained weights",
            )
            init.add(z["initial"])
        need(len(init) == 1, "unpaired starts")
    put(
        JOB / "freeze-ack.json",
        dict(
            frozen_sha=sha(JOB / "frozen.json"),
            fits=8,
            observer_pid=os.getpid(),
            rows_sha=a["rows_sha"],
        ),
    )


def observe():
    owner = get(JOB / "owner.json")
    pid = owner["pid"]
    last = -1
    proven = False
    while not (JOB / "released").exists():
        st = proc(pid)
        if not st or st[0] == "Z" or st[19] != owner["start"]:
            if (JOB / "released").exists():
                break
            raise RuntimeError(
                "detached owner exited before lease release or acquisition; inspect console"
            )
        if (JOB / "progress.json").exists() and (JOB / "initial").exists():
            ini = fields(JOB / "initial")
            p = get(JOB / "progress.json")
            file = ROOT / ".wg/gpu_leases" / ini["gpu"]
            if file.is_file() and p["units"] > last:
                lock = fields(file)
                need(
                    st[1] == "1"
                    and (lock["gpu"], lock["pid"], lock["starttime"], lock["adopted"])
                    == (ini["gpu"], str(pid), owner["start"], "1")
                    and descendant(p["pid"], pid),
                    "foreign lease or worker",
                )
                proof = dict(
                    owner=owner,
                    ppid=1,
                    heartbeat=int(lock["heartbeat"]),
                    worker_start=proc(p["pid"])[19],
                    progress=p,
                )
                with (JOB / "survival.jsonl").open("a") as f:
                    f.write(json.dumps(proof) + "\n")
                proven |= last >= 0 and int(lock["heartbeat"]) > int(ini["heartbeat"])
                last = p["units"]
                if proven and not (JOB / "live.json").exists():
                    put(JOB / "live.json", proof)
                    print("RENEWED AND ADVANCING", last, flush=True)
            if (
                proven
                and (JOB / "frozen.json").exists()
                and not (JOB / "freeze-ack.json").exists()
            ):
                witness_freeze()
                print("ALL FITS WITNESSED", flush=True)
            if (
                proven
                and (JOB / "complete.json").exists()
                and last == get(JOB / "complete.json")["units"]
                and not (JOB / "watch-ack").exists()
            ):
                (JOB / "watch-ack").write_text(str(last) + "\n")
        time.sleep(0.14)
    a = get(JOB / "admission.json")
    term = fields(JOB / "released")
    ini = fields(JOB / "initial")
    fin = fields(JOB / "last")
    need(
        all(ini[k] == fin[k] for k in ("pid", "starttime", "gpu"))
        and ini["gpu"] in ("0", "1")
        and (ini["pid"], ini["starttime"]) == (str(pid), owner["start"]),
        "release identity mismatch",
    )
    s = int(term["released_epoch"]) - int(ini["created"]) + 1
    lock = ROOT / ".wg/gpu_leases" / ini["gpu"]
    released = not lock.exists() or fields(lock)["pid"] != str(pid)
    ok = (
        0 < s <= a["cap"] == CAP and int(term["released_epoch"]) <= a["deadline"] == END
    )
    put(
        JOB / "charge.json",
        dict(
            upper_card_s=s,
            rc=int(term["rc"]),
            release_verified=released,
            renewed_progress=proven,
            within_window=ok,
        ),
    )
    need(
        ok
        and released
        and proven
        and int(term["rc"]) == 0
        and (JOB / "watch-ack").exists(),
        "failed/partial run retained",
    )
    print("RELEASE VERIFIED", s, flush=True)


def monitor():
    while True:
        if (JOB / "watch-failure").exists():
            raise RuntimeError((JOB / "watch-failure").read_text())
        if (JOB / "charge.json").exists():
            c = get(JOB / "charge.json")
            print(c, flush=True)
            need(
                c["rc"] == 0
                and all(
                    c[k]
                    for k in ("release_verified", "renewed_progress", "within_window")
                )
                and (JOB / "watch-ack").exists(),
                "unsuccessful closure",
            )
            return
        time.sleep(0.5)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    os.umask(0o077)
    if action == "check":
        print(admission())
    elif action == "spawn":
        spawn()
    elif action == "prelease":
        pid, start = args
        a = admission()
        need(
            a == get(JOB / "admission.json")
            and get(JOB / "owner.json") == dict(pid=int(pid), start=start)
            and proc(pid)[1] == "1"
            and proc(pid)[19] == start,
            "not the detached admitted owner",
        )
        print(CAP, END)
    elif action == "watch":
        try:
            observe()
        except BaseException as exc:
            (JOB / "watch-failure").write_text(repr(exc))
            raise
    elif action == "monitor":
        monitor()
    else:
        raise ValueError("unknown action")
