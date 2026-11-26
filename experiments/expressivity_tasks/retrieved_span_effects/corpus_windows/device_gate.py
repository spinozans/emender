"""Optional sequential synthetic meter; never enqueue, preempt, retry, or read language data."""

import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..window import BASE, PY, ROOT, fields, get, monitor, need, proc, put, watch
from ..corpus_kernel.device_gate import capacity as previous_capacity

HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/retrieved-span-effects/corpus-windows-1"
PRIOR = ("meter", "natural-meter", "panel", "corpus-kernel-meter")
JOB = "corpus-windows-meter"
D = BASE / JOB
MOD = __package__ + ".device_gate"


def capacity(scope, now, charges):
    need(
        set(charges) == set(PRIOR)
        and all(type(c) is int and c > 0 for c in charges.values()),
        "four positive closed charges required",
    )
    previous = previous_capacity(scope, now, {j: charges[j] for j in PRIOR[:3]})
    cap = min(
        150,
        previous,
        43200 - sum(charges.values()),
        900 - sum(charges[j] for j in PRIOR if j != "panel"),
    )
    need(cap >= 120, "not enough common time or card seconds")
    return cap


def charge(folder, end):
    need(
        (folder / "charge.json").is_file()
        and (folder / "complete.json").is_file()
        and (folder / "watch-ack").is_file(),
        "prior pass has not closed",
    )
    c = get(folder / "charge.json")
    term = fields(folder / "released")
    ad = get(folder / "admission.json")
    ini = fields(folder / "initial")
    last = fields(folder / "last")
    s = int(term["released_epoch"]) - int(ini["created"]) + 1
    need(
        c["rc"] == int(term["rc"]) == 0
        and c["release_verified"]
        and c["renewed_progress"]
        and all(ini[k] == last[k] for k in ("gpu", "pid", "starttime"))
        and ini["gpu"] in ("0", "1")
        and int(last["heartbeat"]) > int(ini["heartbeat"])
        and 0 < c["upper_card_s"] == s <= ad["cap"]
        and int(term["released_epoch"]) <= ad["deadline"] == end,
        "uncertified prior release",
    )
    return s


def authority():
    need(
        __debug__
        and os.uname().nodename == "lambda01"
        and os.getenv("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD")
        and os.getenv("TRITON_INTERPRET") != "1"
        and all(
            os.getenv(k) == "1"
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        ),
        "CPU admission required",
    )
    need(
        (BASE / "campaign/exit.json").is_file()
        and get(BASE / "campaign/exit.json")["rc"] == 0,
        "prior panel not closed successfully",
    )
    w = get(BASE / "window.json")
    end = w["deadline_epoch"]
    cs = {j: charge(BASE / j, end) for j in PRIOR}
    cap = capacity(w, int(time.time()), cs)
    need(shutil.disk_usage(BASE).free >= 20 * 1024**3, "disk admission failed")
    q = get(HERE / "receipts/qualification.json")
    need(
        q["cpu_public"] == (OUT / "cpu-public-ref").read_text().strip()
        and q["cpu_before_port"] == 8
        and q["device_tests"] == 12
        and q["offline_cubins"] == 24
        and q["device_launches"] == 0,
        "no CPU/artifact proof",
    )
    for file, key in [
        ("cpu-first.log", "initial_cpu_sha256"),
        ("window-tests-final.log", "tests_sha256"),
        ("window-cubins-final.json", "codegen_sha256"),
        ("window-clock-final.json", "clock_sha256"),
    ]:
        need(
            hashlib.sha256((OUT / file).read_bytes()).hexdigest() == q[key],
            "qualification changed",
        )
    pub = (OUT / "triton-public-ref").read_text().strip()
    src = HERE.relative_to(ROOT)
    paths = [
        str(src),
        f":(glob){src.parent}/corpus_kernel/*.py",
        str(src.parent / "window.py"),
        str(src.parent / "meter.py"),
        "scripts/gpu_lease.sh",
    ]
    need(
        not subprocess.run(
            ["git", "diff", "--quiet", pub, "--", *paths], cwd=ROOT
        ).returncode
        and not subprocess.check_output(
            ["git", "status", "--porcelain", "--", *paths], cwd=ROOT
        ),
        "unbound meter source",
    )
    return dict(cap=cap, deadline=end, prior_charges=cs, qualified_public=pub)


def spawn():
    a = authority()
    os.umask(0o077)
    D.mkdir()
    put(D / "admission.json", a)

    def detach(cmd, name):
        with (D / name).open("xb") as f:
            return subprocess.Popen(
                cmd,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    owner = detach(["bash", str(HERE / "device_leased.sh")], "console")
    put(D / "owner.json", dict(pid=owner.pid, start=proc(owner.pid)[19]))
    obs = detach([PY, "-m", MOD, "watch"], "watch.log")
    print("detached window meter", owner.pid, "observer", obs.pid, flush=True)


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "check":
        print(authority())
    elif action == "spawn":
        spawn()
    elif action == "prelease":
        pid, start = args
        a = authority()
        st = proc(pid)
        need(
            st
            and st[1] == "1"
            and st[19] == start
            and get(D / "owner.json") == dict(pid=int(pid), start=start),
            "foreign lease owner",
        )
        print(min(a["cap"], get(D / "admission.json")["cap"]), a["deadline"])
    elif action == "watch":
        try:
            watch(JOB)
        except BaseException as exc:
            (D / "watch-failure").write_text(repr(exc))
            raise
    elif action == "monitor":
        monitor(JOB)
    else:
        raise ValueError("unknown action")
