"""Single-use invented meter. No scheduling on pending capacity, retry, or panel modification."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ..window import BASE, ROOT, PY, fields, get, monitor, need, proc, put, watch


HERE = Path(__file__).resolve().parent
OUT = ROOT / "output/retrieved-span-effects/corpus-kernel-1"
JOB = "corpus-kernel-meter"
D = BASE / JOB
MOD = __package__ + ".device_gate"


def capacity(scope, now, charges):
    need(
        scope["deadline_epoch"] == 1795712280
        and scope["aggregate_card_seconds_upper"] == 43200
        and scope["max_cards_at_once"] == 1
        and scope["visible_allowlist"] == "0,1"
        and scope["remote"] is scope["auto_retry"] is False,
        "unknown window",
    )
    need(
        set(charges) == {"meter", "natural-meter", "panel"}
        and all(type(x) is int and x > 0 for x in charges.values()),
        "missing positive terminal charge",
    )
    cap = min(
        180,
        scope["deadline_epoch"] - now,
        43200 - sum(charges.values()),
        900 - charges["meter"] - charges["natural-meter"],
    )
    need(cap >= 120, "not enough time for a device engineering pass")
    return cap


def authority():
    need(
        __debug__
        and os.uname().nodename == "lambda01"
        and os.getenv("CUDA_VISIBLE_DEVICES") == ""
        and not os.getenv("GPU_LEASE_HELD")
        and os.getenv("TRITON_INTERPRET") != "1"
        and all(
            os.getenv(k) == "1"
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        ),
        "local CPU preflight required",
    )
    # No transfer from a failed or still-running panel. Refuse BEFORE creating a run directory.
    need(
        (BASE / "campaign/exit.json").is_file()
        and get(BASE / "campaign/exit.json")["rc"] == 0,
        "previous panel has not closed successfully",
    )
    charges = {}
    scope = get(BASE / "window.json")
    for j in ("meter", "natural-meter", "panel"):
        p = BASE / j
        c = get(p / "charge.json")
        end = fields(p / "released")
        a = get(p / "admission.json")
        ini = fields(p / "initial")
        term = fields(p / "last")
        need(
            c["rc"] == int(end["rc"]) == 0
            and c["release_verified"]
            and c["renewed_progress"]
            and ini["gpu"] in ("0", "1")
            and all(ini[k] == term[k] for k in ("pid", "gpu", "starttime"))
            and (p / "watch-ack").is_file()
            and int(end["released_epoch"]) <= a["deadline"] == scope["deadline_epoch"]
            and 0
            < c["upper_card_s"]
            == int(end["released_epoch"]) - int(ini["created"]) + 1
            <= a["cap"],
            "prior release/charge invalid",
        )
        charges[j] = c["upper_card_s"]
    cap = capacity(scope, int(time.time()), charges)
    need(shutil.disk_usage(BASE).free >= 20 * 1024**3, "need 20 GiB free")
    pub = (OUT / "triton-public-ref").read_text().strip()
    src = HERE.relative_to(ROOT)
    q = get(HERE / "receipts/triton_qualified.json")
    need(
        q["cpu_before_triton"]
        and q["offline_cubins"] == 20
        and q["triton_device_launches"] == 0
        and q["interpreter_cases"] == 8
        and q["cpu_public"] == (OUT / "cpu-public-ref").read_text().strip(),
        "missing qualification",
    )
    import hashlib

    for name, key in [
        ("qualified-internal-all.log", "tests_sha256"),
        ("sm89-qualified.json", "codegen_sha256"),
    ]:
        need(
            hashlib.sha256((OUT / name).read_bytes()).hexdigest() == q[key],
            "qualification changed",
        )
    paths = [
        str(src),
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
        "unbound local source",
    )
    return dict(
        cap=cap,
        deadline=scope["deadline_epoch"],
        prior_charges=charges,
        qualified_public=pub,
    )


def spawn():
    a = authority()
    os.umask(0o077)
    D.mkdir()
    put(D / "admission.json", a)
    with (D / "console").open("xb") as f:
        owner = subprocess.Popen(
            ["bash", str(HERE / "device_leased.sh")],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    put(D / "owner.json", dict(pid=owner.pid, start=proc(owner.pid)[19]))
    with (D / "watch.log").open("xb") as f:
        obs = subprocess.Popen(
            [PY, "-m", MOD, "watch"],
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"detached corpus meter owner={owner.pid} observer={obs.pid}", flush=True)


def prelease(pid, start):
    a = authority()
    need(
        get(D / "owner.json") == dict(pid=int(pid), start=start)
        and proc(pid)[1] == "1"
        and proc(pid)[19] == start,
        "wrong detached owner",
    )
    print(min(a["cap"], get(D / "admission.json")["cap"]), a["deadline"])


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    if action == "watch":
        try:
            watch(JOB)
        except BaseException as e:
            (D / "watch-failure").write_text(repr(e))
            raise
    elif action == "monitor":
        monitor(JOB)
    elif action == "prelease":
        prelease(*args)
    elif action == "spawn":
        spawn()
    elif action == "check":
        print(authority())
    else:
        raise ValueError("unknown action")
