"""SAL-1 campaign gate: staged no-wait leases, witnessed progress and freeze,
closed charges against a campaign ledger. No automatic retry; a failed stage
is a result, not permission to resubmit."""

import json
import os
import shutil
import subprocess
import sys
import time

from ..window import ROOT, PY, fields, proc, descendant, get, need, put
from .settings import (
    ARMS,
    BASE,
    CAMPAIGN_CAP,
    CAMPAIGN_END,
    DATA,
    HERE,
    JOB,
    N,
    SEEDS,
    sha,
    cpu,
    qualified,
)

MOD = __package__ + ".gate"

# The ready.json seal set a campaign's roles must carry before any stage.
READY_KEYS = ("rows", "build", "fit")


def order_sha(s):
    from . import run

    return run.order_sha(s)


def campaign_spent():
    ledger = BASE / "campaign.jsonl"
    if not ledger.exists():
        return 0
    return sum(
        json.loads(line)["upper_card_s"]
        for line in ledger.read_text().splitlines()
        if line.strip()
    )


def admission(stage, cap):
    cpu()
    ref = qualified()
    need(isinstance(cap, int) and 0 < cap <= CAMPAIGN_CAP, "invalid stage cap")
    need(not (JOB / f"charge-{stage}.json").exists(), "stage already run; no retry")
    need(
        not (JOB / f"initial-{stage}").exists(),
        "stage already in flight or completed without a charge",
    )
    spent = campaign_spent()
    need(spent + cap <= CAMPAIGN_CAP, "stage would exceed the campaign cap")
    need(
        time.time() + cap + 60 < CAMPAIGN_END,
        "insufficient campaign window remaining",
    )
    r = get(DATA / "ready.json")
    if stage.startswith("seal"):
        need(
            (DATA / "probe-opened").exists()
            and (JOB / "frozen.json").exists()
            and (JOB / "freeze-ack.json").exists()
            and (JOB / "document-losses.npz").exists(),
            "seal stage requires the completed scoring artifacts",
        )
    else:
        need(
            not (DATA / "probe-opened").exists()
            and r == {k: sha(DATA / f"{k}.json") for k in READY_KEYS},
            "missing data seals or previously opened PROBE",
        )
    free = shutil.disk_usage(BASE).free
    need(free >= 20 * 1024**3, "insufficient disk")
    return dict(
        qualified=ref,
        stage=stage,
        cap=cap,
        ledger=spent,
        deadline=CAMPAIGN_END,
        rows_sha=r["rows"],
        # SAL-3's archive seal is the container receipt (no in-RAM build)
        build_seal=r["build"] if "build" in r else r["container"],
        fit_seal=r["fit"],
        allowlist="0,1",
        automatic_retry=False,
    )


def spawn(stage, cap, module, arms=None):
    a = admission(stage, cap)
    if arms:
        a["arms"] = list(arms)
    if module not in ("device_check", "run", "seal", "sal2run", "sal3run"):
        raise ValueError("unknown stage module")
    JOB.mkdir(mode=0o700, exist_ok=True)
    put(JOB / "admission.json", a)
    (JOB / "stage").write_text(stage + "\n")
    (JOB / "module").write_text(module + "\n")

    def start(args, name):
        with (JOB / f"{stage}-{name}").open("xb") as f:
            return subprocess.Popen(
                args,
                cwd=ROOT,
                start_new_session=True,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
            )

    campaign = (
        "sal3"
        if module.startswith("sal3")
        else ("sal2" if module.startswith("sal2") else None)
    )
    script = {None: "leased.sh", "sal2": "leased2.sh", "sal3": "leased3.sh"}[campaign]
    entry = {None: "gate", "sal2": "sal2", "sal3": "sal3"}[campaign]
    p = start(["bash", str(HERE / script)], "console")
    put(JOB / "owner.json", dict(pid=p.pid, start=proc(p.pid)[19]))
    obs = start([PY, "-m", __package__ + "." + entry, "watch"], "watch.log")
    print("detached owner", p.pid, "observer", obs.pid, flush=True)


def prelease(stage, pid, start):
    a = get(JOB / "admission.json")
    need(
        a["stage"] == stage
        and get(JOB / "owner.json") == dict(pid=int(pid), start=start)
        and proc(int(pid))[1] == "1"
        and proc(int(pid))[19] == start,
        "not the detached admitted owner",
    )
    return a["cap"], a["deadline"]


def witness_freeze():
    f = get(JOB / "frozen.json")
    a = get(JOB / "admission.json")
    need(
        not (DATA / "probe-opened").exists()
        and f["rows_sha"] == a["rows_sha"] == sha(DATA / "rows.json")
        and f["qualified"] == a["qualified"]
        and f["fit_sha"] == sha(DATA / "fit.npy")
        and f["build_sha"] == sha(DATA / "build.npy")
        and set(f["fits"])
        == {f"{s}.{arm}" for s in SEEDS for arm in a.get("arms", ARMS)},
        "invalid freeze",
    )
    arms = a.get("arms", ARMS)
    for s in SEEDS:
        inits = set()
        for arm in arms:
            k = f"{s}.{arm}"
            p = JOB / (k + ".json")
            z = get(p)
            need(
                sha(p) == f["fits"][k]
                and sha(JOB / (k + ".pt")) == z["weights_sha"]
                and (z["seed"], z["arm"], z["updates"]) == (s, arm, 2048)
                and z["targets"] == N["fit"] * 4096
                and z["order_sha"] == order_sha(s),
                "missing or mismatched trained weights",
            )
            inits.add(z["initial"])
        need(len(inits) == 1, "unpaired starts")
    put(
        JOB / "freeze-ack.json",
        dict(
            frozen_sha=sha(JOB / "frozen.json"),
            fits=len(SEEDS) * len(arms),
            observer_pid=os.getpid(),
            rows_sha=a["rows_sha"],
        ),
    )


def observe():
    owner = get(JOB / "owner.json")
    a = get(JOB / "admission.json")
    stage = a["stage"]
    released = JOB / f"released-{stage}"
    initial = JOB / f"initial-{stage}"
    last_file = JOB / f"last-{stage}"
    live = JOB / f"live-{stage}"
    survival = JOB / f"survival-{stage}.jsonl"
    ack = JOB / f"watch-ack-{stage}"
    owner_log = f"{stage}-console"
    pid = owner["pid"]
    last = -1
    proven = False
    while not released.exists():
        st = proc(pid)
        if not st or st[0] == "Z" or st[19] != owner["start"]:
            if released.exists():
                break
            raise RuntimeError(
                "detached owner exited before lease release; inspect " + owner_log
            )
        progress = JOB / f"progress-{stage}.json"
        if progress.exists() and initial.exists():
            ini = fields(initial)
            p = get(progress)
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
                with survival.open("a") as f:
                    f.write(json.dumps(proof) + "\n")
                proven |= last >= 0 and int(lock["heartbeat"]) > int(ini["heartbeat"])
                last = p["units"]
                if proven and not live.exists():
                    put(live, proof)
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
                and stage.startswith("run")
                and (JOB / "complete.json").exists()
                and last == get(JOB / "complete.json")["units"]
                and not ack.exists()
            ):
                ack.write_text(str(last) + "\n")
        time.sleep(0.14)
    term = fields(released)
    ini = fields(initial)
    fin = fields(last_file)
    need(
        all(ini[k] == fin[k] for k in ("pid", "starttime", "gpu"))
        and ini["gpu"] in ("0", "1")
        and (ini["pid"], ini["starttime"]) == (str(pid), owner["start"]),
        "release identity mismatch",
    )
    s = int(term["released_epoch"]) - int(ini["created"]) + 1
    lock = ROOT / ".wg/gpu_leases" / ini["gpu"]
    released_ok = not lock.exists() or fields(lock)["pid"] != str(pid)
    ok = 0 < s <= a["cap"] and int(term["released_epoch"]) <= a["deadline"]
    put(
        JOB / f"charge-{a['stage']}.json",
        dict(
            stage=a["stage"],
            upper_card_s=s,
            rc=int(term["rc"]),
            release_verified=released_ok,
            renewed_progress=proven,
            within_stage=ok,
        ),
    )
    with (BASE / "campaign.jsonl").open("a") as f:
        f.write(
            json.dumps(
                dict(
                    stage=a["stage"],
                    upper_card_s=s,
                    rc=int(term["rc"]),
                    released_sha=sha(released),
                )
            )
            + "\n"
        )
    stage_ok = (
        (JOB / f"meter-{stage}.json").exists()
        if not stage.startswith("run")
        else ack.exists()
    )
    need(
        ok and released_ok and proven and int(term["rc"]) == 0 and stage_ok,
        "failed or partial stage retained",
    )
    print("RELEASE VERIFIED", a["stage"], s, flush=True)


def monitor(stage):
    charge_file = JOB / f"charge-{stage}.json"
    while not charge_file.exists():
        for fail in JOB.glob("watch-failure-*"):
            raise RuntimeError(fail.read_text())
        time.sleep(0.5)
    charge = get(charge_file)
    print(charge, flush=True)
    need(
        charge["rc"] == 0
        and all(
            charge[k] for k in ("release_verified", "renewed_progress", "within_stage")
        ),
        "unsuccessful stage closure",
    )


if __name__ == "__main__":
    action, *args = sys.argv[1:]
    os.umask(0o077)
    if action == "spawn":
        spawn(args[0], int(args[1]), args[2], args[3:] or None)
    elif action == "prelease":
        cap, deadline = prelease(args[0], args[1], args[2])
        print(cap, deadline)
    elif action == "check":
        print(admission(args[0], int(args[1])))
    elif action == "watch":
        try:
            observe()
        except BaseException as exc:
            stage = get(JOB / "admission.json")["stage"]
            (JOB / f"watch-failure-{stage}").write_text(repr(exc))
            raise
    elif action == "monitor":
        monitor(args[0])
    else:
        raise ValueError("unknown action")
