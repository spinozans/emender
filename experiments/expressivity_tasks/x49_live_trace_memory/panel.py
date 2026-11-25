"""One explicit budget, sequential detached jobs, no automatic retry. CPU readiness is not authorization."""

from __future__ import annotations
import argparse
from importlib.metadata import version
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import time
from ..x47_xsa_attention.w3_launch import fields, procstat
from ..x48_muon_continuation.data import ROOT as REPO, load, sha, write_json
from ..x48_muon_continuation.lease import launch, descends
from .meter import HERE, MODULE, PY, source_ok, source_ref
from .panel_spec import (
    ROOT,
    JOBS,
    MIN_DISK_GiB,
    T,
    TOTAL,
    check,
    validate_authorization,
)
from .panels import OUT

MOD = MODULE + ".panel"
GATES = 47
PACKAGES = {
    "torch": "2.9.1",
    "triton": "3.5.1",
    "flash-linear-attention": "0.5.1",
    "numpy": "2.4.6",
}
DEVICE = "NVIDIA RTX 6000 Ada Generation"
SPEC = "93fc4fef2a20f50c0c8be5eb6224650e8cd627f0fad3eea0c4cf69b58a9597f1"
METER = "cb1332f3f6b2b9b592ed5c7f148386168aa6218c1b7e71275c7353e67e899266"


def cpu():
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "CPU/detached supervisor required",
    )


def source():
    source_ok()
    check(
        platform.python_version() == "3.12.3"
        and {k: version(k) for k in PACKAGES} == PACKAGES
        and os.getenv("NDM_PIN_TRITON_AUTOTUNE", "1") == "1",
        "runtime drift from metered implementation",
    )
    check(
        sha(HERE / "PANEL.md") == SPEC
        and sha(HERE / "receipts/meter.json") == METER
        and sum(j.cap for j in JOBS.values()) == TOTAL,
        "changed panel registration or meter",
    )
    return source_ref()


def qualify():
    cpu()
    ref = source()
    d = OUT / ("qualify-panel-" + ref)
    d.mkdir()
    log = d / "gate.log"
    env = dict(os.environ, OMP_NUM_THREADS="4", PYTEST_ADDOPTS="")
    begin = time.monotonic()
    with log.open("xb") as f:
        for cmd in (
            [
                PY,
                "-m",
                "pytest",
                "-q",
                str(HERE),
                "tests/test_memory_program_direct.py",
            ],
            *(
                [PY, "-m", MODULE + "." + m]
                for m in ("native_check", "trace_native", "coupled", "meter_verify")
            ),
        ):
            subprocess.run(
                cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env, check=True
            )
    text = log.read_text()
    passed = re.findall(r"=+ (\d+) passed in [^\n]+ =+", text)
    obj = [json.loads(s) for s in text.splitlines() if s.startswith("{")]
    check(
        len(passed) == 1
        and int(passed[0]) == GATES
        and len(obj) == 4
        and obj[0]["work"]["matched_frontiers"] == 654
        and obj[1]["counts"]["frontiers"] == 542
        and obj[2]["cuda_initialized"] is False
        and obj[3] == load(HERE / "receipts/meter.json"),
        "incomplete qualification",
    )
    from .train_stream import TrainStream

    stream = TrainStream()
    check(len(stream.docs) == 1024 and ref == source(), "wrong stream/source")
    out = dict(
        ref=ref,
        spec=SPEC,
        meter=METER,
        log_sha=sha(log),
        passed=int(passed[0]),
        elapsed_s=time.monotonic() - begin,
        evaluation_windows_opened=0,
    )
    write_json(d / "passed.json", out)
    return out


def authority():
    p = OUT / "natural-panel-authorization.json"
    check(p.is_file(), "X49 natural panel blocked: no separate whole-panel budget")
    ref = source()
    validate_authorization(load(p), ref, SPEC)
    d = OUT / ("qualify-panel-" + ref)
    q = load(d / "passed.json")
    check(
        q["ref"] == ref
        and q["spec"] == SPEC
        and q["meter"] == METER
        and q["passed"] == GATES
        and q["evaluation_windows_opened"] == 0
        and q["log_sha"] == sha(d / "gate.log"),
        "unqualified natural source",
    )
    return ref


def prior(job, *, root=None, jobs=None, total=None):
    root = ROOT if root is None else root
    jobs = JOBS if jobs is None else jobs
    total = TOTAL if total is None else total
    spent = 0
    remaining = 0
    seen = False
    for k in jobs.values():
        if k == job:
            seen = True
        if seen:
            remaining += int(k.kind == "train")
            continue
        d = root / k.name
        c = load(d / "charge.json")
        end = fields(d / "released")
        ini = fields(d / "initial")
        after = fields(d / "after")
        term = load(d / "complete.json")
        check(
            c["job"] == k.name
            and c["rc"] == int(end["rc"]) == 0
            and 0 < c["upper_card_s"] <= k.cap
            and c["upper_card_s"]
            == int(end["released_epoch"]) - int(ini["created"]) + 1
            and c["witnessed_units"] == term["units"] == load(d / "start.json")["units"]
            and c["renewed"]
            and int(after["heartbeat"]) > int(ini["heartbeat"])
            and all(
                ini[f] == after[f]
                for f in ("gpu", "pid", "starttime", "created", "adopted")
            )
            and c["complete_sha"] == sha(d / "complete.json")
            and (d / "watch-ack").is_file(),
            "predecessor partial/foreign: " + k.name,
        )
        spent += c["upper_card_s"]
    check(
        seen
        and spent + job.cap <= total
        and shutil.disk_usage(OUT).free >= (16 << 30) + remaining * (21 << 30),
        "remaining budget or disk inadequate",
    )


def admit(job):
    ref = authority()
    prior(job)
    d = ROOT / job.name
    start = load(d / "start.json")
    check(
        start["name"] == job.name
        and start["ref"] == ref
        and start["cap_s"] == job.cap
        and not (d / "released").exists(),
        "foreign, repeated or released job",
    )
    return d, ref, start


def progress(d, n, targets, *, root=None):
    root = ROOT if root is None else root
    driver = load(root / "driver.json")
    st = procstat(driver["pid"])
    check(
        st
        and st[19] == driver["start"]
        and st[0] != "Z"
        and st[1] == "1"
        and not (root / "stop.json").exists(),
        "lost detached supervisor; retain partial and release",
    )
    write_json(
        d / "progress.json", dict(pid=os.getpid(), units=n, scored_targets=targets)
    )


def enter(job):
    return enter_admitted(job, *admit(job))


def enter_admitted(job, d, ref, start):
    import torch
    from ..x47_xsa_attention.workbench_run import check_gpu_owner

    check(not (d / "identity.json").exists(), "worker already entered this lease")
    owner = check_gpu_owner()
    actual = fields(d / "initial")
    launch_id = load(d / "launch.json")
    check(
        all(
            owner[k] == actual[k]
            for k in ("pid", "starttime", "gpu", "created", "adopted")
        )
        and (int(owner["pid"]), owner["starttime"], procstat(int(owner["pid"]))[1])
        == (launch_id["pid"], launch_id["start"], "1"),
        "foreign live owner",
    )
    check(
        os.getenv("OMP_NUM_THREADS") == "4"
        and os.getenv("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        and all(
            os.getenv(k) == "expandable_segments:True"
            for k in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        ),
        "wrong training environment",
    )
    check(
        str(torch.__version__) == "2.9.1+cu128"
        and torch.cuda.get_device_name() == DEVICE
        and torch.cuda.get_device_properties(0).total_memory == 50876841984,
        "different from metered host device/build; no substitution",
    )
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    check(
        torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]
        is True,
        "allocator inactive",
    )
    torch.cuda.set_per_process_memory_fraction(0.80)
    write_json(
        d / "identity.json",
        dict(
            ref=ref,
            job=job.name,
            trainer=os.getpid(),
            trainer_start=procstat(os.getpid())[19],
            device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            packages=PACKAGES,
            python=platform.python_version(),
            cpu_threads=4,
            host_precision="BF16 autocast/FP32 master",
            owner=owner,
        ),
    )
    return dict(ref=ref, t0=start["t0"], folder=d)


def licensed_split(job):
    # Even the low-level raw evaluation reader must be this stage's admitted live worker.
    return licensed_admitted_split(job, *admit(job))


def licensed_admitted_split(job, d, ref, start):
    check(job.kind in ("calibration", "final_test"), "evaluation scope not licensed")
    return licensed_admitted_reader(job, d, ref, start)


def licensed_admitted_reader(job, d, ref, start):
    """Bind a reader to the current trainer, owner and frozen plan, after campaign-specific admission."""
    ident = load(d / "identity.json")
    from ..x47_xsa_attention.workbench_run import check_gpu_owner

    cur = check_gpu_owner()
    check(
        ident["job"] == job.name
        and ident["ref"] == ref
        and (ident["trainer"], ident["trainer_start"])
        == (os.getpid(), procstat(os.getpid())[19])
        and all(
            cur[k] == ident["owner"][k] for k in ("pid", "starttime", "gpu", "created")
        )
        and sha(d / "frozen.json") == start["freeze_sha"],
        "evaluation scope not licensed",
    )
    return load(d / "frozen.json")


def one(job):
    from .panel_freeze import freeze

    cpu()
    ref = authority()
    prior(job)
    dispatch_one(job, ROOT, ref, freeze)


def dispatch_one(job, root, ref, freeze):
    driver = load(root / "driver.json")
    parent = os.getppid()
    ticket = load(root / (job.name + ".ticket.json"))
    check(
        parent == driver["pid"]
        and procstat(parent)[19] == driver["start"]
        and ticket["job"] == job.name
        and ticket["ref"] == ref
        and 0 < ticket["t0"] <= time.monotonic_ns(),
        "missing live driver admission ticket",
    )
    d = root / job.name
    d.mkdir()
    start = dict(
        name=job.name,
        kind=job.kind,
        cap_s=job.cap,
        t0=ticket["t0"],
        ref=ref,
        units=1024,
    )
    if job.kind != "train":
        start["freeze_sha"] = freeze(job, ref)
        start["units"] = load(d / "frozen.json")["units"]
    write_json(d / "start.json", start)
    launch(
        ["bash", str(HERE / "panel_leased.sh"), job.name, str(job.cap)],
        d / "console",
        d / "launch.json",
    )


def witness(job, *, root=None, targets_for=None):
    d = (ROOT if root is None else root) / job.name
    start = load(d / "start.json")
    la = load(d / "launch.json")
    pid = la["pid"]
    birth = la["start"]
    expected_targets = (
        (lambda n: n * T) if targets_for is None else targets_for(job, d, start)
    )
    last = 0
    renewed = False
    while not (d / "released").exists():
        st = procstat(pid)
        if not st or st[19] != birth or st[0] == "Z":
            check(
                (d / "released").exists(),
                "owner ended before release: inspect refusal/partial, do not reacquire",
            )
            break
        if (d / "initial").is_file() and (d / "progress.json").is_file():
            i = fields(d / "initial")
            prog = load(d / "progress.json")
            n = prog["units"]
            if n > last:
                try:
                    cur = fields(REPO / ".wg/gpu_leases" / i["gpu"])
                except FileNotFoundError:
                    cur = None
                if cur:
                    ident = load(d / "identity.json")
                    trainer = procstat(prog["pid"])
                    check(
                        (
                            int(cur["pid"]),
                            cur["starttime"],
                            cur["gpu"],
                            cur["adopted"],
                            st[1],
                        )
                        == (pid, birth, i["gpu"], "1", "1")
                        and n <= start["units"]
                        and prog["scored_targets"] == expected_targets(n),
                        "foreign progress/owner",
                    )
                    if (
                        trainer
                        and trainer[19] == ident["trainer_start"]
                        and prog["pid"] == ident["trainer"]
                        and descends(prog["pid"], pid)
                    ):
                        renewed |= last > 0 and int(cur["heartbeat"]) > int(
                            i["heartbeat"]
                        )
                        last = n
                        obs = dict(
                            pid=pid,
                            start=birth,
                            ppid=1,
                            trainer=prog["pid"],
                            trainer_start=trainer[19],
                            heartbeat=cur["heartbeat"],
                            units=n,
                        )
                        with (d / "survival.jsonl").open("a") as f:
                            f.write(json.dumps(obs) + "\n")
                        if renewed and not (d / "live.json").exists():
                            write_json(d / "live.json", obs)
                            print(
                                "verified renewing " + job.name + " at unit " + str(n),
                                flush=True,
                            )
        if (
            renewed
            and last == start["units"]
            and (d / "complete.json").is_file()
            and not (d / "watch-ack").exists()
        ):
            term = load(d / "complete.json")
            check(
                term["units"] == last and term["targets"] == expected_targets(last),
                "terminal exposure mismatch",
            )
            (d / "watch-ack").write_text(
                "renewal and terminal actual work independently observed\n"
            )
        time.sleep(0.3)
    i = fields(d / "initial")
    end = fields(d / "released")
    s = int(end["released_epoch"]) - int(i["created"]) + 1
    rc = int(end["rc"])
    out = dict(
        job=job.name,
        rc=rc,
        upper_card_s=s,
        inclusive_wall_s=(int(end["end_monotonic_ns"]) - start["t0"]) * 1e-9,
        owner_and_waited_children_cpu_s=float(end["owner_cpu_s"]),
        witnessed_units=last,
        complete_sha=sha(d / "complete.json")
        if (d / "complete.json").is_file()
        else None,
        renewed=renewed,
    )
    write_json(d / "charge.json", out)
    check(
        rc == 0
        and 0 < s <= job.cap
        and renewed
        and last == start["units"]
        and (d / "watch-ack").is_file(),
        "partial or over cap; no subsequent lease",
    )
    return out


def partial(job, *, root=None):
    """Report independently readable artifacts only. Absence of accounting is unknown, not free."""
    if job is None:
        return {}
    d = (ROOT if root is None else root) / job.name
    out = dict(
        upper_card_s=None,
        complete_recovery_step=None,
        complete_recovery_sha=None,
        known_uncheckpointed_updates=None,
    )
    if (d / "released").exists() and (d / "initial").exists():
        out["upper_card_s"] = (
            int(fields(d / "released")["released_epoch"])
            - int(fields(d / "initial")["created"])
            + 1
        )
    if job.kind == "train":
        last = 0
        if (d / "steps.jsonl").exists():
            for line in (d / "steps.jsonl").read_bytes().splitlines():
                try:
                    r = json.loads(line)
                except (ValueError, UnicodeError):
                    break
                if r["step"] != last + 1:
                    break
                last += 1
        out["last_journaled_update"] = last
        cursor = 0
        if (d / "latest.json").exists():
            ck = load(d / "latest.json")
            path = d / f"step-{ck['step']:04d}.pt"
            if (
                ck["file"] != path.name
                or not path.is_file()
                or ck["sha256"] != sha(path)
                or ck["bytes"] != path.stat().st_size
            ):
                return out
            cursor = ck["step"]
            out.update(
                complete_recovery_step=cursor, complete_recovery_sha=ck["sha256"]
            )
        out["known_uncheckpointed_updates"] = max(0, last - cursor)
    else:
        # Record the raw-access boundary and the last acknowledged real work; a partial
        # is not a license to rerun even an already-open, not-yet-scored final.
        out["source_open_attempted"] = (d / "source-open-attempted").exists()
        out["stream_prepared"] = (d / "stream.json").is_file()
        out["last_progress"] = (
            load(d / "progress.json") if (d / "progress.json").is_file() else None
        )
    return out


def drive():
    from .panel_analysis import finish

    cpu()
    run_schedule(ROOT, authority(), JOBS, MOD, TOTAL, finish)


def run_schedule(root, ref, jobs, mod, total, finish, *, targets_for=None):
    info = load(root / "driver.json")
    check(
        os.getppid() == 1
        and (info["pid"], info["start"]) == (os.getpid(), procstat(os.getpid())[19]),
        "driver not detached",
    )
    charges = {}
    job = None
    try:
        for job in jobs.values():
            ticket = root / (job.name + ".ticket.json")
            check(not ticket.exists(), "job already ticketed")
            write_json(ticket, dict(job=job.name, ref=ref, t0=time.monotonic_ns()))
            subprocess.run([PY, "-m", mod, "one", "--job", job.name], check=True)
            charges[job.name] = witness(job, root=root, targets_for=targets_for)
        check(
            sum(c["upper_card_s"] for c in charges.values()) <= total, "panel over cap"
        )
        write_json(root / "charged.json", charges)
        finish(root)
        write_json(
            root / "done.json",
            dict(
                complete=True,
                charges_sha=sha(root / "charged.json"),
                summary_sha=sha(root / "summary.json"),
                driver_and_short_parents_cpu_s=sum(
                    u.ru_utime + u.ru_stime
                    for u in (
                        resource.getrusage(resource.RUSAGE_SELF),
                        resource.getrusage(resource.RUSAGE_CHILDREN),
                    )
                ),
            ),
        )
    except BaseException as e:
        try:
            retained = partial(job, root=root)
        except Exception as error:
            retained = dict(unverified=str(error))
        write_json(
            root / "stop.json",
            dict(
                job=job.name if job else None,
                error=repr(e),
                partial=retained,
                completed=list(charges),
                no_retry=True,
            ),
        )
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "phase",
        choices=(
            "qualify",
            "check",
            "prelease",
            "inside",
            "one",
            "launch",
            "drive",
            "observe",
        ),
    )
    ap.add_argument("--job", choices=list(JOBS))
    a = ap.parse_args()
    os.umask(0o077)
    os.chdir(REPO)
    job = JOBS.get(a.job)
    check(
        (job is not None) == (a.phase in ("one", "prelease", "inside")),
        "wrong job selector for phase",
    )
    if a.phase == "qualify":
        print(json.dumps(qualify()))
        return
    if a.phase == "check":
        cpu()
        print(authority())
        check(
            shutil.disk_usage(OUT).free >= MIN_DISK_GiB << 30 and not ROOT.exists(),
            "insufficient disk or panel already admitted",
        )
        return
    if a.phase == "prelease":
        cpu()
        d, ref, start = admit(job)
        print(job.cap)
        return
    if a.phase == "one":
        one(job)
        return
    if a.phase == "inside":
        from .panel_train import train
        from .panel_eval import evaluate

        d = ROOT / job.name
        o = train(job) if job.kind == "train" else evaluate(job)
        write_json(d / "complete.json", o)
        for _ in range(40):
            if (d / "watch-ack").is_file():
                return
            time.sleep(0.2)
        raise RuntimeError("no independent completion acknowledgment")
    if a.phase == "launch":
        cpu()
        authority()
        check(shutil.disk_usage(OUT).free >= MIN_DISK_GiB << 30, "need 320 GiB free")
        ROOT.mkdir()
        print(launch([PY, "-m", MOD, "drive"], ROOT / "console", ROOT / "driver.json"))
        return
    if a.phase == "drive":
        for _ in range(50):
            if os.getppid() == 1 and (ROOT / "driver.json").is_file():
                break
            time.sleep(0.1)
        drive()
        return
    observe(ROOT, JOBS)


def observe(root, jobs):
    # Read-only observer; loss of the attended session cannot kill the driver, owner or its witness.
    cpu()
    la = load(root / "driver.json")
    seen = set()
    while not (root / "done.json").exists():
        check(
            not (root / "stop.json").exists(),
            "panel stopped; inspect retained receipts",
        )
        st = procstat(la["pid"])
        check(
            st and st[19] == la["start"] and st[0] != "Z",
            "driver exited; inspect durable status",
        )
        for j in jobs:
            for f in ("live.json", "charge.json"):
                p = root / j / f
                if p.is_file() and p not in seen:
                    seen.add(p)
                    print(j, f, load(p), flush=True)
        time.sleep(2.0)
    print(load(root / "done.json"))


if __name__ == "__main__":
    main()
