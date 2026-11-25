"""Single local synthetic meter. No approval command, no natural trainer, no automatic retry."""

from __future__ import annotations
import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import time
from pathlib import Path
from ..x47_xsa_attention.w3_launch import fields, procstat
from ..x48_muon_continuation.data import ROOT, check, load, sha, write_json
from ..x48_muon_continuation.lease import descends, launch
from .panels import OUT

MODULE = "experiments.expressivity_tasks.x49_live_trace_memory"
SPEC = "f25063016a448669addb2087266f8b47fa0b2069abe45720ae4b24334e27a97f"
CAP = 900
GATES = 37
RUN = OUT / "synthetic-meter-1"
PY = str(ROOT / ".venv/bin/python")
HERE = Path(__file__).resolve().parent


def source_ref():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def source_ok():
    check(
        platform.node() == "lambda01" and sha(HERE / "METER.md") == SPEC,
        "wrong host or meter specification",
    )
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
            "tests/test_memory_program_direct.py",
        ],
        cwd=ROOT,
        check=True,
    )
    check(
        not subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                str(HERE.relative_to(ROOT)),
            ],
            cwd=ROOT,
        ),
        "untracked X49 source",
    )


def validate_approval(a, ref):
    expected = dict(
        scope="x49-synthetic-meter-v1",
        spec_sha256=SPEC,
        qualified_ref=ref,
        card_seconds=CAP,
        gpus=1,
        attempts=1,
        max_physical_updates=16,
        natural_training=False,
        natural_evaluation=False,
        remote=False,
    )
    check(
        type(a) is dict
        and set(a) == set(expected) | {"operator_note"}
        and all(type(a[k]) is type(v) and a[k] == v for k, v in expected.items())
        and isinstance(a["operator_note"], str)
        and a["operator_note"].strip(),
        "different or incomplete X49 synthetic-only authorization",
    )


def qualification():
    check(
        os.getenv("CUDA_VISIBLE_DEVICES") == "" and not os.getenv("GPU_LEASE_HELD"),
        "CPU only",
    )
    source_ok()
    ref = source_ref()
    d = OUT / ("qualify-meter-" + ref)
    d.mkdir()
    log = d / "gate.log"
    with log.open("xb") as f:
        env = dict(os.environ, OMP_NUM_THREADS="4", PYTEST_ADDOPTS="")
        start = time.monotonic()
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
                [PY, "-m", MODULE + "." + s]
                for s in ("native_check", "trace_native", "coupled")
            ),
        ):
            subprocess.run(
                cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, check=True
            )
    text = log.read_text()
    m = re.findall(r"=+ (\d+) passed in [^\n]+ =+", text)
    check(len(m) == 1 and int(m[0]) == GATES, "incomplete pytest gate")
    results = [json.loads(s) for s in text.splitlines() if s.startswith("{")]
    check(
        len(results) == 3
        and results[0]["work"]["matched_frontiers"] == 654
        and results[1]["counts"]["frontiers"] == 542
        and results[2]["cuda_initialized"] is False
        and results[2]["optimizer_state_entries"] == 0,
        "incomplete native/host qualification",
    )
    source_ok()
    check(source_ref() == ref, "source changed")
    o = dict(
        ref=ref,
        spec=SPEC,
        sha256=sha(log),
        passed=int(m[0]),
        elapsed_s=time.monotonic() - start,
    )
    write_json(d / "passed.json", o)
    return o


def authorize():
    p = OUT / "meter-authorization.json"
    check(
        p.is_file(),
        "X49 meter blocked: no explicit separate local card-time authorization",
    )
    ref = source_ref()
    validate_approval(load(p), ref)
    source_ok()
    d = OUT / ("qualify-meter-" + ref)
    q = load(d / "passed.json")
    check(
        q["ref"] == ref
        and q["spec"] == SPEC
        and q["passed"] == GATES
        and q["sha256"] == sha(d / "gate.log"),
        "unqualified meter source",
    )
    check(
        shutil.disk_usage(OUT).free >= 16 << 30,
        "need 16 GiB free on the meter output filesystem",
    )
    return ref


def watch():
    info = load(RUN / "launch.json")
    pid = info["pid"]
    start = info["start"]
    # Reattaching to the *same* owner is allowed, not launching again or rewriting a charge.
    billed = RUN / "charge.json"
    if billed.exists():
        result = load(billed)
        check(
            result["pid"] == pid
            and result["rc"] == 0
            and 0 < result["upper_card_seconds"] <= CAP
            and result["renewal_and_progress"]
            and result["witnessed_updates"] == 16,
            "previously charged partial; no retry",
        )
        return result
    last = 0
    verified = False
    journal = RUN / "survival.jsonl"
    if journal.exists():
        init = fields(RUN / "initial")
        for line in journal.read_text().splitlines():
            old = json.loads(line)
            n = old["updates"]
            check(
                (old["pid"], old["start"], old["ppid"]) == (pid, start, 1)
                and last < n <= 16,
                "invalid retained progress",
            )
            verified |= last > 0 and int(old["heartbeat"]) > int(init["heartbeat"])
            last = n
    while not (RUN / "released").is_file():
        st = procstat(pid)
        if not st or st[19] != start or st[0] == "Z":
            check(
                (RUN / "released").exists(),
                "owner exited before verified release; retain refusal/partial, no retry",
            )
            break
        if (RUN / "initial").is_file() and (RUN / "progress.json").is_file():
            init = fields(RUN / "initial")
            f = ROOT / ".wg/gpu_leases" / init["gpu"]
            r = load(RUN / "progress.json")
            n = r["physical_updates"]
            if n > last:
                try:
                    cur = fields(f)
                except FileNotFoundError:
                    cur = None
                if cur is not None:
                    check(
                        (
                            cur["pid"],
                            cur["starttime"],
                            cur["gpu"],
                            cur["adopted"],
                            st[1],
                        )
                        == (str(pid), start, init["gpu"], "1", "1")
                        and 1 <= n <= 16
                        and r["physical_synthetic_targets"] == 4096 * n,
                        "owner or progress mismatch",
                    )
                    ident = load(RUN / "identity.json")
                    trainer = procstat(r["pid"])
                    if (
                        trainer
                        and trainer[19] == ident["trainer_start"]
                        and r["pid"] == ident["pid"]
                        and descends(r["pid"], pid)
                    ):
                        verified |= last > 0 and int(cur["heartbeat"]) > int(
                            init["heartbeat"]
                        )
                        last = n
                        o = dict(
                            pid=pid,
                            start=start,
                            ppid=1,
                            trainer=r["pid"],
                            trainer_start=trainer[19],
                            updates=n,
                            heartbeat=cur["heartbeat"],
                        )
                        with (RUN / "survival.jsonl").open("a") as out:
                            out.write(json.dumps(o) + "\n")
                        if verified and not (RUN / "live.json").exists():
                            write_json(RUN / "live.json", o)
                            print(
                                "witnessed renewed owner and advancing 4K updates",
                                flush=True,
                            )
        if (
            verified
            and last == 16
            and (RUN / "complete.json").is_file()
            and not (RUN / "watch-ack").exists()
        ):
            (RUN / "watch-ack").write_text(
                "witnessed progress, renewal and 16 actual updates\n"
            )
        time.sleep(0.25)
    init = fields(RUN / "initial")
    release = fields(RUN / "released")
    charge = int(release["released_epoch"]) - int(init["created"]) + 1
    rc = int(release["rc"])
    o = dict(
        rc=rc,
        upper_card_seconds=charge,
        pid=pid,
        gpu=init["gpu"],
        witnessed_updates=last,
        renewal_and_progress=verified,
    )
    write_json(RUN / "charge.json", o)
    check(
        rc == 0
        and 0 < charge <= CAP
        and verified
        and last == 16
        and (RUN / "watch-ack").is_file(),
        "meter partial/over cap; no retry",
    )
    return o


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("qualify", "check", "start", "supervise", "inside", "watch")
    )
    phase = parser.parse_args().phase
    os.umask(0o077)
    os.chdir(ROOT)
    if phase == "qualify":
        print(json.dumps(qualification()))
        return
    if phase == "watch":
        print(json.dumps(watch()))
        return
    if phase == "inside":
        from .meter_run import run

        run()
        return
    ref = authorize()
    if phase == "check":
        print("admitted meter source " + ref)
        return
    check(
        not os.getenv("GPU_LEASE_HELD") and os.getenv("CUDA_VISIBLE_DEVICES") == "",
        "no nested or hand-selected card",
    )
    if phase == "start":
        RUN.mkdir()
        print(
            launch(
                ["bash", str(HERE / "meter_leased.sh")],
                RUN / "console",
                RUN / "launch.json",
            )
        )
        return
    subprocess.run([PY, "-m", MODULE + ".meter", "start"], check=True)
    print(json.dumps(watch()))


if __name__ == "__main__":
    main()
