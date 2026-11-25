"""CPU-only qualification, single-use detached run and independent progress witness."""

import argparse
import os
from pathlib import Path
import platform
import re
import resource
import signal
import subprocess
import sys
from time import monotonic, sleep
import traceback

import numpy as np
import torch

from . import data as d
from . import study as s
from .mount import mount, native_check
from .program import ARMS, T


def proc(pid):
    try:
        row = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        return dict(pid=pid, start=row[19], ppid=int(row[1]), state=row[0])
    except FileNotFoundError:
        return None


def until(fn, seconds, why):
    end = monotonic() + seconds
    while not fn():
        if monotonic() >= end:
            raise TimeoutError(why)
        sleep(0.10)


def qualify():
    s.setup()
    d.registration()
    ref = d.git("rev-parse", "HEAD")
    folder = d.OUT / ("qualify-" + ref)
    folder.mkdir()
    d.check(
        not d.RUN.exists() and not d.git("status", "--porcelain", "--", d.ROW_PREFIX),
        "already started or uncommitted code",
    )
    start = monotonic()
    with (folder / "tests.log").open("x") as f:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(d.HERE)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=d.ROOT,
        )
    matches = re.findall(r"\b(\d+) passed\b", (folder / "tests.log").read_text())
    d.check(matches == ["21"], "incorrect gate count")
    with mount() as pages:
        check = native_check(pages)
        # One geometry/arm on invented bytes only; no natural predictions or fits used.
        rng = np.random.default_rng(5010)
        words = rng.choice(
            np.array([0, 30, 32, 97, 98, 102, 255], np.uint8), (s.BATCH, T + 1)
        )
        timings = {}
        for arm in ARMS:
            from .kernel import Kernel
            from .program import execute, score

            model = s.model_at_birth(5001)
            bank = Kernel(pages)
            t = monotonic()
            tr = execute(
                model, bank, words[:, :-1], arm, torch.Generator().manual_seed(5000)
            )
            obj, _, _ = score(model, tr, words[:, 1:])
            obj.backward()
            timings[arm] = dict(forward_backward_wall_s=monotonic() - t, **bank.counts)
            d.check(
                bank.counts["parent_reads"] == s.BATCH * T
                and bank.counts["child_reads"] == (0 if arm == "ONE" else s.BATCH * T),
                "invented dose",
            )
    q = dict(
        spec=d.SPEC,
        registration=d.REG,
        local_ref=ref,
        sources=d.sources(),
        tests=int(matches[0]),
        test_log_sha=d.sha(folder / "tests.log"),
        native=check,
        invented_full_batches=timings,
        qualification_wall_s=monotonic() - start,
        python=platform.python_version(),
        torch=torch.__version__,
        fit_documents=0,
        probe_documents=0,
        gpu_acquisitions=0,
    )
    d.check(
        not d.RUN.exists() and not (d.OUT / "qualified.json").exists(),
        "changed preaccess state",
    )
    d.write_json(d.OUT / "qualified.json", q)
    print(
        "Qualification passed",
        q["tests"],
        "tests;",
        check,
        "invented one-batch seconds",
        {a: round(timings[a]["forward_backward_wall_s"], 3) for a in ARMS},
        flush=True,
    )


def launch():
    d.qualified()
    d.check(platform.node() == "lambda01", "local CPU scope only")
    d.RUN.mkdir(mode=0o700)
    with (d.RUN / "console.log").open("xb") as f:
        p = subprocess.Popen(
            [sys.executable, "-m", __spec__.parent + ".run", "worker"],
            cwd=d.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    info = proc(p.pid)
    d.check(info and info["state"] != "Z", "lost CPU owner")
    d.write_json(
        d.RUN / "owner.json",
        dict(pid=p.pid, start=info["start"], host=platform.node(), spec=d.SPEC),
    )
    print("Detached CPU PID", p.pid, flush=True)


def worker():
    r = d.RUN
    tic = monotonic()
    s.setup()
    q = d.qualified()
    until(
        lambda: (r / "start-ack.json").exists(),
        120,
        "detachment not independently observed",
    )
    ident = d.read(r / "owner.json")
    stat = proc(os.getpid())
    d.check(
        os.getppid() == 1
        and os.getsid(0) == os.getpid()
        and ident["pid"] == os.getpid()
        and ident["start"] == stat["start"]
        and d.read(r / "start-ack.json")["owner"] == ident,
        "not the registered detached owner",
    )
    # Hard fail-closed backstop for a broken service/pipe; not a compute comparison.
    signal.alarm(3600)
    (r / "worker-once").touch(exist_ok=False)
    d.reserve()
    fit = d.extract("fit")
    t = monotonic()
    total = 0
    nt = 0

    def tick(n):
        nonlocal total, nt
        total += 1
        nt += n
        d.write_json(
            r / "progress.json",
            dict(batches=total, targets=nt, owner=ident, phase=phase, key=key),
        )

    with mount() as pages:
        ms = monotonic() - t
        states = {}
        phase = "fit"
        for seed in d.SEEDS:
            for arm in s.order(seed):
                key = f"{seed}-{arm}"
                model = s.model_at_birth(seed)
                facts, _, _ = s.phase(model, pages, fit, seed, arm, True, tick)
                pt = r / (key + ".pt")
                torch.save(model.state_dict(), pt)
                states[key] = d.sha(pt)
                d.write_json(
                    r / (key + "-fit.json"),
                    dict(facts=facts, state=states[key], fit=d.sha(r / "fit.json")),
                )
        del fit
        d.check(
            set(states) == d.state_keys()
            and total == 12 * s.EPOCHS * d.N["fit"] // s.BATCH
            and not (r / "probe-source-open-attempted").exists(),
            "wrong FIT boundary",
        )
        d.write_json(
            r / "frozen.json",
            dict(
                spec=d.SPEC,
                qualification_sha=d.sha(d.OUT / "qualified.json"),
                rows_sha=d.sha(r / "reservation.json"),
                fit_identity_sha=d.sha(r / "fit.json"),
                states=states,
                fit_batches=total,
                fit_real_targets=nt,
            ),
        )
        until(
            lambda: (r / "freeze-ack.json").exists(),
            120,
            "observer did not witness freeze-before-probe",
        )
        ack = d.read(r / "freeze-ack.json")
        d.check(
            ack["seal"] == d.sha(r / "frozen.json")
            and ack["targets"] == nt
            and ack["owner"] == ident,
            "changed FIT acknowledgement",
        )
        probe = d.extract("probe")
        phase = "probe"
        for seed in d.SEEDS:
            for arm in s.order(seed):
                key = f"{seed}-{arm}"
                pt = r / (key + ".pt")
                d.check(d.sha(pt) == states[key], "wrong eval state")
                model = s.model_at_birth(seed)
                model.load_state_dict(
                    torch.load(pt, weights_only=True, map_location="cpu"), strict=True
                )
                facts, loss, base = s.phase(model, pages, probe, seed, arm, False, tick)
                out = r / (key + "-loss.npz")
                np.savez(out, model=loss, prior=base)
                d.write_json(
                    r / (key + "-probe.json"),
                    dict(
                        facts=facts,
                        state=states[key],
                        scores=d.sha(out),
                        probe=d.sha(r / "probe.json"),
                    ),
                )
    expected = 12 * (s.EPOCHS * d.N["fit"] + d.N["probe"]) * T
    d.check(
        nt == expected
        and nt == total * s.BATCH * T
        and not torch.cuda.is_initialized(),
        "missing terminal targets or unexpected GPU",
    )
    result = s.report()
    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(
        cost=dict(
            natural_wall_s=monotonic() - tic,
            owner_cpu_s=self_usage.ru_utime + self_usage.ru_stime,
            children_cpu_s=child.ru_utime + child.ru_stime,
            owner_peak_rss_kib=self_usage.ru_maxrss,
            max_child_peak_rss_kib=child.ru_maxrss,
            native_open_s=ms,
            gpu_card_s=0,
            reservation_and_other_setup_wall_included=True,
        ),
        local_ref=q["local_ref"],
        fit_source_bytes=d.read(r / "fit.json")["source_bytes"],
        probe_source_bytes=d.read(r / "probe.json")["source_bytes"],
    )
    d.write_json(r / "result.json", result)
    d.write_json(
        r / "done.json",
        dict(
            batches=total, targets=nt, result_sha=d.sha(r / "result.json"), owner=ident
        ),
    )
    signal.alarm(0)


def observe():
    r = d.RUN
    owner = d.read(r / "owner.json")
    pid = owner["pid"]
    acks = []
    start = monotonic()
    last = 0

    def alive():
        p = proc(pid)
        return (
            p
            and p["start"] == owner["start"]
            and p["ppid"] == 1
            and p["state"] != "Z"
            and os.getsid(pid) == pid
        )

    until(alive, 30, "no detached owner")
    d.write_json(r / "start-ack.json", dict(owner=owner))
    while True:
        if (r / "stop.json").exists():
            raise RuntimeError(d.read(r / "stop.json"))
        if (r / "done.json").exists():
            break
        d.check(alive(), "CPU owner died without terminal evidence")
        if (r / "progress.json").exists():
            p = d.read(r / "progress.json")
            n = p["batches"]
            d.check(
                p["owner"] == owner and n >= last and p["targets"] == n * s.BATCH * T,
                "invalid witnessed work",
            )
            if n > last:
                if len(acks) < 2:
                    acks.append(n)
                last = n
        if (r / "frozen.json").exists() and not (r / "freeze-ack.json").exists():
            z = d.frozen()
            d.check(
                len(acks) == 2
                and last == z["fit_batches"]
                and not (r / "probe-source-open-attempted").exists(),
                "unwitnessed FIT boundary",
            )
            d.write_json(
                r / "freeze-ack.json",
                dict(
                    seal=d.sha(r / "frozen.json"),
                    targets=z["fit_real_targets"],
                    owner=owner,
                ),
            )
        d.check(monotonic() - start < 3700, "CPU witness timed out")
        sleep(0.15)
    z = d.read(r / "done.json")
    p = d.read(r / "progress.json")
    d.check(
        len(acks) == 2
        and acks[0] < acks[1]
        and z["owner"] == owner
        and z["batches"] == p["batches"]
        and z["targets"] == p["targets"]
        and z["result_sha"] == d.sha(r / "result.json")
        and not (r / "stop.json").exists(),
        "bad terminal receipt",
    )
    until(lambda: not alive(), 60, "owner did not exit")
    closed = s.report()
    ret = d.read(r / "result.json")
    d.check(
        all(
            ret[k] == v or (k == "seed_order" and tuple(ret[k]) == v)
            for k, v in closed.items()
        ),
        "receipt-only recomputation failed",
    )
    d.write_json(
        r / "witnessed.json",
        dict(
            first_advancing_batches=acks,
            freeze_targets=d.read(r / "freeze-ack.json")["targets"],
            terminal_batches=z["batches"],
            terminal_targets=z["targets"],
            result_sha=z["result_sha"],
            receipt_only_recomputed=True,
            owner_exited=True,
            gpu_acquisitions=0,
        ),
    )
    print(
        "Closed CPU result",
        z["result_sha"],
        ret["policies"],
        ret["contrasts"],
        flush=True,
    )


def main():
    os.umask(0o077)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("qualify", "launch", "worker", "observe"))
    args = ap.parse_args()
    d.cpu()
    if args.action == "worker":
        try:
            worker()
        except BaseException:
            d.write_json(d.RUN / "stop.json", dict(error=traceback.format_exc()))
            raise
    else:
        {"qualify": qualify, "launch": launch, "observe": observe}[args.action]()


if __name__ == "__main__":
    main()
