"""Pre-access CPU qualification, single-use detached fit/probe and independent witness."""

import argparse
import os
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

from experiments.expressivity_tasks.x50_read_tape.mount import mount, native_check
from experiments.expressivity_tasks.x50_read_tape.run import proc, until
from . import data as d, study as s
from .reader import ARMS, T, unigram_counts, unigram_log

FIT_BATCH = len(d.state_keys()) * s.EPOCHS * d.N["fit"] // s.BATCH
TOTAL = FIT_BATCH + len(d.state_keys()) * d.N["probe"] // s.BATCH
UNIT = T * s.BATCH


def qualify():
    s.setup()
    d.registration()
    ref = d.git("rev-parse", "HEAD")
    base = d.OUT / ("qualify-" + ref)
    base.mkdir()
    d.check(
        not d.RUN.exists() and not d.git("status", "--porcelain", "--", d.PREFIX),
        "qualification must precede access and use committed source",
    )
    start = monotonic()
    log = base / "tests.log"
    with log.open("x") as f:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(d.HERE)],
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=d.ROOT,
            check=True,
        )
    passed = re.findall(r"\b(\d+) passed\b", log.read_text())
    d.check(len(passed) == 1 and "skipped" not in log.read_text(), "partial gate")
    rng = np.random.default_rng(5299)
    w = rng.choice(np.array([0, 30, 32, 65, 90, 98, 255], np.uint8), (s.BATCH, T + 1))
    u = unigram_log(unigram_counts(w))
    stamps = {}
    prices = {}
    with mount() as pages:
        native = native_check(pages)
        for arm in ARMS:
            m = s.model_at_birth(d.SEEDS[0])
            stamps[arm] = []
            prices[arm] = {}
            for label, fit in [("fit", True), ("probe", False)]:
                facts, _ = s.phase(
                    m,
                    pages if arm != "NONE" else None,
                    w,
                    u,
                    d.SEEDS[0],
                    arm,
                    fit,
                    lambda n: None,
                )
                stamps[arm].append(facts["refresh_hash"])
                prices[arm][label] = dict(
                    chosen_s=facts["cost"]["chosen_s"], **facts["archive"]
                )
    d.check(
        all(stamps["EVERY"])
        and stamps["EVERY"] == stamps["KEEP4"] == stamps["DISCARD4"]
        and stamps["NONE"] == [None, None],
        "invented native sparse answers differ",
    )
    d.check(
        not d.RUN.exists() and not (d.OUT / "qualified.json").exists(),
        "changed preaccess state",
    )
    q = dict(
        spec=d.SPEC,
        registration=d.REG,
        local_ref=ref,
        sources=d.sources(),
        tests=int(passed[0]),
        test_sha=d.sha(log),
        native=native,
        invented_prices=prices,
        equal_refresh_answers=True,
        natural_documents=0,
        gpu_acquisitions=0,
        wall_s=monotonic() - start,
        python=platform.python_version(),
        torch=torch.__version__,
    )
    d.write_json(d.OUT / "qualified.json", q)
    print(
        "Qualified:",
        q["tests"],
        "tests;",
        native["prefix_and_selected_reads"],
        "native checks; equal scheduled answers",
        flush=True,
    )


def launch():
    d.qualified()
    d.check(platform.node() == "lambda01", "local CPU only")
    d.RUN.mkdir(mode=0o700)
    with (d.RUN / "console.log").open("xb") as f:
        p = subprocess.Popen(
            [sys.executable, "-m", __spec__.parent + ".run", "worker"],
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=d.ROOT,
            start_new_session=True,
            close_fds=True,
        )
    v = proc(p.pid)
    d.check(v and v["state"] != "Z", "lost detached owner")
    d.write_json(
        d.RUN / "owner.json",
        dict(pid=p.pid, start=v["start"], spec=d.SPEC, host=platform.node()),
    )


def worker():
    start = monotonic()
    s.setup()
    q = d.qualified()
    r = d.RUN
    until(lambda: (r / "start-ack.json").exists(), 120, "detachment unconfirmed")
    owner = d.read(r / "owner.json")
    v = proc(os.getpid())
    d.check(
        os.getppid() == 1
        and os.getsid(0) == os.getpid() == owner["pid"]
        and owner["start"] == v["start"]
        and d.read(r / "start-ack.json") == owner,
        "foreign worker",
    )
    signal.alarm(3600)
    (r / "worker-once").touch(exist_ok=False)
    d.reserve()
    fit = d.extract("fit")
    counts = unigram_counts(fit)
    d.check(counts.sum() == d.N["fit"] * T, "bad FIT unigram")
    d.write_json(
        r / "unigram.json", dict(counts=counts.tolist(), fit=d.sha(r / "fit.json"))
    )
    u = unigram_log(counts)
    batches = targets = 0
    states = {}

    def tick(n):
        nonlocal batches, targets
        batches += 1
        targets += n
        d.write_json(
            r / "progress.json",
            dict(owner=owner, batches=batches, targets=targets, phase=label, key=key),
        )

    t = monotonic()
    with mount() as pages:
        open_s = monotonic() - t
        label = "fit"
        for seed in d.SEEDS:
            stamps = {}
            for arm in s.order(seed):
                key = f"{seed}-{arm}"
                m = s.model_at_birth(seed)
                facts, _ = s.phase(
                    m, pages if arm != "NONE" else None, fit, u, seed, arm, True, tick
                )
                stamps[arm] = facts["refresh_hash"]
                pt = r / (key + ".pt")
                torch.save(m.state_dict(), pt)
                states[key] = d.sha(pt)
                d.write_json(
                    r / (key + "-fit.json"),
                    dict(facts=facts, state=states[key], fit=d.sha(r / "fit.json")),
                )
            d.check(
                stamps["NONE"] is None
                and stamps["KEEP4"]
                and stamps["KEEP4"] == stamps["DISCARD4"] == stamps["EVERY"],
                "unequal natural FIT refresh answers",
            )
        del fit
        d.check(
            batches == FIT_BATCH
            and targets == batches * UNIT
            and set(states) == d.state_keys()
            and not (r / "probe-source-open-attempted").exists(),
            "wrong FIT boundary",
        )
        d.write_json(
            r / "frozen.json",
            dict(
                spec=d.SPEC,
                qualification_sha=d.sha(d.OUT / "qualified.json"),
                states=states,
                rows_sha=d.sha(r / "reservation.json"),
                fit_sha=d.sha(r / "fit.json"),
                prior_sha=d.sha(r / "unigram.json"),
                fit_batches=batches,
                fit_targets=targets,
            ),
        )
        until(lambda: (r / "freeze-ack.json").exists(), 120, "unwitnessed freeze")
        probe = d.extract("probe")
        label = "probe"
        table = d.read(r / "unigram.json")
        d.check(
            table["fit"] == d.sha(r / "fit.json")
            and sum(table["counts"]) == d.N["fit"] * T,
            "foreign FIT prior",
        )
        u = unigram_log(table["counts"])
        for seed in d.SEEDS:
            for arm in s.order(seed):
                key = f"{seed}-{arm}"
                pt = r / (key + ".pt")
                d.check(d.sha(pt) == states[key], "foreign state")
                m = s.model_at_birth(seed)
                m.load_state_dict(
                    torch.load(pt, map_location="cpu", weights_only=True), strict=True
                )
                facts, ds = s.phase(
                    m,
                    pages if arm != "NONE" else None,
                    probe,
                    u,
                    seed,
                    arm,
                    False,
                    tick,
                )
                path = r / (key + "-scores.npz")
                np.savez(path, **ds)
                d.write_json(
                    r / (key + "-probe.json"),
                    dict(
                        facts=facts,
                        state=states[key],
                        probe=d.sha(r / "probe.json"),
                        scores=d.sha(path),
                    ),
                )
    d.check(
        batches == TOTAL
        and targets == batches * UNIT
        and not torch.cuda.is_initialized(),
        "incorrect terminal dose or device",
    )
    result = s.score_report()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(
        local_ref=q["local_ref"],
        excluded_unique=d.EXCLUDED,
        fit_source_bytes=d.read(r / "fit.json")["source_bytes"],
        probe_source_bytes=d.read(r / "probe.json")["source_bytes"],
        runtime=dict(
            wall_s=monotonic() - start,
            owner_cpu_s=ru.ru_utime + ru.ru_stime,
            child_cpu_s=ch.ru_utime + ch.ru_stime,
            owner_peak_rss_kib=ru.ru_maxrss,
            max_child_rss_kib=ch.ru_maxrss,
            native_open_s=open_s,
            gpu_card_s=0,
        ),
    )
    d.write_json(r / "result.json", result)
    d.write_json(
        r / "done.json",
        dict(
            owner=owner, batches=batches, targets=targets, sha=d.sha(r / "result.json")
        ),
    )
    signal.alarm(0)


def observe():
    r = d.RUN
    owner = d.read(r / "owner.json")
    pid = owner["pid"]
    seen = []
    last = 0
    start = monotonic()

    def alive():
        v = proc(pid)
        return (
            v
            and v["ppid"] == 1
            and v["start"] == owner["start"]
            and v["state"] != "Z"
            and os.getsid(pid) == pid
        )

    until(alive, 45, "no detached owner")
    d.write_json(r / "start-ack.json", owner)
    while not (r / "done.json").exists():
        if (r / "stop.json").exists():
            raise RuntimeError(d.read(r / "stop.json"))
        d.check(
            alive() and monotonic() - start < 3700, "lost owner or watchdog timeout"
        )
        if (r / "progress.json").exists():
            p = d.read(r / "progress.json")
            n = p["batches"]
            d.check(
                n >= last and p["owner"] == owner and p["targets"] == n * UNIT,
                "invalid witnessed work",
            )
            if n > last and len(seen) < 2:
                seen.append(n)
            last = n
        if (
            last == FIT_BATCH
            and (r / "frozen.json").exists()
            and not (r / "freeze-ack.json").exists()
        ):
            z = d.frozen()
            d.check(
                len(seen) == 2
                and z["fit_batches"] == last
                and z["fit_targets"] == last * UNIT
                and not (r / "probe-source-open-attempted").exists(),
                "unwitnessed FIT boundary",
            )
            d.write_json(
                r / "freeze-ack.json",
                dict(
                    owner=owner, targets=z["fit_targets"], seal=d.sha(r / "frozen.json")
                ),
            )
        sleep(0.1)
    until(lambda: not alive(), 60, "owner did not exit")
    z = d.read(r / "done.json")
    p = d.read(r / "progress.json")
    d.check(
        len(seen) == 2
        and z["owner"] == p["owner"] == owner
        and z["batches"] == p["batches"] == TOTAL
        and z["targets"] == p["targets"] == TOTAL * UNIT
        and z["sha"] == d.sha(r / "result.json")
        and not (r / "stop.json").exists(),
        "bad terminal evidence",
    )
    ret = d.read(r / "result.json")
    rebuilt = s.score_report()
    d.check(all(ret[k] == v for k, v in rebuilt.items()), "receipt-only close differs")
    d.write_json(
        r / "witnessed.json",
        dict(
            owner_exited=True,
            first_advancing_batches=seen,
            freeze_targets=FIT_BATCH * UNIT,
            terminal_batches=TOTAL,
            terminal_targets=TOTAL * UNIT,
            result_sha=z["sha"],
            receipt_only_recomputed=True,
            gpu_acquisitions=0,
        ),
    )
    print("Closed", z["sha"], ret["contrasts"], ret["prices_s_per_million"], flush=True)


def main():
    os.umask(0o077)
    d.cpu()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=("qualify", "worker", "observe", "launch"))
    action = ap.parse_args().action
    if action == "worker":
        try:
            worker()
        except BaseException:
            d.write_json(d.RUN / "stop.json", dict(error=traceback.format_exc()))
            raise
    else:
        {"qualify": qualify, "launch": launch, "observe": observe}[action]()


if __name__ == "__main__":
    main()
