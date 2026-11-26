"""Single-use CPU experiment with an independently witnessed FIT/PROBE boundary."""

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

FIT_BATCHES = len(d.state_keys()) * s.EPOCHS * d.N["fit"] // s.BATCH
TOTAL_BATCHES = FIT_BATCHES + len(d.state_keys()) * d.N["probe"] // s.BATCH


def qualify():
    s.setup()
    d.registration()
    ref = d.git("rev-parse", "HEAD")
    folder = d.OUT / ("qualify-" + ref)
    folder.mkdir()
    d.check(
        not d.RUN.exists() and not d.git("status", "--porcelain", "--", d.PREFIX),
        "started or uncommitted qualification",
    )
    t = monotonic()
    log = folder / "tests.log"
    with log.open("x") as f:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(d.HERE)],
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
            cwd=d.ROOT,
        )
    n = re.findall(r"\b(\d+) passed\b", log.read_text())
    d.check(n == ["10"] and "skipped" not in log.read_text(), "incomplete fixture gate")
    rng = np.random.default_rng(5199)
    words = rng.choice(
        np.array([0, 30, 32, 65, 90, 98, 255], np.uint8), size=(s.BATCH, T + 1)
    )
    prior = unigram_log(unigram_counts(words))
    price = {}
    with mount() as pages:
        native = native_check(pages)
        for arm in ARMS:
            model = s.model_at_birth(d.SEEDS[0])
            fit, _ = s.phase(
                model,
                pages if arm != "NONE" else None,
                words,
                prior,
                d.SEEDS[0],
                arm,
                True,
                lambda n: None,
            )
            probe, _ = s.phase(
                model,
                pages if arm != "NONE" else None,
                words,
                prior,
                d.SEEDS[0],
                arm,
                False,
                lambda n: None,
            )
            price[arm] = dict(
                fit_wall_s_3_batches=fit["cost"]["chosen_wall_s"],
                probe_wall_s_1_batch=probe["cost"]["chosen_wall_s"],
                fit_reads=fit["archive"]["parent_reads"],
                probe_reads=probe["archive"]["parent_reads"],
            )
    q = dict(
        spec=d.SPEC,
        registration=d.REG,
        local_ref=ref,
        sources=d.source_files(),
        tests=int(n[0]),
        tests_sha=d.sha(log),
        native=native,
        invented_prices=price,
        wall_s=monotonic() - t,
        python=platform.python_version(),
        torch=torch.__version__,
        natural_documents=0,
        gpu_acquisitions=0,
    )
    d.check(
        not d.RUN.exists() and not (d.OUT / "qualified.json").exists(),
        "changed preaccess state",
    )
    d.write_json(d.OUT / "qualified.json", q)
    print(
        "Passed qualifier:",
        q["tests"],
        "fixture checks;",
        native["prefix_and_selected_reads"],
        "native comparisons;",
        price,
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
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
        )
    v = proc(p.pid)
    d.check(v and v["state"] != "Z", "missing detached owner")
    d.write_json(
        d.RUN / "owner.json",
        dict(pid=p.pid, start=v["start"], host=platform.node(), spec=d.SPEC),
    )


def worker():
    tic = monotonic()
    s.setup()
    q = d.qualified()
    r = d.RUN
    until(lambda: (r / "start-ack.json").exists(), 120, "no observer")
    owner = d.read(r / "owner.json")
    v = proc(os.getpid())
    d.check(
        os.getppid() == 1
        and os.getsid(0) == os.getpid() == owner["pid"]
        and owner["start"] == v["start"]
        and d.read(r / "start-ack.json") == owner,
        "foreign or attached owner",
    )
    signal.alarm(3600)
    (r / "worker-once").touch(exist_ok=False)
    d.reserve()
    fit = d.extract("fit")
    counts = unigram_counts(fit)
    d.check(counts.sum() == d.N["fit"] * T, "bad unigram exposure")
    d.write_json(
        r / "unigram.json", dict(fit_sha=d.sha(r / "fit.json"), counts=counts.tolist())
    )
    prior = unigram_log(counts)
    batches = nt = 0

    def tick(n):
        nonlocal batches, nt
        batches += 1
        nt += n
        d.write_json(
            r / "progress.json",
            dict(batches=batches, targets=nt, owner=owner, phase=phase, key=key),
        )

    states = {}
    t = monotonic()
    with mount() as pages:
        open_s = monotonic() - t
        phase = "fit"
        for seed in d.SEEDS:
            for arm in s.order(seed):
                key = f"{seed}-{arm}"
                model = s.model_at_birth(seed)
                pt = r / (key + ".pt")
                facts, _ = s.phase(
                    model,
                    pages if arm != "NONE" else None,
                    fit,
                    prior,
                    seed,
                    arm,
                    True,
                    tick,
                )
                torch.save(model.state_dict(), pt)
                states[key] = d.sha(pt)
                d.write_json(
                    r / (key + "-fit.json"),
                    dict(
                        facts=facts,
                        state_sha=states[key],
                        fit_sha=d.sha(r / "fit.json"),
                    ),
                )
        del fit
        d.check(
            states.keys() == d.state_keys()
            and batches == FIT_BATCHES
            and nt == batches * s.BATCH * T
            and not (r / "probe-source-open-attempted").exists(),
            "wrong freeze boundary",
        )
        for seed in d.SEEDS:
            stamps = [
                d.read(r / f"{seed}-{arm}-fit.json")["facts"][
                    "current_answer_and_prior_sha"
                ]
                for arm in ARMS
            ]
            d.check(
                stamps[0] is None and stamps[1] and stamps[1] == stamps[2],
                "FIT current reads differ",
            )
        d.write_json(
            r / "frozen.json",
            dict(
                spec=d.SPEC,
                qualification_sha=d.sha(d.OUT / "qualified.json"),
                rows_sha=d.sha(r / "reservation.json"),
                fit_sha=d.sha(r / "fit.json"),
                prior_sha=d.sha(r / "unigram.json"),
                states=states,
                fit_batches=batches,
                fit_targets=nt,
            ),
        )
        until(lambda: (r / "freeze-ack.json").exists(), 120, "unwitnessed fit freeze")
        probe = d.extract("probe")
        phase = "probe"
        u = d.read(r / "unigram.json")
        d.check(
            u["fit_sha"] == d.sha(r / "fit.json")
            and sum(u["counts"]) == d.N["fit"] * T,
            "foreign unigram",
        )
        prior = unigram_log(u["counts"])
        for seed in d.SEEDS:
            for arm in s.order(seed):
                key = f"{seed}-{arm}"
                pt = r / (key + ".pt")
                d.check(d.sha(pt) == states[key], "changed checkpoint")
                model = s.model_at_birth(seed)
                model.load_state_dict(
                    torch.load(pt, map_location="cpu", weights_only=True), strict=True
                )
                facts, scores = s.phase(
                    model,
                    pages if arm != "NONE" else None,
                    probe,
                    prior,
                    seed,
                    arm,
                    False,
                    tick,
                )
                path = r / (key + "-scores.npz")
                np.savez(path, **scores)
                d.write_json(
                    r / (key + "-probe.json"),
                    dict(
                        facts=facts,
                        state_sha=states[key],
                        probe_sha=d.sha(r / "probe.json"),
                        scores_sha=d.sha(path),
                    ),
                )
    d.check(
        batches == TOTAL_BATCHES
        and nt == batches * s.BATCH * T
        and not torch.cuda.is_initialized(),
        "wrong terminal exposure or GPU",
    )
    result = s.report()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    result.update(
        local_ref=q["local_ref"],
        excluded_unique=46182,
        fit_source_bytes=d.read(r / "fit.json")["source_bytes"],
        probe_source_bytes=d.read(r / "probe.json")["source_bytes"],
        runtime=dict(
            wall_s=monotonic() - tic,
            owner_cpu_s=ru.ru_stime + ru.ru_utime,
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
        dict(owner=owner, batches=batches, targets=nt, sha=d.sha(r / "result.json")),
    )
    signal.alarm(0)


def observe():
    r = d.RUN
    owner = d.read(r / "owner.json")
    pid = owner["pid"]
    first = []
    last = 0
    t = monotonic()

    def alive():
        p = proc(pid)
        return (
            p
            and p["ppid"] == 1
            and p["start"] == owner["start"]
            and p["state"] != "Z"
            and os.getsid(pid) == pid
        )

    until(alive, 45, "no detached owner")
    d.write_json(r / "start-ack.json", owner)
    while not (r / "done.json").exists():
        if (r / "stop.json").exists():
            raise RuntimeError(d.read(r / "stop.json"))
        d.check(alive() and monotonic() - t < 3700, "lost owner or watchdog elapsed")
        if (r / "progress.json").exists():
            p = d.read(r / "progress.json")
            n = p["batches"]
            d.check(
                p["owner"] == owner and n >= last and p["targets"] == n * s.BATCH * T,
                "invalid observed work",
            )
            if n > last and len(first) < 2:
                first.append(n)
            last = n
        if (
            (r / "frozen.json").exists()
            and last == FIT_BATCHES
            and not (r / "freeze-ack.json").exists()
        ):
            z = d.frozen()
            d.check(
                len(first) == 2
                and z["fit_batches"] == last
                and z["fit_targets"] == last * s.BATCH * T
                and not (r / "probe-source-open-attempted").exists(),
                "unwitnessed freeze boundary",
            )
            d.write_json(
                r / "freeze-ack.json",
                dict(
                    owner=owner, targets=z["fit_targets"], seal=d.sha(r / "frozen.json")
                ),
            )
        sleep(0.1)
    until(lambda: not alive(), 60, "owner did not exit")
    done = d.read(r / "done.json")
    p = d.read(r / "progress.json")
    d.check(
        not (r / "stop.json").exists()
        and len(first) == 2
        and done["owner"] == p["owner"] == owner
        and done["batches"] == p["batches"] == TOTAL_BATCHES
        and done["targets"] == p["targets"] == TOTAL_BATCHES * s.BATCH * T
        and done["sha"] == d.sha(r / "result.json"),
        "incomplete terminal receipt",
    )
    rebuilt = s.report()
    result = d.read(r / "result.json")
    d.check(
        all(result[k] == v for k, v in rebuilt.items()), "receipt-only audit differed"
    )
    d.write_json(
        r / "witnessed.json",
        dict(
            first_advancing_batches=first,
            owner_exited=True,
            freeze_targets=FIT_BATCHES * s.BATCH * T,
            terminal_batches=TOTAL_BATCHES,
            terminal_targets=done["targets"],
            result_sha=done["sha"],
            receipt_only_recomputed=True,
            gpu_acquisitions=0,
        ),
    )
    print("Closed", done["sha"], result["policies"], result["contrasts"], flush=True)


def main():
    os.umask(0o077)
    d.parent.cpu()
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("action", choices=("qualify", "launch", "worker", "observe"))
    action = a.parse_args().action
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
