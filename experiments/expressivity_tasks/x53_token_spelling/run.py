"""Publish a synthetic gate first; one detached BUILD/freeze/PROBE with a live witness."""

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
from experiments.expressivity_tasks.x50_read_tape.run import proc, until
from . import data as d, study as s
from .archive import Atlas, Builder


def qualify():
    d.registration()
    ref = d.git("rev-parse", "HEAD")
    base = d.OUT / ("qualify-" + ref)
    base.mkdir()
    d.check(
        not d.RUN.exists() and not d.git("status", "--porcelain", "--", d.PREFIX),
        "uncommitted qualification or prior access",
    )
    tic = monotonic()
    log = base / "tests.log"
    with log.open("x") as f:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(d.HERE),
                str(d.HERE.parent / "token_suffix_fanout"),
            ],
            cwd=d.ROOT,
            stdout=f,
            stderr=subprocess.STDOUT,
            check=True,
        )
    passed = re.findall(r"\b(\d+) passed\b", log.read_text())
    d.check(
        len(passed) == 1 and "skipped" not in log.read_text(), "partial fixture suite"
    )
    # Registered length and batch size, only invented bytes. No payload membership.
    rng = np.random.default_rng(5309)
    builder = Builder()
    vocab = [
        b"hello ",
        b"shell ",
        b"help ",
        b"helm!",
        b"blue_",
        b"\xff",
        b"ship? ",
        b"\x00",
    ]

    def raw(n):
        return b"".join(vocab[int(i)] for i in rng.integers(len(vocab), size=n))[:n]

    for _ in range(16):
        builder.add(raw(d.LENGTH["build"]))
    image = base / "invented.npz"
    builder.save(image)
    del builder
    atlas = Atlas(image)
    facts, sums = s.phase(
        atlas, [raw(d.LENGTH["probe"]) for _ in range(16)], lambda _: None
    )
    z = s.summarize(sums, facts, draws=256)
    s.check_work(facts, z["targets"])
    d.check(z["changed_gap_predictions"] > 0, "invented tail contrast not exercised")
    out = d.OUT / "qualified.json"
    d.check(not out.exists() and not d.RUN.exists(), "preaccess qualification required")
    d.write_json(
        out,
        dict(
            spec=d.SPEC,
            registration=d.REG,
            local_ref=ref,
            sources=d.sources(),
            oracle_sha=d.sha(d.HERE.parent / "token_suffix_fanout/index.py"),
            test_sha=d.sha(log),
            tests=int(passed[0]),
            invented_targets=z["targets"],
            invented_price=z["seconds_per_million"],
            invented_reads=facts["counters"]["SPELL"]["reads"],
            invented_topology_sha=atlas.digest(),
            equal_refresh=True,
            natural_documents=0,
            gpu_acquisitions=0,
            wall_s=monotonic() - tic,
        ),
    )
    print("Invented qualification passed", int(passed[0]), "tests", flush=True)


def launch():
    d.qualified()
    d.check(platform.node() == "lambda01", "local CPU only")
    r = d.RUN
    r.mkdir(mode=0o700)
    with (r / "console.log").open("xb") as f:
        p = subprocess.Popen(
            [sys.executable, "-m", __spec__.parent + ".run", "worker"],
            cwd=d.ROOT,
            stdin=subprocess.DEVNULL,
            stdout=f,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    info = proc(p.pid)
    d.check(info and info["state"] != "Z", "lost owner")
    d.write_json(
        r / "owner.json",
        dict(pid=p.pid, start=info["start"], host=platform.node(), spec=d.SPEC),
    )


def score_receipts():
    r = d.RUN
    f = d.seal()
    facts = d.read(r / "work.json")
    p = d.read(r / "probe.json")
    path = r / "scores.npz"
    d.check(
        facts["scores_sha"] == d.sha(path)
        and p["fit_seal"] == facts["freeze_sha"] == d.sha(r / "frozen.json")
        and facts["probe_sha"] == d.sha(r / "probe.json")
        and facts["windows"] == d.N["probe"]
        and facts["targets"] == p["targets"] == d.N["probe"] * (d.LENGTH["probe"] - 1)
        and p["rows_sha"] == f["rows_sha"],
        "wrong scores or probe",
    )
    with np.load(path, allow_pickle=False) as ds:
        d.check(
            set(ds.files) == {"loss", "sizes", "movement"}
            and np.all(ds["sizes"].sum(1) == d.LENGTH["probe"] - 1),
            "bad document totals",
        )
        sums = dict(ds)
        z = s.summarize(sums, facts)
        alt = s.summarize(sums, facts, use_counts=False)
    for name, cc in z["contrasts"].items():
        for band, v in cc.items():
            w = alt["contrasts"][name][band]
            d.check(
                v["decision"] == w["decision"] and v["valid_draws"] == w["valid_draws"],
                "bootstrap replication mismatch",
            )
            if v["ci95"] is not None:
                np.testing.assert_allclose(v["ci95"], w["ci95"], rtol=1e-11, atol=3e-14)
    s.check_work(facts, z["targets"])
    return z


def worker():
    d.qualified()
    tic = monotonic()
    r = d.RUN
    until(lambda: (r / "start-ack.json").exists(), 120, "unwitnessed detachment")
    owner = d.read(r / "owner.json")
    me = proc(os.getpid())
    d.check(
        me
        and me["ppid"] == 1
        and me["start"] == owner["start"]
        and me["pid"] == os.getsid(0) == owner["pid"]
        and owner == d.read(r / "start-ack.json"),
        "foreign worker",
    )
    signal.alarm(3600)
    (r / "worker-once").touch(exist_ok=False)
    d.reserve()

    def tick(build, probe):
        d.write_json(r / "progress.json", dict(owner=owner, build=build, probe=probe))

    builder = Builder()
    build_s = 0
    for n, raw in enumerate(d.windows("build"), 1):
        start = monotonic()
        builder.add(raw)
        build_s += monotonic() - start
        if n % s.BLOCK == 0:
            tick(n, 0)
        if n == s.BLOCK:
            until(lambda: (r / "early-ack.json").exists(), 120, "unwitnessed BUILD")
    start = monotonic()
    builder.save(r / "atlas.npz")
    del builder
    build_s += monotonic() - start
    start = monotonic()
    atlas = Atlas(r / "atlas.npz")
    mount_s = monotonic() - start
    topo = atlas.digest()
    d.check(
        not (r / "probe-source-open-attempted").exists()
        and atlas.geometry["byte_targets"] == d.read(r / "build.json")["targets"]
        and d.read(r / "early-ack.json") == dict(owner=owner, build=s.BLOCK),
        "build boundary mismatch",
    )
    d.write_json(
        r / "frozen.json",
        dict(
            spec=d.SPEC,
            rows_sha=d.sha(r / "reservation.json"),
            qualification_sha=d.sha(d.OUT / "qualified.json"),
            image_sha=d.sha(r / "atlas.npz"),
            build_sha=d.sha(r / "build.json"),
            topology_sha=topo,
            geometry=atlas.geometry,
            image_bytes=(r / "atlas.npz").stat().st_size,
            build_s=build_s,
            mount_s=mount_s,
        ),
    )
    until(lambda: (r / "freeze-ack.json").exists(), 120, "unwitnessed freeze")
    probe = list(d.windows("probe"))
    facts, sums = s.phase(atlas, probe, lambda n: tick(d.N["build"], n))
    del probe
    d.check(atlas.digest() == topo, "immutable index changed")
    path = r / "scores.npz"
    with path.open("xb") as fp:
        np.savez(fp, **sums)
    facts.update(
        scores_sha=d.sha(path),
        probe_sha=d.sha(r / "probe.json"),
        freeze_sha=d.sha(r / "frozen.json"),
    )
    d.write_json(r / "work.json", facts)
    z = score_receipts()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    z.update(
        spec=d.SPEC,
        freeze_sha=facts["freeze_sha"],
        work_sha=d.sha(r / "work.json"),
        runtime=dict(
            wall_s=monotonic() - tic,
            owner_cpu_s=ru.ru_utime + ru.ru_stime,
            child_cpu_s=ch.ru_utime + ch.ru_stime,
            peak_rss_kib=ru.ru_maxrss,
            gpu_card_s=0,
        ),
    )
    d.write_json(r / "result.json", z)
    d.write_json(
        r / "done.json",
        dict(
            owner=owner,
            targets=facts["targets"] * len(s.ARMS),
            result_sha=d.sha(r / "result.json"),
        ),
    )
    signal.alarm(0)


def observe():
    r = d.RUN
    owner = d.read(r / "owner.json")
    pid = owner["pid"]
    last = (0, 0)
    tic = monotonic()

    def alive():
        v = proc(pid)
        return (
            v
            and v["start"] == owner["start"]
            and v["ppid"] == 1
            and v["state"] != "Z"
            and os.getsid(pid) == pid
        )

    until(alive, 45, "owner not detached")
    d.write_json(r / "start-ack.json", owner)
    while not (r / "done.json").exists():
        if (r / "stop.json").exists():
            raise RuntimeError(d.read(r / "stop.json"))
        d.check(alive() and monotonic() - tic < 3700, "lost worker or deadline")
        if (r / "progress.json").exists():
            v = d.read(r / "progress.json")
            point = v["build"], v["probe"]
            d.check(
                v["owner"] == owner
                and last <= point
                and (not point[1] or point[0] == d.N["build"]),
                "invalid progress",
            )
            last = point
            if point == (s.BLOCK, 0) and not (r / "early-ack.json").exists():
                d.write_json(r / "early-ack.json", dict(owner=owner, build=s.BLOCK))
            if (
                point == (d.N["build"], 0)
                and (r / "frozen.json").exists()
                and not (r / "freeze-ack.json").exists()
            ):
                d.seal()
                d.check(
                    (r / "early-ack.json").exists()
                    and not (r / "probe-source-open-attempted").exists(),
                    "probe not sealed",
                )
                d.write_json(
                    r / "freeze-ack.json",
                    dict(owner=owner, sha=d.sha(r / "frozen.json")),
                )
        sleep(0.1)
    until(lambda: not alive(), 60, "owner still running")
    v = d.read(r / "progress.json")
    done = d.read(r / "done.json")
    d.check(
        v == dict(owner=owner, build=d.N["build"], probe=d.N["probe"])
        and done["owner"] == owner
        and done["result_sha"] == d.sha(r / "result.json")
        and done["targets"] == len(s.ARMS) * d.N["probe"] * (d.LENGTH["probe"] - 1)
        and not (r / "stop.json").exists(),
        "invalid close",
    )
    z = d.read(r / "result.json")
    rebuilt = score_receipts()
    d.check(all(z[k] == v for k, v in rebuilt.items()), "receipt-only replay mismatch")
    d.write_json(
        r / "witnessed.json",
        dict(
            owner_exited=True,
            receipt_only_recomputed=True,
            result_sha=done["result_sha"],
            first_build=s.BLOCK,
            build_at_seal=d.N["build"],
            probe_windows=d.N["probe"],
            terminal_targets=done["targets"],
            gpu_acquisitions=0,
        ),
    )
    print("Closed X53", done["result_sha"], flush=True)


def main():
    os.umask(0o077)
    d.cpu()
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=("qualify", "launch", "worker", "observe"))
    action = p.parse_args().action
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
