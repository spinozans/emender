"""Single synthetic CPU parity gate followed by the registered three-fixture writer timings. No LM data."""

from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from time import perf_counter_ns
import numpy as np
import torch
from . import joint_reader as joint, literal_mark as ls, native_text as nt

HERE = Path(__file__).resolve().parent
TESTS = (
    "test_joint_reader.py",
    "test_literal_mark.py",
    "test_surprise.py",
    "test_alternative.py",
)


def insist(condition, message):
    if not condition:
        raise ValueError(message)


def source():
    insist(
        os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not os.environ.get("GPU_LEASE_HELD")
        and os.environ.get("OPENBLAS_NUM_THREADS")
        == os.environ.get("OMP_NUM_THREADS")
        == "1"
        and platform.node() == "lambda01"
        and not torch.cuda.is_initialized(),
        "registered CPU-only local environment required",
    )
    insist(
        nt.digest(HERE / "JOINT_READER.md") == joint.REG
        and not subprocess.check_output(
            ["git", "status", "--porcelain", "--", str(HERE)]
        ).strip(),
        "changed or dirty joint reader source",
    )
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def write(path, z):
    with path.open("x") as f:
        json.dump(z, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def fixtures():
    w = b"color: blue\ncolor: black\n\x00\xff"
    return {
        "zero": bytes(4096),
        "periodic": (w * 4096)[:4096],
        "uniform": np.random.default_rng(4911)
        .integers(0, 256, 4096, dtype=np.uint8)
        .tobytes(),
    }


def timing(rows):
    p = np.array([row["python_ns"] for row in rows])
    c = np.array([row["native_ns"] for row in rows])
    insist(
        len(rows) == 31
        and np.all(p > 0)
        and np.all(c > 0)
        and [r["first"] for r in rows]
        == ["native" if i % 2 else "python" for i in range(31)],
        "incomplete paired trials",
    )
    return dict(
        python_median_ms=float(np.median(p) * 1e-6),
        native_median_ms=float(np.median(c) * 1e-6),
        python_p90_ms=float(np.quantile(p, 0.9, method="linear") * 1e-6),
        native_p90_ms=float(np.quantile(c, 0.9, method="linear") * 1e-6),
        median_paired_python_over_native=float(np.median(p / c)),
    )


def run():
    ref = source()
    os.umask(0o077)
    torch.set_num_threads(1)
    d = joint.COST
    d.mkdir(mode=0o700)
    folder = d / "build"
    note = nt.build(folder)
    f = d / "parity.log"
    env = dict(
        os.environ, X49_JOINT_TEST_BUILD=str(folder.resolve()), PYTEST_ADDOPTS=""
    )
    with f.open("xb") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                *(str(HERE / test) for test in TESTS),
            ],
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    passed = re.findall(r"=+ (\d+) passed in [^\n]+ =+", f.read_text())
    insist(passed == ["21"], "incomplete joint reader parity and reference tests")
    native = nt._load(folder, note)
    writers = dict(python=ls.text_tape, native=native)
    stats = {}
    times = {}
    words = fixtures()
    for name, word in words.items():
        old, x = ls.text_tape(word)
        new, y = native(word)
        insist(
            torch.equal(old, new) and x == y,
            "incorrect native tape before timing: " + name,
        )
        stats[name] = dict(
            input_sha256=hashlib.sha256(word).hexdigest(),
            tape_sha256=hashlib.sha256(bytes(old.numpy())).hexdigest(),
            work=x,
        )
    for name, word in words.items():
        for _ in range(2):
            for fn in writers.values():
                fn(word)
        rows = []
        for i in range(31):
            order = ("native", "python") if i % 2 else ("python", "native")
            row = dict(first=order[0])
            for label in order:
                fn = writers[label]
                tic = perf_counter_ns()
                out = fn(word)
                elapsed = perf_counter_ns() - tic
                del out
                row[label + "_ns"] = elapsed
            rows.append(row)
        times[name] = rows
    log = d / "paired_writer_times.json"
    write(log, times)
    closed = json.loads(log.read_bytes())
    insist(
        note == json.loads((folder / "build.json").read_bytes())
        and ref == source()
        and nt.digest(folder / nt.BINARY) == note["library_sha256"]
        and times == closed,
        "changed source, binary or trial evidence",
    )
    result = dict(
        scope="CPU-only synthetic exact literal writer, not LM quality or end-to-end throughput",
        qualified_local_source=ref,
        protocol_sha256=joint.REG,
        ls_result_sha256=joint.RESULT,
        build=note,
        tests=list(TESTS),
        pytest_passed=int(passed[0]),
        parity_log_sha256=nt.digest(f),
        paired_trials_sha256=nt.digest(log),
        trials_per_fixture=31,
        warmups_per_writer_fixture=2,
        synthetic_parity=True,
        natural_documents=0,
        acquisitions=0,
        cpu_threads=1,
        cpu_arch=platform.machine(),
        cpu_model=next(
            line.split(":", 1)[1].strip()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("model name")
        ),
        affinity_cpus=len(os.sched_getaffinity(0)),
        python=platform.python_version(),
        numpy=np.__version__,
        torch=str(torch.__version__),
        fixtures={k: dict(**stats[k], **timing(closed[k])) for k in words},
    )
    write(d / "aggregate.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), allow_nan=False, sort_keys=True), flush=True)
