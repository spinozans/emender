"""One separately scoped, fail-closed local run. Never debit or extend the expired window."""

import hashlib
import os
from pathlib import Path
import subprocess

from ..window import ROOT, HERE as PARENT, get, put as put, need

HERE = Path(__file__).resolve().parent
BASE = ROOT / "output/retrieved-span-effects/language-compression-1"
DATA = BASE / "data"
JOB = BASE / "device"
REG = "514df0bd58712e67a44d5e225800346da8c3b42c"
CAP = 900
END = 1795755600
N = {"build": 128, "fit": 256, "probe": 128}
ARMS = ("local", "count", "exact", "indexed")
SEEDS = (5751, 5752)
OLD = "e46d4e122cc2027b56efdb9a7dd39c3c0e6a1cd8bfff0846fe31b743dee5cc98"
MAN = "48b9095eeccb023ff1cb2a4f6da899a8b3e4eb3b76b2804e730f1d6ad031c414"


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu():
    need(
        __debug__
        and os.uname().nodename == "lambda01"
        and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
        and not any(
            k in os.environ
            for k in ("GPU_LEASE_HELD", "PYTHONOPTIMIZE", "TRITON_INTERPRET")
        )
        and all(
            os.environ.get(k) == "1"
            for k in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        ),
        "explicit single-thread CPU checkout required",
    )


def qualified():
    ref = (BASE / "qualified-ref").read_text().strip()
    q = get(HERE / "receipts/qualification.json")
    src = PARENT.relative_to(ROOT)
    p = HERE.relative_to(ROOT)
    deps = [
        str(p),
        str(src / "corpus_windows"),
        str(src / "corpus_kernel"),
        str(src / "window.py"),
        "scripts/gpu_lease.sh",
    ]
    need(
        q["registered"] == REG
        and q["natural_memberships"] == q["natural_documents"] == 0
        and q["cpu_tests"] > 0
        and sha(BASE / "cpu.log") == q["tests_sha256"]
        and subprocess.check_output(
            ["git", "rev-parse", ref + "^"], cwd=ROOT, text=True
        ).strip()
        == REG
        and not subprocess.run(
            ["git", "diff", "--quiet", ref, "--", *deps], cwd=ROOT
        ).returncode
        and not subprocess.check_output(
            ["git", "status", "--porcelain", "--", *deps], cwd=ROOT
        ),
        "no clean public pre-access qualification",
    )
    return ref
