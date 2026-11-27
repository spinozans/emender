"""SAL-1 paths, gates and scope. Never debit or extend any other window."""

import hashlib
import os
import subprocess
from pathlib import Path

from ..window import ROOT, HERE as PARENT, get, need

HERE = Path(__file__).resolve().parent
BASE = ROOT / "output/retrieved-span-effects/sal-1"
DATA = BASE / "data"
JOB = BASE / "device"
N = {"build": 2048, "fit": 512, "probe": 128}
ARMS = ("local", "skeleton", "order2")
STAGE2_ARMS = ("local", "skeleton")  # amendment A1
SEEDS = (5751, 5752)
CAMPAIGN_CAP = 15000
CAMPAIGN_END = 1795845600  # 2026-11-28 06:00 UTC; the operator's one-day mandate


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
    deps = [
        str(HERE.relative_to(ROOT)),
        str(src / "language_compression"),
        str(src / "corpus_windows"),
        str(src / "corpus_kernel"),
        str(src / "window.py"),
        "scripts/gpu_lease.sh",
    ]
    need(
        q["cpu_tests"] > 0
        and q["natural_memberships"] == q["natural_documents"] == 0
        and q["tests_sha256"] == sha(BASE / "sal-phase-a-final.log")
        and subprocess.check_output(
            ["git", "rev-parse", ref + "^"], cwd=ROOT, text=True
        ).strip()
        == q["registered"]
        and not subprocess.run(
            ["git", "diff", "--quiet", ref, "--", *deps], cwd=ROOT
        ).returncode
        and not subprocess.check_output(
            ["git", "status", "--porcelain", "--", *deps], cwd=ROOT, text=True
        ).strip(),
        "no clean public pre-access qualification",
    )
    return ref
