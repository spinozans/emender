"""Check real-device training parity before any same-target update clocks. Synthetic only."""

import os
from pathlib import Path
import re
import subprocess
import sys

import torch
import triton

from ..meter import await_ack
from ..window import ROOT, descendant, fields, get, need, put
from .clock import run
from .device_gate import D


def main(d):
    gpu = os.getenv("GPU_LEASE_HELD")
    need(
        d == D
        and gpu in ("0", "1")
        and os.getenv("CUDA_VISIBLE_DEVICES") == gpu
        and os.getenv("TRITON_INTERPRET") != "1",
        "requires admitted window meter",
    )
    owner = get(d / "owner.json")
    lock = fields(ROOT / ".wg/gpu_leases" / gpu)
    need(
        (lock["pid"], lock["starttime"], lock["adopted"])
        == (str(owner["pid"]), owner["start"], "1")
        and descendant(os.getpid(), owner["pid"])
        and torch.cuda.device_count() == 1,
        "foreign meter worker",
    )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    units = 0

    def tick(label):
        nonlocal units
        units += 1
        put(d / "progress.json", dict(units=units, label=label, pid=os.getpid()))

    target = Path(__file__).with_name("test_kernels.py")
    log = d / "device-tests.log"
    with log.open("xb") as f:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(target)],
            cwd=ROOT,
            check=True,
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(
                os.environ, CORPUS_WINDOWS_TEST_DEVICE="cuda", NVIDIA_TF32_OVERRIDE="0"
            ),
        )
    need(re.search(r"\b12 passed\b", log.read_text()), "device tests missing")
    tick("12 real device fixtures passed")
    clocks = run(device="cuda", progress=tick)
    expected = {
        "rebuild_per_block",
        "all_prefixes_one_bank",
        "packed_indexed",
        "packed_dense",
        "packed_checkpoint",
        "packed_stream",
        "triton_indexed",
        "triton_dense",
        "triton_checkpoint",
        "triton_stream",
        "packed_dense_stream",
        "triton_dense_stream",
    }
    need(
        set(clocks["clocks"]) == set(clocks["same_law_first_update_parity"]) == expected
        and clocks["real_targets"] == 372
        and units == 13,
        "incomplete paired clock",
    )
    put(
        d / "complete.json",
        dict(
            units=units,
            parity_cases=12,
            hardware=torch.cuda.get_device_name(0),
            triton=triton.__version__,
            work=clocks,
        ),
    )
    await_ack(d)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
