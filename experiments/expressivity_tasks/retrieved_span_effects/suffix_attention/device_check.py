"""SAL-1 stage 1: device parity on invented bytes, then full-geometry clocks.

Invented data only: the parity check trains one complete update per arm on a
small archive with a CPU reference, and the clocks run complete updates at the
full 8,388,608-row archive geometry on invented bytes. No natural payload is
opened in this stage. Writes meter.json with per-arm update times, peak
memory, and the projected stage-2 budget.
"""

import os
import statistics
import time

import numpy as np
import torch

from ..window import ROOT, descendant, fields, get, need, put
from .model import SalLM, SalPack
from .settings import JOB, SEEDS


def lease_identity():
    gpu = os.environ.get("GPU_LEASE_HELD")
    owner = get(JOB / "owner.json")
    need(
        gpu in ("0", "1")
        and gpu == os.getenv("CUDA_VISIBLE_DEVICES")
        and torch.cuda.device_count() == 1
        and descendant(os.getpid(), owner["pid"]),
        "real leased device required",
    )
    lock = fields(ROOT / ".wg/gpu_leases" / gpu)
    need(
        (lock["pid"], lock["starttime"], lock["adopted"])
        == (str(owner["pid"]), owner["start"], "1"),
        "foreign lease",
    )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return torch.device("cuda"), owner


def parity(dev):
    """One complete update per arm: CUDA must agree with the CPU reference."""
    from . import run

    rng = np.random.default_rng(5777)
    build = rng.integers(0, 5, (1, 129), dtype=np.uint8)
    live = rng.integers(0, 5, (2, 65), dtype=np.uint8)
    losses = {}
    grads = {}
    for label, device in (("cpu", "cpu"), ("cuda", dev)):
        m = SalLM(5777).to(device)
        pack = SalPack(build, m, batch=2)
        b = pack.bind(live)
        from ..corpus_windows.packet import WindowArchive

        per = {}
        for arm in ("local", "skeleton", "order2"):
            m2 = SalLM(5777).to(device)
            o2 = run.make_optimizer(m2, device)
            arch = WindowArchive(m2, pack.index) if arm == "order2" else None
            y = run.update(m2, o2, b, arm, arch)
            per[arm] = float(y)
            if arm == "order2":
                grads[label] = {
                    k: v.grad.detach().cpu().clone()
                    for k, v in m2.named_parameters()
                    if v.grad is not None
                }
        losses[label] = per
    for arm in ("local", "skeleton", "order2"):
        a, b = losses["cpu"][arm], losses["cuda"][arm]
        need(
            abs(a - b) <= 2e-4 * max(1.0, abs(a)), f"{arm} loss parity failed: {a} {b}"
        )
    for k in grads["cpu"]:
        g1, g2 = grads["cpu"][k], grads["cuda"][k]
        torch.testing.assert_close(g1, g2, rtol=1e-2, atol=1e-4)
    return losses


def clocks(dev):
    """Complete warm updates at full archive geometry, invented bytes."""
    from . import run

    rng = np.random.default_rng(5778)
    build = rng.integers(0, 256, (2048, 4097), dtype=np.uint8)
    live = rng.integers(0, 256, (16, 65), dtype=np.uint8)
    m = SalLM(SEEDS[0]).to(dev)
    t0 = time.monotonic()
    pack = SalPack(build, m, batch=16)
    prep_s = time.monotonic() - t0
    b = pack.bind(live)
    from ..corpus_windows.packet import WindowArchive

    out = {}
    peaks = {}
    for arm in ("local", "skeleton", "order2"):
        o = run.make_optimizer(m, dev)
        arch = WindowArchive(m, pack.index) if arm == "order2" else None
        for _ in range(2 if arm != "order2" else 1):
            run.update(m, o, b, arm, arch)
        if dev.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        samples = []
        for _ in range(3):
            t1 = time.monotonic()
            run.update(m, o, b, arm, arch)
            if dev.type == "cuda":
                torch.cuda.synchronize()
            samples.append(time.monotonic() - t1)
        out[arm] = statistics.median(samples)
        if dev.type == "cuda":
            peaks[arm] = dict(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                incremental_bytes=torch.cuda.max_memory_allocated() - base,
            )
    return dict(
        archive_rows=pack.rows,
        leaves=pack.leaves,
        preparation_s=prep_s,
        update_s=out,
        memory=peaks,
        projected_stage2_card_s=int(
            2 * 2048 * (out["order2"] + out["local"] + out["skeleton"])
            + 512 * (out["order2"] + out["local"] + out["skeleton"])
            + 600
        ),
    )


def main():
    dev, owner = lease_identity()
    stage = get(JOB / "admission.json")["stage"]
    units = 0

    def tick(label):
        nonlocal units
        units += 1
        put(
            JOB / f"progress-{stage}.json",
            dict(units=units, pid=os.getpid(), label=label),
        )

    tick("verified adopted CUDA device")
    losses = parity(dev)
    tick("invented per-arm update parity passed")
    meter = clocks(dev)
    meter["parity_losses"] = losses
    put(JOB / f"meter-{stage}.json", meter)
    tick("full-geometry invented clocks recorded")
    print(meter["update_s"], flush=True)
    print("STAGE 1 OK", flush=True)


if __name__ == "__main__":
    main()
