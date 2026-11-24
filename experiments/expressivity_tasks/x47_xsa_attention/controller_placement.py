"""S8: place only the small controller and its workbench on CPU, not the causal host."""

import argparse
import copy
import json
import os
from pathlib import Path
import torch
from ndm.memory_program.resident_host import ResidentArena
from ndm.memory_program.serving import place_inputs
from .large_host_bench import prepare_large, PREFIX, LENGTH
from .packed_program_check import exercise
from .serving_bench import measure
from .wide_host_bench import Bound
from .workbench_run import dump
from .workbench_witness import digest_state


def check():
    out = exercise("cuda", candidate_device="cpu")
    h = torch.randn(3, 6, device="cuda", requires_grad=True)
    p = torch.randn(3, 256, device="cuda", requires_grad=True)
    x, y = place_inputs(h, p, width=6, device="cpu")
    assert (
        x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr()
        and torch.equal(x, h.cpu())
        and torch.equal(y, p.cpu())
    )
    (x.sum() + y.sum() * 3).backward()
    assert h.grad.eq(1).all() and p.grad.eq(3).all()
    with torch.no_grad():
        h.add_(1)
        assert not torch.equal(x, h.cpu())
    out["single_packet_adjoint"] = True
    return out


def main(run):
    net, eager, fused, selected, windows, manifest = prepare_large(run)
    cpu = copy.deepcopy(eager).cpu()
    assert digest_state(cpu) == digest_state(eager) == digest_state(fused)
    checks = check()
    dump(run / "device-parity.json", checks)
    print("cross-device fixed-program check passed", flush=True)
    models = {"gpu-triton": fused, "gpu-eager": eager, "cpu-eager": cpu}
    total = 0
    with (
        torch.no_grad(),
        ResidentArena(net, 16, 64, (1, 16), byte_budget=256 << 20, private_budget=64 << 20) as pool,
    ):

        def host(model, ss, lives):
            return Bound(pool, model, ss, lives)

        for block in range(2):
            for b in (1, 16):
                ix = list(range(16 * block, 16 * block + b))
                ws = [windows[i] for i in ix]
                arms = [
                    (m, e) for m in ("greedy", "sample") for e in ("gpu-triton", "gpu-eager", "cpu-eager")
                ]
                for m, e in arms[::-1] if block else arms:
                    total += measure(
                        run,
                        net,
                        models[e],
                        f"fused-{m}",
                        block,
                        [selected[i] for i in ix],
                        ws,
                        ix,
                        manifest,
                        total,
                        label=f"{e}-{m}",
                        host_factory=host,
                        length=LENGTH,
                        prefix=PREFIX,
                    )
    dump(
        run / "progress.json", dict(cell="all", phase="complete", total_emitted=total, server_pid=os.getpid())
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    main(p.parse_args().run)
