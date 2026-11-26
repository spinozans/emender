"""Invented work only. Synchronized device+host wall clocks include dispatch."""

import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import triton
from torch.nn import functional as F

from .accelerated import AcceleratedJointLM
from .joint_fixture import JointLM, episodes
from .triton_read import dual_edges, merge, terminal
from .triton_scan import pool


def put(p, obj):
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, allow_nan=False) + "\n")
    tmp.replace(p)


def sync_clock(call, repeats, blocks=7):
    for _ in range(4):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    times = []
    for _ in range(blocks):
        start = time.perf_counter()
        for _ in range(repeats):
            call()
        torch.cuda.synchronize()
        times.append(1000 * (time.perf_counter() - start) / repeats)
    return dict(
        ms=statistics.median(times),
        block_ms=times,
        extra_peak_bytes=torch.cuda.max_memory_allocated() - base,
        base_alloc_bytes=base,
    )


def read_bench(shape):
    g, p, t, k, v = shape
    dev = "cuda"
    torch.manual_seed(7710)
    raw = torch.randn(g, p, t, 3 * k + v, device=dev)
    key, d, r, u = raw.split((k, k, k, v), -1)
    key = F.normalize(key, dim=-1)
    c = torch.cat((key, (d + 2).sigmoid(), r.sigmoid() * key, u), -1).requires_grad_()
    s = torch.randn(g, k, v, device=dev, requires_grad=True)
    q = torch.randn(g, k, device=dev, requires_grad=True)
    weight = torch.randn(g, p, device=dev).softmax(-1).requires_grad_()
    lengths = torch.full((g, p), t, device=dev, dtype=torch.int32)
    up = torch.randn(g, v, device=dev)

    def step(kind):
        if kind == "dense":
            m = s[:, None]
            for z in c.unbind(2):
                a, d, r, u = z.split((k, k, k, v), -1)
                m = m * d.unsqueeze(-1)
                m = m + a.unsqueeze(-1) * (u - (m * r.unsqueeze(-1)).sum(-2)).unsqueeze(
                    -2
                )
            y = torch.einsum("gk,gkv->gv", q, (weight[:, :, None, None] * m).sum(1))
        elif kind == "matrix":
            y = torch.einsum("gk,gkv->gv", q, pool(terminal(c, s, lengths), weight))
        else:
            y = merge(dual_edges(c, q, lengths, int(kind)), weight, s)
        grads = torch.autograd.grad(y, (c, q, weight, s), up)
        return (y.detach(), *grads)

    reference = step("dense")
    out = {}
    for kind in ("dense", "matrix", "1", "4", "8", "16"):
        got = step(kind)
        for z, ref in zip(got, reference, strict=True):
            torch.testing.assert_close(z, ref, atol=8e-5, rtol=4e-4)
        out[kind] = sync_clock(lambda: step(kind), 4 if kind == "dense" else 12)
    return out


def full_steps():
    b = episodes(torch.Generator().manual_seed(7711), 256)
    prompt = b.prompt.cuda()
    pages = b.pages.cuda()
    target = b.answer(b.pages).cuda()
    result = {}
    for kind in ("dense", "matrix", "dual"):
        torch.manual_seed(7712)
        model = (
            JointLM() if kind == "dense" else AcceleratedJointLM(backend=kind)
        ).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=0.003, fused=True)

        def step():
            opt.zero_grad(set_to_none=True)
            y, _ = model(prompt, pages, "effect")
            loss = F.cross_entropy(y, target)
            loss.backward()
            opt.step()
            return loss.detach()

        result[kind] = sync_clock(step, 12)
    return result


def main(folder):
    if (
        not os.environ.get("GPU_LEASE_HELD")
        or os.environ.get("CUDA_VISIBLE_DEVICES") != os.environ["GPU_LEASE_HELD"]
        or os.environ.get("TRITON_INTERPRET") == "1"
    ):
        raise RuntimeError("real leased device required")
    folder = Path(folder)
    num = 0

    def progress(label):
        nonlocal num
        num += 1
        put(folder / "progress.json", dict(units=num, label=label, pid=os.getpid()))

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    env = dict(os.environ, SPAN_TEST_DEVICE="cuda")
    src = Path(__file__).with_name("test_triton_read.py")
    with (folder / "device-tests.log").open("xb") as f:
        subprocess.run(
            [sys.executable, "-m", "pytest", "-q", str(src)],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )
    progress("device VJP and optimizer parity")
    result = {}
    for shape in [(64, 4, 16, 16, 16), (256, 4, 64, 32, 32), (512, 4, 32, 32, 32)]:
        result[str(shape)] = read_bench(shape)
        put(folder / "read-clocks.json", result)
        progress(str(shape))
    steps = full_steps()
    put(folder / "step-clocks.json", steps)
    progress("optimizer steps")
    put(
        folder / "complete.json",
        dict(
            read=result,
            steps=steps,
            units=num,
            device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            triton=triton.__version__,
            dtype="fp32",
            pass_clock=min(steps[k]["ms"] for k in ("matrix", "dual"))
            < steps["dense"]["ms"],
        ),
    )


def await_ack(folder, name="watch-ack"):
    stop = time.monotonic() + 20
    while not (Path(folder) / name).is_file():
        if time.monotonic() > stop:
            raise RuntimeError("missing independent witness: " + name)
        time.sleep(0.1)


if __name__ == "__main__":
    main(sys.argv[1])
    await_ack(sys.argv[1])
