"""Invented device parity and clocks. No implicit acquisition, no quality experiment."""

from copy import deepcopy
from functools import partial
import gc
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from ..meter import await_ack, sync_clock
from ..window import BASE, ROOT, descendant, fields, get, need, put
from .attention import Local, dense, indexed, prepare
from .batch import TokenArchive
from .index import EndpointIndex
from .token_lm import CorpusTokenLM
from .triton_frontier import ResidentIndex


JOB = "corpus-kernel-meter"


def invented(ndocs, per, batch, device):
    rng = np.random.default_rng(5610)
    docs = []
    for i in range(ndocs):
        ids = rng.integers(11, 1023, size=per + 1)
        ids[7::32] = 1
        ids[6::32] = 2 + i % 3
        docs.append((f"ref-{i}", ids.tolist()))
    start = time.perf_counter()
    idx = EndpointIndex(docs, tokenizer="meter-native-ids-v1")
    build = time.perf_counter() - start
    req = []
    for i in range(batch):
        p = rng.integers(11, 1023, size=8 + i % 9).tolist()
        p[-1] = (1, 1023, 11, 15)[i % 4]
        req.append((f"live-{i}", p))
    torch.manual_seed(5611)
    m = CorpusTokenLM(1024, 32, 32, tokenizer=idx.tokenizer, leaf=128).to(device)
    a = TokenArchive(m, idx)
    targets = torch.tensor(rng.integers(0, 1024, size=batch), device=device)
    return m, a, req, targets, build


@torch.no_grad()
def fixed_reads(m, a, req, pkt):
    rows = a.rows(m)
    bank = prepare(*rows, m.omega)
    lk, lv, x = pkt.live.encode(m, need_rows=True, need_last=True)
    q = m.query(x)
    L = Local(lk, lv, q.new_zeros(len(lk)), pkt.live.lengths)
    null = (m.skip(x)[:, 0] + np.log(len(rows[1])), torch.zeros_like(x))
    p = pkt.plan
    funcs = {
        "gemm_exhaustive": partial(dense, q, *rows[1:], local=L, null=null),
        "indexed_torch": partial(
            indexed, q, bank, p, local=L, null=null, work=pkt.work
        ),
        "indexed_triton": partial(
            indexed, q, bank, p, local=L, null=null, work=pkt.work, backend="triton"
        ),
    }
    ref = funcs["indexed_torch"]()
    actual = funcs["indexed_triton"]()
    full = funcs["gemm_exhaustive"]()
    for key in vars(actual):
        torch.testing.assert_close(
            getattr(ref, key), getattr(actual, key), rtol=4e-5, atol=1e-5
        )
    clocks = {k: sync_clock(f, 8, 5) for k, f in funcs.items()}
    lens = torch.tensor(
        [len(seq) for _, seq in req], device=q.device, dtype=torch.int32
    )
    width = max(map(lambda v: len(v[1]), req))
    x = torch.tensor(
        [seq + [0] * (width - len(seq)) for _, seq in req], device=q.device
    )
    ri = ResidentIndex(a.index, q.device)
    seek = partial(ri.intervals, x, lens)
    found = seek().cpu().tolist()
    need(
        found == [list(a.index.frontier(seq).hit_range(1)) for _, seq in req],
        "real GPU locator mismatch",
    )
    return dict(
        fixed_read=clocks,
        locator=sync_clock(seek, 16, 5),
        corpus_share_error_mean=float(
            (actual.shares[:, 0] - full.shares[:, 0]).abs().mean()
        ),
        indexed_v_dense_read_rms=float(
            (actual.read - full.read).square().mean().sqrt()
        ),
        exact_rows=sum(b - z for _, z, b in p.exact),
        tail_edge_rows=sum(b - z for _, z, b in p.edge),
        summary_nodes=len(p.nodes),
        total_row_pairs=p.layout.rows * p.batch,
    )


def updates(origin, pkt, targets):
    saved = None
    clocks = {}
    for name, mode, backend in [
        ("gemm_exhaustive", "dense", "torch"),
        ("indexed_torch", "indexed", "torch"),
        ("indexed_triton", "indexed", "triton"),
    ]:
        m = deepcopy(origin)
        opt = torch.optim.AdamW(m.parameters(), lr=0.0005, fused=True)

        def step():
            opt.zero_grad(set_to_none=True)
            p = pkt.predict(m, mode=mode, backend=backend).probabilities
            loss = -p[torch.arange(len(p), device=p.device), targets].log().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0, foreach=True)
            opt.step()
            return loss.detach()

        loss = step()
        need(torch.isfinite(loss).item(), "nonfinite training step")
        if name == "indexed_torch":
            saved = (
                loss.clone(),
                [(x.detach().clone(), x.grad.clone()) for x in m.parameters()],
            )
        if name == "indexed_triton":
            torch.testing.assert_close(loss, saved[0], rtol=4e-5, atol=1e-5)
            for p, (ref, g) in zip(m.parameters(), saved[1], strict=True):
                torch.testing.assert_close(p, ref, rtol=2e-4, atol=4e-5)
                torch.testing.assert_close(p.grad, g, rtol=4e-4, atol=5e-6)
        clocks[name] = sync_clock(step, 4, 5)
    return clocks


def main(d):
    gpu = os.getenv("GPU_LEASE_HELD")
    need(
        d == BASE / JOB
        and gpu in ("0", "1")
        and os.getenv("CUDA_VISIBLE_DEVICES") == gpu
        and os.getenv("TRITON_INTERPRET") != "1",
        "not the admitted corpus meter lease",
    )
    owner = get(d / "owner.json")
    lock = fields(ROOT / ".wg/gpu_leases" / gpu)
    need(
        descendant(os.getpid(), owner["pid"])
        and (lock["pid"], lock["starttime"]) == (str(owner["pid"]), owner["start"]),
        "lease identity mismatch",
    )
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    unit = 0

    def tick(label):
        nonlocal unit
        unit += 1
        put(d / "progress.json", dict(units=unit, label=label, pid=os.getpid()))

    with (d / "device-tests.log").open("xb") as f:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                str(Path(__file__).with_name("test_triton.py")),
            ],
            stdout=f,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, CORPUS_KERNEL_DEVICE="cuda"),
            check=True,
        )
    tick("device parity and three updates")
    results = {}
    for nd, per, bs in [(128, 128, 64), (512, 256, 128)]:
        m, a, req, y, build = invented(nd, per, bs, "cuda")
        torch.cuda.synchronize()
        start = time.perf_counter()
        pkt = a.batch(req, locator="triton")
        torch.cuda.synchronize()
        planning = time.perf_counter() - start
        tick("cached native-token routes")
        out = fixed_reads(m, a, req, pkt)
        out.update(cpu_index_build_s=build, integer_packet_and_plan_s=planning)
        tick("frozen bank and locator clocks")
        if nd == 128:
            out["full_step"] = updates(m, pkt, y)
            tick("end-to-end current-weight optimizer clocks")
        results[f"{nd * per}x{bs}"] = out
        put(d / "clocks.json", results)
        del pkt, m, a
        gc.collect()
        torch.cuda.empty_cache()
    put(
        d / "complete.json",
        dict(
            units=unit,
            cells=results,
            device=torch.cuda.get_device_name(0),
            dtype="float32",
            torch=str(torch.__version__),
            natural_documents=0,
        ),
    )
    await_ack(d)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
