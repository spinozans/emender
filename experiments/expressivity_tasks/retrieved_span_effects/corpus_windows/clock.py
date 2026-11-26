"""Matched-real-target engineering clock. Invented tokens, fixed routes, one update per cell.

GPU callable only inside a separately admitted, witnessed meter; the standalone
CLI is CPU-only. No implicit lease, natural data, or success threshold on speed.
"""

from copy import deepcopy
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

from ..corpus_kernel.batch import TokenArchive
from ..corpus_kernel.index import EndpointIndex
from ..corpus_kernel.token_lm import CorpusTokenLM
from .packet import WindowArchive


def make(nd, per, live, context, device):
    rng = np.random.default_rng(5687)
    docs = []
    for i in range(nd):
        p = rng.integers(11, 1024, size=per + 1)
        p[6::32] = 2 + i % 3
        p[7::32] = 1
        docs.append((f"building-{i}", p.tolist()))
    req = []
    for i in range(live):
        p = rng.integers(11, 1024, size=context + 1 - i % 3)
        p[6::16] = 2 + i % 3
        p[7::16] = 1
        req.append((f"window-{i}", p.tolist()))
    idx = EndpointIndex(docs, tokenizer="clock-5687")
    torch.manual_seed(5686)
    m = CorpusTokenLM(1024, 32, 32, tokenizer=idx.tokenizer, leaf=128).to(device)
    return m, idx, req


def run(
    *,
    device="cpu",
    nd=128,
    per=128,
    live=12,
    context=32,
    chunk=64,
    blocks=5,
    progress=None,
):
    if device == "cuda" and not (
        os.getenv("GPU_LEASE_HELD") in ("0", "1")
        and os.getenv("GPU_LEASE_HELD") == os.getenv("CUDA_VISIBLE_DEVICES")
        and os.getenv("TRITON_INTERPRET") != "1"
    ):
        raise RuntimeError("meter requires an admitted lease")
    if device not in ("cpu", "cuda"):
        raise ValueError("unknown clock device")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    m, idx, docs = make(nd, per, live, context, device)
    req = [(n, p[:t]) for n, p in docs for t in range(1, len(p))]
    n = len(req)
    labels = torch.tensor([t for _, p in docs for t in p[1:]], device=device)
    t = time.perf_counter()
    arch = WindowArchive(m, idx)
    pkt = arch.windows(
        docs, chunk_targets=chunk, locator="triton" if device == "cuda" else "cpu"
    )
    sync()
    packet_s = time.perf_counter() - t
    before = time.perf_counter()
    orig = TokenArchive(m, idx)
    one = orig.batch(req)
    many = [orig.batch(req[i : i + chunk]) for i in range(0, n, chunk)]
    sync()
    old_setup = time.perf_counter() - before
    clocks = {}
    checks = {}
    references = {}
    cases = [
        "rebuild_per_block",
        "all_prefixes_one_bank",
        "packed_indexed",
        "packed_dense",
        "packed_checkpoint",
        "packed_stream",
        "packed_dense_stream",
    ] + (
        [
            "triton_indexed",
            "triton_dense",
            "triton_checkpoint",
            "triton_stream",
            "triton_dense_stream",
        ]
        if device == "cuda"
        else []
    )
    for name in cases:
        model = deepcopy(m)
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=0.0005,
            **({"fused": True} if device == "cuda" else {}),
        )
        mode = "dense" if "dense" in name.split("_") else "indexed"
        backend = "triton" if name.startswith("triton") else "torch"

        def step():
            opt.zero_grad(set_to_none=True)
            if name in ("rebuild_per_block", "all_prefixes_one_bank"):
                start = 0
                ls = []
                for pack in many if name == "rebuild_per_block" else (one,):
                    pp = pack.predict(model).probabilities
                    y = labels[start : start + len(pp)]
                    loss = -pp.gather(1, y[:, None]).log().sum() / n
                    loss.backward()
                    ls.append(loss.detach())
                    start += len(pp)
                lsum = torch.stack(ls).sum()
            elif name.endswith("stream"):
                lsum = pkt.backward_nll(model, mode=mode, backend=backend)
            else:
                loss = pkt.loss(
                    model,
                    mode=mode,
                    backend=backend,
                    checkpoint_blocks=name.endswith("checkpoint"),
                )
                loss.backward()
                lsum = loss.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=True)
            opt.step()
            return lsum

        init_loss = step()
        assert torch.isfinite(init_loss).item()
        if mode not in references:
            references[mode] = (
                init_loss.clone(),
                [(p.detach().clone(), p.grad.clone()) for p in model.parameters()],
            )

        def check(baseline):
            torch.testing.assert_close(init_loss, baseline[0], rtol=3e-5, atol=3e-6)
            maxg = 0.0
            maxp = 0.0
            for p, (ref, g) in zip(model.parameters(), baseline[1], strict=True):
                torch.testing.assert_close(p.grad, g, rtol=5e-4, atol=3e-6)
                torch.testing.assert_close(p, ref, rtol=2e-4, atol=3e-5)
                maxg = max(maxg, (p.grad - g).abs().max().item())
                maxp = max(maxp, (p - ref).abs().max().item())
            checks[name] = dict(
                loss_abs=(init_loss - baseline[0]).abs().item(),
                gradient_max_abs=maxg,
                parameter_max_abs=maxp,
            )

        check(references[mode])
        step()
        sync()
        times = []
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        for _ in range(blocks):
            start = time.perf_counter()
            step()
            sync()
            times.append((time.perf_counter() - start) * 1000)
        ms = statistics.median(times)
        clocks[name] = dict(
            median_ms=ms, real_targets_per_s=n * 1000 / ms, block_ms=times
        )
        if device == "cuda":
            clocks[name]["extra_peak_bytes"] = torch.cuda.max_memory_allocated() - base
            clocks[name]["base_alloc_bytes"] = base
        if progress is not None:
            progress(name)
    return dict(
        archive_rows=len(idx),
        windows=live,
        real_targets=n,
        query_block_limit=chunk,
        blocks=len(pkt.blocks),
        local_rows_packed=len(pkt.local_rows),
        local_rows_expanded=sum(len(p) - 1 for _, p in req),
        read_blocks_exact_pairs=sum(
            b - a for block in pkt.blocks for _, a, b in block.plan.exact
        ),
        window_packet_and_archive_bind_s=packet_s,
        legacy_bind_both_packets_s=old_setup,
        clocks=clocks,
        same_law_first_update_parity=checks,
        device=device,
        torch=str(torch.__version__),
        natural_documents=0,
        scope="Invented FP32; same ordered targets and one optimizer update. Immutable packets prepared outside clock; learned bank construction, prediction, token loss, all gradients, clipping and AdamW inside.",
    )


if __name__ == "__main__":
    if (
        not __debug__
        or os.getenv("CUDA_VISIBLE_DEVICES") != ""
        or os.getenv("GPU_LEASE_HELD")
        or os.getenv("TRITON_INTERPRET") == "1"
        or any(
            os.getenv(k) != "1"
            for k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS")
        )
    ):
        raise RuntimeError("standalone clock requires one-thread CPU only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = run()
    with Path(sys.argv[1]).open("x") as f:
        json.dump(out, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
