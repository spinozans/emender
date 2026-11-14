"""Block vs block: the matrix memory block against the GDN-2 delta-rule block.

Both blocks are measured with the *same* method at *identical* shapes: one block,
forward + backward of ``out.pow(2).sum()``, eager (no ``torch.compile``, no CUDA
graph), batch 8 x 2048 bytes x width 576, CUDA events, one warm-up then three
reps, mean and p50 reported.  The states are the same size -- GDN-2's state is
``d_k x d_v`` per head = 128 x 128 with two heads, which is exactly the matrix
store's ``rank x d_v`` = 128 x 128 with two heads -- so the two blocks are
comparable on the axis the registration calls capacity.

Measurement boundary: ONE MEMORY BLOCK, not a model and not a step.  Nothing here
is comparable to a whole-arm tokens/s number.

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x45_matrix_emender.bench_blocks \
        --out receipts/block_vs_block.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x45_matrix_emender.model import (
    MatrixShape, MatrixMixer, matrix_config)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent


def measure(block, x, reps, device="cuda"):
    """Forward + backward of one block, timed with CUDA events (mean and p50)."""
    def once():
        out = block(x)
        if isinstance(out, tuple):
            out = out[0]
        torch.autograd.grad(out.square().sum(), [p for p in block.parameters()
                                                 if p.requires_grad])
    once()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for index in range(reps):
        starts[index].record()
        once()
        ends[index].record()
    torch.cuda.synchronize()
    samples = [start.elapsed_time(end) for start, end in zip(starts, ends)]
    # Device time as well as wall time: the matrix path is host-launch bound in
    # eager mode, so the wall number alone is not about the architecture.
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA], record_shapes=False,
        profile_memory=False)
    profiler.start()
    once()
    profiler.stop()
    device_ms = sum(getattr(event, "self_device_time_total", 0) or 0
                    for event in profiler.key_averages()) / 1000.0
    return dict(reps=samples, mean_ms=round(statistics.fmean(samples), 2),
                p50_ms=round(statistics.median(samples), 2),
                device_ms=round(device_ms, 2),
                parameters=sum(p.numel() for p in block.parameters()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--value-dim", type=int, default=128)
    parser.add_argument("--chunk-tokens", type=int, default=128)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3401)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    dim = matrix_config().width          # 576, the registered GDN-2 backbone width
    x = torch.randn(args.batch, args.tokens, dim, device="cuda")

    shape = MatrixShape(width=dim, heads=args.heads, value_dim=args.value_dim,
                        rank=args.rank, rank_max=args.rank,
                        hidden=matrix_config().hidden)
    report = dict(boundary="one memory block, forward+backward, eager, no graph",
                  shape=dict(batch=args.batch, tokens=args.tokens, width=dim,
                             heads=args.heads, value_dim=args.value_dim,
                             rank=args.rank, chunk_tokens=args.chunk_tokens),
                  mode="eager (no torch.compile, no CUDA graph)")

    matrix_block = MatrixMixer(shape, RoutingPolicy(ops=True), args.chunk_tokens,
                               compile_scan=False).cuda()
    report["matrix_ops_block"] = measure(matrix_block, x, args.reps)
    report["matrix_ops_block"]["state_per_head"] = [args.rank, args.value_dim]
    del matrix_block
    torch.cuda.empty_cache()

    append_block = MatrixMixer(shape, RoutingPolicy(ops=False), args.chunk_tokens,
                               compile_scan=False).cuda()
    report["matrix_append_block"] = measure(append_block, x, args.reps)
    report["matrix_append_block"]["state_per_head"] = [args.rank, args.value_dim]
    del append_block
    torch.cuda.empty_cache()

    try:
        from ndm.models.external_gdn2 import GDN2ExternalLayer
        gdn2_block = GDN2ExternalLayer(dim=dim, expansion=1.0, head_dim=args.rank,
                                       num_heads=args.heads, use_conv=True,
                                       d_conv=4).cuda().train()
        report["gdn2_delta_block"] = measure(gdn2_block, x, args.reps)
        report["gdn2_delta_block"]["state_per_head"] = [args.rank, 1 * args.rank]
        report["gdn2_delta_block"]["note"] = (
            "GDN2ExternalLayer(head_dim=rank=128, expansion=1.0, num_heads=2): the "
            "NVIDIA chunked delta-rule Triton kernel, state d_k x d_v = 128 x 128 "
            "per head -- the same state size as the matrix store.")
        report["p50_ratio_ops_over_gdn2"] = round(
            report["matrix_ops_block"]["p50_ms"] / report["gdn2_delta_block"]["p50_ms"], 2)
        report["device_ratio_ops_over_gdn2"] = round(
            report["matrix_ops_block"]["device_ms"]
            / report["gdn2_delta_block"]["device_ms"], 2)
    except Exception as error:                                  # pragma: no cover
        report["gdn2_delta_block"] = dict(status="NOT MEASURED",
                                          reason=repr(error))

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=1))
    print(json.dumps({key: (value.get("mean_ms") if isinstance(value, dict) else value)
                      for key, value in report.items()}, indent=1))


if __name__ == "__main__":
    main()
