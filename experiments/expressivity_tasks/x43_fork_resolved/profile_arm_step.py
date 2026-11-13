"""Attribute the Emender lattice step's time by component (one short run, card leased).

Phase timers (forward / backward / optimizer) plus a torch.profiler kernel
attribution, at the registered arm's real shape. Read-only: trains nothing to
completion, writes nothing but a JSON report.

    eval "$(GPU_LEASE_VISIBLE=0,1 scripts/gpu_lease.sh acquire 1 --wait)"
    "$EMENDER_PYTHON" -m experiments.expressivity_tasks.x43_fork_resolved.profile_arm_step \
        --arm lattice_emender_bounded --steps 5
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped, model_for
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import configure_production


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="lattice_emender_bounded")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--seed", type=int, default=3401)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    L.check_python()
    identity = L.flagship_identity()
    base, policy, _ = L.build_policy(a.arm)
    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    model = model_for(base, policy).cuda().train()
    n_total = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=policy.lr,
                            betas=(.9, .95 if base == "pure" else policy.beta2),
                            weight_decay=.1 if base == "pure" else policy.weight_decay, fused=True)
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    mb = 8 if base == "pure" else policy.microbatch
    batches = iter(stream.batches("train"))

    def one_step(rows, time_it=False):
        n = sum(len(r[3]) for r in rows)
        opt.zero_grad(set_to_none=True)
        t = {}
        if time_it:
            torch.cuda.synchronize(); t0 = time.monotonic()
        loss_sum = 0.0
        groups = list(grouped(rows, mb))
        if time_it:
            torch.cuda.synchronize(); t["fwd"] = time.monotonic() - t0; t0 = time.monotonic()
        for x, y, valid in groups:
            loss = model.loss(x, y)
            if base == "pure":
                loss = loss * (y.numel() / valid) + policy.gate_cost_lambda * model.gate_mean
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        if time_it:
            torch.cuda.synchronize(); t["bwd"] = time.monotonic() - t0; t0 = time.monotonic()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        opt.step()
        if time_it:
            torch.cuda.synchronize(); t["opt"] = time.monotonic() - t0
        return n, loss_sum, t

    for _ in range(a.warmup):
        one_step(next(batches))

    phases = {"fwd": 0.0, "bwd": 0.0, "opt": 0.0}
    tokens = 0
    prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA,
                                              torch.profiler.ProfilerActivity.CPU],
                                  record_shapes=False, profile_memory=False)
    prof.start()
    for _ in range(a.steps):
        rows = next(batches)
        n, loss, t = one_step(rows, time_it=True)
        tokens += n
        for k in phases:
            phases[k] += t[k]
    prof.stop()

    total = sum(phases.values())
    report = dict(
        arm=a.arm, base_arm=base, parameters=n_total,
        microbatches_per_step=max(1, len(rows)),
        windows_per_step=len(rows), context=L.CONTEXT,
        budgets=list(L._budgets(policy)) if base == "pure" else None,
        tokens=tokens, steps=a.steps,
        phase_s={k: round(v, 3) for k, v in phases.items()},
        phase_frac={k: round(v / max(1e-9, total), 4) for k, v in phases.items()},
        total_s=round(total, 3),
        us_per_token=round(1e6 * total / max(1, tokens), 2),
        tokens_per_s=round(tokens / max(1e-9, total), 1),
    )
    # kernel attribution
    rows_k = []
    for e in prof.key_averages():
        cu = getattr(e, "self_device_time_total", 0) or 0
        if cu > 0:
            rows_k.append(dict(name=e.key[:110], cuda_ms=round(cu / 1000, 3),
                               calls=e.count, cuda_frac=None))
    rows_k.sort(key=lambda r: -r["cuda_ms"])
    tot_cu = sum(r["cuda_ms"] for r in rows_k) or 1.0
    for r in rows_k:
        r["cuda_frac"] = round(r["cuda_ms"] / tot_cu, 4)
    report["top_kernels"] = rows_k[:20]
    report["cuda_total_ms"] = round(tot_cu, 1)

    out = Path(a.out or (Path(__file__).resolve().parent / "runs_150m" / f"profile_{a.arm}.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in
                      ("arm", "base_arm", "parameters", "windows_per_step", "budgets",
                       "tokens", "phase_s", "phase_frac", "us_per_token", "tokens_per_s",
                       "cuda_total_ms")}, indent=2))
    print("\ntop kernels by CUDA time:")
    for r in rows_k[:12]:
        print(f"  {r['cuda_ms']:9.1f} ms  {r['cuda_frac']*100:5.1f}%  "
              f"{r['calls']:6d} calls  {r['name']}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
