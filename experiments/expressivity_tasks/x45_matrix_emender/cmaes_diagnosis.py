"""Diagnosis of the matrix_ops arm at 150 M: what the ops actually do.

Runs the probe protocol (``cmaes_search.probe``: fixed 16,777,216-target
exposure, seed 3401, the registered stream and eval boundary) on ONE config
with the instrumentation the search does not pay for, answering the operator's
"figure out what is happening" for both the transferred and the searched
configuration:

* **Op-firing histogram over training** -- append/correct/pop/merge counts per
  chunk, per layer, sampled every 8 steps (plus step 0) by a RECORD PASS: one
  extra no-grad forward of a fixed train window through
  ``whole_scan_record`` (the eager walk with the recorded decision channel the
  K5B gates already use -- the same kernels the training path runs, so the
  recorded schedule IS the training path's).  If the ops rarely fire the arm
  is secretly append-only and the ops story is moot at this scale; if POP
  dominates, the schedule may be thrashing slots.
* **Occupancy / live-slot trajectory** -- live slots per chunk boundary and
  per layer over training (simulated from the recorded schedule's op codes and
  live_after lanes, which is what the scan reproduces bit-exactly).
* **Read-weight statistics** -- entropy, top-1 mass share and total strength
  mass of the read weights over training.
* **Gradient norms per parameter group** -- embed, head, the mixer's
  query/value/out/decay projections, the MLP and the norms, every 4 steps
  (pre-clip, post-backward).  Cheap reductions, no mutation.
* **Per-step loss** -- the full step log the search probes also write.

What the existing harness already logged before this file: per-step
loss/bytes/time (``matched_arms``), and op histograms only in gate fixtures
(``op_histogram`` / ``whole_scan_record`` were test instruments, never run over
training).  Everything sampled here is NEW over-training instrumentation, and
the record pass costs one extra batch-1 eager forward per sample point.

The probe window is the FIRST full 2048-byte window of the train split -- a
window the training loop itself consumes, so the schedule statistics are
measured on the training distribution the search optimises.

Usage (under the search's GPU lease discipline):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.cmaes_diagnosis \\
        run --config '{"lr": ...}' --label transferred --out receipts/cmaes_diagnosis_transferred.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
X45_RECEIPTS = ROOT / "receipts"

# The probe window is fixed across steps, configs and runs.
RECORD_EVERY = 8          # steps between record passes (plus step 0)
GRAD_EVERY = 4            # steps between gradient-norm collections


def _groups(model):
    """Parameter-name groups for gradient norms, resolved to tensors once."""
    names = dict(model.named_parameters())
    groups = {"embed": ["embed.weight"], "head": ["head.weight"],
              "mixer.query": [], "mixer.value": [], "mixer.out": [],
              "mixer.decay": [], "mlp": [], "norms": []}
    for i, block in enumerate(model.blocks):
        for key in ("query", "value", "out", "decay"):
            groups[f"mixer.{key}"].append(f"blocks.{i}.mixer.{key}.weight")
        for key in ("w1", "w2", "w3"):
            groups["mlp"].append(f"blocks.{i}.mlp.{key}.weight")
        groups["norms"] += [f"blocks.{i}.norm1.weight", f"blocks.{i}.norm2.weight"]
    groups["norms"].append("norm.weight")
    return {g: [names[n] for n in members] for g, members in groups.items()}


def _grad_norms(grouped_params):
    import torch
    out = {}
    for g, params in grouped_params.items():
        acc = None
        for p in params:
            if p.grad is None:
                continue
            sq = p.grad.float().pow(2).sum()
            acc = sq if acc is None else acc + sq
        out[g] = float(acc ** 0.5) if acc is not None else 0.0
    return out


def _analyze_schedule(schedule, weights, live, chunk_tokens, rank):
    """Per-layer schedule statistics: op counts per chunk, occupancy, read stats.

    ``schedule`` is the recorded decision channel ``(batch=1, tokens, heads,
    lanes)``; occupancy is simulated from the op codes and ``live_after``
    lanes (the channel the scan reproduces bit-exactly); read statistics are
    over the recorded read weights ``(1, tokens, heads, rank_max)``.
    """
    import torch

    from .algebra import NONE
    ops = schedule.op[0].to("cpu")            # (tokens, heads, lanes)
    slot = schedule.slot[0].to("cpu")
    live_after = schedule.live_after[0].to("cpu")
    tokens, heads, lanes = ops.shape
    n_chunks = (tokens + chunk_tokens - 1) // chunk_tokens
    per_chunk = [[0] * 5 for _ in range(n_chunks)]
    occupancy = [None] * (n_chunks + 1)
    # The liveness simulation the scan itself performs (bit-exactly): a lane
    # with op != NONE sets its slot's liveness to its live_after flag, later
    # lanes winning within a token (the delta-rule order).
    live_mat = torch.zeros(heads, rank, dtype=torch.bool)
    occupancy[0] = [int(v) for v in live_mat.sum(-1)]
    for t in range(tokens):
        c = min(t // chunk_tokens, n_chunks - 1)
        for h in range(heads):
            for lane in range(lanes):
                op = int(ops[t, h, lane])
                if op == NONE:
                    continue
                per_chunk[c][op] += 1
                live_mat[h, int(slot[t, h, lane])] = bool(live_after[t, h, lane])
        if (t + 1) % chunk_tokens == 0 or t == tokens - 1:
            occupancy[min((t + 1) // chunk_tokens, n_chunks)] = \
                [int(v) for v in live_mat.sum(-1)]
    total = [sum(row[op] for row in per_chunk) for op in range(5)]
    # Read-weight statistics over the recorded weights.
    w = weights[0]                                     # (tokens, heads, rank)
    mass = w.sum(-1)                                    # (tokens, heads)
    positive = mass > 0
    safe = w / mass.clamp_min(1e-12).unsqueeze(-1)
    logsafe = torch.where(w > 0, safe, torch.zeros_like(safe))
    entropy = -(logsafe * torch.log(logsafe.clamp_min(1e-12))).sum(-1)
    entropy = entropy[positive]
    top1 = safe.max(-1).values[positive]
    return dict(
        ops_total=total,
        ops_per_chunk=per_chunk,
        occupancy_per_chunk=[sum(row) / heads for row in occupancy],
        occupancy_final=[int(v) for v in live_mat.sum(-1)],
        live_fraction=float(live[0].float().mean()) if live is not None else None,
        read_entropy_mean=float(entropy.mean()) if entropy.numel() else None,
        read_top1_mean=float(top1.mean()) if top1.numel() else None,
        read_zero_mass_fraction=float((~positive).float().mean()),
        read_mass_mean=float(mass.mean()))


def _record_pass(model, x, y):
    """One no-grad forward of the probe window with the schedule recorded.

    Walks the blocks manually so each layer's decision channel is in hand; the
    reads are ``whole_scan_record``'s -- bit-equal to the training path's by
    construction (the same eager body the K5B graph captures), so the recorded
    schedule is the training path's own.
    """
    import torch

    from .k5b_whole_scan import whole_scan_record
    from .model import DECAY_FLOOR
    batch, tokens = x.shape
    mixer0 = model.blocks[0].mixer
    heads, value_dim = mixer0.heads, mixer0.value_dim
    per_layer = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        h = model.embed(x)
        for block in model.blocks:
            mixer = block.mixer
            hn = block.norm1(h)
            q = mixer.query(hn).view(batch, tokens, heads, value_dim)
            p = mixer.value(hn).view(batch, tokens, heads, value_dim)
            d = torch.sigmoid(mixer.decay(hn)).clamp(min=DECAY_FLOOR,
                                                     max=1.0 - 1e-6)
            result = whole_scan_record(
                q, p, d, policy=mixer.policy, rank=mixer.rank,
                rank_max=mixer.rank_max, chunk_tokens=mixer.chunk_tokens)
            h = h + mixer.out(result.reads.reshape(batch, tokens,
                                                    heads * value_dim))
            h = h + block.mlp(block.norm2(h))
            per_layer.append(_analyze_schedule(result.schedule, result.weights,
                                               result.live, mixer.chunk_tokens,
                                               mixer.rank))
        hidden = model.norm(h)
        loss = model.criterion(hidden.to(torch.bfloat16).contiguous(),
                               y.contiguous(),
                               model.head.weight.to(torch.bfloat16))
    # layer-summed per-chunk views; per-layer totals kept for depth contrast
    n_chunks = len(per_layer[0]["ops_per_chunk"])
    ops_per_chunk = [[sum(layer["ops_per_chunk"][c][op] for layer in per_layer)
                      for op in range(5)] for c in range(n_chunks)]
    occupancy_per_chunk = [sum(layer["occupancy_per_chunk"][c]
                              for layer in per_layer) / len(per_layer)
                            for c in range(n_chunks + 1)]
    return dict(
        probe_window_loss=float(loss),
        ops_total_per_layer=[layer["ops_total"] for layer in per_layer],
        ops_per_chunk_summed=ops_per_chunk,
        occupancy_per_chunk_mean=occupancy_per_chunk,
        occupancy_final_per_layer=[layer["occupancy_final"] for layer in per_layer],
        live_fraction_per_layer=[layer["live_fraction"] for layer in per_layer],
        read_entropy_per_layer=[layer["read_entropy_mean"] for layer in per_layer],
        read_top1_per_layer=[layer["read_top1_mean"] for layer in per_layer],
        read_zero_mass_per_layer=[layer["read_zero_mass_fraction"]
                                  for layer in per_layer],
        read_mass_per_layer=[layer["read_mass_mean"] for layer in per_layer])


def run(args) -> dict:
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
    from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
        configure_production)
    from experiments.expressivity_tasks.x45_matrix_emender import cmaes_search as CS
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for as matrix_model_for, parameter_count)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

    config = json.loads(args.config)
    started = time.monotonic()
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None or str(card) != str(visible):
        raise ValueError("GPU lease required (GPU_LEASE_HELD == CUDA_VISIBLE_DEVICES)")

    # The fixed probe window: the first full train window, as its own tensors.
    window = next(w for w in stream.windows("train") if len(w[2]) == L.CONTEXT)
    x_probe = torch.as_tensor(window[2][None, :], device="cuda", dtype=torch.int64)
    y_probe = torch.as_tensor(window[3][None, :], device="cuda", dtype=torch.int64)

    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = RoutingPolicy(ops=True, tau_match=config.get("tau_match", 0.45),
                          tau_merge=config.get("tau_merge", 0.85),
                          tau_pop=config.get("tau_pop", 0.10),
                          read_scale=config.get("read_scale", 2.0))
    shape = matrix_config(rank=CS.RANK)
    model = matrix_model_for(shape, policy, CS.CHUNK_TOKENS, compile_scan=True,
                             fused_whole_scan=True).cuda().train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=config["lr"], betas=(0.9, CS.BETA2),
        weight_decay=config["weight_decay"], fused=True)
    warmup = max(1, int(config["warmup_steps"]))
    grouped_params = _groups(model)

    batches = iter(stream.batches("train"))
    step = targets = 0
    rows, record_points, grad_points = [], [], []
    record_points.append(dict(step=0, **_record_pass(model, x_probe, y_probe)))
    while targets < args.targets:
        torch.cuda.synchronize()
        start = time.monotonic()
        batch_rows = next(batches)
        n = sum(len(r[3]) for r in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = config["lr"] * min(1., (step + 1) / warmup)
        loss_sum = 0.0
        for x, y, valid in grouped(batch_rows, CS.MICROBATCH):
            torch.compiler.cudagraph_mark_step_begin()
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at step {step + 1}")
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        if step % GRAD_EVERY == 0:
            grad_points.append(dict(step=step + 1, grads=_grad_norms(grouped_params)))
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                       error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        step += 1
        targets += n
        rows.append(dict(step=step, bytes=targets, loss=loss_sum))
        if step % RECORD_EVERY == 0 or targets >= args.targets:
            record_points.append(dict(step=step, **_record_pass(model, x_probe,
                                                                y_probe)))

    os.environ["X40_BATCHED_READS"] = "0"
    result = L.evaluate(model, stream, "matrix_ops", "calibration")
    receipt = dict(
        label=args.label, seed=args.seed, probe_targets=args.targets,
        config=config, routing=policy.record(), parameters=parameter_count(shape),
        nll=result["score"], eval_windows=result["eval_windows"],
        eval_targets=result["eval_targets"], steps=step, targets=targets,
        wall_s=round(time.monotonic() - started, 1),
        step_log=rows, record_points=record_points, grad_points=grad_points,
        instrumentation=dict(record_every=RECORD_EVERY, grad_every=GRAD_EVERY,
                            probe_window=dict(doc=window[0], start=window[1],
                                             length=L.CONTEXT),
                            new_instrumentation=("op histograms, occupancy and "
                                                 "read-weight stats over training "
                                                 "(record passes), gradient norms "
                                                 "per group -- the matched-arms "
                                                 "harness logged only per-step "
                                                 "loss/bytes/time"),
                            preexisting=("per-step loss/bytes/time; op_histogram "
                                         "and whole_scan_record existed as gate "
                                         "fixtures only")),
        git=matched_arms.git_state())
    out = Path(args.out)
    atomic_json(out, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "nll", "steps", "targets", "eval_windows")}),
          flush=True)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--config", required=True)
    r.add_argument("--label", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--targets", type=int, default=None)
    r.add_argument("--seed", type=int, default=None)
    a = ap.parse_args()
    from experiments.expressivity_tasks.x45_matrix_emender import cmaes_search as CS
    if a.targets is None:
        a.targets = CS.PROBE_TARGETS
    if a.seed is None:
        a.seed = CS.SEED
    run(a)


if __name__ == "__main__":
    main()
