"""K8 -- the growth quality probe: does the model, given the chance, grow
toward transformer-parity state -- and does it pay?

The three-arm probe (the operator's target).  All three arms are the SAME
``matrix_ops`` model (149,981,808 parameters, the five-op registered policy
constants verbatim, tau_match 0.45 / read_scale 2.0 -- NOT the CMA-ES-tuned
chunk-arm config, which that diagnosis itself scoped to the chunked arm),
built by the same ``matched_arms.build_matrix`` order (seed 3401, the init
shared bit for bit with every banked arm), the same registered flagship
stream, the ONE transferred LR (the gdn2 nomination's -- the CMA-ES diagnosis
confirmed the transferred config was NOT sick: it leads gdn2 then plateaus
architecturally), the same matched-exposure stopping rule (20,000,000
targets), the same honest clock, the same document-disjoint calibration eval:

* ``fixed128``  -- the BANKED K6 per-token probe (receipts/k6_probe.json,
                   2.0278 nats/byte): read, never re-run;
* ``grow``      -- the K8 arm: start rank 128, the registered growth rule
                   (trigger 0.5, window 4, geometric 2.0, stride floor 64,
                   snap set, cap 2048) firing at 64-token statistic
                   boundaries mid-window, per (layer, window) scan;
* ``fixed2048`` -- the capacity control: rank 2048 from token 0, never grows
                   (the C2/C3 claims' arm: capacity at fixed parameters).

Deliverables: ``receipts/k8_probe_grow.json``,
``receipts/k8_probe_fixed2048.json``, the three-arm comparison table in
``K8_GROWTH.md``, and the trigger-event receipts (when growth fired, to what
rank, at what rates -- the registration's "reported per chunk" telemetry,
aggregated per step and per layer).

``smoke`` runs a few real steps at probe shape and verifies the machinery the
long arms depend on BEFORE they are committed to: the trigger fires,
re-dispatch executes, the zero rows are exact, and the backward survives rank
changes -- the short-run confirmation the operator asked for, receipted into
``receipts/k8_smoke.json``.

Usage (see ``run_k8_growth_smoke_leased.sh`` / ``run_k8_probe_leased.sh`` for
the detached leased drivers):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k8_probe_growth smoke
    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k8_probe_growth arm grow
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, OP_NAMES, MatrixSchedule, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k8_growth import (
    GROW_BOUNDARY_TOKENS, GrowingMatrixMixer, convert_model_to_growing,
    growing_scan)
from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k8_runs"
SEED = 3401
EXPOSURE_TARGETS = 20_000_000
PROBE_WINDOWS = 4
SEGMENT_TOKENS = 64
RANK_CAP = 2048
SMOKE_STEPS = 6


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def banked_fixed128_row() -> dict:
    """The banked K6 per-token probe, read from its own receipt (never
    re-run, never rewritten)."""
    r = json.loads((RECEIPTS / "k6_probe.json").read_text())
    return dict(arm="fixed128 (banked K6 per-token probe)",
                eval_nll_nats_per_byte=r["eval_nll_nats_per_byte"],
                exposure_bytes=r["exposure_bytes"],
                train_seconds=r["walls"]["train_seconds"],
                bytes_per_s=r["bytes_per_s"],
                mean_live_slots=r["diagnostics"]["occupancy"]
                ["mean_live_slots"],
                read_entropy=r["diagnostics"]["read_weight_entropy"]["mean"],
                receipt=str(RECEIPTS / "k6_probe.json"))


def growth_snapshot(model) -> dict:
    """The live growth state of every mixer: cheap, host-side, per step.  Each
    mixer's ``growth_events`` are its most recent scan's (the last window of
    the last microbatch) -- a per-step sample of 24 scans."""
    events = 0
    final_ranks = Counter()
    boundaries = Counter()
    for block in model.blocks:
        mixer = block.mixer
        final_ranks[mixer.last_rank] += 1
        events += len(mixer.growth_events)
        for event in mixer.growth_events:
            boundaries[event["boundary"]] += 1
    return dict(mixers=len(model.blocks), events_per_scan_sample=events,
                final_rank_histogram={str(k): v for k, v
                                      in sorted(final_ranks.items())},
                boundary_histogram={str(k): v for k, v
                                    in sorted(boundaries.items())})


# ----------------------------------------------------------------- diagnostics


def probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """Op rates, per-token occupancy, read entropy and growth events of the
    TRAINED model, from the growing collect walk (the same spans the arm
    trains on), under the training's own regime (eval mode, bf16 autocast)."""
    rows = [row for row in stream.windows("calibration")
            if len(row[2]) == L.CONTEXT]
    if len(rows) < windows:
        raise ValueError(f"calibration split has {len(rows)} full windows; "
                         f"the probe needs {windows}")
    rows = rows[:windows]
    model.eval()
    per_layer = []
    handles = []

    def make_hook(layer_index, mixer):
        def hook(_module, args, _output):
            x = args[0]
            batch, tokens, _ = x.shape
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                query = mixer.query(x).view(batch, tokens, mixer.heads,
                                            mixer.value_dim)
                payload = mixer.value(x).view(batch, tokens, mixer.heads,
                                              mixer.value_dim)
                decay = torch.sigmoid(mixer.decay(x)).clamp(
                    min=DECAY_FLOOR, max=1.0 - 1e-6)
                result = growing_scan(
                    query, payload, decay, policy=mixer.policy,
                    start_rank=mixer.rank, rank_cap=mixer.rank_cap,
                    segment_tokens=mixer.segment_tokens, collect=True)
            with torch.no_grad():
                lane0_total = Counter()
                hist_total = Counter()
                decisions = 0
                occupancy_sum = occupancy_n = 0
                entropy_sum = entropy_n = 0
                spans = []
                for span in result.collected:
                    schedule = span["schedule"]
                    lane0 = torch.bincount(
                        schedule.op[..., 0].reshape(-1).to("cpu"),
                        minlength=len(OP_NAMES))
                    lane0_total.update(
                        {OP_NAMES[i]: int(lane0[i])
                         for i in range(len(OP_NAMES))})
                    hist_total.update(op_histogram(schedule))
                    decisions += int(batch * (span["t1"] - span["t0"])
                                     * mixer.heads)
                    occupancy = span["live_pre"].float().sum(-1)
                    occupancy_sum += float(occupancy.sum())
                    occupancy_n += occupancy.numel()
                    w = span["weights"]
                    mass = w.sum(-1)
                    has = mass > 0
                    probs = torch.where(has.unsqueeze(-1),
                                       w / mass.clamp_min(1e-30).unsqueeze(-1),
                                       torch.zeros_like(w))
                    contrib = torch.where(probs > 0, probs * probs.log(),
                                           torch.zeros_like(probs))
                    entropy_sum += float(contrib.sum(-1)[has].sum())
                    entropy_n += int(has.sum())
                    spans.append(dict(t0=span["t0"], t1=span["t1"],
                                       rank=span["rank"]))
                per_layer.append((layer_index, dict(
                    lane0=dict(lane0_total), hist=dict(hist_total),
                    decisions=decisions,
                    occupancy_sum=occupancy_sum, occupancy_n=occupancy_n,
                    entropy_sum=entropy_sum, entropy_n=entropy_n,
                    spans=spans,
                    events=[dict(e) for e in result.stats.events],
                    final_rank=result.rank)))
        return hook

    for index, block in enumerate(model.blocks):
        handles.append(block.mixer.register_forward_hook(
            make_hook(index, block.mixer)))
    started = time.monotonic()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1))
                model.features(x)
    finally:
        for handle in handles:
            handle.remove()
    wall = time.monotonic() - started

    layers = sorted({index for index, _ in per_layer})
    by_layer = {}
    for layer in layers:
        stats = [s for i, s in per_layer if i == layer]
        decisions = sum(s["decisions"] for s in stats)
        by_layer[layer] = dict(
            op_rate_per_token={OP_NAMES[i]: sum(s["lane0"][OP_NAMES[i]]
                                                for s in stats) / decisions
                               for i in range(len(OP_NAMES))},
            spans=stats[0]["spans"],
            growth_events=stats[0]["events"],
            final_rank=stats[0]["final_rank"],
            occupancy_mean_live_slots=float(
                sum(s["occupancy_sum"] for s in stats)
                / max(1, sum(s["occupancy_n"] for s in stats))),
            read_weight_entropy_nats=float(
                sum(s["entropy_sum"] for s in stats)
                / max(1, sum(s["entropy_n"] for s in stats))))
    decisions = sum(s["decisions"] for _, s in per_layer)
    occupancy_mean = (sum(s["occupancy_sum"] for _, s in per_layer)
                      / max(1, sum(s["occupancy_n"] for _, s in per_layer)))
    entropy_n = sum(s["entropy_n"] for _, s in per_layer)
    hist = Counter()
    for _, s in per_layer:
        hist.update(s["hist"])
    # the mean span table and the event aggregate across layers
    span_counter = Counter()
    event_total = 0
    for _, s in per_layer:
        event_total += len(s["events"])
        span_counter[tuple(map(tuple, s["spans"]))] += 1
    return dict(
        probe=dict(split="calibration", windows=len(rows),
                   mixers=len(model.blocks),
                   regime="eval, bf16 autocast, the growing collect walk the "
                          "arm trains on",
                   probe_wall_seconds=round(wall, 1)),
        op_rate_per_token_lane0={OP_NAMES[i]: sum(s["lane0"][OP_NAMES[i]]
                                                  for _, s in per_layer)
                                 / decisions
                                 for i in range(len(OP_NAMES))},
        op_histogram_all_lanes=dict(hist),
        occupancy=dict(
            note=("live slots at every token, per span, from the recorded "
                  "pre-decision liveness"),
            mean_live_slots=float(occupancy_mean)),
        read_weight_entropy=dict(
            units="nats over the live-slot weight distribution, per token",
            mean=float(sum(s["entropy_sum"] for _, s in per_layer)
                       / max(1, entropy_n))),
        growth=dict(
            events_per_probe=round(event_total / len(model.blocks), 2),
            span_tables={"; ".join(f"{t0}-{t1}@{r}" for t0, t1, r in table):
                         count
                         for table, count in sorted(span_counter.items(),
                                                    key=lambda kv: -kv[1])}),
        per_layer=by_layer)


# ------------------------------------------------------------------- the arms


def train_arm(args, arm: str) -> dict:
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None:
        raise ValueError("no GPU lease: set GPU_LEASE_HELD before training")
    if visible is None:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set before launch")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: recorded card {card} vs "
                         f"CUDA_VISIBLE_DEVICES {visible}")

    grow = arm == "grow"
    start_rank = 128 if grow else RANK_CAP
    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", 128, args.seed, k5b=False)
    model = convert_model_to_growing(model, segment_tokens=SEGMENT_TOKENS,
                                     rank_cap=RANK_CAP, grow=grow,
                                     start_rank=start_rank)
    summary = dict(summary)
    summary["scan_path"] = ("k8_growth (per-token decisions, the registered "
                            "growth rule at 64-token boundaries)"
                            if grow else
                            "k8_growth fixed-2048 (per-token, one hi span, "
                            "never grows)")
    summary["start_rank"] = start_rank
    microbatch = optimizer_config["microbatch"]
    model = model.cuda().train()
    out = Path(args.runs_dir) / f"{arm}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    exposure = int(args.bytes)
    stopping_rule = (f"matched-EXPOSURE: train until {exposure} targets seen "
                     "(~20M, the probe exposure); the honest elapsed wall is "
                     "reported first-class")
    config = dict(probe="k8_growth", arm=arm,
                  granularity=("per-token, the registered growth rule at "
                                "64-token statistic boundaries, rank cap "
                                f"{RANK_CAP}" if grow else
                                "per-token, fixed rank "
                                f"{RANK_CAP}, never grows"),
                  seed=args.seed, exposure_targets=exposure,
                  segment_tokens=SEGMENT_TOKENS, optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT,
                  windows_per_step=64, summary=summary, identity=identity,
                  policy=RoutingPolicy(ops=True, grow=grow,
                                       grow_trigger=0.5, grow_window=4,
                                       grow_window_tokens=512,
                                       grow_geometric=2.0,
                                       grow_stride_floor=64).record(),
                  rank_cap=RANK_CAP, start_rank=start_rank,
                  bindings=dict(
                      matched_arms.registration_bindings(),
                      k6_preregistration=str(
                          ROOT / "PREREGISTRATION_SEQUENTIAL_MATRIX.md"),
                      k6_preregistration_sha256=digest(
                          ROOT / "PREREGISTRATION_SEQUENTIAL_MATRIX.md"),
                      k8_gates_receipt=str(RECEIPTS / "k8_gates.json"),
                      k8_gates_sha256=digest(RECEIPTS / "k8_gates.json")),
                  git=matched_arms.git_state(), stopping_rule=stopping_rule,
                  within_sweep=("WITHIN-SWEEP comparison at ONE transferred LR "
                                "(the gdn2 nomination's, verbatim from the "
                                "matched arms)"))
    atomic_json(out / "start.json", dict(
        config, config_sha256=matched_arms.config_sha256(config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(arm=arm, card=card, seed=args.seed,
                          exposure=exposure, start_rank=start_rank,
                          parameters=sum(p.numel()
                                         for p in model.parameters()))),
          flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets = 0
    log_rows = []
    growth_rows = []
    with (out / "training.jsonl").open("w", buffering=1) as log:
        while targets < exposure:
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                rows = next(batches)
            except StopIteration:
                raise RuntimeError("train split exhausted before the budget: "
                                   "refusing to wrap")
            n = sum(len(r[3]) for r in rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config["lr"] * min(
                    1., (step + 1) / max(1, optimizer_config["warmup_steps"]))
            loss_sum = 0.0
            for x, y, valid in grouped(rows, microbatch):
                loss = model.loss(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            step += 1
            targets += n
            row = dict(step=step, active_seconds=elapsed, bytes=targets,
                       loss=loss_sum, grad_norm=float(grad_norm),
                       time_ns=time.time_ns())
            log.write(json.dumps(row) + "\n")
            log_rows.append(row)
            growth_rows.append(dict(step=step, **growth_snapshot(model)))
            if step % 10 == 0 or step <= 3:
                snap = growth_rows[-1]
                print(f"  {arm} step {step} {elapsed:.0f}s "
                      f"{targets / 1e6:.1f}MB loss {loss_sum:.4f} "
                      f"final-rank-hist {snap['final_rank_histogram']}",
                      flush=True)

    diagnostics = probe_diagnostics(model, stream, windows=args.probe_windows)
    os.environ["X40_BATCHED_READS"] = "0"
    eval_started = time.monotonic()
    result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
    eval_wall = time.monotonic() - eval_started

    receipt = dict(
        probe="k8_growth", arm=arm, seed=args.seed,
        granularity=config["granularity"], stopping_rule=stopping_rule,
        within_sweep=config["within_sweep"],
        exposure_bytes=targets, exposure_steps=step,
        eval_nll_nats_per_byte=result["score"], eval_split=result["eval_split"],
        eval_windows=result["eval_windows"], eval_targets=result["eval_targets"],
        walls=dict(train_seconds=round(elapsed, 1),
                   probe_seconds=diagnostics["probe"]["probe_wall_seconds"],
                   eval_seconds=round(eval_wall, 1)),
        bytes_per_s=targets / max(1e-9, elapsed),
        diagnostics=diagnostics,
        growth=dict(
            note=("per-step snapshots of the mixers' live growth state: the "
                  "final-rank histogram over the 24 mixers and the boundary "
                  "histogram of one sampled scan each"),
            first_step=growth_rows[0], last_step=growth_rows[-1],
            step_samples=[g for i, g in enumerate(growth_rows)
                          if i < 3 or (i + 1) % 30 == 0]),
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            log_rows, keep_first=min(5, max(1, len(log_rows)))),
        banked_fixed128=banked_fixed128_row())
    receipt_path = Path(args.receipts_dir) / f"k8_probe_{arm}.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("arm", "eval_nll_nats_per_byte", "exposure_bytes",
                       "exposure_steps", "bytes_per_s")}, indent=1), flush=True)
    return receipt


# --------------------------------------------------------------------- the smoke


def smoke(args) -> dict:
    """A few REAL steps at probe shape: the trigger fires, re-dispatch
    executes, the zero rows are exact, and the backward survives rank
    changes -- the machinery the long arms depend on, verified before they
    are committed to."""
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    if card is None:
        raise ValueError("no GPU lease: set GPU_LEASE_HELD before training")
    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", 128, args.seed, k5b=False)
    model = convert_model_to_growing(model, segment_tokens=SEGMENT_TOKENS,
                                     rank_cap=RANK_CAP, grow=True)
    model = model.cuda().train()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    steps = int(args.steps)
    walls, growth_rows, grad_norms = [], [], []
    loss_values = []
    for step in range(steps):
        rows = next(batches)
        n = sum(len(r[3]) for r in rows)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.monotonic()
        loss_sum = 0.0
        for x, y, valid in grouped(rows, optimizer_config["microbatch"]):
            loss = model.loss(x, y)
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        walls.append(time.monotonic() - t0)
        grad_norms.append(float(grad_norm))
        loss_values.append(loss_sum)
        growth_rows.append(dict(step=step + 1, **growth_snapshot(model)))
        print(f"  smoke step {step + 1}: {walls[-1]:.1f}s loss {loss_sum:.4f} "
              f"grad {grad_norms[-1]:.3f} "
              f"ranks {growth_rows[-1]['final_rank_histogram']}", flush=True)

    # the machinery assertions, on the LIVE model
    last = growth_rows[-1]
    assert any(count > 0 for count in last["boundary_histogram"].values()), \
        "the trigger never fired in the smoke"
    assert last["final_rank_histogram"].get("2048", 0) > 0, \
        f"the rank never reached the cap: {last['final_rank_histogram']}"
    # the zero-row property, exactly, on a collect walk of the first mixer --
    # through a forward hook so the walk sees the mixer's TRUE input
    mixer = model.blocks[0].mixer
    row = next(r for r in stream.windows("calibration")
               if len(r[2]) == L.CONTEXT)
    x, _y, _v = next(grouped([row], 1))
    model.eval()
    captured = {}

    def hook(_module, hook_args, _output):
        hx = hook_args[0]
        batch, tokens, _ = hx.shape
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            query = mixer.query(hx).view(batch, tokens, mixer.heads,
                                         mixer.value_dim)
            payload = mixer.value(hx).view(batch, tokens, mixer.heads,
                                           mixer.value_dim)
            decay = torch.sigmoid(mixer.decay(hx)).clamp(
                min=DECAY_FLOOR, max=1.0 - 1e-6)
            captured["result"] = growing_scan(
                query, payload, decay, policy=mixer.policy,
                start_rank=mixer.rank, rank_cap=mixer.rank_cap,
                segment_tokens=mixer.segment_tokens, collect=True)

    handle = mixer.register_forward_hook(hook)
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            model.features(x)
    finally:
        handle.remove()
    result = captured["result"]
    assert result.stats.events, "the collect walk never grew"
    first_fire = result.stats.events[0]["boundary"]
    zero_rows_ok = True
    prev_rank = result.collected[0]["rank"]
    for index in range(1, len(result.collected)):
        span = result.collected[index]
        live_new = span["live_pre"][:, :, :, prev_rank:]
        schedule = span["schedule"]
        first_write = None
        for t in range(span["t1"] - span["t0"]):
            wrote_new = (schedule.op[:, t, :, 0] != NONE) \
                & (schedule.slot[:, t, :, 0] >= prev_rank)
            if bool(wrote_new.any()):
                first_write = t
                break
        before = live_new[:, :first_write if first_write is not None else None]
        zero_rows_ok = zero_rows_ok and (not before.any()) \
            and (not span["live_pre"][:, 0, :, prev_rank:].any())
        prev_rank = span["rank"]
    model.train()

    receipt = dict(
        probe="k8_growth_smoke",
        purpose=("the operator's short-run confirmation before the long "
                 "arms: does the trigger fire, does re-dispatch execute, are "
                 "the zero rows exact, does the backward survive rank changes"),
        seed=args.seed, steps=steps, walls_s=[round(w, 1) for w in walls],
        loss=loss_values, grad_norms=grad_norms,
        growth_steps=growth_rows,
        assertions=dict(trigger_fired=True,
                        cap_reached=True,
                        backward_survived_rank_changes=all(
                            torch.isfinite(torch.tensor(grad_norms))) and all(
                            g == g for g in grad_norms),
                        zero_rows_exact=bool(zero_rows_ok),
                        first_fire_boundary=first_fire),
        sample_spans=[list(s) for s in result.stats.spans],
        sample_events=result.stats.events,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        parameters=summary["parameters"]["total"],
        card=card, session=matched_arms.session_meta(card),
        git=matched_arms.git_state())
    receipt_path = Path(args.receipts_dir) / "k8_smoke.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps(receipt["assertions"], indent=1), flush=True)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--steps", type=int, default=SMOKE_STEPS)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--receipts-dir", default=str(RECEIPTS))
    a = sub.add_parser("arm")
    a.add_argument("arm", choices=("grow", "fixed2048"))
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--runs-dir", default=str(RUNS))
    a.add_argument("--receipts-dir", default=str(RECEIPTS))
    t = sub.add_parser("table")
    t.add_argument("--receipts-dir", default=str(RECEIPTS))
    args = ap.parse_args()
    if args.cmd == "smoke":
        smoke(args)
        return
    if args.cmd == "arm":
        train_arm(args, args.arm)
        return
    rows = [banked_fixed128_row()]
    for arm in ("grow", "fixed2048"):
        path = Path(args.receipts_dir) / f"k8_probe_{arm}.json"
        if path.exists():
            r = json.loads(path.read_text())
            rows.append(dict(arm=arm,
                             eval_nll_nats_per_byte=r["eval_nll_nats_per_byte"],
                             exposure_bytes=r["exposure_bytes"],
                             train_seconds=r["walls"]["train_seconds"],
                             bytes_per_s=r["bytes_per_s"],
                             mean_live_slots=r["diagnostics"]["occupancy"]
                             ["mean_live_slots"],
                             read_entropy=r["diagnostics"]
                             ["read_weight_entropy"]["mean"],
                             receipt=str(path)))
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
