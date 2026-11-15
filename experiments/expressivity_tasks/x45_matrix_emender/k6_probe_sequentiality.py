"""K6 -- the sequential-matrix quality probe: does PER-TOKEN beat the dial's best?

The question this stage exists for (``PREREGISTRATION_SEQUENTIAL_MATRIX.md``
claim S1): at matched parameters, matched exposure, matched data order and the
ONE transferred LR (the gdn2 nomination's, the sweep's own within-sweep rule),
does the per-token arm's held-out NLL beat the dial's best banked row (chunk 8:
2.2186 nats/byte at 20,022,866 targets)?  The banked sweep gradient
(``RESULT_CHUNK_SEQUENTIALITY.md``) says the quality climbs as chunks shrink
and the per-token limit is where it points; this arm IS the limit.

The protocol is the sweep's verbatim (``sweep_chunk_sequentiality.py``), with
the arm swapped: the SAME ``matrix_ops`` model (149,981,808 parameters, the
five-op registered policy verbatim), built by the same ``matched_arms
.build_matrix`` order (so the init under seed 3401 is the sweep arms' own bit
for bit), then every mixer converted in place to the sequential one (the same
parameter modules -- ``convert_model_to_sequential``), the SAME registered
flagship stream, the SAME transferred optimizer, the SAME matched-exposure
stopping rule (20,000,000 targets), the SAME honest clock (started at step 1,
includes the stream read, fwd/bwd, clip, AdamW), the SAME document-disjoint
calibration eval.  The only difference is the granularity: decisions and reads
against the CURRENT state at every token, one persistent-kernel scan per
(layer, window), the 64-token coarse-checkpoint backward.

Diagnostics (claim S2): per-token op rates (MERGE should recover toward the
fresh-state regime), live-slot occupancy at every TOKEN (the per-token analog
of the sweep's chunk-entry occupancy), and read-weight entropy per token --
all from the recorded channel of the SAME eager walk the arm trains on.

Deliverables: ``receipts/k6_probe.json`` and the comparison table against the
banked sweep rows in ``K6_SEQUENTIAL_MATRIX.md``.

Usage (see ``run_k6_probe_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k6_probe_sequentiality arm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
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
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    DEFAULT_SEGMENT_TOKENS, convert_model_to_sequential, sequential_scan)
from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k6_runs"
SEED = 3401
EXPOSURE_TARGETS = 20_000_000
PROBE_WINDOWS = 4
SEGMENT_TOKENS = DEFAULT_SEGMENT_TOKENS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def banked_sweep_rows() -> list:
    """The dial's banked rows, read from their own receipts (never rewritten)."""
    rows = []
    for path in sorted(RECEIPTS.glob("sweep_chunk_c*.json")):
        if path.name.startswith("sweep_chunk_parity"):
            continue
        try:
            r = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if r.get("exposure_bytes", 0) < 20_000_000:
            continue      # the half-exposure rows are labelled not-comparable
        d = r["diagnostics"]
        rows.append(dict(arm=f"chunk_{r['chunk_tokens']}",
                         chunk_tokens=r["chunk_tokens"],
                         eval_nll_nats_per_byte=r["eval_nll_nats_per_byte"],
                         exposure_bytes=r["exposure_bytes"],
                         train_seconds=r["walls"]["train_seconds"],
                         bytes_per_s=r["bytes_per_s"],
                         merge_rate=d["op_rate_per_token_lane0"]["merge"],
                         occupancy=d["occupancy"]["mean_live_slots_at_chunk_entry"],
                         read_entropy=d["read_weight_entropy"]["mean_populated_store"],
                         receipt=str(path)))
    rows.sort(key=lambda row: -row["chunk_tokens"])
    return rows


# ----------------------------------------------------------------- diagnostics


def probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """Op rates, per-token occupancy and read entropy of the TRAINED model.

    A forward hook on every mixer recomputes the mixer's own query/payload/decay
    from its input and runs the eager walk directly (``_seq_forward_launch(collect=
    True)`` -- the same walk the arm trains on) under the training's own regime
    (bf16 autocast, eval mode).  The probe windows are the calibration split's
    first full windows: the same document-disjoint split the eval uses, the same
    windows for every arm.  Occupancy is the mean live-slot count at EVERY TOKEN
    (from the walk's recorded pre-decision liveness channel -- the per-token
    analog of the sweep's chunk-entry occupancy).
    """
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import _seq_forward_launch, _round_f32
    from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR

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
        pol32 = (_round_f32(mixer.policy.tau_match),
                 _round_f32(mixer.policy.tau_merge),
                 _round_f32(mixer.policy.tau_pop),
                 _round_f32(mixer.policy.tie_break),
                 _round_f32(mixer.policy.read_scale))

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
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                out = _seq_forward_launch(query, payload, decay, state, live0,
                                          pol32, mixer.rank,
                                          mixer.segment_tokens, collect=True,
                                          warps=8, stages=3)
            with torch.no_grad():
                schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                                          payload=out["payload"], op=out["op"],
                                          live_after=out["live_after"])
                lane0 = torch.bincount(
                    schedule.op[..., 0].reshape(-1).to("cpu"),
                    minlength=len(OP_NAMES))
                occupancy = out["live_pre"].float().sum(-1)      # (b, T, h)
                rank = int(out["state"].shape[2])
                w = out["weights"][..., :rank].float()
                mass = w.sum(-1)
                has = mass > 0
                probs = torch.where(has.unsqueeze(-1),
                                    w / mass.clamp_min(1e-30).unsqueeze(-1),
                                    torch.zeros_like(w))
                contrib = torch.where(probs > 0, probs * probs.log(),
                                       torch.zeros_like(probs))
                entropy = -contrib.sum(-1)                       # (b, T, h)
                per_layer.append((layer_index, dict(
                    lane0=[int(v) for v in lane0],
                    hist=op_histogram(schedule),
                    decisions=int(batch * tokens * mixer.heads),
                    occupancy_sum=float(occupancy.sum()),
                    occupancy_n=int(occupancy.numel()),
                    occupancy_first=float(occupancy[:, 0].float().mean()),
                    entropy_sum=float(entropy[has].sum()),
                    entropy_n=int(has.sum()),
                    rank=rank)))
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
            op_rate_per_token={OP_NAMES[i]: sum(s["lane0"][i] for s in stats)
                               / decisions for i in range(len(OP_NAMES))},
            occupancy_mean_live_slots=float(
                sum(s["occupancy_sum"] for s in stats)
                / max(1, sum(s["occupancy_n"] for s in stats))),
            read_weight_entropy_nats=float(
                sum(s["entropy_sum"] for s in stats)
                / max(1, sum(s["entropy_n"] for s in stats))))
    decisions = sum(s["decisions"] for _, s in per_layer)
    occupancy_mean = (sum(s["occupancy_sum"] for _, s in per_layer)
                      / max(1, sum(s["occupancy_n"] for _, s in per_layer)))
    occupancy_first = float(sum(s["occupancy_first"] for _, s in per_layer)
                            / len(per_layer))
    hist = {name: sum(s["hist"][name] for _, s in per_layer) for name in OP_NAMES}
    entropy_n = sum(s["entropy_n"] for _, s in per_layer)
    return dict(
        probe=dict(split="calibration", windows=len(rows),
                   mixers=len(model.blocks),
                   regime="eval, bf16 autocast, the eager walk the arm trains "
                          "on (collect=True)",
                   probe_wall_seconds=round(wall, 1)),
        op_rate_per_token_lane0={OP_NAMES[i]: sum(s["lane0"][i] for _, s
                                                  in per_layer) / decisions
                                 for i in range(len(OP_NAMES))},
        op_histogram_all_lanes=hist,
        decisions_per_token=1.0,
        occupancy=dict(
            note=("live slots at every token, from the walk's recorded "
                  "pre-decision liveness channel (the per-token analog of "
                  "the sweep's chunk-entry occupancy)"),
            mean_live_slots=float(occupancy_mean),
            mean_live_slots_first_token=occupancy_first,
            rank=int(per_layer[0][1]["rank"])),
        read_weight_entropy=dict(
            units="nats over the live-slot weight distribution, per token",
            mean=float(sum(s["entropy_sum"] for _, s in per_layer)
                       / max(1, entropy_n)),
            tokens_with_mass=entropy_n),
        per_layer=by_layer)


# ------------------------------------------------------------------- the arm


def train_arm(args) -> dict:
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

    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", 128, args.seed, k5b=False)
    model = convert_model_to_sequential(model, segment_tokens=SEGMENT_TOKENS)
    summary = dict(summary)
    summary["scan_path"] = "k6_sequential (per-token, the chunk-1 limit)"
    summary["segment_tokens"] = SEGMENT_TOKENS
    summary["chunk_tokens"] = None
    microbatch = optimizer_config["microbatch"]
    model = model.cuda().train()
    out = Path(args.runs_dir) / f"per_token_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    exposure = int(args.bytes)
    within_sweep = ("WITHIN-SWEEP comparison at ONE transferred LR (the gdn2 "
                    "nomination's, verbatim from the matched arms): the same "
                    "rule the banked chunk rows ran under")
    stopping_rule = (f"matched-EXPOSURE: train until {exposure} targets seen "
                     "(~20M, the probe exposure); the honest elapsed wall is "
                     "reported first-class")
    config = dict(probe="k6_sequential_matrix", arm="matrix_ops",
                  granularity="per-token (the chunk-1 limit of the dial)",
                  seed=args.seed, exposure_targets=exposure,
                  segment_tokens=SEGMENT_TOKENS, optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT,
                  windows_per_step=64, summary=summary, identity=identity,
                  bindings=dict(
                      matched_arms.registration_bindings(),
                      k6_preregistration=str(
                          ROOT / "PREREGISTRATION_SEQUENTIAL_MATRIX.md"),
                      k6_preregistration_sha256=digest(
                          ROOT / "PREREGISTRATION_SEQUENTIAL_MATRIX.md"),
                      k6_gates_receipt=str(RECEIPTS / "k6_gates.json"),
                      k6_gates_sha256=digest(RECEIPTS / "k6_gates.json")),
                  git=matched_arms.git_state(), stopping_rule=stopping_rule,
                  within_sweep=within_sweep)
    atomic_json(out / "start.json", dict(
        config, config_sha256=matched_arms.config_sha256(config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(granularity="per-token", card=card, seed=args.seed,
                          exposure=exposure,
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
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            step += 1
            targets += n
            row = dict(step=step, active_seconds=elapsed, bytes=targets,
                       loss=loss_sum, time_ns=time.time_ns())
            log.write(json.dumps(row) + "\n")
            log_rows.append(row)
            if step % 10 == 0:
                print(f"  per-token step {step} {elapsed:.0f}s "
                      f"{targets / 1e6:.1f}MB loss {loss_sum:.4f}", flush=True)

    # diagnostics on the trained model, then the calibration eval.
    diagnostics = probe_diagnostics(model, stream, windows=args.probe_windows)
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation path is recorded-read only
    eval_started = time.monotonic()
    result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
    eval_wall = time.monotonic() - eval_started

    receipt = dict(
        probe="k6_sequential_matrix", arm="matrix_ops",
        granularity="per-token (the chunk-1 limit of the dial)",
        seed=args.seed, within_sweep=within_sweep, stopping_rule=stopping_rule,
        exposure_bytes=targets, exposure_steps=step,
        eval_nll_nats_per_byte=result["score"], eval_split=result["eval_split"],
        eval_windows=result["eval_windows"], eval_targets=result["eval_targets"],
        walls=dict(train_seconds=round(elapsed, 1),
                   probe_seconds=diagnostics["probe"]["probe_wall_seconds"],
                   eval_seconds=round(eval_wall, 1)),
        bytes_per_s=targets / max(1e-9, elapsed),
        diagnostics=diagnostics,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            log_rows, keep_first=min(5, max(1, len(log_rows)))),
        banked_sweep_comparison=banked_sweep_rows(),
    )
    receipt_path = Path(args.receipts_dir) / "k6_probe.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("granularity", "eval_nll_nats_per_byte", "exposure_bytes",
                       "exposure_steps", "bytes_per_s")}, indent=1), flush=True)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--runs-dir", default=str(RUNS))
    a.add_argument("--receipts-dir", default=str(RECEIPTS))
    a.add_argument("--smoke", action="store_true",
                    help="harness shakedown: tiny exposure, smoke paths")
    t = sub.add_parser("table")
    t.add_argument("--receipts-dir", default=str(RECEIPTS))
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.smoke:
            args.runs_dir = str(Path(args.runs_dir) / "smoke")
            args.receipts_dir = str(Path(args.receipts_dir) / "smoke")
            args.bytes = 200_000
        train_arm(args)
        return
    print(json.dumps(banked_sweep_rows(), indent=1))


if __name__ == "__main__":
    main()
