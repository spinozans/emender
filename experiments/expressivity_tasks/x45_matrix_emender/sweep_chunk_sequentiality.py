"""The chunk-size sequentiality sweep: one model, one LR, one seed, ONLY the chunk dial.

Hypothesis under test (the operator's): the matrix form's quality loss comes from
disrupting the SEQUENTIAL character of the algorithm -- decisions and reads taken
against the chunk-ENTRY state at chunk granularity instead of per token.  The
registered read is chunk-constant by construction (``schedule.route_chunk`` takes
its softmax once per chunk against the entry state), and ``K23_READ.md`` section 7
measured what that costs: at the arm's shape, from a populated store, the
per-token-weighted read differs from the registered one by a MEDIAN 13 % / MAX
32 % of the read scale -- a first-order effect, not noise.  Chunk size is the one
dial that interpolates sequentiality WITHOUT touching registered semantics:
chunk 128 = decisions every 128 tokens and reads blind to in-chunk writes for 128
tokens; chunk 8 = near-per-token behaviour at the same thresholds.

Four arms of the SAME ``matrix_ops`` model, ONLY ``chunk_tokens`` varying:
**128, 64, 32, 8**.  Everything else is the matched-arms discipline verbatim
(``matched_arms.py``): the registered flagship stream (hash-bound), the
transferred gdn2-nomination optimizer, seed 3401, the K5B whole-scan fast path
(the host-cost knob whose schedule/reads/state/gradients are bit-equal to the
per-chunk stack by construction), and the document-disjoint calibration eval.
**This is a WITHIN-SWEEP comparison at one transferred LR**: it says what chunk
size does AT THAT LR, not what the best LR does at each chunk size -- the
tuned-LR confirmation is the named follow-up.

Exposure is matched-EXPOSURE, not matched-time: every arm trains to the same
byte count (default 20,000,000 targets, ~22 % of the 2700 s matched arms'
exposure -- large enough that the 13 %-of-read-scale effect of ``K23_READ.md``
section 7 is far above the eval's resolution, small enough that the chunk-8 arm,
which is several-fold slower per token, still fits the session).  The honest
elapsed wall is reported first-class per arm.

Per arm, in order:

1. **Parity, before any timing** (the chunk-128 receipt's discipline,
   ``matched_arms.parity``): the whole-scan op pair bit-equal to the per-chunk
   K5A+K4 arm on the recorded schedule fields, weights, reads, state, liveness
   and the three gradients, at THIS chunk size -- plus, added here, the
   mandatory decision flip test re-run at this chunk size (the K5A router vs
   the registered eager ``route_chunk`` on the same pinned chunk-entry states,
   zero flips), which pins the fast path's decisions to the registered
   router's at this chunk size.
2. Training on the K5B fast path under the matched-exposure rule.
3. **Diagnostics** on the trained model: per-token op-firing rates
   (append/correct/pop/merge/none, lane 0, per layer), live-slot occupancy at
   chunk entries (the scan's own liveness rule), and read-weight entropy --
   the op-rate-vs-chunk-size interaction is part of the answer, not noise.
4. The calibration eval (``lattice.evaluate``, identical boundary to the
   matched arms).

Deliverables: ``receipts/sweep_chunk_parity_c{N}.json`` and
``receipts/sweep_chunk_c{N}.json`` per arm, the running gradient table
``receipts/sweep_chunk_gradient.json``, and the verdict in
``RESULT_CHUNK_SEQUENTIALITY.md``.

Usage (see ``run_sweep_chunk_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.sweep_chunk_sequentiality \\
        arm --chunk-tokens 128
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
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    NONE, OP_NAMES, op_histogram)
from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan import (
    DISPATCH, whole_scan_record)
from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR
from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
    RoutingPolicy, route_chunk)
from experiments.expressivity_tasks.x45_matrix_emender.scan import routed_chunk_scan

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "sweep_runs"
CHUNKS = (128, 64, 32, 8)
# The standing seed, the matched arms' own; identical across every arm, so the
# four arms share init, data order and LR schedule and differ ONLY by chunk size.
SEED = 3401
# ~20M targets per arm (the probe exposure; see module docstring for the defence).
EXPOSURE_TARGETS = 20_000_000
# The diagnostics probe: full calibration windows, document-disjoint from train.
PROBE_WINDOWS = 4


# --------------------------------------------------------------------- parity


def _registered_path_check(chunk_tokens: int, policy: RoutingPolicy) -> dict:
    """The mandatory decision flip test at THIS chunk size (A-phase's bar).

    The banked K5B discipline (``test_k5b_whole_scan``'s flip test),
    parameterized by chunk size: the SAME pinned chunk-entry states from a real
    registered-path routed scan go into both routers -- the registered eager
    ``route_chunk`` and the K5A router (``routing_diff_route_chunk``) whose
    kernels the whole-scan walk runs -- and every lane of every schedule field
    must agree, zero flips.

    Together with ``matched_arms.parity`` above (the whole-scan record bit-equal
    to the per-chunk K5A+K4 arm), this pins the fast path's decisions to the
    registered eager router's at this chunk size.  A bitwise whole-scan
    comparison against the eager registered SCAN is deliberately NOT asserted:
    its state channel differs from the K4 kernel lane's by the banked
    reduction-order floors (K1/K23/K4 receipts), and the registered arm's own
    compiled path carries its banked inductor-vs-eager similarity delta -- so
    whole-scan bitwise equality against the eager scan is neither claimed nor
    true, and the flip test on pinned states is the decision-parity bar the
    registration actually set.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.k5a_routing_diff \
        import routing_diff_route_chunk
    from experiments.expressivity_tasks.x45_matrix_emender.test_k5b_whole_scan \
        import (REG_BATCH, REG_HEADS, REG_RANK, REG_TOKENS, REG_VALUE_DIM,
                SCHEDULE_FIELDS, distinct_stream)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                             REG_VALUE_DIM, 512, 77)
    records = []

    def router(index, offset, state, live, qc, pc, pol, rk):
        records.append((state.detach().clone(), live.detach().clone(), qc, pc))
        return route_chunk(index, offset, state, live, qc, pc, pol, rk)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        routed_chunk_scan(query, payload, decay, router=router, policy=policy,
                          rank=REG_RANK, rank_max=REG_RANK,
                          chunk_tokens=chunk_tokens, fused=False)
    flips = 0
    lanes = 0
    per_field = {name: 0 for name in SCHEDULE_FIELDS}
    for index, (state, live, qc, pc) in enumerate(records):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            eager = route_chunk(index, index * chunk_tokens, state, live, qc,
                                 pc, policy, REG_RANK)
            fused = routing_diff_route_chunk(index, index * chunk_tokens, state,
                                             live, qc, pc, policy, REG_RANK)
        for name in SCHEDULE_FIELDS:
            a, b = getattr(eager[0], name), getattr(fused[0], name)
            field_flips = int((a != b).sum())
            per_field[name] += field_flips
            flips += field_flips
            lanes += a.numel()
    if flips:
        raise AssertionError(f"decision flips vs the registered eager "
                             f"route_chunk path at chunk {chunk_tokens}: "
                             f"{flips} in {lanes} lanes, per field {per_field}")
    return dict(
        gate="decision flip test (A-phase's bar), re-run at this chunk size",
        reference="routing_diff_route_chunk (the router the whole-scan walk "
                  "runs) vs route_chunk (the registered eager router), on the "
                  "same pinned chunk-entry states from a real routed scan",
        fixture=dict(batch=REG_BATCH, tokens=REG_TOKENS, heads=REG_HEADS,
                     rank=REG_RANK, value_dim=REG_VALUE_DIM, distinct=512,
                     seed=77, precision="bf16 autocast"),
        chunks=len(records), flips=flips, lanes=lanes,
        per_field_flips=per_field,
        note=("with the whole-scan-record-vs-per-chunk-arm bit-equality above, "
              "this pins the fast path's decisions to the registered eager "
              "router's at this chunk size; the continuous channels' "
              "deviations against the eager scan are the banked K1/K23/K4 "
              "kernel-lane floors, not re-gated here"))


def parity_receipt(chunk_tokens: int, out_path: Path) -> dict:
    """The per-chunk-size bit-equality receipt, BEFORE any timing.

    Core: ``matched_arms.parity`` verbatim -- the whole-scan op pair (eager
    body, capture+replay, replay) bit-equal to the per-chunk K5A+K4 arm on the
    recorded schedule fields, weights, reads, state, liveness and the three
    gradients, at the registered shape under bf16 autocast, plus the mandatory
    op histogram.  This function adds the decision flip test against the
    registered eager router at this chunk size and walls the whole phase, so
    the receipt states its own cost.
    """
    started = time.monotonic()
    report = matched_arms.parity(chunk_tokens, str(out_path))
    registered = _registered_path_check(chunk_tokens, RoutingPolicy(ops=True))
    report["registered_eager_path_check"] = registered
    report["parity_wall_seconds"] = round(time.monotonic() - started, 1)
    report["sweep_note"] = ("the chunk-size sequentiality sweep's per-arm "
                            "parity gate, run before the arm's clock starts")
    out_path.write_text(json.dumps(report, indent=1) + "\n")
    return report


# ----------------------------------------------------------------- diagnostics


def _occupancy_walk(schedule, chunk_tokens: int, rank_max: int):
    """Live-slot count at every chunk entry, by the scan's own liveness rule.

    Exactly ``chunk_scan``'s integer bookkeeping (the amax over
    ``stamp*2 + live_after``, chunk-entry liveness as the stamp-0 state), walked
    over the recorded schedule.  Returns the per-chunk-entry live count
    ``(chunks, batch, heads)`` on the CPU and the final live count.
    """
    batch, tokens, heads, _lanes = schedule.op.shape
    device = schedule.op.device
    live = torch.zeros(batch, heads, int(rank_max), dtype=torch.int64,
                       device=device)
    entries = []
    for start in range(0, tokens, chunk_tokens):
        stop = min(start + chunk_tokens, tokens)
        entries.append(live.sum(-1))
        op = schedule.op[:, start:stop].permute(0, 2, 1, 3).reshape(batch, heads, -1)
        slot = schedule.slot[:, start:stop].permute(0, 2, 1, 3).reshape(batch, heads, -1)
        after = schedule.live_after[:, start:stop].permute(0, 2, 1, 3).reshape(
            batch, heads, -1)
        steps = op.shape[-1]
        stamp = torch.arange(1, steps + 1, device=device).view(1, 1, steps)
        key = torch.where(op.ne(NONE), stamp * 2 + after.to(torch.int64),
                          torch.zeros_like(stamp))
        live.scatter_reduce_(2, slot, key, reduce="amax", include_self=True)
        live = live % 2          # the key's low bit is the liveness, as in
        # ``chunk_scan``'s ``(last % 2).to(torch.bool)`` exit rule
    return torch.stack(entries).to("cpu"), live.sum(-1).to("cpu")


def _record_stats(record, chunk_tokens: int) -> dict:
    """The diagnostics of one whole-scan record, reduced to CPU numbers."""
    schedule = record.schedule
    batch, tokens, heads, _lanes = schedule.op.shape
    lane0 = torch.bincount(schedule.op[..., 0].reshape(-1).to("cpu"),
                           minlength=len(OP_NAMES))
    entries, final = _occupancy_walk(schedule, chunk_tokens,
                                     int(record.state.shape[2]))
    rank = int(record.rank)
    w = record.weights[..., :rank].float()
    mass = w.sum(-1)
    has = mass > 0
    probs = torch.where(has.unsqueeze(-1), w / mass.clamp_min(1e-30).unsqueeze(-1),
                        torch.zeros_like(w))
    contrib = torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs))
    entropy = -contrib.sum(-1)                                   # (b, t, h)
    populated = (torch.arange(tokens, device=entropy.device)
                 >= chunk_tokens).view(1, tokens, 1)
    return dict(
        lane0=[int(v) for v in lane0],
        hist=op_histogram(schedule),
        entries=entries,                                          # (chunks, b, h)
        final_live=final,
        entropy_sum=float(entropy[has].sum()), entropy_n=int(has.sum()),
        entropy_sum_populated=float(entropy[has & populated].sum()),
        entropy_n_populated=int((has & populated).sum()),
        decisions=int(batch * tokens * heads), rank=rank)


def probe_diagnostics(model, stream, chunk_tokens: int,
                       windows: int = PROBE_WINDOWS) -> dict:
    """Op rates, occupancy and read-weight entropy of the TRAINED model.

    A forward hook on every mixer recomputes the mixer's own query/payload/decay
    from its input and runs ``whole_scan_record`` -- the same eager walk whose
    graph-captured form the arm trained on, with the recorded decision channel
    exposed -- under the training's own regime (bf16 autocast, eval mode).  The
    probe windows are the calibration split's first full windows: the same
    document-disjoint split the eval uses, the same windows for every arm.
    """
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
                record = whole_scan_record(
                    query, payload, decay, policy=mixer.policy,
                    rank=mixer.rank, rank_max=mixer.rank_max,
                    chunk_tokens=mixer.chunk_tokens)
            with torch.no_grad():
                per_layer.append((layer_index, _record_stats(record,
                                                             mixer.chunk_tokens)))
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
            occupancy_mean_live_slots=float(torch.cat(
                [s["entries"].flatten() for s in stats]).float().mean()),
            read_weight_entropy_nats=float(sum(s["entropy_sum"] for s in stats)
                                           / max(1, sum(s["entropy_n"]
                                                        for s in stats))))
    decisions = sum(s["decisions"] for _, s in per_layer)
    rank = per_layer[0][1]["rank"] if per_layer else 0
    entries = torch.cat([s["entries"].flatten() for _, s in per_layer]).float()
    entropy_n = sum(s["entropy_n"] for _, s in per_layer)
    entropy_n_pop = sum(s["entropy_n_populated"] for _, s in per_layer)
    hist = {name: sum(s["hist"][name] for _, s in per_layer)
            for name in OP_NAMES}
    return dict(
        probe=dict(split="calibration", windows=len(rows),
                   mixers=len(model.blocks), regime="eval, bf16 autocast, "
                   "whole_scan_record (the eager walk behind the graph)",
                   probe_wall_seconds=round(wall, 1)),
        op_rate_per_token_lane0={OP_NAMES[i]: sum(s["lane0"][i] for _, s
                                                  in per_layer) / decisions
                                 for i in range(len(OP_NAMES))},
        op_histogram_all_lanes=hist,
        chunk_entry_events_per_token=1.0 / chunk_tokens,
        occupancy=dict(
            rank=int(rank),
            mean_live_slots_at_chunk_entry=float(entries.mean()),
            mean_live_fraction_at_chunk_entry=float(entries.mean()
                                                     / max(1, rank)),
            mean_live_slots_first_entry=float(entries[0].float().mean()),
            mean_live_slots_final=float(torch.cat(
                [s["final_live"].flatten() for _, s in per_layer]).float().mean())),
        read_weight_entropy=dict(
            units="nats over the live-slot weight distribution, per token",
            mean=float(sum(s["entropy_sum"] for _, s in per_layer)
                       / max(1, entropy_n)),
            mean_populated_store=float(sum(s["entropy_sum_populated"]
                                           for _, s in per_layer)
                                       / max(1, entropy_n_pop)),
            tokens_with_mass=entropy_n, tokens_total=decisions),
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
    chunk = int(args.chunk_tokens)
    if chunk not in CHUNKS and not args.smoke:
        raise ValueError(f"chunk {chunk} outside the registered sweep "
                         f"{CHUNKS} (use --smoke for harness shakedowns)")

    # 1. parity, before any timing (the chunk-128 receipt's discipline).
    parity_path = Path(args.receipts_dir) / f"sweep_chunk_parity_c{chunk}.json"
    parity = parity_receipt(chunk, parity_path)

    # 2. the arm: the matched-arms build order verbatim, ONLY chunk varying.
    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", chunk, args.seed, k5b=True)
    microbatch = optimizer_config["microbatch"]
    out = Path(args.runs_dir) / f"chunk_{chunk}"
    out.mkdir(parents=True, exist_ok=False)
    exposure = int(args.bytes)
    stopping_rule = (f"matched-EXPOSURE: train until {exposure} targets seen "
                     "(~20M, the probe exposure; see the module docstring for "
                     "the defence); the honest elapsed wall is reported "
                     "first-class")
    within_sweep = ("WITHIN-SWEEP comparison at ONE transferred LR (the gdn2 "
                    "nomination's, verbatim from the matched arms): the sweep "
                    "says what chunk size does at that LR; the tuned-LR "
                    "confirmation is the named follow-up")
    config = dict(sweep="chunk_sequentiality", arm="matrix_ops",
                  chunk_tokens=chunk, seed=args.seed, exposure_targets=exposure,
                  chunks_registered_sweep=list(CHUNKS), optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT, windows_per_step=64,
                  summary=summary, identity=identity,
                  bindings=matched_arms.registration_bindings(),
                  git=matched_arms.git_state(), stopping_rule=stopping_rule,
                  within_sweep=within_sweep,
                  parity_receipt=str(parity_path),
                  parity_receipt_sha256=matched_arms.digest(parity_path))
    atomic_json(out / "start.json", dict(
        config, config_sha256=matched_arms.config_sha256(config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(chunk_tokens=chunk, card=card, seed=args.seed,
                          exposure=exposure,
                          parameters=sum(p.numel()
                                         for p in model.parameters()))), flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets = 0
    log_rows = []
    with (out / "training.jsonl").open("w", buffering=1) as log:
        # the honest clock of the matched arms, verbatim: started at step 1,
        # includes the stream read, compile/capture, fwd/bwd, clip, AdamW; the
        # parity phase above is a separate receipted phase whose shared kernel
        # warm-up is identical across arms.
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
                # the K5B discipline: required before every graphed invocation
                torch.compiler.cudagraph_mark_step_begin()
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
                print(f"  chunk {chunk} step {step} {elapsed:.0f}s "
                      f"{targets / 1e6:.1f}MB loss {loss_sum:.4f}", flush=True)
    dispatch = dict(DISPATCH)

    # 3. diagnostics on the trained model, then 4. the calibration eval.
    diagnostics = probe_diagnostics(model, stream, chunk,
                                    windows=args.probe_windows)
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation path is recorded-read only
    eval_started = time.monotonic()
    result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
    eval_wall = time.monotonic() - eval_started

    receipt = dict(
        sweep="chunk_sequentiality", arm="matrix_ops", chunk_tokens=chunk,
        seed=args.seed, within_sweep=within_sweep, stopping_rule=stopping_rule,
        exposure_bytes=targets, exposure_steps=step,
        exposure_note=("~20M targets ~= 22% of the 2700 s matched arms' "
                       "exposure (91.9 MB): enough that the 13%-of-read-scale "
                       "chunk-constant-read effect (K23_READ.md section 7) is "
                       "far above the 14,336-target eval's resolution, and "
                       "small enough that the several-fold-slower chunk-8 arm "
                       "fits the session"),
        eval_nll_nats_per_byte=result["score"], eval_split=result["eval_split"],
        eval_windows=result["eval_windows"], eval_targets=result["eval_targets"],
        walls=dict(parity_seconds=parity["parity_wall_seconds"],
                   train_seconds=round(elapsed, 1),
                   probe_seconds=diagnostics["probe"]["probe_wall_seconds"],
                   eval_seconds=round(eval_wall, 1)),
        bytes_per_s=targets / max(1e-9, elapsed),
        diagnostics=diagnostics, whole_scan_dispatch=dispatch,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            log_rows, keep_first=min(5, max(1, len(log_rows)))),
        parity=dict(receipt=str(parity_path),
                    sha256=matched_arms.digest(parity_path),
                    bit_equal_channels=parity["bit_equal_channels"],
                    bit_equal_gradients=parity["bit_equal_gradients"],
                    decision_flip_test=parity["registered_eager_path_check"]))
    receipt_path = Path(args.receipts_dir) / f"sweep_chunk_c{chunk}.json"
    atomic_json(receipt_path, receipt)
    _update_gradient_table(args.receipts_dir)
    print(json.dumps({k: receipt[k] for k in
                      ("chunk_tokens", "eval_nll_nats_per_byte", "exposure_bytes",
                       "exposure_steps", "bytes_per_s")},
                     indent=1), flush=True)
    return receipt


def _update_gradient_table(receipts_dir: str) -> dict:
    """The running gradient table: every landed arm, one row each.

    Updated after each arm so the sweep's evidence survives a session ceiling
    even before the prose verdict is written.
    """
    rows = []
    for path in sorted(Path(receipts_dir).glob("sweep_chunk_c*.json")):
        if path.name.startswith("sweep_chunk_parity"):
            continue
        r = json.loads(path.read_text())
        d = r["diagnostics"]
        rows.append(dict(chunk_tokens=r["chunk_tokens"],
                         eval_nll_nats_per_byte=r["eval_nll_nats_per_byte"],
                         exposure_bytes=r["exposure_bytes"],
                         exposure_steps=r["exposure_steps"],
                         train_seconds=r["walls"]["train_seconds"],
                         bytes_per_s=r["bytes_per_s"],
                         op_rate_per_token_lane0=d["op_rate_per_token_lane0"],
                         occupancy_mean_live_slots=(
                             d["occupancy"]["mean_live_slots_at_chunk_entry"]),
                         read_weight_entropy_nats=(
                             d["read_weight_entropy"]["mean_populated_store"]),
                         receipt=str(path)))
    rows.sort(key=lambda row: -row["chunk_tokens"])
    table = dict(sweep="chunk_sequentiality", seed=SEED, chunks=list(CHUNKS),
                 within_sweep="one transferred LR (see the receipts); the "
                              "tuned-LR confirmation is the named follow-up",
                 complete=len(rows) == len(CHUNKS), rows=rows)
    atomic_json(Path(receipts_dir) / "sweep_chunk_gradient.json", table)
    return table


def table(args) -> dict:
    L.check_python()
    data = _update_gradient_table(args.receipts_dir)
    print(json.dumps(data, indent=1))
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--chunk-tokens", type=int, required=True)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--runs-dir", default=str(RUNS))
    a.add_argument("--receipts-dir", default=str(RECEIPTS))
    a.add_argument("--smoke", action="store_true",
                   help="harness shakedown: any chunk, write-once dirs still "
                        "enforced, receipts named sweep_chunk_c{N}.smoke.json")
    t = sub.add_parser("table")
    t.add_argument("--receipts-dir", default=str(RECEIPTS))
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.smoke:
            args.runs_dir = str(Path(args.runs_dir) / "smoke")
            args.receipts_dir = str(Path(args.receipts_dir) / "smoke")
        train_arm(args)
        return
    table(args)


if __name__ == "__main__":
    main()
