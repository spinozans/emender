"""The 2700 s chunk-8 board arm: the sequentiality dial measured at board scale.

The extension lane's own verdict (``RESULT_CHUNK_SEQUENTIALITY.md``, next-steps
item 2): the chunk-8 board arm is worth its GPU hour because chunk 8 is NOT at
the dial's plateau (the 8->4 probe step still buys -0.0395).  This driver runs
that arm -- the board-scale measurement of the chunk-size sequentiality dial,
the question the probe sweep could not answer: **how much of the 0.5600-nat
board gap to GDN2 (1.1976 same-session) does the dial close at 150M scale?**

The arm is the banked board arm (``matched_arms.py``, commit b52d4472: the
``matrix_ops`` arm, 1.7576 nats/byte, 832 steps, 91.87 MB, 2702.7 s) with
**only ``chunk_tokens`` changed to 8** -- everything else identical by
construction: seed 3401, the transferred gdn2-nomination optimizer (the matched
arms' standing deconfound), the K5B whole-scan fast path, the registered
flagship stream, the honest 2700-active-second stopping rule (the clock starts
at step 1; includes the stream read, any compile/capture, fwd/bwd, clip, AdamW;
no untimed warmup steps), the 150M board's own calibration eval boundary
(``lattice.evaluate``, full 2048-byte windows, batch 1, recorded-read path),
microbatch 8, 64 windows per step.

**Matched-TIME, not matched-exposure -- stated because it biases AGAINST this
arm**: chunk 8 runs the K5B graph at ~19.1k tok/s (the probe sweep's measured
rate) against the chunk-128 arm's ~34.0k, so at matched 2700 s this arm sees
only ~56 % of the chunk-128 arm's bytes.  The exposure shortfall is a
handicap, not a confound removed: any dial gain this arm shows is a LOWER bound
on the dial's matched-exposure effect at board scale.

Per-arm protocol (the sweep's ``arm`` discipline, only the stopping rule
changed to the board's seconds rule):

1. **Parity, before any timing** (``sweep_chunk_sequentiality.parity_receipt``):
   the whole-scan op pair bit-equal to the per-chunk K5A+K4 arm on the recorded
   schedule fields, weights, reads, state, liveness and the three gradients at
   chunk 8, PLUS the mandatory decision flip test re-run at chunk 8 (the K5A
   router vs the registered eager ``route_chunk`` on pinned chunk-entry states,
   0 flips).  A NEW receipt (``receipts/board_chunk8_parity.json``); the banked
   sweep parity receipt (``receipts/sweep_chunk_parity_c8.json``) is not
   touched.
2. Training on the K5B fast path under the board's honest 2700 s clock (the
   ``matched_arms.train`` loop verbatim, only ``chunk_tokens`` = 8).
3. **Diagnostics on the trained model** (``sweep_chunk_sequentiality.
   probe_diagnostics``): per-token op-firing rates, live-slot occupancy at chunk
   entries, read-weight entropy -- the op-rate columns the board table reports.
4. The calibration eval, identical boundary to the board arms.

Deliverables: ``receipts/board_chunk8_parity.json`` and
``receipts/board_chunk8.json`` (the arm's receipt: config hash, stopping-rule
ledger, eval NLL, bytes/steps/tok/s, op-rate diagnostics), the run directory
``board_runs/matrix_ops_c8/``, and the three-row addendum table in
``RESULT_BOARD_CHUNK8.md``.

Usage (see ``run_board_chunk8_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.board_chunk8 arm
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
from experiments.expressivity_tasks.x45_matrix_emender import (
    sweep_chunk_sequentiality as sweep)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "board_runs"
# The board arm's chunk: the extension lane's own verdict (RESULT_CHUNK_
# SEQUENTIALITY.md next-steps item 2) -- chunk 8 is the right matched-time
# board point (~56 % of the chunk-128 arm's bytes at 19.1k tok/s; chunk 4
# would see only ~35 % and cost 2.8x the wall at matched exposure).
CHUNK_TOKENS = 8
SECONDS = 2700.
SEED = 3401


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

    # 1. parity, before any timing (a NEW receipt; the banked sweep parity
    #    receipts are not modified).
    parity_path = Path(args.receipts_dir) / "board_chunk8_parity.json"
    parity = sweep.parity_receipt(chunk, parity_path)
    parity["board_note"] = ("the chunk-8 board arm's parity gate, run before "
                            "the arm's honest clock starts; the same chunk-8 "
                            "discipline as the sweep's "
                            "sweep_chunk_parity_c8.json, re-receipted for the "
                            "board arm")
    parity_path.write_text(json.dumps(parity, indent=1) + "\n")

    # 2. the arm: the matched-arms build order verbatim, ONLY chunk varying.
    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", chunk, args.seed, k5b=True)
    microbatch = optimizer_config["microbatch"]
    out = Path(args.runs_dir) / f"matrix_ops_c{chunk}"
    out.mkdir(parents=True, exist_ok=False)
    stopping_rule = (f"{args.seconds:g} active seconds (the registered rule, "
                     "the banked board arms' own): the clock starts at step 1 "
                     "and includes the stream read, any compile/capture, "
                     "fwd/bwd, clip, AdamW; no untimed warmup steps for any "
                     "arm")
    matched = ("matched-TIME, not matched-exposure: steps and bytes are "
               "first-class deconfound columns.  Chunk 8 runs the K5B graph "
               "several-fold slower per token than chunk 128, so this arm "
               "sees FEWER bytes at matched time -- the shortfall biases "
               "AGAINST the chunk-8 arm; any dial gain it shows is a lower "
               "bound on the dial's matched-exposure effect at board scale")
    within_sweep = ("WITHIN-BOARD comparison at ONE transferred LR (the gdn2 "
                    "nomination's, verbatim from the banked board arms): the "
                    "arm says what chunk size does at that LR at board scale; "
                    "the tuned-LR confirmation is the CMA-ES lane's, running")
    config = dict(board="chunk8_matched_time", arm="matrix_ops",
                  chunk_tokens=chunk, seed=args.seed, seconds=args.seconds,
                  bytes_target=None, eval_split=args.eval_split,
                  optimizer=optimizer_config, microbatch=microbatch,
                  context=L.CONTEXT, windows_per_step=64, summary=summary,
                  identity=identity,
                  bindings=matched_arms.registration_bindings(),
                  git=matched_arms.git_state(), stopping_rule=stopping_rule,
                  matched=matched, within_sweep=within_sweep,
                  parity_receipt=str(parity_path),
                  parity_receipt_sha256=matched_arms.digest(parity_path))
    atomic_json(out / "start.json", dict(
        config, config_sha256=matched_arms.config_sha256(config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(arm="matrix_ops", chunk_tokens=chunk, card=card,
                          seed=args.seed, seconds=args.seconds,
                          parameters=sum(p.numel() for p in model.parameters()))),
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
        # the honest clock of the matched arms, verbatim: started at step 1,
        # includes the stream read, compile/capture, fwd/bwd, clip, AdamW; the
        # parity phase above is a separate receipted phase whose shared kernel
        # warm-up sits outside every arm's clock.
        while elapsed < args.seconds:
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
                # the K5B discipline (gates.py one_step): required before
                # every graphed invocation in a training loop.
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
            if step % 25 == 0:
                print(f"  chunk {chunk} step {step} {elapsed:.0f}s "
                      f"{targets / 1e6:.1f}MB loss {loss_sum:.4f}", flush=True)
    from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan \
        import DISPATCH
    dispatch = dict(DISPATCH)

    # 3. diagnostics on the trained model, then 4. the calibration eval
    #    (identical boundary to the banked board arms).
    diagnostics = sweep.probe_diagnostics(model, stream, chunk,
                                          windows=args.probe_windows)
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation path is recorded-read only
    eval_started = time.monotonic()
    result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
    eval_wall = time.monotonic() - eval_started

    receipt = dict(
        board="chunk8_matched_time", arm="matrix_ops", chunk_tokens=chunk,
        seed=args.seed, within_sweep=within_sweep, stopping_rule=stopping_rule,
        matched=matched,
        exposure_bytes=targets, exposure_steps=step,
        exposure_note=("matched-TIME arm: chunk 8's K5B throughput is "
                       "several-fold below chunk 128's, so this arm sees fewer "
                       "bytes than the banked chunk-128 board arm at the same "
                       "2700 s -- the exposure shortfall biases AGAINST the "
                       "chunk-8 arm (a lower bound on the dial's "
                       "matched-exposure effect at board scale)"),
        eval_nll_nats_per_byte=result["score"], eval_split=result["eval_split"],
        eval_windows=result["eval_windows"], eval_targets=result["eval_targets"],
        walls=dict(parity_seconds=parity["parity_wall_seconds"],
                   train_seconds=round(elapsed, 1),
                   probe_seconds=diagnostics["probe"]["probe_wall_seconds"],
                   eval_seconds=round(eval_wall, 1)),
        bytes_per_s=targets / max(1e-9, elapsed),
        stopping_ledger=dict(
            rule="active seconds", requested=args.seconds, actual=elapsed,
            steps=step, bytes=targets,
            last_step_row=dict(log_rows[-1]) if log_rows else None),
        diagnostics=diagnostics, whole_scan_dispatch=dispatch,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(log_rows),
        parity=dict(receipt=str(parity_path),
                    sha256=matched_arms.digest(parity_path),
                    bit_equal_channels=parity["bit_equal_channels"],
                    bit_equal_gradients=parity["bit_equal_gradients"],
                    decision_flip_test=parity["registered_eager_path_check"]))
    receipt_path = Path(args.receipts_dir) / "board_chunk8.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("chunk_tokens", "eval_nll_nats_per_byte", "exposure_bytes",
                       "exposure_steps", "bytes_per_s", "walls")},
                     indent=1), flush=True)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    a.add_argument("--seconds", type=float, default=SECONDS)
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=sweep.PROBE_WINDOWS)
    a.add_argument("--runs-dir", default=str(RUNS))
    a.add_argument("--receipts-dir", default=str(RECEIPTS))
    args = ap.parse_args()
    if args.cmd == "arm":
        train_arm(args)


if __name__ == "__main__":
    main()
