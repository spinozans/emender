"""The 150M matched-quality arms: gdn2 vs matrix_append vs matrix_ops, 2700 s.

The experiment the speed program was built to enable
(``PREREGISTRATION_MATRIX_EMENDER.md``, Wave A at fixed rank): do the five
management ops earn their cost in QUALITY at matched wall-clock, at parameter
parity?  This harness runs the three parameter-matched (~149.9 M) arms on the
registered flagship stream under the registered honest stopping rule -- **2700
active seconds**, the ``x42_pile_scale/lattice.py`` convention mirrored here
without touching that file (it stays read-only, the registered runner) -- and
evaluates each on the registered calibration split with the 150 M board's eval
boundary (24 full windows max, batch 1, recorded-read path).

Arms:

``gdn2``           the registered ``lattice_gdn2`` configuration verbatim: its
                   own implementation (external Triton chunk kernel, eager),
                   the searched nomination's optimizer, microbatch 8.
``matrix_append``  the x45 matrix arm, APPEND only (``RoutingPolicy(ops=False)``).
``matrix_ops``     the x45 matrix arm, the five-op registered policy.

Both matrix arms run **with the K5B fast path enabled**
(``fused_whole_scan=True`` behind the compiled wrapper): a host-cost knob whose
schedule, reads, state, liveness and gradients are bit-equal to the registered
per-chunk path by construction (``receipts/k5b_whole_scan.json``).  The
recorded gate-D receipts (``cd_*.json``) measured this model at 13.2 k / 15.9 k
tok/s against gdn2's 28.1 k; the K5B path removes the host gap, which is what
makes matched-time arms no longer confounded by a 3x throughput handicap.

Matched-TIME, not matched-exposure (stated as the 150 M board did): every arm
trains for the same 2700 active seconds, so the faster arm sees more bytes.
Steps and bytes are reported as first-class deconfound columns.  The clock is
the registered one: it starts at step 1 and includes everything the arm's
process does per step -- the flagship-stream read, any compile/capture the
arm's shipped mode performs on its first steps, forward/backward, clip, AdamW.
There are NO untimed warmup steps, for any arm.

Single seed (3401, the standing seed), stated in every receipt.

Subcommands:

``train``    one arm, one 2700 s run, ``start.json`` + ``training.jsonl`` +
             ``result.json`` in the output directory (the ``runs_150m``
             convention).
``parity``   the K5B gate-A construction claim re-asserted at the run's chunk
             length (the banked receipts are at chunk 64; the cd-protocol
             configuration is chunk 128): the whole-scan op pair bit-equal to
             the per-chunk K5A+K4 arm on reads, state, liveness and the three
             gradients, at the registered shape under bf16 autocast.

Usage (see ``run_matched_arms_leased.sh`` for the detached leased launcher):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.matched_arms train \\
        --arm matrix_ops --seconds 2700 --seed 3401 --output runs_150m_matched/matrix_ops
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
    grouped, model_for)
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
    configure_production)
from experiments.expressivity_tasks.x45_matrix_emender.model import (
    COMPILE_MODE, matrix_config, model_for as matrix_model_for,
    parameter_count)
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
X43 = REPO / "experiments/expressivity_tasks/x43_fork_resolved"
RUNS = X43 / "runs_150m_matched"
MATRIX_REG = X43 / "PREREGISTRATION_MATRIX_EMENDER.md"
X45_RECEIPTS = ROOT / "receipts"
ARMS = ("gdn2", "matrix_append", "matrix_ops")
# The registered matrix-arm rank (the precompiled-set rank the cd gate receipts
# and both parameter counts use): capacity at fixed parameters, not a parameter.
RANK = 128
# The cd-protocol routing granularity (cd_*_c128.json); the K5B parity row for
# this chunk length is banked by the ``parity`` subcommand before any arm runs.
CHUNK_TOKENS = 128


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def session_meta(card: str) -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(host=os.uname().nodename, card=card,
                device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2 ** 30, 2),
                torch=torch.__version__,
                triton=__import__("triton").__version__)


def git_state() -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    relevant = ["experiments/expressivity_tasks/x42_pile_scale/lattice.py",
                "experiments/expressivity_tasks/x45_matrix_emender/"]
    dirty = git("status", "--porcelain", "--", *relevant)
    return dict(commit=git("rev-parse", "HEAD"),
                relevant_paths_dirty=dirty, branch=git("branch", "--show-current"))


def registration_bindings() -> dict:
    """Hash-bind every authority these arms run under, before measuring."""
    k5b = X45_RECEIPTS / "k5b_whole_scan.json"
    k5b_step = X45_RECEIPTS / "k5b_step_metric.json"
    k5b_doc = ROOT / "K5B_WHOLE_SCAN.md"
    return dict(
        matrix_registration=str(MATRIX_REG),
        matrix_registration_sha256=digest(MATRIX_REG),
        lattice_registration=L.flagship_identity()["lattice_registration"],
        lattice_registration_sha256=L.flagship_identity()["lattice_sha256"],
        k5b_whole_scan_receipt=str(k5b), k5b_whole_scan_sha256=digest(k5b),
        k5b_step_metric_receipt=str(k5b_step),
        k5b_step_metric_sha256=digest(k5b_step),
        k5b_doc=str(k5b_doc), k5b_doc_sha256=digest(k5b_doc),
        recorded_cd_receipts={
            "gdn2_c128_rerun": digest(X45_RECEIPTS / "cd_gdn2_c128_rerun.json"),
            "matrix_append_c128": digest(X45_RECEIPTS / "cd_matrix_append_c128.json"),
            "matrix_ops_c128": digest(X45_RECEIPTS / "cd_matrix_ops_c128.json")},
        recorded_150m_gdn2_result=digest(
            X43 / "runs_150m/lattice_gdn2/result.json"),
    )


def config_sha256(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def build_gdn2(seed: int):
    """The registered lattice_gdn2 arm, the lattice.train build order verbatim."""
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    configure_production()
    base, policy, nom = L.build_policy("lattice_gdn2")
    model = model_for(base, policy).cuda().train()
    optimizer = dict(lr=policy.lr, betas=(0.9, policy.beta2),
                    weight_decay=policy.weight_decay,
                    warmup_steps=policy.warmup_steps,
                    microbatch=policy.microbatch)
    summary = dict(base_arm=base,
                   parameters=dict(total=sum(p.numel() for p in model.parameters())),
                   nomination={k: nom[k] for k in
                               ("arm", "receipt", "receipt_sha256", "generation",
                                "candidate", "search_score")},
                   policy=dict(lr=policy.lr, microbatch=policy.microbatch,
                               beta2=policy.beta2, weight_decay=policy.weight_decay,
                               warmup_steps=policy.warmup_steps,
                               init_scale=policy.init_scale,
                               decay_rate_scale=policy.decay_rate_scale,
                               dt_bias_shift=policy.dt_bias_shift,
                               expansion=policy.expansion))
    return model, optimizer, summary


def build_matrix(arm: str, chunk_tokens: int, seed: int):
    """A matrix arm with the K5B fast path enabled, the gates.build order verbatim.

    The optimizer is the registered gdn2 nomination's settings (the x45 lane's
    standing deconfound, identical numbers to the cd gate receipts).
    """
    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _, nomination_policy, _ = L.build_policy("lattice_gdn2")
    policy = RoutingPolicy(ops=arm == "matrix_ops")
    shape = matrix_config(rank=RANK)
    model = matrix_model_for(shape, policy, chunk_tokens, compile_scan=True,
                             fused_whole_scan=True).cuda().train()
    optimizer = dict(lr=nomination_policy.lr,
                    betas=(0.9, nomination_policy.beta2),
                    weight_decay=nomination_policy.weight_decay,
                    warmup_steps=nomination_policy.warmup_steps,
                    microbatch=nomination_policy.microbatch)
    summary = dict(base_arm="matrix", shape=shape.__dict__,
                   parameters=parameter_count(shape), routing=policy.record(),
                   chunk_tokens=chunk_tokens, compile_scan=True,
                   compile_mode=COMPILE_MODE, fused_whole_scan=True,
                   fused_scan=False, fused_backward=False,
                   fused_routing=False, fused_routing_diff=False,
                   optimizer_note=("the registered gdn2 nomination's optimizer "
                                   "settings (gates.py / profile_step_vs_gdn2.py)"))
    return model, optimizer, summary


def trajectory_sample(log_rows: list, keep_first=5, keep_last=5, stride=25) -> list:
    steps = {r["step"]: dict(step=r["step"], active_seconds=r["active_seconds"],
                            bytes=r["bytes"], loss=r["loss"]) for r in log_rows}
    keep = set(range(1, keep_first + 1))
    keep |= {s for s in steps if s % stride == 0}
    keep |= {s for s in steps if s > max(steps, default=0) - keep_last}
    return [steps[s] for s in sorted(keep)]


def train(args) -> dict:
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = args.device or os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None:
        raise ValueError("no GPU lease: set GPU_LEASE_HELD (or --device) before training")
    if visible is None:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set before launch")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: recorded card {card} vs "
                         f"CUDA_VISIBLE_DEVICES {visible}")

    if args.arm == "gdn2":
        model, optimizer_config, summary = build_gdn2(args.seed)
        microbatch = optimizer_config["microbatch"]
    elif args.arm in ("matrix_append", "matrix_ops"):
        model, optimizer_config, summary = build_matrix(args.arm, args.chunk_tokens,
                                                         args.seed)
        microbatch = optimizer_config["microbatch"]
    else:
        raise ValueError(f"unknown arm {args.arm!r}; expected {ARMS}")

    config = dict(arm=args.arm, seed=args.seed, seconds=args.seconds,
                  eval_split=args.eval_split, optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT,
                  windows_per_step=64, summary=summary, identity=identity,
                  bindings=registration_bindings(), git=git_state(),
                  stopping_rule=("2700 active seconds (the registered rule): the "
                                 "clock starts at step 1 and includes the stream "
                                 "read, any compile/capture, fwd/bwd, clip, AdamW; "
                                 "no untimed warmup steps for any arm"),
                  matched=("matched-TIME, not matched-exposure: steps and bytes "
                           "are first-class deconfound columns"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        config, config_sha256=config_sha256(config), pid=os.getpid(),
        session=session_meta(card), started_ns=time.time_ns(),
        parameters=dict(total=sum(p.numel() for p in model.parameters()),
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(arm=args.arm, card=card, seed=args.seed,
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
    is_matrix = args.arm != "gdn2"
    with (out / "training.jsonl").open("w", buffering=1) as log:
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
                if is_matrix:
                    # The K5B discipline (gates.py one_step): required before
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
                print(f"  {args.arm} step {step} {elapsed:.0f}s {targets/1e6:.1f}MB "
                      f"loss {loss_sum:.4f}", flush=True)
    if is_matrix:
        from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan \
            import DISPATCH
        dispatch = dict(DISPATCH)
    else:
        dispatch = None
    os.environ["X40_BATCHED_READS"] = "0"  # evaluation path is recorded-read only
    result = L.evaluate(model, stream, args.arm, args.eval_split)
    result.update(status="completed", score_eligible=True, arm=args.arm,
                  seed=args.seed, units="nats/byte",
                  train_seconds=elapsed, steps=step, bytes=targets,
                  bytes_per_s=targets / max(1e-9, elapsed),
                  identity=identity, config_sha256=config_sha256(config),
                  config=config, session=session_meta(card),
                  peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
                  whole_scan_dispatch=dispatch,
                  trajectory=trajectory_sample(log_rows))
    atomic_json(out / "result.json", result)
    print(json.dumps({k: result[k] for k in
                      ("arm", "score", "eval_split", "eval_windows", "eval_targets",
                       "steps", "bytes", "train_seconds", "bytes_per_s")}, indent=2),
          flush=True)
    return result


def parity(chunk_tokens: int, out_path: str) -> dict:
    """K5B gate A at the run's chunk length: the construction claim, re-asserted.

    The banked K5B parity receipts are at chunk 64; the cd-protocol
    configuration this board runs is chunk 128.  The whole-scan op pair (eager
    body, capture+replay, replay) must be bit-equal to the per-chunk K5A+K4
    arm on the recorded schedule fields, weights, reads, state, liveness and
    the three gradients, at the registered shape under bf16 autocast.
    """
    from experiments.expressivity_tasks.x45_matrix_emender.test_k5b_whole_scan \
        import (REG_BATCH, REG_HEADS, REG_RANK, REG_TOKENS, REG_VALUE_DIM,
                 SCHEDULE_FIELDS, assert_graph_bit_equal_to_arm,
                 assert_record_bit_equal_to_arm, distinct_stream)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra \
        import op_histogram
    from experiments.expressivity_tasks.x45_matrix_emender.k5b_whole_scan \
        import DISPATCH
    L.check_python()
    configure_production()
    torch.set_num_threads(1)
    policy = RoutingPolicy(ops=True)
    query, payload, decay = distinct_stream(REG_BATCH, REG_TOKENS, REG_HEADS,
                                             REG_VALUE_DIM, 512, 77)
    generator = torch.Generator().manual_seed(1234)
    upstream = torch.randn(REG_BATCH, REG_TOKENS, REG_HEADS, REG_VALUE_DIM,
                           generator=generator).cuda()
    _, arm, record = assert_record_bit_equal_to_arm(
        query, payload, decay, policy, REG_RANK, REG_RANK, chunk_tokens,
        f"chunk{chunk_tokens}", upstream)
    outputs = assert_graph_bit_equal_to_arm(
        query, payload, decay, policy, REG_RANK, REG_RANK, chunk_tokens,
        f"chunk{chunk_tokens}", upstream)
    report = dict(
        gate="A (re-asserted at the board's chunk length)",
        chunk_tokens=chunk_tokens,
        fixture=dict(batch=REG_BATCH, tokens=REG_TOKENS, heads=REG_HEADS,
                     rank=REG_RANK, value_dim=REG_VALUE_DIM, distinct=512,
                     seed=77, precision="bf16 autocast"),
        bit_equal_channels=list(SCHEDULE_FIELDS) + ["weights", "reads", "state",
                                                   "live"],
        bit_equal_gradients=["dquery", "dpayload", "ddecay"],
        call_forms_checked=["eager", "capture+replay", "replay"],
        calls_checked=len(outputs),
        op_histogram=op_histogram(record.schedule),
        dispatch_ledger=dict(DISPATCH),
        bindings=registration_bindings(), git=git_state(),
        note=("the K5B construction claim (receipts/k5b_whole_scan.json) is "
              "chunk-agnostic: the whole-scan body is the per-chunk stack "
              "walked in one function; this row re-asserts it at chunk 128, "
              "the cd-protocol configuration the 2700 s arms run"))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in
                      ("chunk_tokens", "bit_equal_channels", "op_histogram")},
                     indent=1), flush=True)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--arm", required=True, choices=ARMS)
    t.add_argument("--seconds", type=float, default=2700.)
    t.add_argument("--seed", type=int, default=3401)
    t.add_argument("--output", required=True)
    t.add_argument("--eval-split", default="calibration")
    t.add_argument("--device", default=None)
    t.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    p = sub.add_parser("parity")
    p.add_argument("--chunk-tokens", type=int, default=CHUNK_TOKENS)
    p.add_argument("--out", default=str(RUNS / "k5b_chunk128_parity.json"))
    a = ap.parse_args()
    if a.cmd == "train":
        train(a)
        return
    parity(a.chunk_tokens, a.out)


if __name__ == "__main__":
    main()
