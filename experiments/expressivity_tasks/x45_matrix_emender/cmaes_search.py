"""CMA-ES hyperparameter search for the matrix_ops arm at the 150 M board.

Motivation (receipted): the matched-arms run measured ``matrix_ops`` at 1.7576
vs ``gdn2``'s 1.1976 nats/byte (2700 s, matched time), but the arms' optimizer
settings were TRANSFERRED from the gdn2 nomination while gdn2's own came from
the 144-candidate byte-velocity search -- the tuning asymmetry the 150 M
lattice registration lists as caveat 1 ("hyperparameter transfer").  This
module closes that asymmetry for the matrix arm before the 0.56-nat gap is read
as architectural: a CMA-ES search over the matrix arm's own optimizer/policy
knobs, then a diagnosis of what the ops actually do at this scale.

The probe (fixed exposure, not fixed time)
------------------------------------------
Every evaluation trains the ``matrix_ops`` arm (the registered 149,981,808-
parameter shape, rank 128, chunk 128, K5B fast path -- the matched-arms
configuration verbatim except the searched knobs) for a FIXED TARGET BUDGET of
16,777,216 bytes (~128 steps of 131,072 targets) on the registered flagship
train split, then evaluates on the calibration split with the registered
boundary (full 2048-byte windows, batch 1, recorded-read path -- the same rule
the 2700 s arms were scored by).  Budget defence: the K5B steady rate is ~34.0k
bytes/s (the matched-arms receipt), so the training leg is ~495 s plus
compile/capture and eval -- about 9 minutes per evaluation, which sizes the
search: ~26 evaluations fit the ~4 h GPU budget.  A fixed byte budget (not
seconds) makes exposure identical across evaluations: the stream, the batch
order and the stop rule are all deterministic, so every probe sees byte-for-
byte the same windows.

The seed is FIXED (3401, the standing seed) for every evaluation.  Training
stochasticity is the enemy of a search on 9-minute probes, and a fixed seed
plus the deterministic stream removes init/data noise entirely; the stated
cost is that the search tunes the seed too -- the incumbent's NLL is a
seed-3401 number.  Cross-process kernel-choice noise (inductor autotuning) is
the residual nondeterminism and is stated in the receipt, not hidden.

The search
----------
Phase 1 searches 4 dimensions: learning rate (log), weight decay (log), warmup
steps (log, capped at the probe horizon), read_scale (linear).  Phase 2 (run
only if phase 1's best does not close most of the gap) adds the policy
thresholds tau_match and tau_pop; tau_merge is added only if the diagnosis
shows merges matter.  pycma (already importable in the approved venv,
4.4.4) drives the search from a TORCH-FREE driver process: every evaluation
is a separate probe subprocess, so a diverged config or a CUDA fault cannot
kill the search, and every evaluation is appended to the receipt log as it
lands.  CMA-ES is rank-based by construction; the objective is deterministic
by design, so no noise handling beyond that is claimed.  Population size,
generations, sigma0 and the CMA-ES seed are recorded in the result receipt.

Honest comparison rule: gdn2 is NOT tuned here.  The registered gdn2
nomination runs through the identical probe exactly once, as an anchor; the
deliverable reports the gap before tuning (the 2700 s receipts), the gap
after (probe exposure, both arms' anchors vs the matrix incumbent), how much
of 0.56 nats was tuning, and what remains architectural.

Subcommands:

``probe``    one evaluation: train to the fixed target budget, evaluate, write
             the receipt, print a one-line JSON result.
``search``   the driver: evaluate the transferred anchor + the gdn2 anchor,
             then run CMA-ES, logging and committing as evaluations land.
``confirm``  one 2700-active-second run of one config through the matched-arms
             protocol (``runs_150m`` receipt convention, same stopping rule and
             eval boundary as the 2700 s arms), so the searched incumbent can be
             compared to the 2700 s receipts directly.  Does NOT touch
             ``matched_arms.py`` (the arms lane's file) -- the loop is mirrored
             here, and the receipt says which knobs differ from the transfer.

Usage (see ``run_cmaes_search_leased.sh`` for the detached leased launcher):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.cmaes_search search \\
        --evals receipts/cmaes_evals.jsonl --result receipts/cmaes_result.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# The probe subprocess sets these BEFORE importing torch (the matched-arms
# discipline; configure_production() fail-closes on a missing allocator knob).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
X45_RECEIPTS = ROOT / "receipts"
PROBE_RECEIPTS = X45_RECEIPTS / "cmaes_probes"

# --------------------------------------------------------------------- the space

# The transferred configuration: the registered gdn2 nomination's optimizer
# settings, exactly what the 2700 s matched-arms matrix_ops run used (its
# build_matrix, which is what "hyperparameter transfer" means here).  beta2
# and microbatch stay at the transferred values in every phase.
TRANSFERRED = {
    "lr": 5.420185524740253e-4,
    "weight_decay": 0.20059436743465126,
    "warmup_steps": 69,
    "read_scale": 2.0,
    "tau_match": 0.45,
    "tau_pop": 0.10,
    "tau_merge": 0.85,
}
BETA2 = 0.9392004369596886      # transferred, fixed
MICROBATCH = 8                  # transferred, fixed
RANK = 128                      # the registered matrix-arm rank (not a parameter)
CHUNK_TOKENS = 128              # the cd-protocol routing granularity
PROBE_TARGETS = 16_777_216       # ~128 steps, ~9 min at the K5B steady rate
SEED = 3401                      # the standing seed, fixed for EVERY evaluation

# (name, scale, lo, hi); u in [0, 1] decodes left-to-right, log scales first.
PHASE1_DIMS = [
    ("lr", "log", 1e-4, 3e-3),
    ("weight_decay", "log", 1e-3, 0.6),
    ("warmup_steps", "logint", 2, 120),
    ("read_scale", "linear", 0.5, 4.0),
]
# Phase 2 adds the policy thresholds; tau_merge only if the diagnosis demands
# it (the merge counts decide, not this file).
PHASE2_EXTRA_DIMS = [
    ("tau_match", "linear", 0.2, 0.8),
    ("tau_pop", "linear", -0.2, 0.4),
]


def dims_for(phase: int) -> list:
    return list(PHASE1_DIMS) + (list(PHASE2_EXTRA_DIMS) if phase >= 2 else [])


def decode(dims, u) -> dict:
    """Unit cube -> real hyperparameters (clipped; CMA-ES proposes outside)."""
    out = dict(TRANSFERRED)
    for (name, scale, lo, hi), v in zip(dims, u):
        v = min(1.0, max(0.0, float(v)))
        if scale == "log":
            value = lo * (hi / lo) ** v
        elif scale == "logint":
            value = round(lo * (hi / lo) ** v)
        else:
            value = lo + v * (hi - lo)
        out[name] = int(value) if scale == "logint" else float(value)
    return out


def encode(dims, params: dict) -> list:
    """Real hyperparameters -> unit cube (the inverse of :func:`decode`)."""
    u = []
    for name, scale, lo, hi in dims:
        v = float(params[name])
        if scale in ("log", "logint"):
            v = min(hi, max(lo, v))
            u.append(math.log(v / lo) / math.log(hi / lo))
        else:
            u.append(min(1.0, max(0.0, (v - lo) / (hi - lo))))
    return u


# ------------------------------------------------------------------ the probe


def confirm(args) -> dict:
    """One 2700-active-second run of one config, the matched-arms protocol.

    The loop, the stopping rule (the clock starts at step 1 and includes the
    stream read, compile/capture, fwd/bwd, clip, AdamW), the per-step log and
    the eval boundary are ``matched_arms.train``'s verbatim; only the searched
    knobs differ, and the receipt records exactly which.
    """
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
    from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
        configure_production)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for as matrix_model_for, parameter_count)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

    config = json.loads(args.config)
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None or str(card) != str(visible):
        raise ValueError("GPU lease required (GPU_LEASE_HELD == CUDA_VISIBLE_DEVICES)")
    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    policy = RoutingPolicy(ops=True, tau_match=config.get("tau_match", 0.45),
                          tau_merge=config.get("tau_merge", 0.85),
                          tau_pop=config.get("tau_pop", 0.10),
                          read_scale=config.get("read_scale", 2.0))
    shape = matrix_config(rank=RANK)
    model = matrix_model_for(shape, policy, CHUNK_TOKENS, compile_scan=True,
                             fused_whole_scan=True).cuda().train()
    optimizer_config = dict(lr=config["lr"], betas=(0.9, BETA2),
                            weight_decay=config["weight_decay"],
                            warmup_steps=int(config["warmup_steps"]),
                            microbatch=MICROBATCH)
    summary = dict(base_arm="matrix", shape=shape.__dict__,
                   parameters=parameter_count(shape), routing=policy.record(),
                   chunk_tokens=CHUNK_TOKENS, compile_scan=True,
                   fused_whole_scan=True,
                   knobs_that_differ_from_transfer={
                       k: dict(transferred=TRANSFERRED[k], searched=config[k])
                       for k in config if k in TRANSFERRED and config[k] != TRANSFERRED[k]})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        label=args.label, seed=args.seed, seconds=args.seconds, summary=summary,
        optimizer=optimizer_config, identity=identity,
        bindings=matched_arms.registration_bindings(), git=matched_arms.git_state(),
        session=matched_arms.session_meta(card), started_ns=time.time_ns(),
        config=config, note=("the CMA-ES incumbent through the matched-arms "
                             "2700 s protocol; matched_arms.py untouched")))
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets = 0
    rows = []
    warmup = max(1, int(optimizer_config["warmup_steps"]))
    with (out / "training.jsonl").open("w", buffering=1) as log:
        while elapsed < args.seconds:
            torch.cuda.synchronize()
            start = time.monotonic()
            batch_rows = next(batches)
            n = sum(len(r[3]) for r in batch_rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config["lr"] * min(1., (step + 1) / warmup)
            loss_sum = 0.0
            for x, y, valid in grouped(batch_rows, MICROBATCH):
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
            rows.append(row)
            if step % 25 == 0:
                print(f"  {args.label} step {step} {elapsed:.0f}s "
                      f"{targets/1e6:.1f}MB loss {loss_sum:.4f}", flush=True)
    os.environ["X40_BATCHED_READS"] = "0"
    result = L.evaluate(model, stream, "matrix_ops", "calibration")
    result.update(status="completed", arm="matrix_ops", label=args.label,
                  seed=args.seed, units="nats/byte", train_seconds=elapsed,
                  steps=step, bytes=targets,
                  bytes_per_s=targets / max(1e-9, elapsed), identity=identity,
                  config=config, summary=summary,
                  trajectory=matched_arms.trajectory_sample(rows),
                  peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
    atomic_json(out / "result.json", result)
    print(json.dumps({k: result[k] for k in
                      ("label", "score", "eval_windows", "eval_targets", "steps",
                       "bytes", "train_seconds")}), flush=True)
    return result


def probe(args) -> dict:
    """One evaluation at the fixed exposure.  Writes the receipt, prints JSON."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
        configure_production)
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        matrix_config, model_for as matrix_model_for, parameter_count)
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy

    config = json.loads(args.config)
    arm = config.get("arm", "matrix_ops")
    started = time.time()

    def finish(status, nll, steps, targets, rows, detail):
        receipt = dict(
            label=args.label, arm=arm, seed=args.seed, status=status,
            probe_targets=args.targets, config=config,
            steps=steps, targets=targets,
            train_seconds=round(time.time() - started, 1),
            nll=nll, detail=detail, step_log=rows,
            git=matched_arms.git_state())
        if args.out:
            atomic_json(Path(args.out), receipt)
        print(json.dumps({k: receipt[k] for k in
                          ("label", "arm", "status", "nll", "steps", "targets",
                           "probe_targets")}), flush=True)
        return receipt

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("no GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES required")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: {card} vs CUDA_VISIBLE_DEVICES {visible}")

    configure_production()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if arm == "gdn2":
        model, optimizer_config, summary = matched_arms.build_gdn2(args.seed)
    else:
        policy = RoutingPolicy(ops=True, tau_match=config.get("tau_match", 0.45),
                              tau_merge=config.get("tau_merge", 0.85),
                              tau_pop=config.get("tau_pop", 0.10),
                              read_scale=config.get("read_scale", 2.0))
        shape = matrix_config(rank=RANK)
        model = matrix_model_for(shape, policy, CHUNK_TOKENS, compile_scan=True,
                                 fused_whole_scan=True).cuda().train()
        optimizer_config = dict(lr=config["lr"], betas=(0.9, BETA2),
                                weight_decay=config["weight_decay"],
                                warmup_steps=int(config["warmup_steps"]),
                                microbatch=MICROBATCH)
        summary = dict(base_arm="matrix", shape=shape.__dict__,
                       parameters=parameter_count(shape), routing=policy.record(),
                       chunk_tokens=CHUNK_TOKENS, compile_scan=True,
                       fused_whole_scan=True, optimizer_note="CMA-ES probe")

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    microbatch = optimizer_config["microbatch"]
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets = 0
    rows = []
    is_matrix = arm != "gdn2"
    warmup = max(1, int(optimizer_config["warmup_steps"]))
    diverged_at = None
    while targets < args.targets:
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            batch_rows = next(batches)
        except StopIteration:
            raise RuntimeError("train split exhausted before the probe budget")
        n = sum(len(r[3]) for r in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = optimizer_config["lr"] * min(1., (step + 1) / warmup)
        loss_sum = 0.0
        diverged = False
        for x, y, valid in grouped(batch_rows, microbatch):
            if is_matrix:
                torch.compiler.cudagraph_mark_step_begin()
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        if diverged:
            elapsed += time.monotonic() - start
            finish("diverged", float("inf"), step, targets, rows,
                   dict(step=step + 1, active_seconds=elapsed))
            return {}
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                       error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        elapsed += time.monotonic() - start
        step += 1
        targets += n
        rows.append(dict(step=step, seconds=round(elapsed, 3), bytes=targets,
                         loss=loss_sum))

    os.environ["X40_BATCHED_READS"] = "0"   # evaluation is recorded-read only
    result = L.evaluate(model, stream, arm, "calibration")
    receipt = finish("completed", result["score"], step, targets, rows,
                     dict(eval_windows=result["eval_windows"],
                          eval_targets=result["eval_targets"],
                          bytes_per_s=targets / max(1e-9, elapsed),
                          identity=identity, session=matched_arms.session_meta(card)))
    return receipt


# ----------------------------------------------------------------- the driver


def append_eval(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_evals(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def commit(paths, message):
    # Only paths that exist can be staged; a not-yet-written receipt (e.g. the
    # result json mid-search) must not abort the whole add.
    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        return
    subprocess.run(["git", "add", "--", *paths], cwd=REPO, check=False)
    subprocess.run(["git", "commit", "-m", message, "--", *paths], cwd=REPO,
                   check=False)


def run_probe_subprocess(cfg: dict, label: str, eval_id: int, out: Path,
                         targets: int, seed: int, timeout: int) -> dict:
    cmd = [sys.executable, "-m",
           "experiments.expressivity_tasks.x45_matrix_emender.cmaes_search",
           "probe", "--config", json.dumps(cfg), "--targets", str(targets),
           "--seed", str(seed), "--label", label, "--out", str(out)]
    started = time.monotonic()
    try:
        done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=timeout)
        if done.returncode != 0:
            return dict(label=label, status="crashed", nll=float("inf"),
                        returncode=done.returncode,
                        stderr_tail=done.stderr[-2000:])
        line = done.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except subprocess.TimeoutExpired:
        return dict(label=label, status="timeout", nll=float("inf"))
    except Exception as exc:                       # defensive: the search survives
        return dict(label=label, status="error", nll=float("inf"), error=repr(exc))


def search(args) -> dict:
    import cma

    evals_path = Path(args.evals)
    result_path = Path(args.result)
    PROBE_RECEIPTS.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    dims = dims_for(args.phase)
    log = load_evals(evals_path)

    def remaining_wall():
        return args.max_wall - (time.monotonic() - t0)

    def evaluate(cfg, label, timeout=2400):
        """Run one probe, append the receipt line, return the record."""
        nonlocal log
        eval_id = len(log)
        out = PROBE_RECEIPTS / f"eval_{eval_id:03d}.json"
        record = run_probe_subprocess(cfg, label, eval_id, out, args.targets,
                                      args.seed, timeout)
        record.update(eval_id=eval_id, params={k: cfg[k] for k in cfg if k != "arm"},
                      arm=cfg.get("arm", "matrix_ops"),
                      wall_s=round(time.monotonic() - t0, 1))
        log.append(record)
        append_eval(evals_path, record)
        print(f"[{label}] eval {eval_id}: nll={record.get('nll')} "
              f"steps={record.get('steps')} wall={record['wall_s']}s", flush=True)
        if len(log) % args.commit_every == 0:
            commit([evals_path, result_path, PROBE_RECEIPTS],
                   f"x45 CMA-ES search: evals {len(log)} (log receipt append)")
        return record

    # ---- anchors: the transferred matrix config and the untuned gdn2 arm.
    transferred_label = f"transferred_p{args.phase}"
    gdn2_label = "gdn2_anchor"
    if not any(r.get("label") == transferred_label for r in log):
        evaluate(dict(TRANSFERRED), transferred_label)
    if not any(r.get("label") == gdn2_label for r in log):
        evaluate(dict(TRANSFERRED, arm="gdn2"), gdn2_label)
    transferred_nll = min(r["nll"] for r in log
                          if r.get("label") == transferred_label
                          and r.get("nll") is not None)
    gdn2_nll = min(r["nll"] for r in log if r.get("label") == gdn2_label
                   and r.get("nll") is not None)

    # ---- the incumbent CMA-ES starts from: --x0 (a saved config), else the
    # best matrix eval already in the log, else the transferred anchor.
    x0_cfg = TRANSFERRED
    if args.x0:
        x0_cfg = json.loads(Path(args.x0).read_text())
        print(f"[search] x0 from {args.x0}", flush=True)
    elif any(r.get("arm") == "matrix_ops" and r.get("status") == "completed"
             for r in log):
        best = min((r for r in log if r.get("arm") == "matrix_ops"
                    and r.get("status") == "completed"),
                   key=lambda r: r["nll"])
        x0_cfg = best["params"]
        print(f"[search] x0 resumed from log eval {best['eval_id']} "
              f"(nll {best['nll']:.4f})", flush=True)
    x0 = encode(dims, x0_cfg)

    es = cma.CMAEvolutionStrategy(
        x0, args.sigma0,
        dict(bounds=[0, 1], popsize=args.popsize, seed=args.cma_seed,
             verbose=-9, CMA_active=True))
    generation = 0
    while generation < args.generations:
        est = args.popsize * 10 * 60       # ~10 min/eval, honest worst case
        if remaining_wall() < est * 1.05:
            print(f"[search] wall budget short ({remaining_wall():.0f}s < "
                  f"{est * 1.05:.0f}s): stopping before generation "
                  f"{generation + 1}", flush=True)
            break
        xs = es.ask()
        print(f"[search] generation {generation + 1}/{args.generations} "
              f"popsize {len(xs)}", flush=True)
        cfgs, recs = [], []
        for i, x in enumerate(xs):
            cfg = decode(dims, x)
            cfgs.append(cfg)
            recs.append(evaluate(cfg, f"p{args.phase}_g{generation + 1}_{i}"))
        es.tell(cfgs and xs, [r.get("nll") if r.get("nll") is not None
                              else float("inf") for r in recs])
        generation += 1

    # ---- assemble the result receipt (incumbent trajectory from the log).
    matrix_evals = [r for r in log if r.get("arm") == "matrix_ops"
                    and r.get("status") == "completed"]
    trajectory, best = [], None
    for r in matrix_evals:
        if best is None or r["nll"] < best["nll"]:
            best = r
        trajectory.append(dict(eval_id=r["eval_id"], nll=r["nll"],
                               best_so_far=best["nll"]))
    result = dict(
        phase=args.phase, space=[list(d) for d in dims],
        settings=dict(popsize=args.popsize, generations_requested=args.generations,
                      generations_run=generation, sigma0=args.sigma0,
                      cma_seed=args.cma_seed, probe_seed=args.seed,
                      probe_targets=args.targets, cma_version=cma.__version__,
                      fixed_seed_note=("every evaluation uses seed 3401 and the "
                                       "same deterministic stream prefix: the "
                                       "search tunes the seed and the data draw "
                                       "as well as the hyperparameters")),
        transferred_anchor=dict(nll=transferred_nll, config=TRANSFERRED),
        gdn2_anchor=dict(nll=gdn2_nll, note="the registered gdn2 nomination "
                        "verbatim through the identical probe; NOT tuned here"),
        incumbent_trajectory=trajectory,
        best=None if best is None else dict(nll=best["nll"], config=best["params"],
                                            eval_id=best["eval_id"]),
        n_evaluations=len(log), wall_s=round(time.monotonic() - t0, 1))
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
    atomic_json(result_path, result)
    commit([evals_path, result_path, PROBE_RECEIPTS],
           f"x45 CMA-ES search phase {args.phase}: {len(log)} evaluations, "
           f"best nll {None if best is None else round(best['nll'], 4)}")
    print(json.dumps(dict(best=None if best is None else best["nll"],
                          evaluations=len(log)), ), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--config", required=True,
                   help="JSON hyperparameters (arm=gdn2 runs the gdn2 anchor)")
    p.add_argument("--targets", type=int, default=PROBE_TARGETS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--label", default="probe")
    p.add_argument("--out", default=None)
    c = sub.add_parser("confirm")
    c.add_argument("--config", required=True)
    c.add_argument("--seconds", type=float, default=2700.)
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--label", default="cmaes_confirm")
    c.add_argument("--output", required=True)
    s = sub.add_parser("search")
    s.add_argument("--evals", default=str(X45_RECEIPTS / "cmaes_evals.jsonl"))
    s.add_argument("--result", default=str(X45_RECEIPTS / "cmaes_result.json"))
    s.add_argument("--phase", type=int, default=1, choices=(1, 2))
    s.add_argument("--popsize", type=int, default=8)
    s.add_argument("--generations", type=int, default=3)
    s.add_argument("--sigma0", type=float, default=0.20)
    s.add_argument("--cma-seed", type=int, default=20261115)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--targets", type=int, default=PROBE_TARGETS)
    s.add_argument("--max-wall", type=float, default=4 * 3600)
    s.add_argument("--commit-every", type=int, default=10)
    s.add_argument("--x0", default=None,
                   help="JSON config to start the incumbent from (e.g. phase-1 best)")
    a = ap.parse_args()
    if a.cmd == "probe":
        probe(a)
        return
    if a.cmd == "confirm":
        confirm(a)
        return
    search(a)


if __name__ == "__main__":
    main()
