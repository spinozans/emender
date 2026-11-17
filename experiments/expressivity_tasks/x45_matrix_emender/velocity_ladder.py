"""The matched-exposure velocity ladder: CMA-ES tuning of the K10 winner.

The operator's lane: the velocity ladder ranks the arms at ONE protocol -- the
K6/K10 matched-exposure probe (20,022,866 targets, seed 3401, the registered
flagship stream, the document-disjoint calibration eval at 7 windows / 14,336
targets) -- and the current ladder is

    chunked registered 2.4173 -> chunked tuned 2.2564 (CMA-ES, banked)
        -> per-token 2.0278 (K6) -> per-token + keys 1.8418 (K10, the winner).

Two deliverables live here:

* **Part B -- tune the winner.**  The CMA-ES discipline of ``cmaes_search.py``
  (the chunked arm's search, ``CMAES_TUNING.md``) re-run on the K10 arm: the
  SAME fixed-exposure objective (16,777,216 targets), the SAME fixed-seed-per-
  eval rule (the stated cost: the search tunes the seed and the data draw as
  well as the hyperparameters), the same per-evaluation receipt log committed
  as it lands, the same phases (phase 1: lr / weight_decay / warmup /
  read_scale; phase 2 adds tau_match / tau_pop only if phase 1 does not
  converge).  The probe entry point is the K10 arm's own
  (``matched_arms.build_matrix("matrix_ops", 128, seed, k5b=False)`` then
  ``convert_model_to_separated_keys`` -- ``k10_probe_separated_keys.py``'s
  build verbatim, the parameter-neutral swap asserted at 149,981,808), with
  the searched policy knobs set on the converted mixers.  ``cmaes_search.py``
  is imported, never edited; ``k10_*`` sources are imported, never edited.
* **The confirmation cell.**  The tuned incumbent through the probe protocol
  itself (the 20,000,000-target matched-exposure rule -> 20,022,866 targets,
  seed 3401, the same calibration eval, the same honest clock): the
  tuned-winner ladder row, with the K10 mechanism diagnostics
  (``probe_diagnostics``) receipted alongside.

Part A -- the GDN2 ladder row -- is the registered ``matched_arms.py`` harness
itself (``train --arm gdn2 --bytes 20000000``: the registered lattice_gdn2
nomination config verbatim, its 144-candidate-search-tuned optimizer), run
detached under the lease; its receipt is the run directory this doc's driver
points at.  ``matched_arms.py`` is not edited either.

Budget honesty (the sizing constraint that shapes this search): the K10 arm's
measured probe throughput is 10.7k bytes/s (``receipts/k10_probe.json``)
against the chunked arm's 34.0k, so one 16,777,216-target evaluation costs
~25-30 min where the chunked search's cost ~8.6 min.  A ~4-5 h GPU budget
therefore buys the transferred anchor plus roughly ONE CMA-ES generation at
the prior search's popsize (8) -- the wall guard below stops cleanly before
any generation that does not fit, and the result receipt states exactly what
ran.  The gdn2 anchor at probe exposure is CITED from the banked chunked-arm
search (``receipts/cmaes_result_phase2.json``: 1.5163 at the identical fixed
exposure / seed / stream), not re-run.

Subcommands:

``probe``     one K10 evaluation at the fixed probe exposure: train, evaluate,
              write the per-eval receipt, print a one-line JSON result.
``search``    the driver: the transferred anchor, then CMA-ES (pycma,
              torch-free driver, one subprocess per evaluation), logging and
              committing as evaluations land.
``confirm``   the tuned incumbent at the probe protocol (matched exposure,
              the K10 arm's own training loop and mechanism diagnostics):
              the tuned-winner ladder row.

Usage (see ``run_velocity_ladder_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder search \\
        --evals receipts/velocity_k10_cmaes_evals.jsonl \\
        --result receipts/velocity_k10_cmaes_result.json
"""
from __future__ import annotations

import argparse
import json
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
PROBE_RECEIPTS = X45_RECEIPTS / "velocity_k10_probes"
RUNS = ROOT / "velocity_runs"

# The CMA-ES machinery, imported verbatim from the banked chunked-arm search:
# the transferred config (the gdn2 nomination's optimizer + the registered
# policy thresholds -- exactly what the banked K10 probe ran), the search
# space, the unit-cube codecs, and the receipt-log/commit helpers.
from experiments.expressivity_tasks.x45_matrix_emender.cmaes_search import (
    BETA2, MICROBATCH, PROBE_TARGETS, RANK, SEED, TRANSFERRED,
    append_eval, commit, decode, dims_for, encode, load_evals)

# The probe protocol's matched-exposure rule (k10_probe_sequentiality's
# EXPOSURE_TARGETS: train until 20,000,000 targets seen, landing at
# 20,022,866 on the registered stream -- byte-for-byte the K6/K10 landing).
EXPOSURE_TARGETS = 20_000_000
PARAMETERS_AUDITED = 149_981_808
# The banked anchors the ladder doc cites (never rewritten here).
K10_BANKED = dict(nll=1.841756020273481, exposure_bytes=20_022_866, steps=180,
                  train_seconds=1865.8, bytes_per_s=10731.3,
                  receipt=str(X45_RECEIPTS / "k10_probe.json"))
GDN2_PROBE_ANCHOR_BANKED = dict(
    nll=1.5162515129361833,
    note="the banked gdn2 anchor of the chunked-arm CMA-ES search "
         "(receipts/cmaes_result_phase2.json), at the IDENTICAL fixed probe "
         "exposure / seed / stream; cited, not re-run",
    receipt=str(X45_RECEIPTS / "cmaes_result_phase2.json"))


# ------------------------------------------------------------- the K10 probe


def _build_k10(config: dict, seed: int):
    """The K10 arm at the banked probe's build order, with searched knobs.

    ``k10_probe_separated_keys.train_arm``'s build verbatim (the same
    ``matched_arms.build_matrix`` call, the same conversion, the same
    parameter-neutrality assertion, the same post-conversion ``.cuda()``),
    with the searched policy thresholds set on the converted mixers and the
    searched optimizer knobs returned in the optimizer config.  ``k5b=False``
    is the banked probe's own choice (the per-token machines run the
    registered cd-protocol path, not the K5B whole-scan op the chunked arm
    takes).
    """
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.schedule import (
        RoutingPolicy)
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import (DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS,
                MIMETIC_SEED as K10_MIMETIC_SEED, convert_model_to_separated_keys)

    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", RANK, seed, k5b=False)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_separated_keys(model, key_dim=DEFAULT_KEY_DIM,
                                            ctx=DEFAULT_CONTEXT,
                                            segment_tokens=DEFAULT_SEGMENT_TOKENS,
                                            seed=K10_MIMETIC_SEED)
    parameters_after = sum(p.numel() for p in model.parameters())
    if (parameters_after != PARAMETERS_AUDITED
            or parameters_after != parameters_before):
        raise ValueError(f"the parameter-neutral swap must hold: "
                         f"{parameters_before} -> {parameters_after} "
                         f"(the audited count is {PARAMETERS_AUDITED})")
    policy = RoutingPolicy(ops=True,
                           tau_match=config.get("tau_match", 0.45),
                           tau_merge=config.get("tau_merge", 0.85),
                           tau_pop=config.get("tau_pop", 0.10),
                           read_scale=config.get("read_scale", 2.0))
    for block in model.blocks:
        block.mixer.policy = policy
    model = model.cuda().train()
    summary = dict(summary)
    summary["scan_path"] = "k10_separated_keys (per-token, the key channel)"
    summary["key_dim"] = DEFAULT_KEY_DIM
    summary["key_context"] = DEFAULT_CONTEXT
    summary["segment_tokens"] = DEFAULT_SEGMENT_TOKENS
    summary["key_init"] = ("mimetic at the fixed orthonormal family (QR, seed "
                           f"{K10_MIMETIC_SEED} + head), free drift")
    optimizer_config = dict(lr=config["lr"], betas=(0.9, BETA2),
                            weight_decay=config["weight_decay"],
                            warmup_steps=int(config["warmup_steps"]),
                            microbatch=MICROBATCH)
    return model, optimizer_config, summary


def _train_to_targets(model, optimizer_config, stream, targets: int, label: str,
                      log_path=None):
    """The K10 training loop (k10_probe_sequentiality's own, verbatim shape):
    matched-EXPOSURE stopping, honest clock, per-step log rows (written live
    when ``log_path`` is given -- the probe-protocol runs' convention)."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = seen = 0
    rows = []
    diverged = False
    log = Path(log_path).open("w", buffering=1) if log_path else None
    try:
        while seen < targets:
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                batch_rows = next(batches)
            except StopIteration:
                raise RuntimeError("train split exhausted before the probe "
                                   "budget")
            n = sum(len(r[3]) for r in batch_rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config["lr"] * min(
                    1., (step + 1) / max(1, optimizer_config["warmup_steps"]))
            loss_sum = 0.0
            for x, y, valid in grouped(batch_rows, optimizer_config["microbatch"]):
                loss = model.loss(x, y)
                if not torch.isfinite(loss):
                    diverged = True
                    break
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            if diverged:
                elapsed += time.monotonic() - start
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows)
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            step += 1
            seen += n
            rows.append(dict(step=step, active_seconds=elapsed, bytes=seen,
                             loss=loss_sum))
            if log is not None:
                log.write(json.dumps(rows[-1]) + "\n")
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {loss_sum:.4f}", flush=True)
    finally:
        if log is not None:
            log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                rows=rows)


def probe(args) -> dict:
    """One K10 evaluation at the fixed probe exposure (the cmaes_search
    discipline: fixed targets, fixed seed, the same stream)."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms

    config = json.loads(args.config)
    started = time.time()

    def finish(status, nll, steps, seen, rows, detail):
        receipt = dict(
            label=args.label, arm="k10_separated_keys", seed=args.seed,
            status=status, probe_targets=args.targets, config=config,
            steps=steps, targets=seen,
            train_seconds=round(time.time() - started, 1),
            nll=nll, detail=detail, step_log=rows,
            git=matched_arms.git_state())
        if args.out:
            atomic_json(Path(args.out), receipt)
        print(json.dumps({k: receipt[k] for k in
                          ("label", "status", "nll", "steps", "targets",
                           "probe_targets")}), flush=True)
        return receipt

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES "
                         "required")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: {card} vs CUDA_VISIBLE_DEVICES "
                         f"{visible}")

    model, optimizer_config, summary = _build_k10(config, args.seed)
    run = _train_to_targets(model, optimizer_config, stream, args.targets,
                            args.label)
    if run["status"] != "completed":
        return finish("diverged", float("inf"), run["step"], run["seen"],
                      run["rows"], dict(step=run["step"] + 1,
                                        active_seconds=run["elapsed"]))
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation is recorded-read only
    result = L.evaluate(model, stream, "matrix_ops", "calibration")
    return finish("completed", result["score"], run["step"], run["seen"],
                  run["rows"],
                  dict(eval_windows=result["eval_windows"],
                       eval_targets=result["eval_targets"],
                       train_active_seconds=round(run["elapsed"], 1),
                       bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
                       parameters=PARAMETERS_AUDITED, identity=identity,
                       session=matched_arms.session_meta(card)))


# ------------------------------------------------------------------- confirm


def confirm(args) -> dict:
    """The tuned incumbent at the probe protocol: the matched-exposure rule
    (20,000,000 targets -> the 20,022,866 landing), seed 3401, the same
    document-disjoint calibration eval, the same honest clock -- the
    tuned-winner ladder row.  The K10 mechanism diagnostics
    (``probe_diagnostics``) are receipted alongside, imported unchanged."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys \
        import probe_diagnostics

    config = json.loads(args.config)
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES "
                         "required")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: {card} vs CUDA_VISIBLE_DEVICES "
                         f"{visible}")

    model, optimizer_config, summary = _build_k10(config, args.seed)
    exposure = int(args.bytes)
    stopping_rule = (f"matched-EXPOSURE: train until {exposure} targets seen "
                     "(~20M, the probe exposure); the honest elapsed wall is "
                     "reported first-class")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    run_config = dict(probe="velocity_ladder confirm", arm="k10_separated_keys",
                      seed=args.seed, exposure_targets=exposure,
                      optimizer=optimizer_config, summary=summary,
                      knobs_that_differ_from_transfer={
                          k: dict(transferred=TRANSFERRED[k], searched=config[k])
                          for k in config
                          if k in TRANSFERRED and config[k] != TRANSFERRED[k]},
                      identity=identity,
                      bindings=dict(
                          matched_arms.registration_bindings(),
                          k10_preregistration=str(
                              ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
                          k10_gates_receipt=str(X45_RECEIPTS / "k10_gates.json"),
                          k10_banked_probe=str(X45_RECEIPTS / "k10_probe.json")),
                      git=matched_arms.git_state(), stopping_rule=stopping_rule)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=PARAMETERS_AUDITED, trainable=PARAMETERS_AUDITED)))
    print(json.dumps(dict(addressing="separated-keys", card=card,
                          seed=args.seed, exposure=exposure)), flush=True)

    run = _train_to_targets(model, optimizer_config, stream, exposure,
                            args.label, log_path=out / "training.jsonl")
    if run["status"] != "completed":
        raise FloatingPointError(f"the confirm cell diverged at step "
                                  f"{run['step'] + 1}; no receipt is written")
    diagnostics = probe_diagnostics(model, stream)
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation path is recorded-read only
    eval_started = time.monotonic()
    result = L.evaluate(model, stream, "matrix_ops", "calibration")
    eval_wall = time.monotonic() - eval_started
    receipt = dict(
        probe="velocity_ladder confirm", arm="k10_separated_keys",
        addressing="separated write-context keys (key_dim 16, mimetic birth, "
                   "free drift, keys undecayed) -- the tuned winner",
        label=args.label, seed=args.seed, stopping_rule=stopping_rule,
        exposure_bytes=run["seen"], exposure_steps=run["step"],
        eval_nll_nats_per_byte=result["score"],
        eval_split=result["eval_split"], eval_windows=result["eval_windows"],
        eval_targets=result["eval_targets"],
        walls=dict(train_seconds=round(run["elapsed"], 1),
                   probe_seconds=diagnostics["probe"]["probe_wall_seconds"],
                   eval_seconds=round(eval_wall, 1)),
        bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
        parameters=PARAMETERS_AUDITED, searched_config=config,
        k10_banked=K10_BANKED, diagnostics=diagnostics,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        config=run_config, session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            run["rows"], keep_first=min(5, max(1, len(run["rows"])))))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("eval_nll_nats_per_byte", "exposure_bytes",
                       "exposure_steps", "bytes_per_s")}), flush=True)
    return receipt


# ----------------------------------------------------------------- the driver


def run_probe_subprocess(cfg: dict, label: str, out: Path, targets: int,
                         seed: int, timeout: int) -> dict:
    """cmaes_search's isolation verbatim: one probe per subprocess, so a
    diverged config or a CUDA fault cannot kill the search."""
    cmd = [sys.executable, "-m",
           "experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder",
           "probe", "--config", json.dumps(cfg), "--targets", str(targets),
           "--seed", str(seed), "--label", label, "--out", str(out)]
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

    def evaluate(cfg, label):
        """Run one probe, append the receipt line, commit as it lands."""
        nonlocal log
        eval_id = len(log)
        out = PROBE_RECEIPTS / f"eval_{eval_id:03d}.json"
        record = run_probe_subprocess(cfg, label, out, args.targets,
                                      args.seed, args.timeout)
        record.update(eval_id=eval_id, params={k: cfg[k] for k in cfg},
                      arm="k10_separated_keys",
                      wall_s=round(time.monotonic() - t0, 1))
        log.append(record)
        append_eval(evals_path, record)
        print(f"[{label}] eval {eval_id}: nll={record.get('nll')} "
              f"steps={record.get('steps')} wall={record['wall_s']}s",
              flush=True)
        commit([evals_path, result_path, PROBE_RECEIPTS, ROOT / "velocity_ladder.py",
                ROOT / "run_velocity_ladder_leased.sh"],
               f"x45 velocity ladder: K10 CMA-ES eval {eval_id} "
               f"({label}) landed (log receipt append)")
        return record

    # ---- the anchor: the transferred K10 config (the banked probe's own
    # knobs) at this probe exposure.  The gdn2 anchor is cited from the banked
    # chunked-arm search (identical exposure/seed/stream), not re-run.
    anchor_label = f"transferred_k10_p{args.phase}"
    if not any(r.get("label") == anchor_label for r in log):
        evaluate(dict(TRANSFERRED), anchor_label)
    anchor_nll = min(r["nll"] for r in log
                     if r.get("label") == anchor_label
                     and r.get("nll") is not None)

    # ---- the incumbent starts from --x0 (a saved config), else the best
    # completed K10 eval in the log, else the transferred anchor.
    x0_cfg = TRANSFERRED
    if args.x0:
        x0_cfg = json.loads(Path(args.x0).read_text())
        print(f"[search] x0 from {args.x0}", flush=True)
    elif any(r.get("status") == "completed" and r.get("nll") is not None
             for r in log):
        best = min((r for r in log if r.get("status") == "completed"
                    and r.get("nll") is not None), key=lambda r: r["nll"])
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
        est = args.popsize * args.eval_est_seconds   # honest per-eval wall
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
    completed = [r for r in log if r.get("status") == "completed"
                 and r.get("nll") is not None]
    trajectory, best = [], None
    for r in completed:
        if best is None or r["nll"] < best["nll"]:
            best = r
        trajectory.append(dict(eval_id=r["eval_id"], label=r.get("label"),
                               nll=r["nll"], best_so_far=best["nll"]))
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    result = dict(
        arm="k10_separated_keys", phase=args.phase,
        space=[list(d) for d in dims],
        settings=dict(popsize=args.popsize,
                      generations_requested=args.generations,
                      generations_run=generation, sigma0=args.sigma0,
                      cma_seed=args.cma_seed, probe_seed=args.seed,
                      probe_targets=args.targets, cma_version=cma.__version__,
                      eval_est_seconds=args.eval_est_seconds,
                      budget_note=(f"the K10 arm's probe throughput is ~10.7k "
                                   f"bytes/s, so one evaluation costs ~25-30 "
                                   f"min: the {args.max_wall/3600:.1f} h wall "
                                   f"budget buys the anchor plus roughly one "
                                   f"generation at popsize {args.popsize}; the "
                                   f"wall guard stopped cleanly at "
                                   f"{generation} generation(s) run"),
                      fixed_seed_note=("every evaluation uses seed 3401 and "
                                       "the same deterministic stream prefix: "
                                       "the search tunes the seed and the data "
                                       "draw as well as the hyperparameters")),
        transferred_anchor=dict(nll=anchor_nll, config=TRANSFERRED,
                                note="the banked K10 probe's own knobs at this "
                                     "probe exposure"),
        gdn2_anchor=GDN2_PROBE_ANCHOR_BANKED,
        incumbent_trajectory=trajectory,
        best=None if best is None else dict(nll=best["nll"],
                                            config=best["params"],
                                            eval_id=best["eval_id"],
                                            label=best.get("label")),
        n_evaluations=len(log), wall_s=round(time.monotonic() - t0, 1))
    atomic_json(result_path, result)
    commit([evals_path, result_path, PROBE_RECEIPTS],
           f"x45 velocity ladder: K10 CMA-ES phase {args.phase} complete, "
           f"{len(log)} evaluations, best nll "
           f"{None if best is None else round(best['nll'], 4)}")
    print(json.dumps(dict(best=None if best is None else best["nll"],
                          evaluations=len(log))), flush=True)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--config", required=True)
    p.add_argument("--targets", type=int, default=PROBE_TARGETS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--label", default="k10_probe")
    p.add_argument("--out", default=None)
    c = sub.add_parser("confirm")
    c.add_argument("--config", required=True)
    c.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    c.add_argument("--seed", type=int, default=SEED)
    c.add_argument("--label", default="k10_tuned_confirm")
    c.add_argument("--output",
                   default=str(RUNS / "k10_tuned_confirm_seed3401"))
    s = sub.add_parser("search")
    s.add_argument("--evals", default=str(X45_RECEIPTS
                                          / "velocity_k10_cmaes_evals.jsonl"))
    s.add_argument("--result", default=str(X45_RECEIPTS
                                           / "velocity_k10_cmaes_result.json"))
    s.add_argument("--phase", type=int, default=1, choices=(1, 2))
    s.add_argument("--popsize", type=int, default=8)
    s.add_argument("--generations", type=int, default=2)
    s.add_argument("--sigma0", type=float, default=0.20)
    s.add_argument("--cma-seed", type=int, default=20261115)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--targets", type=int, default=PROBE_TARGETS)
    s.add_argument("--timeout", type=int, default=3000,
                   help="per-evaluation subprocess timeout (s); the K10 "
                        "probe leg is ~25-30 min")
    s.add_argument("--eval-est-seconds", type=int, default=1800,
                   help="honest worst-case per-evaluation wall, for the "
                        "generation wall guard")
    s.add_argument("--max-wall", type=float, default=5 * 3600)
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
