"""OPTIMIZER/SCHEDULE BAKE-OFF on the winner -- AdamW-vs-SF-vs-Muon at the
exact probe protocol, with the checkpoint-eval curve on GDN2's banked curve.

THE OPERATOR'S QUESTION: schedule-free? Muon? what is the best
optimizer/schedule for THIS machine?  The side-by-side dissection
(``RESULT_SIDE_BY_SIDE_0239.md``) receipts that the remaining gap to GDN2 is
earned in the FIRST ~10-20 STEPS (deficit 0.36-0.42 at steps 10-20, narrowing
to a flat 0.162 +/- 0.018 over steps 60-160) -- a warmup/early-schedule-shaped
hole -- and that tuning (C1) owns <= 0.02-0.03 nats of the 0.239 headline.
This bake-off answers the optimizer question with receipts: every arm runs the
WINNER (K10 per-token + separated keys) at the EXACT probe protocol verbatim
(20,000,000-target matched-exposure rule -> the 20,022,866 landing, seed 3401,
the registered flagship stream, the document-disjoint calibration eval at 7
windows / 14,336 targets, the honest clock), differing ONLY in the optimizer
and (where the family demands it) the schedule.

DIAGNOSIS-FLAVORED COMPARISON ONLY -- optimizers are training-surface; no
model, no semantics, no ``k10_*``/``k6_*``/side-by-side source, no
PREREGISTRATION, no ``lattice.py`` is edited.  The banked builder
``velocity_ladder._build_k10`` is imported VERBATIM.

THE ARMS (conventions approved by the supervisor; symmetry rule: same stream,
same clip, same stopping rule, same eval; only optimizer family and schedule
differ, receipted per arm):

* ``adamw``     the incumbent itself -- NOT re-run here (the banked row
                1.822737353188651 and the side-by-side curve are cited); this
                arm exists in the driver only so a same-session curve could be
                produced if ever demanded.
* ``sf``        SF-AdamW (schedule-free: primal averaging, NO warmup, NO
                decay) at the incumbent's own tuned knobs -- lr 3.3427e-4
                (the tuned peak LR; the mapping is "same LR, schedule
                removed", the cleanest isolation of the schedule axis),
                betas (0.9, 0.9392) (the probe's transferred beta2),
                weight_decay 0.17422 (the tuned incumbent's), r 0,
                weight_lr_power 2 (package defaults).  Pre-stated prediction:
                if the early hole is warmup-shaped, this arm ELIMINATES the
                first-10-steps hole by design; refuted if it opens the same
                hole or worse.  The package is the SAME ``schedulefree`` the
                prior invertible line receipted.
* ``muon1x`` / ``muon3x``  hybrid Muon (``bakeoff_optimizers.HybridMuonX45``:
                the prior line's pinned KellerJordan quintic VERBATIM via
                ``hybrid_muon.scaled_update``, Muon on embeddings +
                projections, AdamW on norms/decay-gate) at the incumbent's
                tuned base LR x1 and x3 -- the prior receipt's bracket (3x =
                Muon's neutral point, 10x collapses) -- with the incumbent's
                warmup-45 schedule shared by BOTH groups (the prior line's
                convention: both hybrid groups scale equally), aux AdamW at
                the probe's betas and the incumbent's weight decay.
* ``sf_prior``  SF-AdamW at the prior invertible line's own point (lr 1e-3,
                betas (0.9, 0.95), wd 0.01) -- a one-arm LR/beta bracket for
                the SF family, run after the required arms if budget allows.
* ``sf_muon``   the exploratory Muon+SF combination
                (``schedulefree.ScheduleFreeWrapper`` primal averaging around
                the hybrid) -- unstudied interactions; optional, run only
                after the required arms land clean.

Every arm's curve is instrumented at the same checkpoints as the side-by-side
(steps 1/5/10/20/40/80/120/160/180, the early-transient readout is the
primary metric): the model state is snapshotted (CPU clones) inside the
training loop -- for the schedule-free arms, the snapshot is the PRIMAL
AVERAGE (the eval point, computed non-destructively, see
``bakeoff_optimizers.sf_x_snapshot``) -- and every eval runs on restored
snapshots after training, so the training path is unperturbed by construction.

Deliverables: per-arm receipts under ``bakeoff_runs/<label>_seed3401/``,
``receipts/optimizer_bakeoff_summary.json`` (the ``assemble`` subcommand) and
``RESULT_OPTIMIZER_BAKEOFF.md``.

Usage (see ``run_optimizer_bakeoff_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff arm \\
        --label sf --output bakeoff_runs/sf_seed3401
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

# The probe subprocess sets these BEFORE importing torch (the matched-arms
# discipline; configure_production() fail-closes on a missing allocator knob).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "bakeoff_runs"
SIDE_BY_SIDE_RUNS = ROOT / "side_by_side_runs"

SEED = 3401
EXPOSURE_TARGETS = 20_000_000
PARAMETERS_AUDITED = 149_981_808
# The checkpoint plan: the early-transient readout (the side-by-side's C2
# receipt: the deficit peaks at steps 10-20 and closes by step 60) is the
# primary metric; steps are the task's own list.
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
PROBE_WINDOWS = 4

# The tuned winner's config (the ladder row 6 / side-by-side TUNED_INCUMBENT):
# the K10 arm + CMA-ES incumbent (eval 4 of velocity_k10_cmaes_result.json).
TUNED_WINNER = {
    "lr": 0.00033427141575371284, "weight_decay": 0.17422187309224696,
    "warmup_steps": 45, "read_scale": 2.7870458737509636,
    "tau_match": 0.45, "tau_pop": 0.1, "tau_merge": 0.85,
}
BETA2 = 0.9392004369596886      # transferred, fixed (cmaes_search.py)
MICROBATCH = 8                  # transferred, fixed
GRAD_CLIP = 1.0                 # the probe protocol's own clip, every arm

# The banked ladder rows this bake-off compares against (cited, never re-run).
BANKED = {
    "k10_tuned": 1.822737353188651,      # the incumbent arm itself
    "k10_untuned": 1.841756020273481,
    "gdn2": 1.5836738858904158,
}

# The arms' optimizer conventions (approved by the supervisor).  Every arm
# holds the probe protocol's non-optimizer knobs at the incumbent's values;
# the "note" fields are receipted verbatim in each arm's result.json.
ARMS = {
    "adamw": dict(
        family="adamw",
        spec=dict(optimizer="torch.optim.AdamW(fused=True)",
                  lr=TUNED_WINNER["lr"], betas=(0.9, BETA2),
                  weight_decay=TUNED_WINNER["weight_decay"],
                  warmup_steps=TUNED_WINNER["warmup_steps"]),
        note="the incumbent, banked at the protocol (1.8227); the driver arm "
             "exists only for a demanded same-session curve"),
    "sf": dict(
        family="sf-adamw",
        spec=dict(optimizer="schedulefree.AdamWScheduleFree",
                  lr=TUNED_WINNER["lr"], betas=(0.9, BETA2),
                  weight_decay=TUNED_WINNER["weight_decay"],
                  warmup_steps=0, r=0.0, weight_lr_power=2.0),
        note="schedule-free: NO warmup, NO decay (primal averaging instead); "
             "lr mapped to the incumbent's tuned PEAK lr -- same LR, schedule "
             "removed, the cleanest isolation of the schedule axis (the "
             "package's own guidance: no scheduler is needed; warmup would be "
             "its warmup_steps knob, set 0 by design)"),
    "muon1x": dict(
        family="hybrid-muon",
        spec=dict(optimizer="bakeoff_optimizers.HybridMuonX45",
                  lr=TUNED_WINNER["lr"], lr_multiplier=1,
                  betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
                  warmup_steps=TUNED_WINNER["warmup_steps"],
                  muon_params="embeddings + projections (embed, head, mixer "
                              "value/key/query_key/out, mlp w1/w2/w3)",
                  momentum=0.95, nesterov=True,
                  update_scaling="0.2*sqrt(max(shape)) (hybrid_muon, pinned)"),
        note="the prior line's pinned KellerJordan quintic, both hybrid groups "
             "on the incumbent's warmup-45 schedule (the prior line scaled "
             "both groups equally), 1x = the incumbent's tuned LR"),
    "muon3x": dict(
        family="hybrid-muon",
        spec=dict(optimizer="bakeoff_optimizers.HybridMuonX45",
                  lr=TUNED_WINNER["lr"] * 3, lr_multiplier=3,
                  betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
                  warmup_steps=TUNED_WINNER["warmup_steps"],
                  muon_params="embeddings + projections (embed, head, mixer "
                              "value/key/query_key/out, mlp w1/w2/w3)",
                  momentum=0.95, nesterov=True,
                  update_scaling="0.2*sqrt(max(shape)) (hybrid_muon, pinned)"),
        note="the prior receipt's bracket: 3x = Muon's neutral point, 10x "
             "collapses"),
    "sf_prior": dict(
        family="sf-adamw",
        spec=dict(optimizer="schedulefree.AdamWScheduleFree",
                  lr=1e-3, betas=(0.9, 0.95), weight_decay=0.01,
                  warmup_steps=0, r=0.0, weight_lr_power=2.0),
        note="the SF family's LR/beta bracket at the PRIOR invertible line's "
             "own point (run_invertible_stage10_optimizers_8m.py: lr .001, "
             "betas (.9,.95), wd .01) -- different model, different regime, "
             "cited as the prior, bracket only"),
    "sf_muon": dict(
        family="sf-wrapped-hybrid-muon",
        spec=dict(optimizer="schedulefree.ScheduleFreeWrapper("
                            "HybridMuonX45)",
                  lr=TUNED_WINNER["lr"], betas=(0.9, BETA2),
                  base_weight_decay=0.0, warmup_steps=0,
                  wrapper_momentum=0.9,
                  weight_decay_at_y=TUNED_WINNER["weight_decay"]),
        note="the OPTIONAL exploratory combination: primal averaging around "
             "the hybrid Muon via the package's own wrapper; the wrapper's "
             "momentum interacts with Muon's own momentum in unstudied ways "
             "and no published convention exists -- receipted as exploratory "
             "if run at all"),
}

# The pre-stated SF prediction (the task's own): if the C1/C2 early hole is
# warmup-shaped, the schedule-free arm ELIMINATES the first-10-steps hole by
# design; refuted if it opens the same hole or worse.
SF_PREDICTION = dict(
    arm="sf",
    direction="eliminates the first-10-steps hole (the step<=20 deficit "
               "vs gdn2) by design",
    refuted_if="opens the same hole or worse (step-10 deficit at or above the "
               "incumbent's 0.3555)",
    incumbent_step10_deficit=0.35550427436828613,   # side_by_side_summary.json
    incumbent_step20_deficit=0.41776544707162033,
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require_lease() -> str:
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES "
                         "required before launch")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: {card} vs CUDA_VISIBLE_DEVICES "
                         f"{visible}")
    return card


def _versions() -> dict:
    import torch
    try:
        sf_version = importlib.metadata.version("schedulefree")
    except importlib.metadata.PackageNotFoundError:
        sf_version = "unknown"
    return dict(torch=torch.__version__, schedulefree=sf_version,
                hybrid_muon_sha256=digest(REPO / "experiments/expressivity_tasks"
                                               / "hybrid_muon.py"),
                bakeoff_optimizers_sha256=digest(Path(__file__).resolve().parent
                                                 / "bakeoff_optimizers.py"))


# ------------------------------------------------------------- the optimizer


def _make_optimizer(arm: str, model):
    """Build the arm's optimizer + the per-family loop adapters.  Returns
    (optimizer, adapter) where the adapter carries the family's three protocol
    facts: how the per-step LR is set (the schedule), how the checkpoint
    snapshot is taken (live weights vs the schedule-free primal average), and
    any pre-loop call the family requires."""
    import torch
    import schedulefree
    from experiments.expressivity_tasks.x45_matrix_emender import \
        bakeoff_optimizers as BO

    spec = ARMS[arm]["spec"]
    trainable = [p for p in model.parameters() if p.requires_grad]
    if ARMS[arm]["family"] == "adamw":
        optimizer = torch.optim.AdamW(
            trainable, lr=spec["lr"], betas=tuple(spec["betas"]),
            weight_decay=spec["weight_decay"], fused=True)

        def set_lr(step):
            for group in optimizer.param_groups:
                group["lr"] = spec["lr"] * min(
                    1., (step + 1) / max(1, spec["warmup_steps"]))
        return optimizer, dict(set_lr=set_lr, snapshot=lambda:
                               {k: v.detach().cpu().clone()
                                for k, v in model.state_dict().items()},
                               pre_loop=None, swap_beta=None,
                               muon_metrics=False)
    if ARMS[arm]["family"] == "sf-adamw":
        optimizer = schedulefree.AdamWScheduleFree(
            trainable, lr=spec["lr"], betas=tuple(spec["betas"]),
            weight_decay=spec["weight_decay"], warmup_steps=spec["warmup_steps"],
            r=spec["r"], weight_lr_power=spec["weight_lr_power"])
        return optimizer, dict(
            set_lr=lambda step: None,       # NO schedule, by design
            snapshot=lambda: BO.sf_x_snapshot(model, optimizer,
                                               spec["betas"][0]),
            pre_loop=optimizer.train, swap_beta=spec["betas"][0],
            muon_metrics=False)
    if ARMS[arm]["family"] == "hybrid-muon":
        muon_params, adamw_params, muon_names, adamw_names = BO.split_parameters(
            model)
        optimizer = BO.HybridMuonX45(muon_params, adamw_params, lr=spec["lr"],
                                      betas=tuple(spec["betas"]),
                                      weight_decay=spec["weight_decay"])

        def set_lr(step):
            for group in optimizer.param_groups:
                group["lr"] = spec["lr"] * min(
                    1., (step + 1) / max(1, spec["warmup_steps"]))
        return optimizer, dict(
            set_lr=set_lr,
            snapshot=lambda: {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()},
            pre_loop=None, swap_beta=None, muon_metrics=True,
            routing=dict(muon=muon_names, adamw=adamw_names))
    if ARMS[arm]["family"] == "sf-wrapped-hybrid-muon":
        muon_params, adamw_params, muon_names, adamw_names = BO.split_parameters(
            model)
        base = BO.HybridMuonX45(muon_params, adamw_params, lr=spec["lr"],
                                betas=tuple(spec["betas"]),
                                weight_decay=spec["base_weight_decay"])
        optimizer = schedulefree.ScheduleFreeWrapper(
            base, momentum=spec["wrapper_momentum"],
            weight_decay_at_y=spec["weight_decay_at_y"])
        return optimizer, dict(
            set_lr=lambda step: None,       # schedule-free by design
            snapshot=lambda: BO.sf_x_snapshot(model, optimizer,
                                              spec["wrapper_momentum"]),
            pre_loop=optimizer.train, swap_beta=spec["wrapper_momentum"],
            muon_metrics=True, routing=dict(muon=muon_names, adamw=adamw_names))
    raise ValueError(f"unregistered arm family: {ARMS[arm]['family']}")


# ------------------------------------------------------------ the training


def _train_with_snapshots(model, arm, stream, targets, label, log_path,
                          checkpoint_steps):
    """The probe protocol's training loop (the side-by-side's
    ``_train_with_snapshots_k10`` shape VERBATIM -- matched-exposure stopping,
    honest clock, per-step log rows written live, the protocol's own grad
    clip), with the optimizer family the arm's own.  The loop's ONLY addition
    over the banked loops is the CPU state snapshot at each checkpoint step
    (after the row is logged; no eval, no telemetry, no RNG draw inside the
    loop)."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)

    optimizer, adapter = _make_optimizer(arm, model)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    opt_seconds = []
    step = seen = 0
    rows = []
    snapshots = {}
    diverged = False
    last_muon_metrics = None
    if adapter["pre_loop"] is not None:
        adapter["pre_loop"]()
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
            adapter["set_lr"](step)
            loss_sum = 0.0
            for x, y, valid in grouped(batch_rows, MICROBATCH):
                loss = model.loss(x, y)
                if not torch.isfinite(loss):
                    diverged = True
                    break
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            if diverged:
                elapsed += time.monotonic() - start
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots,
                            opt_seconds=opt_seconds,
                            muon_metrics=last_muon_metrics)
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                           error_if_nonfinite=True)
            torch.cuda.synchronize()
            opt_start = time.monotonic()
            optimizer.step()
            torch.cuda.synchronize()
            opt_seconds.append(time.monotonic() - opt_start)
            if adapter["muon_metrics"]:
                # ScheduleFreeWrapper forwards state/param_groups but not
                # arbitrary attributes; the hybrid's metrics live on .base.
                holder = optimizer
                if not hasattr(holder, "last_metrics") \
                        and getattr(holder, "base", None) is not None:
                    holder = holder.base
                last_muon_metrics = getattr(holder, "last_metrics", None)
            elapsed += time.monotonic() - start
            step += 1
            seen += n
            row = dict(step=step, active_seconds=elapsed, bytes=seen,
                       loss=loss_sum, time_ns=time.time_ns())
            rows.append(row)
            if log is not None:
                log.write(json.dumps(row) + "\n")
            if step in checkpoint_steps:
                snapshots[step] = adapter["snapshot"]()
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {loss_sum:.4f}", flush=True)
    finally:
        if log is not None:
            log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                rows=rows, snapshots=snapshots, opt_seconds=opt_seconds,
                muon_metrics=last_muon_metrics)


# ------------------------------------------------------------------- the arm


def run_arm(args) -> dict:
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys \
        import probe_diagnostics
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    if args.label not in ARMS:
        raise ValueError(f"unregistered arm label: {args.label}")
    config = dict(TUNED_WINNER)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    model, optimizer_config, summary = _build_k10(config, args.seed)
    arm = ARMS[args.label]
    run_config = dict(
        probe="optimizer bake-off", arm="k10_separated_keys (the winner)",
        label=args.label, family=arm["family"], spec=arm["spec"],
        conventions_note=arm["note"], seed=args.seed,
        exposure_targets=int(args.bytes),
        model_build="velocity_ladder._build_k10 verbatim (the tuned winner's "
                    "config)", optimizer_knobs_held_at_incumbent=dict(
                        betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
                        warmup_steps=TUNED_WINNER["warmup_steps"],
                        grad_clip=GRAD_CLIP,
                        microbatch=MICROBATCH),
        schedule_axis=dict(incumbent="warmup 45 then constant",
                          sf_arms="NO schedule by design (warmup_steps=0)"),
        sf_lr_mapping="lr := the incumbent's tuned PEAK lr (same LR, schedule "
                      "removed); no scheduler is needed with schedule-free "
                      "(the package's own guidance), so there is no "
                      "warmup/decay curve to tune against the probe's FIXED "
                      "180-step budget -- the averaging is the schedule",
        symmetry_rule="same stream, same data order, same matched-exposure "
                      "stopping rule, same honest clock, same grad clip, "
                      "same calibration eval; only the optimizer family (and "
                      "its schedule, where the family demands one) differs",
        optimizer=optimizer_config, summary=summary, identity=identity,
        checkpoint_steps=list(args.checkpoint_steps),
        checkpoint_note=("state snapshots taken inside the training loop at "
                         "each checkpoint step (the schedule-free arms "
                         "snapshot the PRIMAL-AVERAGED eval point, computed "
                         "non-destructively); every eval runs on restored "
                         "snapshots after training, so the training path is "
                         "unperturbed by construction"),
        versions=_versions(),
        bindings=dict(matched_arms.registration_bindings(),
                      banked_rows=BANKED,
                      side_by_side=str(RECEIPTS / "side_by_side_summary.json")),
        git=matched_arms.git_state(),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the probe exposure); the honest "
                       "elapsed wall is reported first-class"))
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=PARAMETERS_AUDITED,
                        trainable=PARAMETERS_AUDITED)))
    print(json.dumps(dict(arm=args.label, family=arm["family"], card=card,
                          seed=args.seed, parameters=PARAMETERS_AUDITED)),
          flush=True)

    run = _train_with_snapshots(
        model, args.label, stream, int(args.bytes), args.label,
        out / "training.jsonl", tuple(args.checkpoint_steps))
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 else run["opt_seconds"]

    # ---- the checkpoint pass (restored snapshots; training is over)
    rows_by_step = {r["step"]: r for r in run["rows"]}
    checkpoint_rows = []
    for step in args.checkpoint_steps:
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        row = dict(step=step, bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"], eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        checkpoint_rows.append(row)
        print(f"  checkpoint step {step}: bytes {row['bytes']} "
              f"train {row['train_loss']:.4f} eval {row['eval_nll']:.4f}",
              flush=True)
    with (out / "checkpoints.jsonl").open("w", buffering=1) as f:
        for row in checkpoint_rows:
            f.write(json.dumps(row) + "\n")

    diagnostics = None
    if run["status"] == "completed" and run["snapshots"]:
        final_step = run["snapshots"][max(run["snapshots"])]
        model.load_state_dict(final_step)
        diagnostics = probe_diagnostics(model, stream, windows=args.probe_windows)

    receipt = dict(
        probe="optimizer bake-off", label=args.label, family=arm["family"],
        spec=arm["spec"], conventions_note=arm["note"],
        status=run["status"], seed=args.seed,
        exposure_bytes=run["seen"], exposure_steps=run["step"],
        train_seconds=round(run["elapsed"], 1),
        bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
        optimizer_step_seconds=dict(
            mean_steady=(sum(steady) / len(steady)) if steady else None,
            note="steps 10+; the honest clock's own measure, per step"),
        parameters=PARAMETERS_AUDITED,
        final_eval_nll=(checkpoint_rows[-1]["eval_nll"]
                        if checkpoint_rows else None),
        banked=dict(k10_tuned=BANKED["k10_tuned"], gdn2=BANKED["gdn2"]),
        checkpoints=checkpoint_rows, diagnostics=diagnostics,
        muon_update_rms=run["muon_metrics"],
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        config=run_config, session=matched_arms.session_meta(card))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "status", "exposure_bytes", "exposure_steps",
                       "final_eval_nll", "bytes_per_s")}), flush=True)
    return receipt


# ------------------------------------------------------------------- assemble


def assemble(args) -> dict:
    """Read the bake-off arms' checkpoint curves + the side-by-side's banked
    curves (gdn2, the tuned and transferred incumbents) and emit the overlay
    summary: the gap-vs-phase curve per arm against gdn2's own, the
    first-10-steps hole per arm, the steady read, and the SF prediction
    verdict."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    def load_checkpoints(path: Path):
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    gdn2_rows = load_checkpoints(
        SIDE_BY_SIDE_RUNS / "gdn2_armed_seed3401/checkpoints.jsonl")
    gdn2_by_step = {r["step"]: r for r in gdn2_rows}
    incumbent_rows = load_checkpoints(
        SIDE_BY_SIDE_RUNS / "k10_tuned_armed_seed3401/checkpoints.jsonl")
    incumbent_by_step = {r["step"]: r for r in incumbent_rows}

    arms = {}
    labels = args.arms or [p.name[: -len("_seed3401")]
                           for p in RUNS.glob("*_seed3401")
                           if (p / "checkpoints.jsonl").exists()]
    for label in labels:
        rows = load_checkpoints(RUNS / f"{label}_seed3401/checkpoints.jsonl")
        if not rows:
            continue
        curve, deficits_early, deficits_steady = [], [], []
        for row in rows:
            g = gdn2_by_step.get(row["step"])
            entry = dict(step=row["step"], bytes=row["bytes"],
                        arm_eval_nll=row["eval_nll"],
                        arm_train_loss=row["train_loss"])
            if g is not None:
                entry["gdn2_eval_nll"] = g["eval_nll"]
                entry["deficit_vs_gdn2"] = row["eval_nll"] - g["eval_nll"]
                if row["step"] <= 20:
                    deficits_early.append(entry["deficit_vs_gdn2"])
                if 80 <= row["step"] <= 160:
                    deficits_steady.append(entry["deficit_vs_gdn2"])
            inc = incumbent_by_step.get(row["step"])
            if inc is not None:
                entry["incumbent_eval_nll"] = inc["eval_nll"]
                entry["arm_minus_incumbent"] = (row["eval_nll"]
                                                - inc["eval_nll"])
            curve.append(entry)
        result_path = RUNS / f"{label}_seed3401/result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() \
            else {}
        arms[label] = dict(
            family=result.get("family"), status=result.get("status"),
            spec=result.get("spec"), train_seconds=result.get("train_seconds"),
            bytes_per_s=result.get("bytes_per_s"),
            final_eval_nll=result.get("final_eval_nll"),
            final_vs_banked_incumbent=(result.get("final_eval_nll")
                                       - BANKED["k10_tuned"]
                                       if result.get("final_eval_nll")
                                       else None),
            first_10_steps_hole=dict(
                step10_deficit=next((e["deficit_vs_gdn2"] for e in curve
                                     if e["step"] == 10), None),
                step20_deficit=next((e["deficit_vs_gdn2"] for e in curve
                                     if e["step"] == 20), None),
                max_deficit_steps_le_20=(max(deficits_early)
                                         if deficits_early else None),
                note="deficit vs gdn2's banked curve (arm NLL minus gdn2 NLL "
                     "at the same step); positive = the arm is WORSE"),
            steady_deficit_mean_steps_80_160=(sum(deficits_steady)
                                              / len(deficits_steady)
                                              if deficits_steady else None),
            curve=curve)
    hole = arms.get("sf", {}).get("first_10_steps_hole", {}) if "sf" in arms \
        else None
    if hole is None:
        verdict = None
    else:
        step10 = hole.get("step10_deficit")
        if step10 is None:
            verdict = "UNMEASURED (no step-10 checkpoint)"
        elif step10 < 0.10:
            verdict = ("MET: the hole is ELIMINATED (the step-10 deficit "
                       f"{step10:.4f} is at/below the steady-state band, "
                       "not the incumbent's 0.3555 hole)")
        elif step10 >= SF_PREDICTION["incumbent_step10_deficit"]:
            verdict = ("REFUTED: the schedule-free arm opens the same hole or "
                      f"worse (step-10 deficit {step10:.4f} >= the "
                      "incumbent's 0.3555)")
        else:
            verdict = ("PARTIAL: the hole is reduced but present (step-10 "
                      f"deficit {step10:.4f}, between elimination and the "
                      "incumbent's hole)")
    summary = dict(
        probe="optimizer bake-off assemble", banked=BANKED,
        sf_prediction=dict(SF_PREDICTION,
                          measured=hole,
                          verdict=verdict),
        arms=arms)
    atomic_json(Path(args.out), summary)
    print(json.dumps({label: dict(
        final=arms[label]["final_eval_nll"],
        hole10=arms[label]["first_10_steps_hole"]["step10_deficit"])
        for label in arms}, indent=1), flush=True)
    return summary


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--label", required=True, choices=sorted(ARMS))
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--output", required=True)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    a.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, 2 checkpoint steps")
    s = sub.add_parser("assemble")
    s.add_argument("--arms", nargs="*", default=None,
                   help="arm labels to assemble (default: every run dir)")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "optimizer_bakeoff_summary.json"))
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.smoke:
            args.bytes = 200_000
            args.checkpoint_steps = [1, 2, 3]
            args.probe_windows = 2
        run_arm(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
