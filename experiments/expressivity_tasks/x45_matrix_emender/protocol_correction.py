"""PROTOCOL CORRECTION -- the literature-backed scoring fixes applied to the
velocity probe, and the recipe-correct (Moonlight) Muon arms, on BOTH machines.

THE PROBLEM THIS LANE CLOSES (the operator's directive, from the research
brief): the constant-LR fixed-step velocity probe is a BIASED estimator of
long-horizon final loss -- the river-valley oscillation penalty (Wen et al.,
ICLR 2025: constant-LR loss stays elevated by bouncing on the steep valley
walls; the decay phase removes the bounce), fixed by appending a short
cooldown (Haegele et al., NeurIPS 2024: constant-LR-then-cooldown matches
cosine at full scale) and/or by scoring averaged iterates.  And the Muon arms
the probe selected are SPEEDRUN-recipe configs -- raw 3x-5x LR multiples,
the short-horizon speedrun convention (Keller Jordan's own page: speed
records, "no claims about better final loss") -- rather than the
long-horizon-correct MOONLIGHT recipe (Liu et al., arXiv:2502.16982: with
the update RMS matched to AdamW's, Muon directly reuses the LR and weight
decay tuned for AdamW, i.e. ~1x the AdamW-tuned point, WITH weight decay).

THE KEY CODEBASE FACT (the lane's own finding, approved by the supervisor):
our pinned hybrid Muon is ALREADY RMS-matched by construction --
``hybrid_muon.scaled_update`` scales the orthogonalized update by
``0.2 * sqrt(max(shape))``, the Moonshot scaling whose own docstring says it
"matches AdamW's approximate .2 update RMS"; the measured Muon update RMS on
every banked arm is 0.168-0.185, i.e. at Moonlight's ~0.2 target (AdamW's
typical 0.2-0.4), and weight decay is already in the hybrid (decoupled, at
each machine's own tuned value).  So the Moonlight recipe's only unmet
element is the LR point: the four arms below isolate the LR under matched
update scale -- a purer test than the brief implied -- and the driver ADDS a
read-only per-step measurement of the aux AdamW group's update RMS so the
"AdamW-equivalent RMS" receipt is measured on OUR machines, not assumed from
the paper.

THE FOUR ARMS (2 machines x {speedrun recipe re-scored, Moonlight recipe}),
every one under the corrected scoring, at the exact probe protocol verbatim
(the 20,000,000-target matched-exposure rule -> the 20,022,866 landing, the
registered flagship stream, the document-disjoint calibration eval at 7
windows / 14,336 targets, the honest clock, the protocol's own grad-clip):

* ``winner_speedrun_rescored``  OUR banked best row re-scored: the K10 winner
  + hybrid Muon at 4x its own tuned AdamW LR, the banked B-analog routing
  (``routing_symmetry.split_parameters_winner(model, "banked_gates_adamw")``,
  imported verbatim), seed 3402 -- the 1.4899 row's own config, so the
  step-180 constant-LR eval must reproduce the banked final and the corrected
  scores differ from the banked row ONLY by the instrument.
* ``winner_moonlight``          the SAME machine/routing/seed at the Moonlight
  LR point: 1x the tuned AdamW LR (3.3427e-4), own betas/weight-decay/warmup.
* ``gdn2_speedrun_rescored``   GDN2's banked best row re-scored: option B
  (gates-on-AdamW, ``muon_control.split_parameters_gdn2`` imported verbatim)
  at 5x its own banked AdamW LR, seed 3401 -- the 1.3951 row's own config.
* ``gdn2_moonlight``            the same at the Moonlight LR point: 1x GDN2's
  banked AdamW LR (5.4201855e-4), own betas/weight-decay/warmup.

THE CORRECTED SCORING (supervisor-approved, pre-stated here and receipted in
every arm):

* PRIMARY -- post-COOLDOWN eval: after the constant-LR phase completes the
  banked protocol verbatim (180 steps, the 20,022,866 landing), append a
  LINEAR cooldown of 24 steps (13.3% of 180, inside the brief's 10-15%
  band), continuing the same registered stream/data order, LR factor
  ``(25 - k)/24`` for cooldown step k = 1..24 (1.0 -> 1/24, never zero),
  both hybrid groups scaling equally (the lanes' convention).  All arms see
  the same ~2.7M extra targets (~13%) -- a stated protocol EXTENSION,
  matched across arms; the step-180 constant-LR eval is retained so every
  banked row stays comparable.
* SECONDARY -- EMA iterate at step 180 (no extra data): EMA decay 0.99
  (window ~100 steps, 55% of the budget), initialized at the init weights
  and updated after every constant-phase optimizer step, scored on the same
  calibration eval.  Ranking-only: its absolute number is biased toward
  mid-run weights (stated, not smoothed); the cross-check is whether its
  RANKING agrees with the cooldown score's.

PRE-STATED, NO DIRECTIONAL PREDICTION (the directive's own rule, receipted
in every arm): the literature does not say which machine benefits more from
the corrected instrument or from the Moonlight LR point -- the gate-surface
difference (our 27.7K learned gate params vs GDN2's 27.4M) makes both
directions live.  This lane measures; any reversal is reported as plainly as
any confirmation.

PROVENANCE (the lanes' own rules, restated): no model/semantics change,
scoring/optimizer-surface only; new files only -- no ``k10_*``/``k11_*``
source, no PREREGISTRATION, no ``lattice.py``, no GDN2 model code is
edited, only imported (``velocity_ladder._build_k10`` and
``matched_arms.build_gdn2`` verbatim, ``bakeoff_optimizers.HybridMuonX45``
verbatim through a metrics-only subclass, the prior lanes' own routing
splits).  Protocol-identical starts, receipted: every arm's step-1 train
loss is bit-identical to its banked same-seed row's (asserted in-code
against the banked run's own training.jsonl).  Runs on the protocol
correction lane's own cards 4/5 (the K11 lane's 0/1 and the op-audit lane's
2/3 untouched).

Deliverables: per-arm receipts under
``protocol_correction_runs/<label>/`` (each arm's own seed is part of its
label), the assemble summary ``receipts/protocol_correction_summary.json``,
and ``RESULT_PROTOCOL_CORRECTION.md``.

Usage (see ``run_protocol_correction_leased.sh`` for the detached leased
driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.protocol_correction arm \\
        --label winner_speedrun_rescored \\
        --output protocol_correction_runs/winner_speedrun_rescored
"""
from __future__ import annotations

import argparse
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
RUNS = ROOT / "protocol_correction_runs"
ROUTING_SYMMETRY_RUNS = ROOT / "routing_symmetry_runs"
GDN2_LR_BRACKET_RUNS = ROOT / "gdn2_lr_bracket_runs"

EXPOSURE_TARGETS = 20_000_000
# The bake-off's own checkpoint plan (the curve-overlay grid every banked
# curve is read at) -- the constant-LR phase's own eval grid.
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
PROBE_WINDOWS = 4
GRAD_CLIP = 1.0                 # the probe protocol's own clip, every arm

# ---- the corrected scoring (supervisor-approved conventions, pre-stated)
COOLDOWN_STEPS = 24             # 13.3% of the 180-step budget (the brief's
                                # 10-15% band; Haegele: constant-LR+cooldown
                                # matches cosine at full scale)
EMA_DECAY = 0.99                # window ~100 steps, 55% of the budget

CORRECTED_SCORING = dict(
    primary="post-cooldown eval: a linear cooldown of 24 steps (13.3% of the "
            "180-step constant-LR budget) appended after the banked "
            "matched-exposure landing, same stream/data order, LR factor "
            "(25 - k)/24 for cooldown step k (1.0 -> 1/24, never zero), both "
            "hybrid groups scaling equally; the corrected score is the "
            "post-cooldown calibration eval",
    secondary="EMA iterate at step 180 (no extra data): decay 0.99, "
              "initialized at the init weights, updated after every "
              "constant-phase optimizer step, NOT updated during the "
              "cooldown; RANKING-ONLY -- its absolute number is biased "
              "toward mid-run weights (stated, not smoothed)",
    cooldown_steps=COOLDOWN_STEPS,
    cooldown_fraction=round(COOLDOWN_STEPS / 180, 4),
    cooldown_form="linear",
    ema_decay=EMA_DECAY,
    exposure_note="the cooldown steps see ~2.7M more targets (~13%), "
                  "identical across arms -- a stated protocol EXTENSION; "
                  "the step-180 constant-LR eval is retained so every banked "
                  "row stays comparable",
    citations=dict(
        river_valley="Wen et al., ICLR 2025, arXiv:2410.05192 -- constant-LR "
                     "loss carries an LR-dependent oscillation penalty that "
                     "decay/averaging removes",
        cooldown="Haegele et al., NeurIPS 2024, arXiv:2405.18392 -- "
                 "constant-LR + short cooldown matches cosine at full scale",
        averaging="Defazio et al., arXiv:2405.15682 -- iterate averaging as "
                  "the schedule-free fix for the same penalty",
        moonlight="Liu et al. (Moonshot AI), arXiv:2502.16982 -- with the "
                  "update RMS matched to AdamW's (~0.2; AdamW typical "
                  "0.2-0.4), Muon directly reuses the LR and weight decay "
                  "tuned for AdamW, i.e. ~1x the AdamW-tuned LR, WITH "
                  "weight decay",
        horizon="Bjorck et al., ICLR 2025 -- the optimal LR shrinks with "
                "token horizon; probe-tuned LRs transfer only downscaled"),
)

# The banked best rows being re-scored (cited, never re-run; receipted
# against the banked run dirs by the tests).
OUR_BEST = dict(
    label="muon4x_banked_seed3402", machine="winner", seed=3402,
    final=1.4899048124040877, lr_multiplier=4,
    run_dir=str(ROUTING_SYMMETRY_RUNS / "muon4x_banked_seed3402"))
GDN2_BEST = dict(
    label="gdn2_muon5x_gates_adamw_seed3401", machine="gdn2", seed=3401,
    final=1.3950541530336653, lr_multiplier=5,
    run_dir=str(GDN2_LR_BRACKET_RUNS / "gdn2_muon5x_gates_adamw_seed3401"))
BANKED_MARGIN = OUR_BEST["final"] - GDN2_BEST["final"]   # 0.0949, GDN2 ahead

# The bands every margin is read against (the side-by-side's endpoint band;
# the winner's own measured two-seed band -- the seed evidence this box has).
ENDPOINT_VARIANCE_BAND = 0.075
WINNER_SEED_DELTA = 0.011948040553501738

# Other banked reference rows (cited).
BANKED_REFS = dict(
    our_adamw=1.822737353188651,
    gdn2_adamw=1.5836738858904158,
    gdn2_muon3x_gates_adamw=1.4150957890919276,
)

# The pre-stated no-prediction rule (the directive's own; receipted in every
# arm and applied verbatim by the assemble).
NO_PREDICTION = dict(
    rule="NO DIRECTIONAL PREDICTION is pre-stated for any arm: the "
         "literature does not say which machine benefits more from the "
         "corrected instrument or from the Moonlight LR point -- the "
         "gate-surface difference (our 27.7K learned gate params vs GDN2's "
         "27.4M) makes both directions live; this lane measures, and any "
         "reversal is reported as plainly as any confirmation")

# The horizon-transfer rule for the board runs (Bjorck-style; the
# supervisor-approved statement, to be receipted in the RESULT doc; the
# alpha fit is DEFERRED to the board-prep stage -- a two-budget LR bracket
# is a multi-arm search that this lane's budget does not admit, labelled as
# such, per the supervisor's scope note).
HORIZON_TRANSFER_RULE = dict(
    rule="board runs transfer probe-optimal LRs DOWNSCALED: "
         "eta_board = eta_probe_opt * (D_probe / D_board)^alpha, with alpha "
         "FIT at our scale from a two-budget bracket (re-run the top config "
         "at 2x probe budget and take the local optimum shift) -- Bjorck et "
         "al.'s power-law FORM, never their imported exponent (fitted at far "
         "larger scale); PLUS the hard gate: top-k configs must hold rank "
         "stability at 2x probe budget under the corrected scoring before "
         "any board transfer",
    probe_targets=EXPOSURE_TARGETS,
    board_seconds=2700,
    board_targets_estimate="~91.7M (2700 s at gdn2's banked 33,958 bytes/s), "
                           "~4.6x the probe exposure",
    alpha_provenance="DEFERRED to the board-prep stage: the two-budget "
                     "bracket that fits alpha is a multi-arm search outside "
                     "this lane's budget; the rule is pre-stated now and the "
                     "fit is labelled as not-yet-run",
    hard_gate="rank stability at 2x probe budget under the corrected "
              "scoring is a HARD gate for any board transfer",
)

ARMS = {
    "winner_speedrun_rescored": dict(
        machine="winner", seed=3402, lr_multiplier=4,
        routing="banked_gates_adamw",
        recipe="speedrun",
        recipe_note="the banked best row's own recipe: RMS-matched Muon (the "
                    "pinned Moonshot 0.2*sqrt(max(shape)) scaling, measured "
                    "update RMS 0.168-0.185 on the banked arms) at the "
                    "SPEEDRUN LR point -- 4x the machine's own tuned AdamW "
                    "LR -- the config the constant-LR velocity probe "
                    "selected",
        note="DELIVERABLE 1, our side: the 1.4899 row's own config "
             "(B-analog routing, 4x, seed 3402) re-scored under the "
             "corrected instrument -- the corrected score differs from the "
             "banked row ONLY by the instrument"),
    "winner_moonlight": dict(
        machine="winner", seed=3402, lr_multiplier=1,
        routing="banked_gates_adamw",
        recipe="moonlight",
        recipe_note="the Moonlight recipe: the SAME RMS-matched Muon at ~1x "
                    "the machine's own tuned AdamW LR (Moonlight: with the "
                    "update RMS matched to AdamW's, Muon directly reuses "
                    "the LR and weight decay tuned for AdamW), weight decay "
                    "retained at the machine's own tuned value (already in "
                    "the hybrid, decoupled)",
        note="DELIVERABLE 2, our side: the recipe-correct comparison arm at "
             "the machine-seed pairing of the banked best row (3402)"),
    "gdn2_speedrun_rescored": dict(
        machine="gdn2", seed=3401, lr_multiplier=5, gates_on_adamw=True,
        routing="option B (gates-on-AdamW, the control's own arm)",
        recipe="speedrun",
        recipe_note="the banked best row's own recipe: RMS-matched Muon at "
                    "the SPEEDRUN LR point -- 5x GDN2's own banked AdamW LR "
                    "-- the config the constant-LR velocity probe selected",
        note="DELIVERABLE 1, GDN2's side: the 1.3951 row's own config "
             "(option B, 5x, seed 3401) re-scored under the corrected "
             "instrument"),
    "gdn2_moonlight": dict(
        machine="gdn2", seed=3401, lr_multiplier=1, gates_on_adamw=True,
        routing="option B (gates-on-AdamW, the control's own arm)",
        recipe="moonlight",
        recipe_note="the Moonlight recipe: the SAME RMS-matched Muon at ~1x "
                    "GDN2's own banked AdamW LR, weight decay retained at "
                    "GDN2's own tuned value",
        note="DELIVERABLE 2, GDN2's side: the recipe-correct comparison arm "
             "at the machine-seed pairing of the banked best row (3401)"),
}


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def _banked_step1_loss(machine: str, seed: int) -> float:
    """The banked same-seed row's step-1 train loss (the protocol-identical
    start receipt: the optimizer cannot touch the first forward, so a new arm
    at the same seed must reproduce it bit-for-bit)."""
    if machine == "winner":
        path = ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{seed}" \
            / "training.jsonl"
    else:
        path = GDN2_LR_BRACKET_RUNS / f"gdn2_muon5x_gates_adamw_seed{seed}" \
            / "training.jsonl"
    if not path.exists():
        raise ValueError(f"the banked same-seed row is missing: {path}")
    return json.loads(path.read_text().splitlines()[0])["loss"]


# --------------------------------------------------------- the optimizer hook


def _make_measured_hybrid():
    """``bakeoff_optimizers.HybridMuonX45`` VERBATIM through a metrics-only
    subclass: the aux AdamW group's update RMS (m_hat / (sqrt(v_hat) + eps))
    is computed AFTER each step from the aux optimizer's own state, so the
    Moonlight RMS-matching receipt (Muon's update RMS vs the machine's own
    AdamW update RMS) is MEASURED, not assumed.  No update path is touched."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender import (
        bakeoff_optimizers as BO)

    class _MeasuredHybridMuonX45(BO.HybridMuonX45):
        @torch.no_grad()
        def step(self, closure=None):
            super().step(closure)
            squared, count = 0.0, 0
            for group in self.aux.param_groups:
                beta1, beta2 = group["betas"]
                for param in group["params"]:
                    state = self.aux.state.get(param)
                    if not state or "step" not in state:
                        continue
                    raw_step = state["step"]
                    step_n = int(raw_step.item()) if torch.is_tensor(
                        raw_step) else int(raw_step)
                    m_hat = state["exp_avg"] / (1 - beta1 ** step_n)
                    v_hat = state["exp_avg_sq"] / (1 - beta2 ** step_n)
                    update = m_hat / (v_hat.sqrt() + group["eps"])
                    squared += float(update.square().sum())
                    count += update.numel()
            self.last_metrics = dict(
                self.last_metrics,
                adamw_update_rms=(squared / count) ** .5 if count else 0.)
            return None

    return _MeasuredHybridMuonX45


# ------------------------------------------------------------ the training


def cooldown_factor(k: int, cooldown_steps: int = COOLDOWN_STEPS) -> float:
    """The approved linear cooldown form: factor (C + 1 - k)/C for cooldown
    step k = 1..C -- 1.0 at the first cooldown step, 1/C (never zero) at
    the last."""
    if not 1 <= k <= cooldown_steps:
        raise ValueError(f"cooldown step {k} outside 1..{cooldown_steps}")
    return (cooldown_steps + 1 - k) / cooldown_steps


def _train_corrected(model, optimizer, peak_lr, warmup, microbatch, stream,
                     targets, label, log_path, checkpoint_steps,
                     cooldown_steps, ema_decay):
    """The lanes' armed loop (matched-exposure stopping, honest clock with
    time_ns rows, the protocol's own grad clip, the optimizer-step seconds
    receipted separately, CPU state snapshots inside the loop) with the two
    corrections: (a) the constant-LR phase runs the banked protocol verbatim
    to the matched-exposure landing, then a linear cooldown of
    ``cooldown_steps`` steps continues the SAME stream/data order; (b) an EMA
    of the parameters (decay ``ema_decay``) is maintained over the
    constant-LR phase only, initialized at the init weights, updated after
    every constant-phase optimizer step.  No eval, no telemetry, no RNG draw
    inside the loop -- every eval runs on restored snapshots after
    training."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)

    batches = iter(stream.batches("train"))
    elapsed = 0.0
    opt_seconds = []
    step = seen = 0
    rows = []
    snapshots = {}
    diverged = False
    last_metrics = None
    ema = {name: param.detach().float().clone()
           for name, param in model.named_parameters()
           if param.requires_grad}

    def lr_at(step0, cooldown_k=0):
        factor = min(1., (step0 + 1) / max(1, warmup))
        if cooldown_k:
            factor *= cooldown_factor(cooldown_k, cooldown_steps)
        return peak_lr * factor

    def do_step(cooldown_k):
        """One training step (the lanes' own body); returns the logged row
        or None on divergence."""
        nonlocal elapsed, step, seen, last_metrics, diverged
        torch.cuda.synchronize()
        start = time.monotonic()
        try:
            batch_rows = next(batches)
        except StopIteration:
            raise RuntimeError("train split exhausted before the probe budget")
        n = sum(len(r[3]) for r in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        lr = lr_at(step, cooldown_k)
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for x, y, valid in grouped(batch_rows, microbatch):
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                diverged = True
                break
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        if diverged:
            elapsed += time.monotonic() - start
            return None
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                       error_if_nonfinite=True)
        torch.cuda.synchronize()
        opt_start = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        opt_seconds.append(time.monotonic() - opt_start)
        last_metrics = getattr(optimizer, "last_metrics", None)
        elapsed += time.monotonic() - start
        step += 1
        seen += n
        row = dict(step=step, active_seconds=elapsed, bytes=seen,
                   loss=loss_sum, time_ns=time.time_ns(),
                   phase="cooldown" if cooldown_k else "constant",
                   lr_factor=(lr / peak_lr))
        rows.append(row)
        return row

    log = Path(log_path).open("w", buffering=1) if log_path else None
    try:
        # ---- phase 1: the banked protocol verbatim (constant LR)
        while seen < targets:
            row = do_step(0)
            if row is None:
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots,
                            opt_seconds=opt_seconds, ema=None,
                            constant_steps=step, muon_metrics=last_metrics)
            if log is not None:
                log.write(json.dumps(row) + "\n")
            with torch.no_grad():      # the EMA (constant phase only)
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        ema[name].mul_(ema_decay).add_(param.detach().float(),
                                                       alpha=1 - ema_decay)
            if step in checkpoint_steps:
                snapshots[step] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {row['loss']:.4f}", flush=True)
        constant_steps = step
        constant_elapsed = elapsed
        # ---- phase 2: the linear cooldown (the corrected scoring's own
        # extension; the same stream continues, matched across arms)
        for k in range(1, cooldown_steps + 1):
            row = do_step(k)
            if row is None:
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots,
                            opt_seconds=opt_seconds, ema=ema,
                            constant_steps=constant_steps,
                            muon_metrics=last_metrics)
            if log is not None:
                log.write(json.dumps(row) + "\n")
            if step % 25 == 0 or k == cooldown_steps:
                print(f"  {label} cooldown {k}/{cooldown_steps} "
                      f"{elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {row['loss']:.4f}", flush=True)
        snapshots["final"] = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
    finally:
        if log is not None:
            log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                constant_steps=constant_steps,
                constant_elapsed=constant_elapsed, rows=rows,
                snapshots=snapshots, opt_seconds=opt_seconds, ema=ema,
                muon_metrics=last_metrics)


# ------------------------------------------------------------------- the arm


def run_arm(args) -> dict:
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
        GDN2_BANKED_ADAMW, split_parameters_gdn2)
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import BETA2, MICROBATCH, TUNED_WINNER
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import split_parameters_winner
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    if args.label not in ARMS:
        raise ValueError(f"unregistered arm label: {args.label}")
    arm = ARMS[args.label]
    if args.seed != arm["seed"]:
        raise ValueError(f"the {args.label} arm's approved seed is "
                         f"{arm['seed']} (the banked best row's own), got "
                         f"{args.seed}")
    machine = arm["machine"]

    # ---- the machine, the banked builders verbatim, knobs asserted
    if machine == "winner":
        model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER),
                                                      args.seed)
        expected = dict(lr=TUNED_WINNER["lr"], betas=(0.9, BETA2),
                        weight_decay=TUNED_WINNER["weight_decay"],
                        warmup_steps=TUNED_WINNER["warmup_steps"],
                        microbatch=MICROBATCH)
        adam_lr, betas, weight_decay = (TUNED_WINNER["lr"], (0.9, BETA2),
                                        TUNED_WINNER["weight_decay"])
        warmup = TUNED_WINNER["warmup_steps"]
        eval_arm = "matrix_ops"
        muon_params, adamw_params, muon_names, adamw_names = \
            split_parameters_winner(model, arm["routing"])
    else:
        model, optimizer_config, summary = matched_arms.build_gdn2(args.seed)
        expected = GDN2_BANKED_ADAMW
        adam_lr, betas, weight_decay = (optimizer_config["lr"],
                                        tuple(optimizer_config["betas"]),
                                        optimizer_config["weight_decay"])
        warmup = optimizer_config["warmup_steps"]
        eval_arm = "gdn2"
        muon_params, adamw_params, muon_names, adamw_names = \
            split_parameters_gdn2(model, gates_on_adamw=arm["gates_on_adamw"])
    for key, want in expected.items():
        actual = (tuple(optimizer_config[key]) if key == "betas"
                  else optimizer_config[key])
        if actual != want:
            raise ValueError(f"the machine's config drifted on {key}: "
                             f"{actual!r} != the banked {want!r}")

    peak_lr = arm["lr_multiplier"] * adam_lr
    optimizer = _make_measured_hybrid()(
        muon_params, adamw_params, lr=peak_lr, betas=tuple(betas),
        weight_decay=weight_decay)

    parameters_total = sum(p.numel() for p in model.parameters())
    run_config = dict(
        probe="the protocol correction", label=args.label, machine=machine,
        recipe=arm["recipe"], recipe_note=arm["recipe_note"],
        conventions_note=arm["note"], no_prediction=NO_PREDICTION,
        corrected_scoring=CORRECTED_SCORING, seed=args.seed,
        exposure_targets=int(args.bytes),
        model_build=("velocity_ladder._build_k10 verbatim (the tuned "
                     "winner's config, the banked bake-off build)"
                     if machine == "winner" else
                     "matched_arms.build_gdn2 verbatim (the registered "
                     "lattice_gdn2 arm, its searched nomination's knobs)"),
        optimizer=dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim "
                      "through a metrics-only subclass that records the aux "
                      "AdamW group's update RMS; no update path touched)",
            peak_lr=peak_lr, lr_multiplier=arm["lr_multiplier"],
            adamw_tuned_lr=adam_lr,
            lr_convention=(f"{arm['lr_multiplier']}x the machine's own "
                           "tuned AdamW LR -- the speedrun point the "
                           "constant-LR probe selected"
                           if arm["lr_multiplier"] > 1 else
                           "~1x the machine's own tuned AdamW LR -- the "
                           "Moonlight recipe (arXiv:2502.16982: with the "
                           "update RMS matched to AdamW's, Muon directly "
                           "reuses the LR and weight decay tuned for "
                           "AdamW)"),
            betas=tuple(betas), weight_decay=weight_decay,
            weight_decay_note="decoupled, at the machine's own tuned value "
                              "(already in the hybrid; Moonlight requires "
                              "weight decay)",
            warmup_steps=warmup,
            warmup_note="both hybrid groups scale equally (the prior line's "
                        "convention, every banked hybrid arm's own)",
            microbatch=optimizer_config["microbatch"], grad_clip=GRAD_CLIP,
            muon_params=f"{len(muon_names)} tensors "
                        f"({sum(p.numel() for p in muon_params)} params)",
            adamw_params=f"{len(adamw_names)} tensors "
                         f"({sum(p.numel() for p in adamw_params)} params)",
            rms_matching="the pinned Moonshot 0.2*sqrt(max(shape)) scaling "
                         "(hybrid_muon.scaled_update) matches the "
                         "orthogonalized update's RMS to AdamW's ~0.2 "
                         "class BY CONSTRUCTION; measured on every arm "
                         "here (muon_update_rms vs adamw_update_rms)"),
        routing=arm["routing"],
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        horizon_transfer_rule=HORIZON_TRANSFER_RULE,
        checkpoint_steps=list(args.checkpoint_steps),
        checkpoint_note=("state snapshots taken inside the training loop at "
                         "each constant-phase checkpoint step, at the "
                         "post-cooldown final step, plus the EMA iterate at "
                         "step 180 (constant-phase only); every eval runs on "
                         "restored snapshots after training, so the "
                         "training path is unperturbed by construction"),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear cooldown "
                       "(the corrected-scoring extension, same stream); the "
                       "honest elapsed wall is reported first-class, with "
                       "the constant-phase wall reported separately"),
        symmetry_rule="same stream, same data order, same matched-exposure "
                      "stopping rule, same honest clock, same grad clip, "
                      "same calibration eval as the banked rows; the ONLY "
                      "differences are the LR point (speedrun vs Moonlight) "
                      "and the scoring instrument (constant-LR step-180 eval "
                      "vs post-cooldown vs EMA), receipted per arm",
        summary=summary, identity=identity, versions=_versions(),
        bindings=dict(matched_arms.registration_bindings(),
                      banked_best_rows=dict(ours=OUR_BEST, gdn2=GDN2_BEST,
                                            refs=BANKED_REFS,
                                            banked_margin=BANKED_MARGIN),
                      bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                                 winner_seed_delta=WINNER_SEED_DELTA)),
        git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=parameters_total,
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(arm=args.label, machine=machine, card=card,
                          seed=args.seed, recipe=arm["recipe"],
                          parameters=parameters_total,
                          muon_params=len(muon_names),
                          adamw_params=len(adamw_names))), flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, peak_lr, warmup, optimizer_config["microbatch"],
        stream, int(args.bytes), args.label, out / "training.jsonl",
        tuple(args.checkpoint_steps), cooldown_steps, EMA_DECAY)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]

    # the protocol-identical start receipt: the optimizer cannot touch the
    # first forward, so step 1 must reproduce the BANKED same-seed row's
    # step-1 train loss bit-for-bit (the smoke shakedown is exempt: its
    # tiny exposure is not the protocol)
    banked_step1 = _banked_step1_loss(machine, args.seed)
    if not args.smoke and run["rows"][0]["loss"] != banked_step1:
        raise ValueError(
            f"step-1 train loss {run['rows'][0]['loss']!r} != the banked "
            f"same-seed row's {banked_step1!r} (the protocol start is not "
            f"identical)")

    # ---- the scoring pass (restored snapshots; training is over)
    rows_by_step = {r["step"]: r for r in run["rows"]}
    scores = []
    for step in args.checkpoint_steps:            # the constant-LR curve
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, eval_arm, args.eval_split)
        row = dict(kind="constant", step=step, bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"],
                   eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        scores.append(row)
        print(f"  constant step {step}: eval {row['eval_nll']:.4f}", flush=True)

    # the SECONDARY corrected score: the EMA iterate at step 180 (the
    # constant-phase EMA overlaid on the step-180 snapshot; buffers at their
    # step-180 values, parameters replaced by the EMA)
    if run["snapshots"] and run["ema"] is not None and \
            run["constant_steps"] in run["snapshots"]:
        model.load_state_dict(run["snapshots"][run["constant_steps"]])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(run["ema"][name].to(param.dtype))
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, eval_arm, args.eval_split)
        ema_row = dict(kind="ema_180", step=run["constant_steps"],
                       bytes=rows_by_step[run["constant_steps"]]["bytes"],
                       eval_nll=result["score"],
                       eval_windows=result["eval_windows"],
                       eval_targets=result["eval_targets"],
                       ema_decay=EMA_DECAY,
                       note="ranking-only: biased toward mid-run weights "
                            "(stated, not smoothed)")
        scores.append(ema_row)
        print(f"  EMA at step {run['constant_steps']}: "
              f"eval {ema_row['eval_nll']:.4f}", flush=True)

    # the PRIMARY corrected score: the post-cooldown final state
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, eval_arm, args.eval_split)
        final_row = dict(kind="post_cooldown", step=run["step"],
                         bytes=run["seen"], eval_nll=result["score"],
                         eval_windows=result["eval_windows"],
                         eval_targets=result["eval_targets"])
        scores.append(final_row)
        print(f"  post-cooldown final (step {run['step']}): "
              f"eval {final_row['eval_nll']:.4f}", flush=True)
    with (out / "scores.jsonl").open("w", buffering=1) as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    # ---- the mechanism-unchanged receipt at the post-cooldown final state
    diagnostics = None
    if run["status"] == "completed" and "final" in run["snapshots"]:
        if machine == "winner":
            from experiments.expressivity_tasks.x45_matrix_emender.\
                k10_probe_separated_keys import probe_diagnostics
            diagnostics = probe_diagnostics(model, stream,
                                            windows=args.probe_windows)
        else:
            from experiments.expressivity_tasks.x45_matrix_emender.\
                side_by_side import gdn2_gate_telemetry
            diagnostics = gdn2_gate_telemetry(model, stream,
                                              windows=args.probe_windows)

    constant_final = next((r["eval_nll"] for r in scores
                           if r["kind"] == "constant"
                           and r["step"] == run["constant_steps"]), None)
    post_cooldown = next((r["eval_nll"] for r in scores
                          if r["kind"] == "post_cooldown"), None)
    ema_final = next((r["eval_nll"] for r in scores if r["kind"] == "ema_180"),
                     None)
    banked_final = (OUR_BEST if machine == "winner" else GDN2_BEST)["final"]

    receipt = dict(
        probe="the protocol correction", label=args.label, machine=machine,
        recipe=arm["recipe"], recipe_note=arm["recipe_note"],
        conventions_note=arm["note"], no_prediction=NO_PREDICTION,
        corrected_scoring=CORRECTED_SCORING, status=run["status"],
        seed=args.seed,
        exposure_bytes=run["constant_steps"] and rows_by_step[
            run["constant_steps"]]["bytes"],
        exposure_steps=run["constant_steps"],
        total_bytes=run["seen"], total_steps=run["step"],
        cooldown_steps=COOLDOWN_STEPS,
        train_seconds_constant=round(run.get("constant_elapsed", 0.0), 1),
        train_seconds_total=round(run["elapsed"], 1),
        bytes_per_s=(rows_by_step[run["constant_steps"]]["bytes"]
                     / max(1e-9, run.get("constant_elapsed", 1.0))),
        optimizer_step_seconds=dict(
            mean_steady=(sum(steady) / len(steady)) if steady else None,
            note="steps 10+; the honest clock's own measure, per step"),
        parameters=parameters_total,
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=run["rows"][0]["loss"],
        step1_bit_identical_to_banked=bool(
            args.smoke or run["rows"][0]["loss"] == banked_step1),
        banked=dict(banked_final=banked_final,
                    rerun_minus_banked=(constant_final - banked_final
                                        if constant_final is not None
                                        else None),
                    note="the banked same-seed row's constant-LR final; the "
                         "re-run's step-180 score is receipted against it "
                         "(GPU nondeterminism may move later steps; step 1 "
                         "is asserted bit-identical)"),
        cooldown_steps_run=cooldown_steps,
        cooldown_note=("the smoke shakedown runs 3 cooldown steps; every real "
                       "arm runs the approved 24 (13.3% of 180)"),
        scores=scores, diagnostics=diagnostics,
        muon_update_rms=(run["muon_metrics"] or {}).get("muon_update_rms"),
        adamw_update_rms=(run["muon_metrics"] or {}).get("adamw_update_rms"),
        rms_matching_note="measured on THIS arm: the Muon group's update RMS "
                          "(the hybrid's own metric) vs the aux AdamW "
                          "group's update RMS (the metrics-only subclass) "
                          "-- the Moonlight receipt on our machines",
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        config=run_config, session=matched_arms.session_meta(card))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "status", "exposure_steps", "total_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll")}),
          flush=True)
    return receipt


# ------------------------------------------------------------------- assemble


def _final_of(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text())
    return result


def assemble(args) -> dict:
    """The corrected ladder: both machines x both recipes x all three
    scorings (constant-180 re-run, post-cooldown PRIMARY, EMA-180 secondary),
    the banked comparability reads, and the pre-stated no-prediction
    adjudication -- every margin read against the endpoint-variance and seed
    bands, reversals reported as plainly as confirmations."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    arms = {}
    for label in ARMS:
        path = RUNS / label
        if not (path / "result.json").exists():
            continue
        r = _final_of(path)
        arms[label] = dict(
            machine=r["machine"], recipe=r["recipe"], seed=r["seed"],
            status=r["status"],
            constant_lr=r["final_eval_nll_constant_lr"],
            post_cooldown=r["final_eval_nll_post_cooldown"],
            ema_180=r["ema_eval_nll"],
            cooldown_drop=r["cooldown_drop"],
            banked_final=r["banked"]["banked_final"],
            rerun_minus_banked=r["banked"]["rerun_minus_banked"],
            muon_update_rms=r["muon_update_rms"],
            adamw_update_rms=r["adamw_update_rms"],
            train_seconds_constant=r["train_seconds_constant"],
            train_seconds_total=r["train_seconds_total"])

    def margin(field, ours_label, gdn2_label):
        if ours_label not in arms or gdn2_label not in arms:
            return None
        return (arms[ours_label][field] - arms[gdn2_label][field]
                if arms[ours_label][field] is not None
                and arms[gdn2_label][field] is not None else None)

    def read_margin(m):
        if m is None:
            return None
        if m > ENDPOINT_VARIANCE_BAND:
            return dict(margin=m, verdict="GDN2 AHEAD, outside the "
                         f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance "
                         "band")
        if m < -ENDPOINT_VARIANCE_BAND:
            return dict(margin=m, verdict="OUR WINNER AHEAD, outside the "
                         f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance "
                         "band")
        return dict(margin=m, verdict="PARITY WITHIN THE "
                    f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance band "
                    "-- no leads/behind claim bankable at this margin")

    def read_recipe_delta(machine):
        speedrun = next((l for l, a in arms.items()
                         if a["machine"] == machine and a["recipe"] == "speedrun"),
                        None)
        moonlight = next((l for l, a in arms.items()
                          if a["machine"] == machine
                          and a["recipe"] == "moonlight"), None)
        if speedrun is None or moonlight is None:
            return None
        out = {}
        for field, name in (("post_cooldown", "post_cooldown"),
                            ("ema_180", "ema_180"),
                            ("constant_lr", "constant_lr")):
            s, m = arms[speedrun][field], arms[moonlight][field]
            out[name] = dict(
                speedrun=s, moonlight=m,
                speedrun_minus_moonlight=(s - m
                                          if s is not None and m is not None
                                          else None))
        out["note"] = ("speedrun minus moonlight, per scoring; positive = "
                       "the Moonlight recipe scores BETTER (lower NLL) on "
                       "this machine under that scoring")
        return out

    def banked_comparability(machine):
        row = next((a for a in arms.values() if a["machine"] == machine
                    and a["recipe"] == "speedrun"), None)
        if row is None:
            return None
        return dict(
            banked_final=row["banked_final"],
            rerun_constant_lr=row["constant_lr"],
            rerun_minus_banked=row["rerun_minus_banked"],
            note="the re-scored speedrun arm re-ran the banked best row's "
                 "own config/seed: the constant-LR step-180 score must land "
                 "on the banked final (within GPU-nondeterminism noise) for "
                 "the corrected scores to differ from the banked row ONLY "
                 "by the instrument")

    summary = dict(
        probe="the protocol correction assemble",
        no_prediction=NO_PREDICTION, corrected_scoring=CORRECTED_SCORING,
        horizon_transfer_rule=HORIZON_TRANSFER_RULE,
        banked=dict(our_best=OUR_BEST, gdn2_best=GDN2_BEST,
                    refs=BANKED_REFS, banked_margin=BANKED_MARGIN),
        bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                   winner_seed_delta=WINNER_SEED_DELTA),
        arms=arms,
        the_gap=dict(
            question="does the 0.0949 banked best-vs-best gap survive the "
                     "instrument fix?",
            banked_margin=BANKED_MARGIN,
            corrected_speedrun_best_vs_best=read_margin(
                margin("post_cooldown", "winner_speedrun_rescored",
                       "gdn2_speedrun_rescored")),
            moonlight_best_vs_best=read_margin(
                margin("post_cooldown", "winner_moonlight", "gdn2_moonlight")),
            ema_crosscheck=dict(
                speedrun=read_margin(margin("ema_180",
                                            "winner_speedrun_rescored",
                                            "gdn2_speedrun_rescored")),
                moonlight=read_margin(margin("ema_180", "winner_moonlight",
                                             "gdn2_moonlight")),
                note="the secondary instrument's ranking cross-check; its "
                     "absolute numbers are biased toward mid-run weights")),
        recipe_delta=dict(ours=read_recipe_delta("winner"),
                         gdn2=read_recipe_delta("gdn2")),
        banked_comparability=dict(ours=banked_comparability("winner"),
                                  gdn2=banked_comparability("gdn2")))
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(
        arms={k: dict(machine=v["machine"], recipe=v["recipe"],
                      constant=v["constant_lr"],
                      post_cooldown=v["post_cooldown"],
                      ema=v["ema_180"]) for k, v in arms.items()},
        the_gap={k: summary["the_gap"][k] for k in
                 ("banked_margin", "corrected_speedrun_best_vs_best",
                  "moonlight_best_vs_best")}), indent=1), flush=True)
    return summary


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--label", required=True, choices=sorted(ARMS))
    a.add_argument("--seed", type=int, default=None)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--output", required=True)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    a.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, 3 checkpoint "
                        "steps, 3 cooldown steps")
    s = sub.add_parser("assemble")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "protocol_correction_summary.json"))
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.seed is None:
            args.seed = ARMS[args.label]["seed"]
        if args.smoke:
            args.bytes = 300_000
            args.checkpoint_steps = [1, 2, 3]
            args.probe_windows = 2
        run_arm(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
