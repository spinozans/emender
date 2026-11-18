"""THE SYMMETRIC CONTROL -- GDN2's missing Muon arm, and the winner's second
seed, on the bake-off's own protocol and pinned Muon implementation.

THE CLAIM UNDER TEST (``RESULT_OPTIMIZER_BAKEOFF.md``): hybrid Muon at 3x LR
lands our winner at **1.514617647443499** vs GDN2's banked AdamW row
**1.5836738858904158** -- a 0.0691-nat lead at matched exposure.  But GDN2
never got a Muon arm: the banked comparison is our-best-optimizer vs
GDN2's-AdamW.  This control closes that hole with two arms (plus one
informative appendix arm if budget allows):

* ``gdn2_muon3x``            THE SYMMETRIC CONTROL: GDN2 trained by the bake-
                             off's own ``bakeoff_optimizers.HybridMuonX45``
                             (the prior line's pinned KellerJordan quintic
                             VERBATIM) at the SAME 3x-of-its-own-AdamW-LR
                             convention, on GDN2's own warmup schedule (both
                             hybrid groups scale equally -- the prior line's
                             convention, exactly as the bake-off's muon3x arm).
                             Routing OPTION A (approved by the supervisor):
                             Muon on EVERY 2D hidden-layer matrix of GDN2's
                             own inventory -- embed/head, the gdn2 core's
                             q/k/v/b/w projections, f_proj.0/1, g_proj.0/1,
                             o_proj, and the SwiGLU mlp w1/w2/w3 (149.68M of
                             149.86M params, 99.9%); AdamW on the true
                             non-matrix params -- the RMSNorms, o_norm,
                             A_log/dt_bias (the decay-gate parameters),
                             g_proj.1.bias, and the 3D depthwise q/k/v
                             short-conv kernels (173K params).  The maximal
                             Muon surface, the conservative direction for OUR
                             lead claim: a lead that survives GDN2's
                             maximal-Muon control is receipted against the
                             strongest possible control.
* ``muon3x`` at seed 3402    the single-seed caveat on the 1.5146 number: the
                             BANKED ``optimizer_bakeoff.py`` driver, imported
                             VERBATIM and run UNMODIFIED with ``--seed 3402``
                             (receipts under ``bakeoff_runs/muon3x_seed3402``;
                             this module only launches and assembles it).
* ``gdn2_muon3x_gates_adamw`` the informative appendix (approved as optional,
                             does NOT gate the verdict): the same arm with the
                             convention-mirror routing OPTION B -- GDN2's
                             gate-producing projections (b/w_proj,
                             f_proj.0/1, g_proj.0/1) stay on AdamW, mirroring
                             the bake-off's K10 decay-gate exclusion.

THE VERDICT RULE, pre-stated (the supervisor's approval): the lead is
RETRACTED if ``gdn2_muon3x`` lands BELOW the banked winner+muon3x final
(1.514617647443499) -- the honest result is then "Muon helps both; GDN2
remains ahead"; RECEIPTED otherwise.

PROVENANCE (the bake-off's own rules, restated): GDN2's model code is
UNTOUCHED -- ``matched_arms.build_gdn2`` (the registered lattice_gdn2 arm,
its searched nomination's optimizer knobs) is imported verbatim; the Muon is
``bakeoff_optimizers.HybridMuonX45`` imported verbatim (the same
``hybrid_muon.scaled_update`` the bake-off receipted); ``k10_*``/``k6_*``/
bakeoff/side-by-side sources, the PREREGISTRATIONs and ``lattice.py`` are NOT
edited.  New files only.  Every arm runs the EXACT probe protocol: the
20,000,000-target matched-exposure rule (the 20,022,866 landing), seed 3401
(the control arms; the seed axis is the second arm's own variable), the
registered flagship stream (identity asserted), the document-disjoint
calibration eval at 7 windows / 14,336 targets, the honest clock, the
protocol's own grad-clip (1.0, error_if_nonfinite), and the bake-off's own
checkpoint plan (steps 1/5/10/20/40/80/120/160/180, CPU state snapshots
inside the loop, every eval on restored snapshots after training -- the
training path unperturbed by construction).

Deliverables: per-arm receipts under ``muon_control_runs/<label>_seed3401/``
(and ``bakeoff_runs/muon3x_seed3402`` for the seed arm), the assemble summary
``receipts/muon_control_summary.json``, and ``RESULT_MUON_CONTROL.md``.

Usage (see ``run_muon_control_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.muon_control arm \\
        --label gdn2_muon3x --output muon_control_runs/gdn2_muon3x_seed3401
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

# The probe subprocess sets these BEFORE importing torch (the matched-arms
# discipline; configure_production() fail-closes on a missing allocator knob).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "muon_control_runs"
BAKEOFF_RUNS = ROOT / "bakeoff_runs"
SIDE_BY_SIDE_RUNS = ROOT / "side_by_side_runs"
VELOCITY_RUNS = ROOT / "velocity_runs"

SEED = 3401
SECOND_SEED = 3402
EXPOSURE_TARGETS = 20_000_000
# The bake-off's own checkpoint plan (the curve-overlay grid the banked
# bake-off curves are read at).
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
PROBE_WINDOWS = 4
GRAD_CLIP = 1.0                 # the probe protocol's own clip, every arm

# GDN2's banked AdamW row (side_by_side_runs/gdn2_armed_seed3401/start.json's
# optimizer block -- the searched nomination's knobs, asserted against
# matched_arms.build_gdn2 at launch so a nomination drift fails loudly).
GDN2_BANKED_ADAMW = dict(
    lr=5.420185524740253e-4, betas=(0.9, 0.9392004369596886),
    weight_decay=0.20059436743465126, warmup_steps=69, microbatch=8)
# The banked gdn2 probe's own first training row (the protocol-identical
# start receipt: the optimizer cannot touch the first forward, so the control
# arm's step-1 train loss must be bit-identical).
GDN2_BANKED_STEP1_LOSS = 5.948033220866492
GDN2_BANKED_FINAL = 1.5836738858904158
# The lead claim's number (the bake-off's muon3x arm, seed 3401).
WINNER_MUON3X_BANKED_FINAL = 1.514617647443499
MUON_LR_MULTIPLIER = 3         # the bake-off's own convention (3x = the prior
                               # receipt's neutral point; 10x collapses)

VERDICT_RULE = dict(
    rule="the lead is RETRACTED if gdn2+hybrid-muon (3x, option A) final "
         "eval NLL lands BELOW the banked winner+muon3x final; RECEIPTED "
         "otherwise (the pre-stated supervisor-approved rule)",
    retracted_if="gdn2_muon3x final_eval_nll < "
                 f"{WINNER_MUON3X_BANKED_FINAL!r}",
    winner_muon3x_banked_final=WINNER_MUON3X_BANKED_FINAL,
    gdn2_adamw_banked_final=GDN2_BANKED_FINAL,
    banked_margin=GDN2_BANKED_FINAL - WINNER_MUON3X_BANKED_FINAL,
)

# The GDN2 routing (OPTION A, approved): Muon owns every 2D hidden-layer
# matrix of GDN2's own inventory.  The patterns are source-named and the
# inventory is ENUMERATED -- any parameter that matches none of them fails
# loudly instead of silently routing.  (``model.`` prefix: matched_arms'
# Baseline wrapper owns the GDNModel as ``self.model``.)
_P = r"(?:model\.)?"
_GDN2_EMBED_HEAD = tuple(
    re.compile(_P + name + "$") for name in (r"embed\.weight", r"head\.weight"))
_GDN2_CORE = re.compile(
    _P + r"blocks\.\d+\.layer\.gdn2\.gdn2\.(q_proj|k_proj|v_proj|o_proj)\.weight$")
_GDN2_MLP = re.compile(_P + r"blocks\.\d+\.layer\.mlp\.w[123]\.weight$")
# GDN2's gate-PRODUCING projections (full 2D hidden-layer maps): the erase
# gate b_proj, the write gate w_proj, the decay pre-activation MLP f_proj.0/1
# and the output SiLU-gate MLP g_proj.0/1.  Option A routes them to Muon (the
# approved reading: the bake-off's K10 exclusion was about the decay-gate
# PARAMETER, a small non-matrix-class param, not gate-producing matrices);
# Option B (the informative appendix) keeps them on AdamW.
_GDN2_GATE_PROJ = re.compile(
    _P + r"blocks\.\d+\.layer\.gdn2\.gdn2\."
         r"(b_proj|w_proj|f_proj\.[01]|g_proj\.[01])\.weight$")
# GDN2's documented non-matrix inventory (Option A's AdamW side): the three
# RMSNorms + o_norm, the decay-gate parameters A_log/dt_bias, the g_proj.1
# bias, and the 3D depthwise short-conv kernels (not full matrices -- the
# Newton-Schulz convention requires 2D).
_GDN2_NONMATRIX = re.compile(
    _P + r"(norm\.weight|blocks\.\d+\.norm\.weight"
         r"|blocks\.\d+\.layer\.norm_2\.weight"
         r"|blocks\.\d+\.layer\.gdn2\.gdn2\.(o_norm\.weight|A_log|dt_bias"
         r"|g_proj\.1\.bias|[qkv]_conv1d\.weight))$")


def split_parameters_gdn2(model, gates_on_adamw: bool = False):
    """Route GDN2's parameters to (muon, adamw) by name, the bake-off's hybrid
    convention applied to GDN2's own param inventory.

    Option A (``gates_on_adamw=False``, the approved control): Muon on every
    2D hidden-layer matrix; AdamW on the true non-matrix params.  Option B
    (``gates_on_adamw=True``, the informative appendix): the gate-producing
    projections move to AdamW, mirroring the bake-off's K10 decay-gate
    exclusion.  Fails loudly on a Muon-owned name that is not a full 2D
    matrix, on distinct parameters sharing storage (the prior line's own
    guard), and on ANY parameter outside the enumerated inventory.
    """
    muon, adamw = [], []
    storage = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.device.type != "meta":
            key = (param.device, param.untyped_storage().data_ptr())
            if key in storage:
                raise ValueError(f"distinct parameters share storage: {name}")
            storage.add(key)
        is_gate_proj = _GDN2_GATE_PROJ.fullmatch(name) is not None
        is_muon = (any(r.fullmatch(name) for r in _GDN2_EMBED_HEAD)
                   or _GDN2_CORE.fullmatch(name) is not None
                   or _GDN2_MLP.fullmatch(name) is not None
                   or (is_gate_proj and not gates_on_adamw))
        if is_muon:
            if param.ndim != 2:
                raise ValueError(f"Muon-owned parameter is not a full 2D "
                                 f"matrix: {name}")
            muon.append(name)
        elif is_gate_proj or _GDN2_NONMATRIX.fullmatch(name) is not None:
            adamw.append(name)
        else:
            raise ValueError(
                f"unclassified GDN2 parameter: {name} (the inventory is "
                f"enumerated; a drift must fail loudly, not silently route)")
    if not muon or not adamw:
        raise ValueError("the hybrid requires both semantic groups non-empty")
    params = dict(model.named_parameters())
    return ([params[n] for n in muon], [params[n] for n in adamw],
            muon, adamw)


ARMS = {
    "gdn2_muon3x": dict(
        family="hybrid-muon",
        lr_multiplier=MUON_LR_MULTIPLIER, gates_on_adamw=False,
        routing="option A (approved): Muon on every 2D hidden-layer matrix "
                "(embed, head, gdn2 q/k/v/b/w projections, f_proj.0/1, "
                "g_proj.0/1, o_proj, mlp w1/w2/w3); AdamW on the RMSNorms, "
                "o_norm, A_log/dt_bias, g_proj.1.bias and the 3D depthwise "
                "q/k/v short-conv kernels",
        note="THE SYMMETRIC CONTROL: GDN2 + the bake-off's own hybrid Muon "
             "at 3x GDN2's own AdamW LR, on GDN2's own warmup schedule "
             "(both hybrid groups scale equally, the prior line's "
             "convention); the verdict arm"),
    "gdn2_muon3x_gates_adamw": dict(
        family="hybrid-muon",
        lr_multiplier=MUON_LR_MULTIPLIER, gates_on_adamw=True,
        routing="option B (informative convention-mirror): as option A but "
                "GDN2's gate-producing projections (b/w_proj, f_proj.0/1, "
                "g_proj.0/1) stay on AdamW, mirroring the bake-off's K10 "
                "decay-gate exclusion",
        note="the informative appendix arm: does NOT gate the verdict (a "
             "lead must survive option A; this arm only informs the "
             "convention discussion)"),
}


def _require_lease() -> str:
    # the bake-off's own lease guard, imported discipline (same check, same
    # error contract -- optimizer_bakeoff._require_lease).
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


# ------------------------------------------------------------ the training


def _train_with_snapshots_gdn2_hybrid(model, optimizer, set_lr, microbatch,
                                      stream, targets, label, log_path,
                                      checkpoint_steps):
    """The gdn2 armed loop (side_by_side's ``_train_with_snapshots_gdn2``
    shape: matched-exposure stopping, honest clock with time_ns rows, the
    protocol's own grad clip), with the hybrid optimizer the arm's own and
    the bake-off's two additions: the optimizer-step seconds (Muon's own
    cost, receipted separately) and the CPU state snapshot at each
    checkpoint step (after the row is logged; no eval, no telemetry, no RNG
    draw inside the loop)."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)

    batches = iter(stream.batches("train"))
    elapsed = 0.0
    opt_seconds = []
    step = seen = 0
    rows = []
    snapshots = {}
    last_muon_metrics = None
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
            set_lr(step)
            loss_sum = 0.0
            for x, y, valid in grouped(batch_rows, microbatch):
                loss = model.loss(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                           error_if_nonfinite=True)
            torch.cuda.synchronize()
            opt_start = time.monotonic()
            optimizer.step()
            torch.cuda.synchronize()
            opt_seconds.append(time.monotonic() - opt_start)
            last_muon_metrics = getattr(optimizer, "last_metrics", None)
            elapsed += time.monotonic() - start
            step += 1
            seen += n
            row = dict(step=step, active_seconds=elapsed, bytes=seen,
                       loss=loss_sum, time_ns=time.time_ns())
            rows.append(row)
            if log is not None:
                log.write(json.dumps(row) + "\n")
            if step in checkpoint_steps:
                snapshots[step] = {k: v.detach().cpu().clone()
                                   for k, v in model.state_dict().items()}
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
    from experiments.expressivity_tasks.x45_matrix_emender import (
        bakeoff_optimizers as BO, matched_arms)
    from experiments.expressivity_tasks.x45_matrix_emender.side_by_side import (
        gdn2_gate_telemetry)

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    if args.label not in ARMS:
        raise ValueError(f"unregistered arm label: {args.label}")
    arm = ARMS[args.label]

    model, optimizer_config, summary = matched_arms.build_gdn2(args.seed)
    # the banked row's own knobs, asserted (a nomination drift must fail
    # loudly, not silently change the control's baseline)
    for key, expected in GDN2_BANKED_ADAMW.items():
        actual = (tuple(optimizer_config[key]) if key == "betas"
                  else optimizer_config[key])
        if actual != expected:
            raise ValueError(f"the gdn2 nomination drifted on {key}: "
                             f"{actual!r} != the banked {expected!r}")

    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_gdn2(model, gates_on_adamw=arm["gates_on_adamw"])
    lr = arm["lr_multiplier"] * optimizer_config["lr"]
    optimizer = BO.HybridMuonX45(
        muon_params, adamw_params, lr=lr,
        betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"])
    warmup = optimizer_config["warmup_steps"]

    def set_lr(step):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1., (step + 1) / max(1, warmup))

    parameters_total = sum(p.numel() for p in model.parameters())
    run_config = dict(
        probe="the symmetric Muon control", arm="gdn2 (the banked row 5 model)",
        label=args.label, family=arm["family"], routing=arm["routing"],
        conventions_note=arm["note"],
        verdict_rule=VERDICT_RULE,
        seed=args.seed, exposure_targets=int(args.bytes),
        model_build="matched_arms.build_gdn2 verbatim (the registered "
                    "lattice_gdn2 arm, its searched nomination's knobs)",
        optimizer=dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            lr=lr, lr_multiplier=arm["lr_multiplier"],
            lr_convention="3x GDN2's own banked AdamW LR (the bake-off's "
                          "own 3x convention, applied optimizer-for-optimizer)",
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            warmup_steps=warmup,
            warmup_note="both hybrid groups scale equally (the prior line's "
                        "convention, the bake-off's muon3x arm's own)",
            microbatch=optimizer_config["microbatch"],
            grad_clip=GRAD_CLIP,
            muon_params=f"{len(muon_names)} tensors "
                        f"({sum(p.numel() for p in muon_params)} params)",
            adamw_params=f"{len(adamw_names)} tensors "
                         f"({sum(p.numel() for p in adamw_params)} params)"),
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        symmetry_rule="same stream, same data order, same matched-exposure "
                      "stopping rule, same honest clock, same grad clip, "
                      "same calibration eval as the banked gdn2 row; the "
                      "ONLY difference is the optimizer family (hybrid Muon "
                      "at 3x LR vs fused AdamW), the bake-off's own "
                      "symmetry rule applied to GDN2",
        optimizer_knobs_held_at_gdn2=dict(
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            warmup_steps=warmup, microbatch=optimizer_config["microbatch"]),
        checkpoint_steps=list(args.checkpoint_steps),
        checkpoint_note=("state snapshots taken inside the training loop at "
                         "each checkpoint step (the bake-off's own plan); "
                         "every eval runs on restored snapshots after "
                         "training, so the training path is unperturbed by "
                         "construction"),
        summary=summary, identity=identity, versions=_versions(),
        bindings=dict(matched_arms.registration_bindings(),
                      banked_rows=dict(
                          gdn2_adamw=GDN2_BANKED_FINAL,
                          winner_muon3x=WINNER_MUON3X_BANKED_FINAL),
                      banked_gdn2_probe_log=str(
                          VELOCITY_RUNS / "gdn2_probe_seed3401/training.jsonl"),
                      side_by_side=str(RECEIPTS / "side_by_side_summary.json"),
                      optimizer_bakeoff=str(
                          RECEIPTS / "optimizer_bakeoff_summary.json")),
        git=matched_arms.git_state(),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the probe exposure); the honest "
                       "elapsed wall is reported first-class"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=parameters_total,
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(arm=args.label, card=card, seed=args.seed,
                          parameters=parameters_total,
                          muon_params=len(muon_names),
                          adamw_params=len(adamw_names))), flush=True)

    run = _train_with_snapshots_gdn2_hybrid(
        model, optimizer, set_lr, optimizer_config["microbatch"], stream,
        int(args.bytes), args.label, out / "training.jsonl",
        tuple(args.checkpoint_steps))
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]

    # the protocol-identical start receipt: the optimizer cannot touch the
    # first forward, so step 1 must reproduce the banked gdn2 row bit-for-bit
    if not args.smoke:
        if run["rows"][0]["loss"] != GDN2_BANKED_STEP1_LOSS:
            raise ValueError(
                f"step-1 train loss {run['rows'][0]['loss']!r} != the banked "
                f"gdn2 row's {GDN2_BANKED_STEP1_LOSS!r} (the protocol start "
                f"is not identical)")

    # ---- the checkpoint pass (restored snapshots; training is over)
    rows_by_step = {r["step"]: r for r in run["rows"]}
    checkpoint_rows = []
    for step in args.checkpoint_steps:
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, "gdn2", args.eval_split)
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

    # ---- the mechanism-unchanged receipt at the final checkpoint: gdn2's
    # own gate telemetry, read non-invasively (its source untouched).
    gates = None
    if run["snapshots"]:
        model.load_state_dict(run["snapshots"][max(run["snapshots"])])
        gates = gdn2_gate_telemetry(model, stream, windows=args.probe_windows)

    receipt = dict(
        probe="the symmetric Muon control", label=args.label,
        family=arm["family"], routing=arm["routing"],
        conventions_note=arm["note"], verdict_rule=VERDICT_RULE,
        status=run["status"], seed=args.seed,
        exposure_bytes=run["seen"], exposure_steps=run["step"],
        train_seconds=round(run["elapsed"], 1),
        bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
        optimizer_step_seconds=dict(
            mean_steady=(sum(steady) / len(steady)) if steady else None,
            note="steps 10+; the honest clock's own measure, per step"),
        parameters=parameters_total,
        final_eval_nll=(checkpoint_rows[-1]["eval_nll"]
                        if checkpoint_rows else None),
        step1_train_loss=run["rows"][0]["loss"],
        step1_bit_identical_to_banked=bool(
            run["rows"][0]["loss"] == GDN2_BANKED_STEP1_LOSS),
        banked=dict(gdn2_adamw=GDN2_BANKED_FINAL,
                    winner_muon3x=WINNER_MUON3X_BANKED_FINAL),
        checkpoints=checkpoint_rows, gdn2_gates_final=gates,
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


def _load_checkpoints(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]


def assemble(args) -> dict:
    """Overlay the control arms on the banked curves (gdn2's and the
    incumbent's side-by-side curves, the bake-off's muon3x seed-3401 curve,
    the seed-3402 arm) and emit the control summary: the overlay table, the
    pre-stated verdict on the lead, and the second-seed variance read
    honestly against the 0.069 margin."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    gdn2_rows = _load_checkpoints(
        SIDE_BY_SIDE_RUNS / "gdn2_armed_seed3401/checkpoints.jsonl")
    gdn2_by_step = {r["step"]: r for r in gdn2_rows}
    incumbent_rows = _load_checkpoints(
        SIDE_BY_SIDE_RUNS / "k10_tuned_armed_seed3401/checkpoints.jsonl")
    incumbent_by_step = {r["step"]: r for r in incumbent_rows}

    def arm_curve(path):
        rows = _load_checkpoints(path)
        return {r["step"]: r for r in rows}

    curves = dict(
        muon3x_seed3401=arm_curve(BAKEOFF_RUNS / "muon3x_seed3401"
                                  "/checkpoints.jsonl"),
        muon3x_seed3402=arm_curve(BAKEOFF_RUNS / f"muon3x_seed{SECOND_SEED}"
                                  "/checkpoints.jsonl"),
        gdn2_muon3x=arm_curve(RUNS / "gdn2_muon3x_seed3401/checkpoints.jsonl"),
    )
    if (RUNS / "gdn2_muon3x_gates_adamw_seed3401/checkpoints.jsonl").exists():
        curves["gdn2_muon3x_gates_adamw"] = arm_curve(
            RUNS / "gdn2_muon3x_gates_adamw_seed3401/checkpoints.jsonl")

    overlay = []
    for step in CHECKPOINT_STEPS:
        entry = dict(step=step,
                     bytes=(curves["gdn2_muon3x"][step]["bytes"]
                            if step in curves["gdn2_muon3x"]
                            else gdn2_by_step[step]["bytes"]))
        g = gdn2_by_step.get(step)
        if g is not None:
            entry["gdn2_adamw_eval_nll"] = g["eval_nll"]
        inc = incumbent_by_step.get(step)
        if inc is not None:
            entry["incumbent_adamw_eval_nll"] = inc["eval_nll"]
        for label, curve in curves.items():
            if step in curve:
                entry[f"{label}_eval_nll"] = curve[step]["eval_nll"]
                if g is not None:
                    entry[f"{label}_deficit_vs_gdn2_adamw"] = \
                        curve[step]["eval_nll"] - g["eval_nll"]
        overlay.append(entry)

    def final_of(label):
        result_path = {"muon3x_seed3401": BAKEOFF_RUNS / "muon3x_seed3401",
                       "muon3x_seed3402": BAKEOFF_RUNS
                       / f"muon3x_seed{SECOND_SEED}",
                       "gdn2_muon3x": RUNS / "gdn2_muon3x_seed3401",
                       "gdn2_muon3x_gates_adamw":
                           RUNS / "gdn2_muon3x_gates_adamw_seed3401"}[label]
        result = json.loads((result_path / "result.json").read_text())
        return dict(final_eval_nll=result["final_eval_nll"],
                    status=result.get("status"),
                    train_seconds=result.get("train_seconds"),
                    exposure_bytes=result.get("exposure_bytes"),
                    exposure_steps=result.get("exposure_steps"))

    finals = {label: final_of(label) for label in curves}

    # ---- the pre-stated verdict
    control = finals.get("gdn2_muon3x")
    if control is None:
        verdict = "UNMEASURED (the gdn2_muon3x arm has not landed)"
        margin = None
    else:
        margin = control["final_eval_nll"] - WINNER_MUON3X_BANKED_FINAL
        if control["final_eval_nll"] < WINNER_MUON3X_BANKED_FINAL:
            verdict = ("RETRACTED: GDN2+Muon lands below the winner+Muon row "
                       "-- the honest result is 'Muon helps both; GDN2 "
                       "remains ahead'")
        else:
            verdict = ("RECEIPTED: GDN2+Muon does not beat the winner+Muon "
                       "row; the winner's Muon lead over GDN2's banked "
                       "AdamW row survives the symmetric control")

    # ---- the second-seed variance, stated against the banked margin
    seed_3402 = finals.get("muon3x_seed3402")
    if seed_3402 is None:
        seed_variance = None
    else:
        delta = seed_3402["final_eval_nll"] - WINNER_MUON3X_BANKED_FINAL
        seed_variance = dict(
            muon3x_seed3401_final=WINNER_MUON3X_BANKED_FINAL,
            muon3x_seed3402_final=seed_3402["final_eval_nll"],
            seed_delta=delta,
            seed3402_vs_gdn2_adamw=seed_3402["final_eval_nll"]
            - GDN2_BANKED_FINAL,
            banked_margin=VERDICT_RULE["banked_margin"],
            note="the seed axis varies the model-init draw only (the "
                 "stream/data order and the eval windows are fixed by the "
                 "registration); read honestly against the "
                 f"{VERDICT_RULE['banked_margin']:.4f} margin")

    summary = dict(probe="the symmetric Muon control assemble",
                   verdict_rule=VERDICT_RULE,
                   banked=dict(gdn2_adamw=GDN2_BANKED_FINAL,
                               winner_muon3x=WINNER_MUON3X_BANKED_FINAL),
                   finals=finals, overlay=overlay,
                   verdict=dict(verdict=verdict, control_margin_vs_winner_muon3x=margin),
                   seed_variance=seed_variance)
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(verdict=verdict,
                          finals={k: v["final_eval_nll"]
                                  for k, v in finals.items()}), indent=1),
          flush=True)
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
                   help="harness shakedown: tiny exposure, 3 checkpoint steps")
    s = sub.add_parser("assemble")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "muon_control_summary.json"))
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.smoke:
            args.bytes = 300_000
            args.checkpoint_steps = [1, 2, 3]
            args.probe_windows = 2
        run_arm(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
