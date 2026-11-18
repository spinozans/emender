"""GDN2'S LR BRACKET -- the last config-symmetry arm.

THE ASYMMETRY THIS LANE CLOSES (the symmetric claim rule's last hole): the
routing-symmetry lane gave OUR winner an LR bracket and found 4x beats its
inherited 3x point (1.4899/1.4907 vs 1.5146/1.5027, two seeds) -- but GDN2's
best arm (option B, gates-on-AdamW: **1.4150957890919276**) still sits at its
INHERITED 3x point, unbracketed.  The claim rule (no 'leads'/'behind' except
best-vs-best with the config search symmetric) therefore still has one open
asymmetry -- in GDN2's favor this time -- and this lane closes it: GDN2 gets
the same upward bracket ours got.

THE ARMS (the exact probe protocol verbatim, 20,022,866 targets, seed 3401
first, checkpoint curves at the banked steps 1/5/10/20/40/80/120/160/180):

* ``gdn2_muon4x_gates_adamw``  GDN2 + hybrid Muon at **4x** its own banked
                               AdamW LR (4 x 5.4201855e-4 = 2.1680742e-3),
                               at its best routing (option B, gates-on-AdamW
                               -- the control's own arm), everything else
                               the control's conventions verbatim.
* ``gdn2_muon5x_gates_adamw``  the same arm at **5x** (2.7100927e-3) --
                               brackets the far side, exactly the shape our
                               machine's bracket took above its inherited
                               point (our 4x won over 3x, so GDN2's optimum
                               may move up too; 5x brackets the far side).

THE PRE-STATED SECOND-SEED RULE: if either arm lands BELOW the banked
option-B final (1.4150957890919276) by MORE than the winner's own measured
seed band (0.0119) it is a CLEAR new best and owes a second seed (3402); if
it lands below by less than that band it is the new best-of-table but the
margin is parity-within-band and the second-seed call is the operator's, not
this lane's.  If neither arm beats 1.4151, no second seed is owed -- the
inherited 3x point stands as GDN2's best.

PROVENANCE (the prior lanes' own rules, restated): GDN2's model code is
UNTOUCHED -- ``matched_arms.build_gdn2`` and the control's own
``split_parameters_gdn2`` / ``_train_with_snapshots_gdn2_hybrid`` /
``bakeoff_optimizers.HybridMuonX45`` are imported VERBATIM; no
``k10_*``/``k6_*``/bakeoff/muon-control/routing-symmetry source, no
PREREGISTRATION and no ``lattice.py`` is edited.  New files only.  Every arm
runs the EXACT probe protocol: the 20,000,000-target matched-exposure rule
(the 20,022,866 landing, 180 steps, every arm identical), the registered
flagship stream (identity asserted), the document-disjoint calibration eval
at 7 windows / 14,336 targets (recorded-read path), the honest clock, the
protocol's own grad-clip (1.0, error_if_nonfinite), and the bake-off's own
checkpoint plan (CPU state snapshots inside the loop, every eval on restored
snapshots after training -- the training path unperturbed by construction).
The ONLY difference vs the banked option-B arm is the peak Muon LR (4x/5x of
GDN2's own AdamW LR instead of 3x); the warmup (69 steps, both hybrid groups
scaling equally), betas, weight decay and microbatch are the banked row's
own, asserted at launch.

Deliverables: per-arm receipts under ``gdn2_lr_bracket_runs/<label>_seed<seed>/``,
the assemble summary ``receipts/gdn2_lr_bracket_summary.json``, and
``RESULT_GDN2_LR_BRACKET.md``.

Usage (see ``run_gdn2_lr_bracket_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.gdn2_lr_bracket arm \\
        --label gdn2_muon4x_gates_adamw \\
        --output gdn2_lr_bracket_runs/gdn2_muon4x_gates_adamw_seed3401
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

from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    CHECKPOINT_STEPS, EXPOSURE_TARGETS, GDN2_BANKED_ADAMW,
    GDN2_BANKED_FINAL, GDN2_BANKED_STEP1_LOSS, GRAD_CLIP, PROBE_WINDOWS,
    SEED, SECOND_SEED, _require_lease, _train_with_snapshots_gdn2_hybrid,
    _versions, split_parameters_gdn2)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "gdn2_lr_bracket_runs"
MUON_CONTROL_RUNS = ROOT / "muon_control_runs"
BAKEOFF_RUNS = ROOT / "bakeoff_runs"
ROUTING_SYMMETRY_RUNS = ROOT / "routing_symmetry_runs"
SIDE_BY_SIDE_RUNS = ROOT / "side_by_side_runs"

# GDN2's banked option-B (gates-on-AdamW) row -- the bracket's reference and
# the best number any lane has measured on this box (receipted against
# receipts/muon_control_summary.json by the tests and the assemble).
GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL = 1.4150957890919276
GDN2_MUON3X_MAX_BANKED_FINAL = 1.4702048471995763       # option A, cited
# Our side's measured-best rows (the routing-symmetry lane's bracket, cited:
# the symmetric best-vs-best's our-side inputs, receipted against
# receipts/routing_symmetry_summary.json by the tests and the assemble).
OUR_BEST_LABEL = "muon4x_banked_seed3402"
OUR_BEST_FINAL = 1.4899048124040877
WINNER_MUON3X_BANKED_FINAL = 1.514617647443499          # our B-analog, cited
# The bands every margin is read against (the side-by-side's endpoint band;
# the winner's own measured two-seed band -- the seed evidence this box has).
ENDPOINT_VARIANCE_BAND = 0.075
WINNER_SEED_DELTA = 0.011948040553501738

# The pre-stated second-seed rule (receipted in every arm's result.json; the
# assemble applies it exactly as stated).
SECOND_SEED_RULE = dict(
    rule="if either bracket arm lands below the banked option-B final "
         "(1.4150957890919276) by MORE than the winner's measured seed band "
         "(0.0119) it is a CLEAR new best and owes a second seed (3402); "
         "below by less than the band: new best-of-table but "
         "parity-within-band, and the second-seed call is the operator's; "
         "neither arm beats 1.4151: no second seed owed -- the inherited 3x "
         "point stands as GDN2's best",
    reference=GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL,
    seed_band=WINNER_SEED_DELTA,
    clear_new_best_if="final < 1.4150957890919276 - 0.011948040553501738",
    second_seed=SECOND_SEED)

# The routing is the control's own option B, imported VERBATIM (its banked
# arm's routing string -- not restated here; a drift in the control's wording
# fails the tests, not silently passes).
from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    ARMS as _CONTROL_ARMS)
ROUTING_OPTION_B = _CONTROL_ARMS["gdn2_muon3x_gates_adamw"]["routing"]

ARMS = {
    "gdn2_muon4x_gates_adamw": dict(
        family="hybrid-muon",
        lr_multiplier=4, gates_on_adamw=True, routing=ROUTING_OPTION_B,
        note="THE UPWARD BRACKET POINT: GDN2 + the bake-off's own hybrid "
             "Muon at 4x GDN2's own banked AdamW LR (2.1680742e-3), at its "
             "best routing (option B, gates-on-AdamW -- the control's own "
             "arm); everything else the control's conventions verbatim. "
             "Our machine's bracket found 4x beats its inherited 3x at both "
             "seeds -- this arm tests whether GDN2's optimum moves up too"),
    "gdn2_muon5x_gates_adamw": dict(
        family="hybrid-muon",
        lr_multiplier=5, gates_on_adamw=True, routing=ROUTING_OPTION_B,
        note="THE FAR-SIDE BRACKET: the same arm at 5x (2.7100927e-3) -- "
             "brackets the far side of GDN2's LR axis exactly the shape our "
             "machine's bracket took above its inherited point"),
}


def _sibling_step1_loss(seed: int, own_label: str):
    """The first already-landed bracket arm's step-1 train loss at this seed
    (the cross-arm protocol-identical start receipt: the optimizer cannot
    touch the first forward, so every arm at the same seed must reproduce it
    bit-for-bit).  Seed 3401 additionally receipts against the banked gdn2
    probe's own first row (``GDN2_BANKED_STEP1_LOSS``); other seeds have no
    banked GDN2 row, so the first arm of the seed is the reference."""
    if seed == SEED:
        return GDN2_BANKED_STEP1_LOSS
    for run_dir in sorted(RUNS.glob(f"*_seed{seed}")):
        m = re.fullmatch(r"(?P<label>.+)_seed(?P<seed>\d+)", run_dir.name)
        if not m or m["label"].startswith("smoke_") \
                or m["label"] == own_label:
            continue
        first = json.loads(
            (run_dir / "training.jsonl").read_text().splitlines()[0])
        return first["loss"]
    return None


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
    # loudly, not silently change the bracket's baseline)
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
        probe="the GDN2 LR bracket", arm="gdn2 (the banked row 5 model)",
        label=args.label, family=arm["family"], routing=arm["routing"],
        conventions_note=arm["note"],
        second_seed_rule=SECOND_SEED_RULE,
        seed=args.seed, exposure_targets=int(args.bytes),
        model_build="matched_arms.build_gdn2 verbatim (the registered "
                    "lattice_gdn2 arm, its searched nomination's knobs)",
        optimizer=dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            lr=lr, lr_multiplier=arm["lr_multiplier"],
            lr_convention=f"{arm['lr_multiplier']}x GDN2's own banked AdamW "
                          "LR (the bake-off's own 3x convention, applied "
                          "optimizer-for-optimizer and bracketed upward the "
                          "way our machine's bracket went)",
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
                      "same calibration eval, same routing (option B) and "
                      "same warmup/betas/wd/microbatch as the banked "
                      "option-B arm; the ONLY difference is the peak Muon "
                      "LR (the bracket's own axis)",
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
                          gdn2_muon3x_gates_adamw=
                              GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL,
                          gdn2_muon3x_max=GDN2_MUON3X_MAX_BANKED_FINAL,
                          winner_muon3x=WINNER_MUON3X_BANKED_FINAL,
                          our_best=dict(label=OUR_BEST_LABEL,
                                        final=OUR_BEST_FINAL)),
                      muon_control_summary=str(
                          RECEIPTS / "muon_control_summary.json"),
                      routing_symmetry_summary=str(
                          RECEIPTS / "routing_symmetry_summary.json")),
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
    # first forward, so step 1 must reproduce the banked row (seed 3401) or
    # the first same-seed bracket arm bit-for-bit
    step1_reference = _sibling_step1_loss(args.seed, args.label)
    step1_bit_identical = None
    if not args.smoke and step1_reference is not None:
        if run["rows"][0]["loss"] != step1_reference:
            raise ValueError(
                f"step-1 train loss {run['rows'][0]['loss']!r} != the "
                f"same-seed reference {step1_reference!r} (the protocol "
                f"start is not identical)")
        step1_bit_identical = True

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
        probe="the GDN2 LR bracket", label=args.label,
        family=arm["family"], routing=arm["routing"],
        conventions_note=arm["note"], second_seed_rule=SECOND_SEED_RULE,
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
            args.seed == SEED
            and run["rows"][0]["loss"] == GDN2_BANKED_STEP1_LOSS),
        step1_reference=step1_reference,
        step1_bit_identical_to_same_seed_reference=step1_bit_identical,
        step1_receipt_note=("seed 3401: the banked gdn2 probe's own first "
                            "row (the optimizer cannot touch the first "
                            "forward); other seeds: the first same-seed "
                            "bracket arm's own first row (no banked GDN2 "
                            "row exists beyond seed 3401)"
                            if args.seed == SEED else
                            "no banked GDN2 row exists at this seed; the "
                            "reference is the first same-seed bracket "
                            "arm's own first row"),
        banked=dict(gdn2_adamw=GDN2_BANKED_FINAL,
                    gdn2_muon3x_gates_adamw=GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL,
                    gdn2_muon3x_max=GDN2_MUON3X_MAX_BANKED_FINAL,
                    winner_muon3x=WINNER_MUON3X_BANKED_FINAL,
                    our_best=dict(label=OUR_BEST_LABEL, final=OUR_BEST_FINAL)),
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


def _final_of(path: Path) -> dict:
    result = json.loads((path / "result.json").read_text())
    return dict(final_eval_nll=result["final_eval_nll"],
                status=result.get("status"),
                train_seconds=result.get("train_seconds"),
                exposure_bytes=result.get("exposure_bytes"),
                exposure_steps=result.get("exposure_steps"),
                routing=result.get("routing"))


def assemble(args) -> dict:
    """The GDN2 LR bracket table (4x/5x at option B vs the banked 3x
    option-B row, with our machine's bracket cited), the pre-stated
    second-seed rule applied exactly as stated, and the symmetric
    best-vs-best with every margin read against the endpoint-variance and
    seed bands."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    def arm_curve(path: Path):
        if not (path / "checkpoints.jsonl").exists():
            return {}
        return {r["step"]: r for r in _load_checkpoints(
            path / "checkpoints.jsonl")}

    curves = dict(
        gdn2_adamw=arm_curve(SIDE_BY_SIDE_RUNS / "gdn2_armed_seed3401"),
        gdn2_muon3x_gates_adamw=arm_curve(
            MUON_CONTROL_RUNS / "gdn2_muon3x_gates_adamw_seed3401"),
        winner_muon3x_seed3401=arm_curve(BAKEOFF_RUNS / "muon3x_seed3401"),
        winner_muon3x_seed3402=arm_curve(
            BAKEOFF_RUNS / f"muon3x_seed{SECOND_SEED}"),
        winner_muon4x_banked=arm_curve(
            ROUTING_SYMMETRY_RUNS / "muon4x_banked_seed3401"),
        winner_muon4x_banked_seed3402=arm_curve(
            ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{SECOND_SEED}"),
    )
    # this lane's arms: <label>_seed<seed> run dirs (smoke receipts are never
    # arms of the table)
    lane_dirs = {}
    for run_dir in sorted(list(RUNS.glob("*_seed3401"))
                          + list(RUNS.glob("*_seed3402"))):
        m = re.fullmatch(r"(?P<label>.+)_seed(?P<seed>\d+)", run_dir.name)
        if not m or m["label"].startswith("smoke_"):
            continue
        key = (m["label"] if m["seed"] == "3401"
               else f"{m['label']}_seed{m['seed']}")
        lane_dirs[key] = run_dir
        curves[key] = arm_curve(run_dir)

    overlay = []
    for step in CHECKPOINT_STEPS:
        entry = dict(step=step)
        for label, curve in curves.items():
            if step in curve:
                entry[f"{label}_eval_nll"] = curve[step]["eval_nll"]
                if step in curves["gdn2_adamw"]:
                    entry[f"{label}_deficit_vs_gdn2_adamw"] = \
                        curve[step]["eval_nll"] - \
                        curves["gdn2_adamw"][step]["eval_nll"]
        overlay.append(entry)

    def final_of(label):
        paths = {
            "gdn2_adamw": SIDE_BY_SIDE_RUNS / "gdn2_armed_seed3401",
            "gdn2_muon3x_gates_adamw":
                MUON_CONTROL_RUNS / "gdn2_muon3x_gates_adamw_seed3401",
            "gdn2_muon3x_max": MUON_CONTROL_RUNS / "gdn2_muon3x_seed3401",
            "winner_muon3x_seed3401": BAKEOFF_RUNS / "muon3x_seed3401",
            "winner_muon3x_seed3402": BAKEOFF_RUNS
            / f"muon3x_seed{SECOND_SEED}",
            "winner_muon4x_banked":
                ROUTING_SYMMETRY_RUNS / "muon4x_banked_seed3401",
            "winner_muon4x_banked_seed3402":
                ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{SECOND_SEED}",
            "winner_muon2x_banked":
                ROUTING_SYMMETRY_RUNS / "muon2x_banked_seed3401",
            "winner_muon3x_max":
                ROUTING_SYMMETRY_RUNS / "muon3x_max_seed3401",
            "winner_muon3x_gates_adamw":
                ROUTING_SYMMETRY_RUNS / "muon3x_gates_adamw_seed3401",
        }
        if label in paths:
            return _final_of(paths[label])
        if label in lane_dirs:
            return _final_of(lane_dirs[label])
        return None

    labels = set(curves) | {"gdn2_muon3x_max", "winner_muon2x_banked",
                            "winner_muon3x_max",
                            "winner_muon3x_gates_adamw"}
    finals = {label: final_of(label) for label in sorted(labels)}

    # ---- the bracket read: the new arms vs the banked 3x option-B row
    bracket = {}
    banked_3x = finals["gdn2_muon3x_gates_adamw"]["final_eval_nll"]
    for label in sorted(lane_dirs):
        arm_final = finals[label]["final_eval_nll"]
        delta = arm_final - banked_3x
        steady = [r for r in overlay if r["step"] in (80, 120, 160)
                  and f"{label}_deficit_vs_gdn2_adamw" in r]
        bracket[label] = dict(
            final_eval_nll=arm_final, delta_vs_banked_3x_option_b=delta,
            steady_deficit_vs_gdn2_adamw=(
                sum(r[f"{label}_deficit_vs_gdn2_adamw"] for r in steady)
                / len(steady)) if steady else None,
            second_seed_rule=SECOND_SEED_RULE,
            second_seed_owed=bool(arm_final < banked_3x - WINNER_SEED_DELTA),
            parity_within_band=bool(
                banked_3x - WINNER_SEED_DELTA
                <= arm_final < banked_3x + WINNER_SEED_DELTA),
            note=("CLEAR new best (below the banked 3x option-B row by more "
                  "than the 0.0119 seed band): owes the second seed"
                  if arm_final < banked_3x - WINNER_SEED_DELTA else
                  "below the banked 3x option-B row but inside the seed "
                  "band: new best-of-table, parity-within-band, the "
                  "second-seed call is the operator's"
                  if arm_final < banked_3x else
                  "does not beat the banked 3x option-B row: no second seed "
                  "owed, the inherited 3x point stands as GDN2's best"))

    # ---- the symmetric best-vs-best (the config search now symmetric on
    # BOTH axes: routing on both machines, LR bracket on both machines)
    our_candidates = {label: f["final_eval_nll"] for label, f in
                      finals.items() if f and f.get("final_eval_nll")
                      is not None and label.startswith("winner_")}
    gdn2_candidates = {label: f["final_eval_nll"] for label, f in
                       finals.items() if f and f.get("final_eval_nll")
                       is not None and label.startswith("gdn2")}
    our_best_label = min(our_candidates, key=our_candidates.get) \
        if our_candidates else None
    gdn2_best_label = min(gdn2_candidates, key=gdn2_candidates.get) \
        if gdn2_candidates else None
    symmetric = None
    if our_best_label and gdn2_best_label:
        our_best = our_candidates[our_best_label]
        gdn2_best = gdn2_candidates[gdn2_best_label]
        margin = our_best - gdn2_best
        if margin > ENDPOINT_VARIANCE_BAND:
            verdict = (f"GDN2 AHEAD by {margin:.4f} (outside the "
                       f"{ENDPOINT_VARIANCE_BAND} endpoint-variance band)")
        elif margin < -ENDPOINT_VARIANCE_BAND:
            verdict = (f"OUR WINNER AHEAD by {-margin:.4f} (outside the "
                       f"{ENDPOINT_VARIANCE_BAND} endpoint-variance band)")
        else:
            verdict = (f"PARITY WITHIN THE BAND (margin {margin:+.4f} vs the "
                       f"{ENDPOINT_VARIANCE_BAND} endpoint-variance band; "
                       f"the winner's own seed band is "
                       f"{WINNER_SEED_DELTA:.4f}) -- no leads/behind claim "
                       "is bankable at this margin")
        symmetric = dict(
            rule="no 'leads'/'behind' claims except best-vs-best with the "
                 "config search symmetric (routing AND LR bracket on both "
                 "machines): our best measured routing x best measured LR "
                 "point vs GDN2's best measured routing x best measured LR "
                 "point; margins inside the "
                 f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance band or "
                 f"the {WINNER_SEED_DELTA:.4f} seed band are stated as "
                 "parity-within-band",
            our_best=dict(label=our_best_label, final=our_best),
            gdn2_best=dict(label=gdn2_best_label, final=gdn2_best),
            margin=margin, verdict=verdict,
            endpoint_variance_band=ENDPOINT_VARIANCE_BAND,
            winner_seed_band=WINNER_SEED_DELTA)

    summary = dict(probe="the GDN2 LR bracket assemble",
                   banked=dict(
                       gdn2_adamw=GDN2_BANKED_FINAL,
                       gdn2_muon3x_gates_adamw=
                           GDN2_MUON3X_GATES_ADAMW_BANKED_FINAL,
                       gdn2_muon3x_max=GDN2_MUON3X_MAX_BANKED_FINAL,
                       winner_muon3x_seed3401=WINNER_MUON3X_BANKED_FINAL,
                       our_best=dict(label=OUR_BEST_LABEL,
                                     final=OUR_BEST_FINAL)),
                   second_seed_rule=SECOND_SEED_RULE,
                   finals=finals, overlay=overlay, bracket=bracket,
                   symmetric_best_vs_best=symmetric)
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(bracket={k: dict(final=v["final_eval_nll"],
                                           note=v["note"])
                                   for k, v in bracket.items()},
                          symmetric=(symmetric or {}).get("verdict"),
                          gdn2_best=(symmetric or {}).get("gdn2_best"),
                          our_best=(symmetric or {}).get("our_best")),
                     indent=1), flush=True)
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
                                        / "gdn2_lr_bracket_summary.json"))
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
