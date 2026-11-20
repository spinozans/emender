"""K16 -- THE CONFIG BRACKET ON THE COMPOSED MACHINE (the K15 adjudication's
mandatory follow-up).

THE ASYMMETRY THIS LANE CLOSES (the supervisor's ruling of 2026-11-24, the
named follow-up 1): the composed machine -- per-token + separated keys + the
K13 ADDITIVE READ + the K11B p3 per-channel gate stack, the program's best
absolute rows (post-cooldown **1.3811104467936925 / 1.383232252938407**,
+0.029/+0.031 remaining vs GDN2's **1.3521734476089478**) -- has NEVER been
tuned: it inherits the banked winner's hybrid Muon 4x point, which was tuned
for the SOFTMAX-read machine.  The read-form change moved the machine
substantially (the additive read trains ~10% faster and its surface may
differ), so part of the remaining ~0.03 may be config -- and the
symmetric-claim rule requires our best-vs-best (ours searched, GDN2 searched)
to be SEARCH-SYMMETRIC before any final gap is quoted.  This lane brackets the
composed machine the way both machines' brackets went (the
routing-symmetry/GDN2-LR-bracket conventions verbatim) and re-quotes the gap.

THE ARMS (the prior bracket conventions: ``RESULT_ROUTING_SYMMETRY.md`` /
``RESULT_GDN2_LR_BRACKET.md`` / the committed ``hybrid_muon.py``), every one
at the EXACT probe protocol (the 20,022,866-target matched-exposure landing,
180 constant-LR steps, the registered flagship stream, the document-disjoint
calibration eval, the honest clock, the protocol's own grad-clip) under the
CORRECTED scoring (24-step linear cooldown PRIMARY, EMA-0.99 secondary,
constant-180 retained for banked comparability), seed 3401 first:

* ``muon2x``  the composed machine + hybrid Muon at **2x** the tuned AdamW
              base (2 x 3.3427141575371284e-4 = 6.685428315074257e-4), the
              banked winner's own p3 routing (the K11B probe's own split,
              imported verbatim) -- the bracket's near side.
* ``muon3x``  the same at **3x** (1.0028142472611385e-3) -- the inherited
              3x point both machines' brackets started from.
* (4x)        the INHERITED point -- the K15 composition arm's own landed
              rows at BOTH seeds (1.3811/1.3832 post-cooldown), CITED, never
              re-run (a re-run would be identical by construction).
* ``muon6x``  the same at **6x** (2.0056284945222770e-3) -- the far side
              past the inherited 4x.
* ``adamw``   the composed machine under PLAIN AdamW at its own tuned
              point (1x = 3.3427141575371284e-4, fused, the bake-off's own
              adamw arm's convention) -- the optimizer-family check: the
              read-form change may have moved the optimizer ranking.

THE PRE-STATED RULES (receipted in every arm's result.json; the assemble
applies them exactly as stated):

* SECOND_SEED: the best cell of the bracket -- the lowest post-cooldown row
  at seed 3401 across muon2x / muon3x / muon4x (cited) / muon6x / adamw --
  is the composed machine's search-symmetric best cell and owes its second
  seed (3402), UNLESS it is the 4x cell (the K15 composition arm already
  landed at both seeds; the pair is complete and no new run is owed).
* THE BAND DISCIPLINE (the K15 ruling's own clause): any bracket cell whose
  post-cooldown row would put the composed machine AT OR PAST GDN2's
  1.3521734476089478 is reported with BOTH scorings (post-cooldown PRIMARY
  and the constant-180 retained row) and BOTH seeds before any claim; the
  ~0.075-nat endpoint-variance band and the winner's 0.0119 seed band are
  quoted with every margin.  NO directional prediction is pre-stated on the
  bracket's outcome: if the bracket moves the best row by >= 0.01 the
  remaining-gap quote is updated and the write-form attribution inherits a
  smaller residual; if the inherited 4x point was already optimal, that
  receipts too (the config-symmetry duty discharged either way).

PRE-STATE (no directional prediction): the bracket may move the composed
machine's best row either way; every conclusion below keys on the measured
rows against the two bands, and any result that inverts the standing
quality conclusions (the optimizer-family ranking, or a cell at/past GDN2's
row) is ESCALATED, not adjudicated here.

PROVENANCE (the lane's own rules): the K15 ENGINE IS COMMITTED AND READ-ONLY
-- ``k15_composition.py`` (the composed oracle, the one new forward kernel,
``convert_model_to_composition``, ``composed_parameter_delta``) is imported
VERBATIM, never modified; ``k15_probe_composition.py``'s own machinery
(``k15_probe_diagnostics`` -- the mechanism instrument -- the anchors, the
step-1 nearest-kin references) is imported, not restated.  The machine is the
K15 composition VERBATIM: ``velocity_ladder._build_k10`` (the tuned winner's
config) + ``convert_model_to_composition`` (the gates born at the registered
scalar equivalents; the birth IS the additive machine, receipted by the K15
lane at BOTH seeds -- 0 flips in 516,734,976 lanes on the probe's own first
train batch, the step-1 loss bit-identical to the K13 additive arm's own
row).  No ``k10_*``/``k11*``/``k12_*``/``k13_*``/``k15_*`` source, no
committed PREREGISTRATION, no ``lattice.py`` is edited.  The ONLY difference
vs the K15 composition arm is the OPTIMIZER CONFIG (the peak LR multiple,
or the family on the adamw arm); every other knob is the banked row's own
(betas (0.9, 0.9392), weight decay 0.17422, warmup 45 on both hybrid groups,
microbatch 8, grad-clip 1.0), asserted at launch.  The protocol-identical
start receipt: every arm's step-1 train loss must be BIT-IDENTICAL to the
K13 additive arm's own same-seed row (the composed machine at birth is the
additive machine; the optimizer cannot touch the first forward).

Deliverables: per-arm receipts under ``k16_runs/<label>_seed<seed>/``, the
assemble summary ``receipts/k16_config_bracket_summary.json``, and
``RESULT_CONFIG_BRACKET.md``.

Usage (see ``run_k16_config_bracket_leased.sh`` for the detached leased
driver, cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \\
        arm --arm muon2x --seed 3401 --output k16_runs/muon2x_seed3401
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

from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff import (
    BETA2, GRAD_CLIP, MICROBATCH, TUNED_WINNER)
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction \
    import (COOLDOWN_STEPS, CORRECTED_SCORING, EMA_DECAY,
            ENDPOINT_VARIANCE_BAND, WINNER_SEED_DELTA, _train_corrected)
from experiments.expressivity_tasks.x45_matrix_emender.\
    k15_probe_composition import (BANKED, CHECKPOINT_STEPS, EXPOSURE_TARGETS,
                                  K13_ADDITIVE, PRIMARY_SEED, PROBE_WINDOWS,
                                  SECOND_SEED, _banked_mechanism_anchors,
                                  _banked_step1_loss, _k13_additive_step1,
                                  k15_probe_diagnostics)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k16_runs"
K15_RUNS = ROOT / "k15_runs"

# The tuned AdamW base the bracket's multiples are taken of (the banked
# winner's own tuned LR -- the ONLY LR the composed machine's AdamW side has
# ever been tuned at).
TUNED_ADAMW_LR = TUNED_WINNER["lr"]
INHERITED_LR_MULTIPLIER = 4      # the banked winner lineage (the K15 arm's own)

# The composed machine's banked 4x rows -- the K15 composition arm's OWN
# landed rows at BOTH seeds (cited, never re-run).  The loader reads the
# committed assemble receipt; the constants are the fallback the tests pin.
COMPOSED_4X_FALLBACK = dict(
    post_cooldown={3401: 1.3811104467936925, 3402: 1.383232252938407},
    constant_lr={3401: 1.433674931526184, 3402: 1.4341776371002197},
    ema_180={3401: 1.424389226096017, 3402: 1.4385861498968942},
    step1_train_loss={3401: 5.782775391046648, 3402: 5.720172563202926},
    note="the K15 composition arm's own landed rows (the inherited 4x point, "
         "tuned for the SOFTMAX-read machine): receipts/"
         "k15_composition_summary.json")


def composed_4x_banked() -> dict:
    """The composed machine's banked 4x rows, loaded from the K15 assemble
    receipt (the constants above are the pinned fallback)."""
    out = dict(COMPOSED_4X_FALLBACK)
    path = RECEIPTS / "k15_composition_summary.json"
    if path.exists():
        summary = json.loads(path.read_text())
        for seed in (PRIMARY_SEED, SECOND_SEED):
            row = summary["arms"][f"composition_seed{seed}"]
            for key in ("post_cooldown", "constant_lr", "ema_180",
                        "step1_train_loss"):
                out[key][seed] = row[key]
    return out


# The arms (the bracket around the inherited 4x point + the family check).
# 4x itself is NOT an arm: its rows exist at both seeds (cited above).
ARMS = {
    "muon2x": dict(
        family="hybrid-muon", lr_multiplier=2,
        note="the composed machine + hybrid Muon at 2x the tuned AdamW base "
             "-- the bracket's near side"),
    "muon3x": dict(
        family="hybrid-muon", lr_multiplier=3,
        note="the composed machine + hybrid Muon at 3x -- the inherited "
             "bracket point both machines' brackets started from"),
    "muon6x": dict(
        family="hybrid-muon", lr_multiplier=6,
        note="the composed machine + hybrid Muon at 6x -- the far side past "
             "the inherited 4x point"),
    "adamw": dict(
        family="adamw", lr_multiplier=1,
        note="the composed machine under PLAIN AdamW at its own tuned point "
             "(1x the tuned base, fused, the bake-off's own adamw arm's "
             "convention) -- the optimizer-family check: the read-form "
             "change may have moved the optimizer ranking"),
}

# The pre-stated second-seed rule (receipted in every arm's result.json; the
# assemble applies it exactly as stated).
SECOND_SEED_RULE = dict(
    rule="the best cell of the bracket -- the lowest post-cooldown row at "
         "seed 3401 across muon2x / muon3x / muon4x (cited) / muon6x / adamw "
         "-- is the composed machine's search-symmetric best cell and owes "
         "its second seed (3402), UNLESS it is the 4x cell (the K15 "
         "composition arm already landed at both seeds; the pair is "
         "complete and no new run is owed)",
    bracket_seed=PRIMARY_SEED, second_seed=SECOND_SEED,
    cited_cell="muon4x (the K15 composition arm's own rows at both seeds)",
    escalation="any bracket cell whose post-cooldown row lands at or below "
               "GDN2's 1.3521734476089478 is reported with BOTH scorings "
               "and BOTH seeds before any claim -- and escalated, not "
               "adjudicated here")

# The bands every margin is read against (the standing amendments' own).
BANDS = dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
             winner_seed_delta=WINNER_SEED_DELTA)

GDN2_POST_COOLDOWN = BANKED["gdn2"]["post_cooldown"]
BANKED_CORRECTED_ANCHOR = BANKED["post_cooldown"]["seed3402"]


# --------------------------------------------------------------- the receipts


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def headline_rows(receipt: dict) -> dict:
    """The arm's own headline rows: vs the inherited 4x point's SAME-SEED
    banked row (the config yield), and vs GDN2 under the corrected scoring
    with the band caveats attached (the contact rule's own flag)."""
    seed = receipt["seed"]
    post = receipt.get("final_eval_nll_post_cooldown")
    constant = receipt.get("final_eval_nll_constant_lr")
    four_x = composed_4x_banked()
    out = {}
    if post is not None:
        out["config_yield_vs_inherited_4x_same_seed"] = dict(
            measured_post_cooldown=post,
            inherited_4x_post_cooldown=four_x["post_cooldown"][seed],
            improvement=four_x["post_cooldown"][seed] - post,
            note="the bracket cell's own row vs the inherited point's "
                 "same-seed row -- the config-symmetry yield, read against "
                 "the endpoint-variance band "
                 f"({ENDPOINT_VARIANCE_BAND}) and the winner's seed band "
                 f"({WINNER_SEED_DELTA:.4f})")
        out["vs_gdn2_post_cooldown"] = post - GDN2_POST_COOLDOWN
        out["fraction_of_corrected_gap_closed"] = max(
            0.0, BANKED_CORRECTED_ANCHOR - post) / (
            BANKED_CORRECTED_ANCHOR - GDN2_POST_COOLDOWN)
        out["gdn2_contact_flag"] = bool(post <= GDN2_POST_COOLDOWN)
        out["gdn2_contact_rule"] = (
            "at-or-past GDN2's row: BOTH scorings (post-cooldown PRIMARY, "
            "constant-180 retained) and BOTH seeds are reported before any "
            "claim; the endpoint-variance band caveat stands on every gap "
            "quote")
        if constant is not None:
            out["both_scorings"] = dict(
                post_cooldown_primary=post, constant_lr_retained=constant,
                ema_180_ranking_only=receipt.get("ema_eval_nll"))
    out["pre_state"] = ("NO directional prediction was pre-stated on the "
                        "bracket's outcome (the K15 ruling's own pre-state); "
                        "every reading keys on the measured rows against "
                        "the two bands, and any quality-conclusion inversion "
                        "is escalated")
    return out


# ------------------------------------------------------------------- the arm


def run_arm(args) -> dict:
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import HybridMuonX45
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import (_routing_cross_check,
                                         gate_tensor_names,
                                         split_parameters_gated)
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import (composed_parameter_delta, convert_model_to_composition)

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r} (expected one of "
                         f"{sorted(ARMS)})")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed} (the bracket runs "
                         f"seed {PRIMARY_SEED} first; the best cell owes "
                         f"{SECOND_SEED} per the pre-stated rule)")
    arm = ARMS[args.arm]
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    seed = args.seed

    # ---- the machine: the K15 composition VERBATIM (the banked winner
    # build + the K13 additive read + the K11B p3 gate stack at the
    # mimetic-neutral birth) -- the machine is NOT this lane's variable
    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_composition(model)
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = composed_parameter_delta(
        depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta:
        raise ValueError(f"the parameter accounting must hold: "
                         f"{parameters_before} + {delta} != {parameters_after}")
    if delta != 8_657_040 or parameters_after != 158_638_848:
        raise ValueError(f"the composed machine's parameters must be the "
                         f"K15 receipt's own (149,981,808 + 8,657,040); got "
                         f"delta {delta}, total {parameters_after}")
    # the banked knobs, asserted (a config drift must fail loudly, not
    # silently change the arm's baseline)
    expected = dict(betas=(0.9, BETA2),
                    weight_decay=TUNED_WINNER["weight_decay"],
                    warmup_steps=TUNED_WINNER["warmup_steps"],
                    microbatch=MICROBATCH)
    for key, want in expected.items():
        actual = (tuple(optimizer_config[key]) if key == "betas"
                  else optimizer_config[key])
        if actual != want:
            raise ValueError(f"the winner's config drifted on {key}: "
                             f"{actual!r} != the banked {want!r}")

    # ---- the optimizer: the bracket's own axis (the ONLY difference vs
    # the K15 composition arm), the banked routing on the hybrid arms
    peak_lr = arm["lr_multiplier"] * TUNED_ADAMW_LR
    routing_receipt = None
    if arm["family"] == "hybrid-muon":
        muon_params, adamw_params, muon_names, adamw_names = \
            split_parameters_gated(model, "p3")
        routing_receipt = _routing_cross_check(model, "p3", seed, False)
        optimizer = HybridMuonX45(
            muon_params, adamw_params, lr=peak_lr, betas=(0.9, BETA2),
            weight_decay=TUNED_WINNER["weight_decay"])
        routing_inventory = dict(muon=muon_names, adamw=adamw_names)
        routing_note = ("banked_gates_adamw + the p3 gate tensors on AdamW "
                        "(the k11b probe's own generic split, imported "
                        "verbatim -- the K15 arm's own routing, unchanged)")
        optimizer_block = dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            peak_lr=peak_lr, lr_multiplier=arm["lr_multiplier"],
            lr_convention=(f"{arm['lr_multiplier']}x the tuned AdamW base "
                           f"({TUNED_ADAMW_LR!r}) -- the prior bracket "
                           "lanes' own axis (multiples of the machine's own "
                           "tuned AdamW LR)"),
            betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
            warmup_steps=TUNED_WINNER["warmup_steps"],
            warmup_note="both hybrid groups scale equally (the prior line's "
                        "convention)",
            microbatch=MICROBATCH, grad_clip=GRAD_CLIP)
    elif arm["family"] == "adamw":
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=peak_lr, betas=(0.9, BETA2),
            weight_decay=TUNED_WINNER["weight_decay"], fused=True)
        # the p3 gate tensors must all be present and trainable (the
        # single-group family's own inventory receipt)
        names = {n for n, _ in model.named_parameters()}
        gate_names = sorted(
            f"blocks.{i}.mixer.{g}"
            for i in range(depth) for g in gate_tensor_names("p3"))
        missing = [n for n in gate_names if n not in names]
        if missing:
            raise ValueError(f"the composed machine's gate tensors are "
                             f"missing from the inventory: {missing[:6]}...")
        routing_inventory = dict(
            single_group=f"{len(trainable)} trainable tensors "
                         f"({sum(p.numel() for p in trainable)} params)")
        routing_note = ("plain AdamW over every trainable tensor (the "
                        "bake-off's own adamw arm's convention) -- NO "
                        "routing; the p3 gate tensors ride the same group")
        optimizer_block = dict(
            optimizer="torch.optim.AdamW(fused=True)",
            peak_lr=peak_lr, lr_multiplier=1,
            lr_convention=("1x the tuned AdamW base (the machine's own "
                           "tuned point -- the ONLY LR the composed "
                           "machine's AdamW side has ever been tuned at)"),
            betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
            warmup_steps=TUNED_WINNER["warmup_steps"],
            microbatch=MICROBATCH, grad_clip=GRAD_CLIP,
            gate_tensors_present=len(gate_names))
    else:
        raise ValueError(f"unregistered family: {arm['family']!r}")

    four_x = composed_4x_banked()
    anchors = dict(
        banked=BANKED, k13_additive=K13_ADDITIVE, bands=BANDS,
        composed_4x_banked=four_x,
        banked_mechanism=_banked_mechanism_anchors())
    run_config = dict(
        probe="k16_config_bracket", arm=args.arm, seed=seed,
        family=arm["family"], arm_note=arm["note"],
        corrected_scoring=CORRECTED_SCORING,
        second_seed_rule=SECOND_SEED_RULE,
        exposure_targets=int(args.bytes),
        model_build="the K15 composition VERBATIM: velocity_ladder._build_k10 "
                    "(the tuned winner's config, the banked bake-off build) "
                    "+ convert_model_to_composition (the K15 conversion; the "
                    "gates born at the registered scalar equivalents) -- the "
                    "K15 lane's own machine, byte for byte; the birth "
                    "receipts (BORN-AS-THE-ADDITIVE-MACHINE, 0 flips in "
                    "516,734,976 lanes; the gates born at the registered "
                    "values) are the K15 lane's own, receipted at BOTH seeds",
        optimizer=optimizer_block,
        routing=routing_note, routing_receipt=routing_receipt,
        routing_inventory=routing_inventory,
        parameters=dict(before=parameters_before, added=delta,
                        total=parameters_after,
                        note="p3's own gate stack (the read channel is K13's "
                             "own committed construction, instantiated once)"),
        checkpoint_steps=list(args.checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear cooldown "
                       "(the corrected-scoring extension, same stream); the "
                       "step-180 constant-LR eval is retained for banked "
                       "comparability"),
        protocol_identical_start="the step-1 train loss is asserted "
                                  "BIT-IDENTICAL to the K13 additive arm's "
                                  "own same-seed row (the composed machine at "
                                  "birth IS the additive machine; the "
                                  "optimizer cannot touch the first forward)",
        both_bands=BANDS,
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                      k15_summary=str(RECEIPTS / "k15_composition_summary.json"),
                      k15_gates_receipt=str(RECEIPTS / "k15_gates.json"),
                      k15_result_document=str(ROOT / "RESULT_COMPOSITION.md")),
        summary=summary, identity=identity, versions=_versions(),
        git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns()))
    print(json.dumps(dict(arm=args.arm, card=card, seed=seed,
                          family=arm["family"], peak_lr=peak_lr,
                          parameters=parameters_after)), flush=True)

    banked_step1 = _banked_step1_loss(seed)
    k13_step1 = _k13_additive_step1(seed)
    run_config["banked_step1_loss_reference"] = banked_step1
    run_config["k13_additive_step1_loss_reference"] = k13_step1

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, peak_lr, optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"{args.arm}_seed{seed}", out / "training.jsonl",
        tuple(args.checkpoint_steps), cooldown_steps, EMA_DECAY)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # ---- the nearest-kin step-1 parity receipt (the K15 lane's own): the
    # arm's step-1 train loss must be BIT-IDENTICAL to the K13 additive
    # arm's own same-seed row (and hence to the K15 composition arm's own
    # same-seed row -- the same function at birth, optimizer-invariant).
    step1 = run["rows"][0]["loss"]
    print(f"  step-1 train loss {step1!r} (the K13 additive arm's own row's "
          f"is {k13_step1!r} -- asserted bit-identical; the inherited 4x "
          f"composition row's is {four_x['step1_train_loss'][seed]!r})",
          flush=True)
    if not args.smoke and step1 != k13_step1:
        raise ValueError(
            f"step-1 train loss {step1!r} != the K13 additive arm's own "
            f"same-seed row's {k13_step1!r} -- the arm's machine is not the "
            f"K15 composition byte for byte (escalate, not excuse)")
    if not args.smoke and step1 != four_x["step1_train_loss"][seed]:
        raise ValueError(
            f"step-1 train loss {step1!r} != the K15 composition arm's own "
            f"same-seed row's {four_x['step1_train_loss'][seed]!r} -- the "
            f"build order drifted (escalate, not excuse)")

    # ---- the scoring pass (restored snapshots; training is over)
    scores = []
    for step in args.checkpoint_steps:            # the constant-LR curve
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        row = dict(kind="constant", step=step,
                   bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"],
                   eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        scores.append(row)
        print(f"  constant step {step}: eval {row['eval_nll']:.4f}", flush=True)
    if run["snapshots"] and run["ema"] is not None and \
            run["constant_steps"] in run["snapshots"]:
        model.load_state_dict(run["snapshots"][run["constant_steps"]])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(run["ema"][name].to(param.dtype))
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        scores.append(dict(kind="ema_180", step=run["constant_steps"],
                           bytes=rows_by_step[run["constant_steps"]]["bytes"],
                           eval_nll=result["score"],
                           eval_windows=result["eval_windows"],
                           eval_targets=result["eval_targets"],
                           ema_decay=EMA_DECAY,
                           note="ranking-only: biased toward mid-run weights"))
        print(f"  EMA at step {run['constant_steps']}: "
              f"eval {result['score']:.4f}", flush=True)
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        scores.append(dict(kind="post_cooldown", step=run["step"],
                           bytes=run["seen"], eval_nll=result["score"],
                           eval_windows=result["eval_windows"],
                           eval_targets=result["eval_targets"]))
        print(f"  post-cooldown final (step {run['step']}): "
              f"eval {result['score']:.4f}", flush=True)
    with (out / "scores.jsonl").open("w", buffering=1) as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    # ---- the mechanism receipts at BOTH states + the read-mask sweep
    # (the K15 lane's own instrument, imported verbatim)
    from experiments.expressivity_tasks.x45_matrix_emender.side_by_side \
        import k10_read_mask_sweep
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
    if run["status"] == "completed":
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            diagnostics_180 = k15_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_180 = k10_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = k15_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_final = k10_read_mask_sweep(model, stream)

    constant_final = next((r["eval_nll"] for r in scores
                           if r["kind"] == "constant"
                           and r["step"] == run["constant_steps"]), None)
    post_cooldown = next((r["eval_nll"] for r in scores
                          if r["kind"] == "post_cooldown"), None)
    ema_final = next((r["eval_nll"] for r in scores if r["kind"] == "ema_180"),
                     None)
    receipt = dict(
        probe="k16_config_bracket", arm=args.arm, seed=seed,
        family=arm["family"], arm_note=arm["note"],
        lr_multiplier=arm["lr_multiplier"], peak_lr=peak_lr,
        corrected_scoring=CORRECTED_SCORING, status=run["status"],
        exposure_bytes=rows_by_step[run["constant_steps"]]["bytes"],
        exposure_steps=run["constant_steps"],
        total_bytes=run["seen"], total_steps=run["step"],
        cooldown_steps=cooldown_steps,
        train_seconds_constant=round(run.get("constant_elapsed", 0.0), 1),
        train_seconds_total=round(run["elapsed"], 1),
        bytes_per_s=(rows_by_step[run["constant_steps"]]["bytes"]
                     / max(1e-9, run.get("constant_elapsed", 1.0))),
        optimizer_step_seconds=dict(
            mean_steady=(sum(steady) / len(steady)) if steady else None,
            note="steps 10+; the honest clock's own measure, per step"),
        muon_update_rms=run.get("muon_metrics"),
        parameters=parameters_after,
        parameters_added=delta,
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=step1,
        step1_receipt=dict(
            arm_step1=step1, k13_additive_same_seed_step1=k13_step1,
            k13_parity=bool(args.smoke or step1 == k13_step1),
            composed_4x_same_seed_step1=four_x["step1_train_loss"][seed],
            composed_4x_parity=bool(
                args.smoke or step1 == four_x["step1_train_loss"][seed]),
            banked_same_seed_step1=banked_step1,
            note="the nearest-kin parity vs the K13 additive arm's own row "
                 "(== the K15 composition arm's own same-seed row) is "
                 "ASSERTED; the banked row's step-1 is a different "
                 "function's value and none is asserted"),
        scores=scores,
        anchors=anchors,
        inherited_4x_same_seed=dict(
            post_cooldown=four_x["post_cooldown"][seed],
            constant_lr=four_x["constant_lr"][seed],
            ema_180=four_x["ema_180"][seed],
            note="the K15 composition arm's own same-seed row at the "
                 "inherited 4x point (cited, never re-run)"),
        diagnostics_step180=diagnostics_180,
        diagnostics_post_cooldown=diagnostics_final,
        read_mask_sweep_step180=sweep_180,
        read_mask_sweep_post_cooldown=sweep_final,
        routing_receipt=routing_receipt,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        config=run_config, session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            run["rows"], keep_first=min(5, max(1, len(run["rows"])))))
    receipt["headline_rows"] = headline_rows(receipt)
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("arm", "seed", "status", "exposure_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll")}),
          flush=True)
    return receipt


# ------------------------------------------------------------------- assemble


def _arm_row(path: Path) -> dict | None:
    if not (path / "result.json").exists():
        return None
    r = json.loads((path / "result.json").read_text())
    diag = r["diagnostics_post_cooldown"] or {}
    return dict(
        arm=r["arm"], seed=r["seed"], family=r["family"],
        lr_multiplier=r["lr_multiplier"], peak_lr=r["peak_lr"],
        status=r["status"], post_cooldown=r["final_eval_nll_post_cooldown"],
        constant_lr=r["final_eval_nll_constant_lr"],
        ema_180=r["ema_eval_nll"], step1_train_loss=r["step1_train_loss"],
        step1_parity=r["step1_receipt"]["k13_parity"],
        train_seconds_constant=r["train_seconds_constant"],
        parameters=r["parameters"],
        effective_rows=diag.get("effective_read_rows"),
        top_row_share=(diag.get("top_row_w_mass_share") or {}).get("mean"),
        occupancy=(diag.get("occupancy") or {}).get("mean_live_slots"),
        identity_share=diag.get("identity_rewrite_share"),
        gate_drifts={k: (diag.get("gate_telemetry") or {}).get(k) for k in (
            "decay_gate_mean_distance_from_registered_scalar",
            "write_gate_mean_distance_from_1",
            "read_gate_mean_distance_from_1")},
        muon_update_rms=(r.get("muon_update_rms") or {}).get("muon_update_rms")
        if isinstance(r.get("muon_update_rms"), dict) else None)


def assemble(args) -> dict:
    """The bracket table (the three run cells + the cited 4x rows + the
    adamw family check), the pre-stated second-seed rule applied exactly as
    stated, the search-symmetric ours-best, and the re-quoted gap vs GDN2
    under the corrected scoring with the honest band caveats.  The final
    gap-quote phrasing is the supervisor's to adjudicate (the K13
    clause-conflict precedent): this assemble states the measured rows and
    the flags, not the final wording."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    four_x = composed_4x_banked()
    cells: dict[str, dict] = {
        "muon4x": dict(
            family="hybrid-muon", lr_multiplier=INHERITED_LR_MULTIPLIER,
            cited=True,
            note="the K15 composition arm's own landed rows (the inherited "
                 "point, both seeds banked) -- cited, never re-run",
            seeds={PRIMARY_SEED: dict(
                post_cooldown=four_x["post_cooldown"][PRIMARY_SEED],
                constant_lr=four_x["constant_lr"][PRIMARY_SEED],
                ema_180=four_x["ema_180"][PRIMARY_SEED],
                step1_train_loss=four_x["step1_train_loss"][PRIMARY_SEED]),
                SECOND_SEED: dict(
                post_cooldown=four_x["post_cooldown"][SECOND_SEED],
                constant_lr=four_x["constant_lr"][SECOND_SEED],
                ema_180=four_x["ema_180"][SECOND_SEED],
                step1_train_loss=four_x["step1_train_loss"][SECOND_SEED])})}
    escalation_flags = []
    for label in sorted(ARMS):
        cell = dict(family=ARMS[label]["family"],
                    lr_multiplier=ARMS[label]["lr_multiplier"],
                    cited=False, note=ARMS[label]["note"], seeds={})
        for seed in (PRIMARY_SEED, SECOND_SEED):
            row = _arm_row(RUNS / f"{label}_seed{seed}")
            if row is None or row["status"] != "completed":
                continue
            cell["seeds"][seed] = dict(
                post_cooldown=row["post_cooldown"],
                constant_lr=row["constant_lr"], ema_180=row["ema_180"],
                step1_train_loss=row["step1_train_loss"],
                step1_parity=row["step1_parity"],
                train_seconds_constant=row["train_seconds_constant"],
                effective_rows=row["effective_rows"],
                top_row_share=row["top_row_share"],
                occupancy=row["occupancy"],
                identity_share=row["identity_share"],
                gate_drifts=row["gate_drifts"],
                muon_update_rms=row["muon_update_rms"])
            if row["post_cooldown"] is not None \
                    and row["post_cooldown"] <= GDN2_POST_COOLDOWN:
                escalation_flags.append(
                    f"the {label} cell at seed {seed} landed at/past GDN2's "
                    f"{GDN2_POST_COOLDOWN} (post-cooldown "
                    f"{row['post_cooldown']}) -- the band-discipline clause: "
                    f"BOTH scorings and BOTH seeds are reported before any "
                    f"claim, and the result is escalated")
        if cell["seeds"]:
            cells[label] = cell
    if not any(not c["cited"] for c in cells.values()):
        raise ValueError("no completed bracket arm receipts to assemble")

    # the pre-stated second-seed rule, applied exactly as stated
    seed1_cells = {label: c["seeds"][PRIMARY_SEED]["post_cooldown"]
                   for label, c in cells.items()
                   if PRIMARY_SEED in c["seeds"]
                   and c["seeds"][PRIMARY_SEED]["post_cooldown"] is not None}
    if not seed1_cells:
        raise ValueError("no seed-3401 post-cooldown rows in the table")
    best_cell = min(seed1_cells, key=seed1_cells.get)
    best_seed_pairs = {label: (SECOND_SEED in cells[label]["seeds"])
                       for label in cells}
    second_seed_owed = (best_cell != "muon4x"
                        and not best_seed_pairs.get(best_cell, False))
    second_seed_adjudication = dict(
        best_cell_at_seed_3401=best_cell,
        best_post_cooldown_at_seed_3401=seed1_cells[best_cell],
        rule=SECOND_SEED_RULE,
        second_seed_owed=bool(second_seed_owed),
        second_seed_already_banked=bool(best_cell == "muon4x"),
        second_seed_already_run=bool(best_seed_pairs.get(best_cell, False)),
        note=("the 4x cell's pair is complete (the K15 arm landed at both "
              "seeds); any other best cell owes seed 3402") if best_cell \
            == "muon4x" else
            ("the best cell owes its second seed (3402) -- run it before "
             "the final gap quote") if second_seed_owed else
            ("the best cell's second seed has already landed"))

    # the optimizer-family read (the adamw arm's own flag, escalated not
    # adjudicated if it inverts the standing ranking)
    family_rows = {label: (cells[label]["seeds"].get(PRIMARY_SEED)
                           or {}).get("post_cooldown")
                   for label in cells if label != "adamw"}
    adamw_row = (cells.get("adamw", {}).get("seeds", {}).get(PRIMARY_SEED)
                 or {}).get("post_cooldown")
    family_inversion = bool(
        adamw_row is not None and family_rows
        and adamw_row < min(v for v in family_rows.values()
                            if v is not None))
    if family_inversion:
        escalation_flags.append(
            f"the OPTIMIZER-FAMILY RANKING INVERTED on the composed machine: "
            f"plain AdamW at 1x ({adamw_row}) beat every hybrid-Muon cell at "
            f"seed {PRIMARY_SEED} -- a standing-quality-conclusion inversion, "
            f"escalated per the lane's rules")

    # the search-symmetric ours-best: the best measured row across the
    # composed machine's whole bracket surface (all cells, all run seeds)
    ours_best = None
    for label, cell in cells.items():
        for seed, row in cell["seeds"].items():
            if row["post_cooldown"] is None:
                continue
            if ours_best is None \
                    or row["post_cooldown"] < ours_best["post_cooldown"]:
                ours_best = dict(cell=label, seed=seed,
                                 post_cooldown=row["post_cooldown"],
                                 constant_lr=row.get("constant_lr"),
                                 ema_180=row.get("ema_180"))
    gap_quote = None
    if ours_best is not None:
        post = ours_best["post_cooldown"]
        gap_quote = dict(
            ours_best_post_cooldown=post, cell=ours_best["cell"],
            seed=ours_best["seed"],
            gdn2_post_cooldown=GDN2_POST_COOLDOWN,
            remaining_gap=post - GDN2_POST_COOLDOWN,
            fraction_of_corrected_gap_closed=max(
                0.0, BANKED_CORRECTED_ANCHOR - post)
            / (BANKED_CORRECTED_ANCHOR - GDN2_POST_COOLDOWN),
            bands=BANDS,
            caveat="the endpoint-variance band (0.075) and the winner's "
                   "seed delta (0.0119) are quoted with every margin; the "
                   "composed-machine-vs-GDN2 row is a cross-machine "
                   "context row at the same probe protocol, not a "
                   "controlled comparison; the final gap-quote phrasing is "
                   "the supervisor's to adjudicate (the K13 clause-conflict "
                   "precedent)",
            config_yield_vs_inherited_4x=dict(
                same_seed_improvement=(four_x["post_cooldown"][
                    ours_best["seed"]] - post)
                if ours_best["cell"] != "muon4x" else 0.0,
                note="how much of the remaining ~0.03 the bracket's config "
                     "axis owned (the same-seed read vs the inherited 4x "
                     "row); if the inherited point was already optimal this "
                     "is 0 and the config-symmetry duty is discharged by "
                     "receipt"))

    summary = dict(
        probe="k16_config_bracket assemble",
        corrected_scoring=CORRECTED_SCORING,
        score_form="the K15 composed machine (the K13 additive read under "
                   "the K11B p3 per-channel gate stack) at the bracket's "
                   "LR/optimizer points -- the machine is NOT the lane's "
                   "variable, the optimizer config is",
        arms=cells, best_cell_at_seed_3401=best_cell,
        second_seed_adjudication=second_seed_adjudication,
        optimizer_family_inversion=family_inversion,
        escalation_flags=escalation_flags,
        search_symmetric_ours_best=ours_best,
        gap_quote_vs_gdn2=gap_quote,
        notes=dict(
            scoring="PRIMARY: post-cooldown (24-step linear); SECONDARY: "
                    "EMA-0.99 at step 180 (ranking-only); constant-180 "
                    "retained for banked comparability",
            bands="every margin is read against BOTH the "
                  f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance band and "
                  f"the winner's {WINNER_SEED_DELTA:.4f} seed delta",
            pre_state="NO directional prediction was pre-stated on the "
                      "bracket's outcome; the config-symmetry duty is "
                      "discharged by receipt either way (the inherited point "
                      "optimal is a receipt, not a null result)",
            escalation="the final gap-quote phrasing and any "
                       "quality-conclusion inversion are escalated to the "
                       "supervisor (the K13 clause-conflict precedent "
                       "governs adjudication)"))
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(cells={k: {s: v["post_cooldown"] for s, v in
                                     c["seeds"].items()}
                                 for k, c in cells.items()},
                          best_cell=best_cell,
                          second_seed_owed=second_seed_owed,
                          ours_best=ours_best,
                          escalation_flags=escalation_flags), indent=1),
          flush=True)
    return summary


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--arm", required=True, choices=sorted(ARMS))
    a.add_argument("--seed", type=int, default=PRIMARY_SEED)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--output", required=True)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    a.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, 3 checkpoint "
                        "steps, 3 cooldown steps, no parity assertions")
    s = sub.add_parser("assemble")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "k16_config_bracket_summary.json"))
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
