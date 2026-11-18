"""THE SYMMETRIC ROUTING EXTENSION -- OUR winner's missing Muon routings.

THE ASYMMETRY THIS LANE CLOSES (the operator's catch): the Muon control gave
GDN2 TWO Muon routings -- max-Muon (option A, 1.4702) and gates-on-AdamW
(option B, 1.4151, GDN2's better by 0.055) -- but OUR winner only ONE (the
bake-off convention, 1.5146/1.5027 two-seed).  "Our-best vs GDN2's-best-of-
two" is asymmetric in GDN2's favor; under MATCHED (A-style) routing the gap is
0.044, near the endpoint-variance band.  Nothing is settled: our machine may
prefer a different routing just as GDN2 did.

THE PARAMETER-INVENTORY MAPPING (supervisor-approved, M1 -- the faithful
functional analog of the control's own GDN2 enumeration): our winner's
inventory is 267 tensors / 149,981,808 params, enumerated BY FUNCTION:

* **the gate-VALUE producer (the only learned gate):** ``mixer.decay.weight``
  (2x576) + ``mixer.decay.bias`` (2) -- the DECAY gate, the direct analog of
  GDN2's ``f_proj`` (decay pre-activation), which GDN2's option B moved to
  AdamW.  The DECAY pair is 24 x (1,152+2) = 27,696 params (the 2D weight
  alone: 27,648, 0.018% of the machine) -- vs GDN2's 27.4M gate-matrix
  surface (18% of its matrix params).
* **the write/erase/read admission:** the five-op selection chain -- threshold
  comparisons on HYPERPARAMETERS (tau_match/tau_merge/tau_pop, not params)
  against the similarity of write_keys with the stored key table; the read is
  a softmax addressed by query_keys.  ``mixer.key`` / ``mixer.query_key``
  produce ADDRESSES -- the functional analogs of GDN2's k_proj/q_proj, which
  stayed on Muon under BOTH of GDN2's routings.  Our machine has NO learned
  write/erase gate value.
* **everything else:** embed/head, ``mixer.value``/``mixer.out``, the SwiGLU
  mlp w1/w2/w3 (all 2D, Muon under every routing here), the RMSNorms (1D).

THE CONSEQUENCE (stated in the doc, not hidden): under the approved mapping
the banked bake-off routing (Muon on embed/head/mixer value/key/query_key/out
and mlp w1-w3, 170 tensors / 149,925,888 params; AdamW on the norms +
``decay.weight`` + ``decay.bias``, 97 tensors / 55,920 params) ALREADY IS our
gates-on-AdamW routing --
the banked 1.5146/1.5027 rows ARE our B-analog, cited not re-run (a re-run
would be bit-identical by construction).  The arms this lane RUNS:

* ``muon3x_max``        THE A-ANALOG (max-Muon): every 2D matrix of our
                        inventory under Muon -- the banked routing plus
                        ``decay.weight`` (194 tensors / 149,953,536 params,
                        99.98%); AdamW only on the true non-matrix residue
                        (the 1D norms + ``decay.bias``, 73 tensors / 28,272
                        params).  The direct analog of GDN2's option A.
* ``muon3x_gates_adamw`` THE BROAD-MAPPING APPENDIX (the M2 reading,
                        supervisor-approved as an arm so the broad reading
                        cannot be said untested): the gate producer AND the
                        address producers move to AdamW --
                        ``decay.weight``/``.bias`` + ``mixer.key`` +
                        ``mixer.query_key`` on AdamW (145 tensors / 3,594,864
                        params), Muon on embed/head/value/out and the mlp
                        (122 tensors / 146,386,944 params).  GDN2's own
                        precedent (addressing stays Muon in both its
                        routings) predicts neutral-to-worse.
* ``muon{2x,4x}_<routing>``  THE LR BRACKET at the best routing: 3x is an
                        inherited bracket on BOTH machines, not a search;
                        {2x, 4x} around it, honestly scoped.

PRE-STATED PREDICTIONS (supervisor-approved, receipted in every arm):

* ``routing_spread`` (the A-analog): with 0.018% of the machine moving, the
  max-Muon arm lands within 0.02 of the banked B-analog same-seed row
  (1.5146) -- our machine has no meaningful routing surface to exploit.
  Outside the band = a genuine decay-gate optimization-sensitivity finding
  (a 27K-param gate steering a 150M machine would be remarkable), reported
  as such either way.
* ``broad_mapping`` (the appendix): key/query_key on AdamW is
  NEUTRAL-TO-WORSE than the banked B-analog at the same seed -- GDN2's own
  precedent that addressing prefers Muon.

PROVENANCE (the lane's own rules): no model/semantics change, optimizer
routing only; new files only -- ``k10_*``/``k6_*``/bakeoff/muon_control
sources, the PREREGISTRATIONs and ``lattice.py`` are NOT edited, only
imported (``velocity_ladder._build_k10`` verbatim, ``bakeoff_optimizers.
HybridMuonX45`` verbatim -- the same pinned hybrid the bake-off and the
control receipted).  Every arm runs the EXACT probe protocol: the
20,000,000-target matched-exposure rule (the 20,022,866 landing, 180 steps),
seed 3401 first (the seed axis an arm's own variable), the registered
flagship stream (identity asserted), the document-disjoint calibration eval
at 7 windows / 14,336 targets (recorded-read path), the honest clock, the
protocol's own grad-clip (1.0, error_if_nonfinite), and the bake-off's own
checkpoint plan (steps 1/5/10/20/40/80/120/160/180, CPU state snapshots
inside the loop, every eval on restored snapshots after training -- the
training path unperturbed by construction).  Protocol-identical starts,
receipted: every arm's step-1 train loss is bit-identical to the BANKED
muon3x arm's at the same seed (asserted in-code against the banked run's
own training.jsonl).

Deliverables: per-arm receipts under ``routing_symmetry_runs/<label>_seed<seed>/``,
the assemble summary ``receipts/routing_symmetry_summary.json``, and
``RESULT_ROUTING_SYMMETRY.md``.

Usage (see ``run_routing_symmetry_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry arm \\
        --label muon3x_max --output routing_symmetry_runs/muon3x_max_seed3401
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
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "routing_symmetry_runs"
BAKEOFF_RUNS = ROOT / "bakeoff_runs"
MUON_CONTROL_RUNS = ROOT / "muon_control_runs"
SIDE_BY_SIDE_RUNS = ROOT / "side_by_side_runs"

SEED = 3401
SECOND_SEED = 3402
EXPOSURE_TARGETS = 20_000_000
# The bake-off's own checkpoint plan (the curve-overlay grid every banked
# curve is read at).
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
PROBE_WINDOWS = 4
GRAD_CLIP = 1.0                 # the probe protocol's own clip, every arm
MUON_LR_MULTIPLIER = 3         # the inherited bracket point both machines ran

# The banked rows this lane's table is built from (cited, never re-run).
BANKED = {
    "winner_muon3x_seed3401": 1.514617647443499,   # = OUR B-analog (the banked
    "winner_muon3x_seed3402": 1.5026696068899972,  #   bake-off routing, M1)
    "winner_adamw": 1.822737353188651,
    "gdn2_adamw": 1.5836738858904158,
    "gdn2_muon3x_max": 1.4702048471995763,         # GDN2's option A
    "gdn2_muon3x_gates_adamw": 1.4150957890919276, # GDN2's option B
}
# The endpoint-variance band the side-by-side receipts (~0.075 nats) and the
# winner's own two-seed band (|delta| 0.0119) -- every margin is read against
# both.
ENDPOINT_VARIANCE_BAND = 0.075
WINNER_SEED_DELTA = abs(BANKED["winner_muon3x_seed3402"]
                        - BANKED["winner_muon3x_seed3401"])

# The pre-stated, supervisor-approved predictions (receipted in every arm's
# result.json; the assemble verdicts apply them exactly as stated).
ROUTING_SPREAD_BAND = 0.02
PREDICTIONS = dict(
    routing_spread=dict(
        arm="muon3x_max",
        prediction="the max-Muon (A-analog) final lands within 0.02 of the "
                   "banked B-analog same-seed row (1.5146): 0.018% of the "
                   "machine moves, so there is no meaningful routing surface "
                   "to exploit",
        band=ROUTING_SPREAD_BAND,
        reference=BANKED["winner_muon3x_seed3401"],
        confirmed_if="abs(final - 1.514617647443499) <= 0.02",
        outside_band_is="a genuine decay-gate optimization-sensitivity "
                        "finding (a 27K-param gate steering a 150M machine "
                        "would be remarkable), reported as such either way"),
    broad_mapping=dict(
        arm="muon3x_gates_adamw",
        prediction="key/query_key on AdamW is NEUTRAL-TO-WORSE than the "
                   "banked B-analog at the same seed -- GDN2's own precedent "
                   "that addressing prefers Muon (its q/k/v/o stayed on Muon "
                   "under both routings)",
        band=ROUTING_SPREAD_BAND,
        reference=BANKED["winner_muon3x_seed3401"],
        confirmed_if="final >= 1.514617647443499 - 0.02",
        refuted_is="the broad-mapping arm BEATS the banked B-analog by more "
                   "than the 0.02 band -- the addressing projections train "
                   "better under AdamW on this machine"),
)

# The symmetric claim rule, pre-stated (the task's own, restated): no 'leads'
# or 'behind' claims except against best-vs-best with the ROUTING SEARCH
# SYMMETRIC -- our-best-of-routings vs GDN2's-best-of-routings -- with every
# margin read against the endpoint-variance band and the seed band.
CLAIM_RULE = dict(
    rule="no 'leads'/'behind' claims except best-vs-best with the routing "
         "search symmetric (our best measured routing x best measured LR "
         "point vs GDN2's best measured routing, both at 3x-of-own-AdamW "
         "unless the bracket says otherwise); margins inside the "
         f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance band or the "
         f"{WINNER_SEED_DELTA:.4f} seed band are stated as parity-within-band",
    our_best_of_routings="min over the banked B-analog rows (two seeds) and "
                         "this lane's A-analog / broad-mapping arms, at the "
                         "measured-best LR point",
    gdn2_best_of_routings="min(GDN2 option A, GDN2 option B) = 1.4150957890919276",
)

# ------------------------------------------------- the winner's inventory

# Source-named, ENUMERATED (the muon_control discipline: any parameter outside
# the documented inventory fails loudly instead of silently routing).
_EMBED_HEAD = tuple(re.compile(name + "$") for name in
                    (r"embed\.weight", r"head\.weight"))
_MIXER_MATRIX = re.compile(
    r"blocks\.\d+\.mixer\.(value|key|query_key|out)\.weight$")
_MLP = re.compile(r"blocks\.\d+\.mlp\.w[123]\.weight$")
# the gate-VALUE producer (the only learned gate): the DECAY pair.
_DECAY_WEIGHT = re.compile(r"blocks\.\d+\.mixer\.decay\.weight$")
_DECAY_BIAS = re.compile(r"blocks\.\d+\.mixer\.decay\.bias$")
# the true non-matrix residue: the 1D RMSNorm weights.
_NORMS = re.compile(r"(norm|blocks\.\d+\.norm[12])\.weight$")

ROUTINGS = {
    "banked_gates_adamw":
        "the banked bake-off routing (OUR B-analog under the approved M1 "
        "mapping): Muon on embed/head, mixer value/key/query_key/out, mlp "
        "w1/w2/w3 (170 tensors / 149,925,888 params); AdamW on the norms, "
        "decay.weight and decay.bias (97 tensors / 55,920 params) -- the "
        "gate-VALUE producer already on AdamW, exactly GDN2's option-B "
        "shape",
    "max_muon":
        "the A-analog (max-Muon): every 2D matrix of our inventory under "
        "Muon -- the banked routing plus decay.weight (194 tensors / "
        "149,953,536 params, 99.98%); AdamW only on the true non-matrix "
        "residue (the 1D norms + decay.bias, 73 tensors / 28,272 params)",
    "gates_adamw_broad":
        "the broad-mapping appendix (M2, run so the broad reading cannot be "
        "said untested): the gate producer AND the address producers on "
        "AdamW -- decay.weight/.bias + mixer.key + mixer.query_key (145 "
        "tensors / 3,594,864 params); Muon on embed/head, mixer "
        "value/out, mlp w1/w2/w3 (122 tensors / 146,386,944 params)",
}


def split_parameters_winner(model, routing: str):
    """Route the converted winner's parameters to (muon, adamw) by name, the
    approved M1 mapping applied to our own enumerated inventory.

    Fails loudly on a Muon-owned name that is not a full 2D matrix (the
    Newton-Schulz convention requires it), on distinct parameters sharing
    storage (the prior line's own guard), and on ANY parameter outside the
    enumerated inventory (a drift must fail, not silently route).
    """
    if routing not in ROUTINGS:
        raise ValueError(f"unregistered routing: {routing}")
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
        kind = None
        if any(r.fullmatch(name) for r in _EMBED_HEAD):
            kind = "embed_head"
        elif _MIXER_MATRIX.fullmatch(name) is not None:
            kind = ("mixer_address"
                    if _MIXER_MATRIX.fullmatch(name).group(1)
                    in ("key", "query_key") else "mixer_core")
        elif _MLP.fullmatch(name) is not None:
            kind = "mlp"
        elif _DECAY_WEIGHT.fullmatch(name) is not None:
            kind = "decay_weight"
        elif _DECAY_BIAS.fullmatch(name) is not None:
            kind = "decay_bias"
        elif _NORMS.fullmatch(name) is not None:
            kind = "norm"
        else:
            raise ValueError(
                f"unclassified winner parameter: {name} (the inventory is "
                f"enumerated; a drift must fail loudly, not silently route)")
        # Muon owns every 2D matrix except the routing's own exclusions:
        #   banked_gates_adamw excludes the DECAY pair (the gate-VALUE
        #     producer, GDN2's option-B shape);
        #   max_muon excludes nothing 2D (GDN2's option-A shape);
        #   gates_adamw_broad additionally excludes the ADDRESS producers
        #     (key/query_key, the M2 broad reading).
        excluded = {"banked_gates_adamw": {"decay_weight", "decay_bias"},
                    "max_muon": {"decay_bias"},
                    "gates_adamw_broad": {"decay_weight", "decay_bias",
                                          "mixer_address"}}[routing]
        is_muon = kind not in excluded and kind != "norm"
        if is_muon:
            if param.ndim != 2:
                raise ValueError(f"Muon-owned parameter is not a full 2D "
                                 f"matrix: {name}")
            muon.append(name)
        else:
            adamw.append(name)
    if not muon or not adamw:
        raise ValueError("the hybrid requires both semantic groups non-empty")
    params = dict(model.named_parameters())
    return ([params[n] for n in muon], [params[n] for n in adamw],
            muon, adamw)


def _arm_routing_note(routing: str) -> str:
    return ROUTINGS[routing]


ARMS = {
    "muon3x_max": dict(
        routing="max_muon", lr_multiplier=MUON_LR_MULTIPLIER,
        note="THE A-ANALOG (max-Muon): every 2D matrix under Muon, the "
             "direct analog of GDN2's option A; carries the pre-stated "
             "routing_spread prediction"),
    "muon3x_gates_adamw": dict(
        routing="gates_adamw_broad", lr_multiplier=MUON_LR_MULTIPLIER,
        note="the broad-mapping appendix (M2): decay + the address "
             "producers on AdamW, run so the broad reading cannot be said "
             "untested; carries the pre-stated broad_mapping prediction"),
    "muon2x_max": dict(
        routing="max_muon", lr_multiplier=2,
        note="the LR bracket below the inherited 3x, at the max-Muon routing"),
    "muon4x_max": dict(
        routing="max_muon", lr_multiplier=4,
        note="the LR bracket above the inherited 3x, at the max-Muon routing"),
    "muon2x_gates_adamw": dict(
        routing="gates_adamw_broad", lr_multiplier=2,
        note="the LR bracket below the inherited 3x, at the broad-mapping "
             "routing"),
    "muon4x_gates_adamw": dict(
        routing="gates_adamw_broad", lr_multiplier=4,
        note="the LR bracket above the inherited 3x, at the broad-mapping "
             "routing"),
    "muon2x_banked": dict(
        routing="banked_gates_adamw", lr_multiplier=2,
        note="the LR bracket below the inherited 3x, at the banked (B-analog) "
             "routing -- the routing the two banked seeds ran"),
    "muon4x_banked": dict(
        routing="banked_gates_adamw", lr_multiplier=4,
        note="the LR bracket above the inherited 3x, at the banked "
             "(B-analog) routing -- the routing the two banked seeds ran"),
}


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def _banked_step1_loss(seed: int) -> float:
    """The banked muon3x arm's step-1 train loss at this seed (the
    protocol-identical start receipt: the optimizer cannot touch the first
    forward, so a new arm at the same seed must reproduce it bit-for-bit)."""
    path = BAKEOFF_RUNS / f"muon3x_seed{seed}" / "training.jsonl"
    if not path.exists():
        raise ValueError(f"the banked same-seed row is missing: {path}")
    first = json.loads(path.read_text().splitlines()[0])
    return first["loss"]


# ------------------------------------------------------------ the training


def _train_with_snapshots(model, optimizer, set_lr, stream, targets, label,
                          log_path, checkpoint_steps):
    """The winner's armed loop (optimizer_bakeoff's ``_train_with_snapshots``
    shape VERBATIM: matched-exposure stopping, honest clock with time_ns
    rows, the protocol's own grad clip, the optimizer-step seconds receipted
    separately), with the routing arm's own optimizer wired in."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)

    microbatch = 8                # the transferred, fixed probe knob
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    opt_seconds = []
    step = seen = 0
    rows = []
    snapshots = {}
    diverged = False
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
    from experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys \
        import probe_diagnostics
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import BETA2, MICROBATCH, TUNED_WINNER
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

    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER),
                                                  args.seed)
    # the banked row's own knobs, asserted (a config drift must fail loudly,
    # not silently change the arm's baseline)
    expected = dict(lr=TUNED_WINNER["lr"],
                    betas=(0.9, BETA2),
                    weight_decay=TUNED_WINNER["weight_decay"],
                    warmup_steps=TUNED_WINNER["warmup_steps"],
                    microbatch=MICROBATCH)
    for key, want in expected.items():
        actual = (tuple(optimizer_config[key]) if key == "betas"
                  else optimizer_config[key])
        if actual != want:
            raise ValueError(f"the winner's config drifted on {key}: "
                             f"{actual!r} != the banked {want!r}")

    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_winner(model, arm["routing"])
    lr = arm["lr_multiplier"] * TUNED_WINNER["lr"]
    optimizer = BO.HybridMuonX45(
        muon_params, adamw_params, lr=lr, betas=(0.9, BETA2),
        weight_decay=TUNED_WINNER["weight_decay"])
    warmup = TUNED_WINNER["warmup_steps"]

    def set_lr(step):
        for group in optimizer.param_groups:
            group["lr"] = lr * min(1., (step + 1) / max(1, warmup))

    parameters_total = sum(p.numel() for p in model.parameters())
    run_config = dict(
        probe="the symmetric routing extension", arm="k10_separated_keys "
        "(the winner)", label=args.label, family="hybrid-muon",
        routing=arm["routing"], routing_inventory=dict(
            muon=muon_names, adamw=adamw_names),
        routing_note=_arm_routing_note(arm["routing"]),
        conventions_note=arm["note"], predictions=PREDICTIONS,
        claim_rule=CLAIM_RULE, seed=args.seed,
        exposure_targets=int(args.bytes),
        model_build="velocity_ladder._build_k10 verbatim (the tuned winner's "
                    "config, the banked bake-off build)",
        optimizer=dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            lr=lr, lr_multiplier=arm["lr_multiplier"],
            lr_convention=(f"{arm['lr_multiplier']}x the tuned winner's "
                           "AdamW LR (3x is the inherited bracket point both "
                           "machines ran, not a search; the bracket arms "
                           "honesty-scope it)"),
            betas=(0.9, BETA2),
            weight_decay=TUNED_WINNER["weight_decay"],
            warmup_steps=warmup,
            warmup_note="both hybrid groups scale equally (the prior line's "
                        "convention, the bake-off's muon arms' own)",
            microbatch=MICROBATCH, grad_clip=GRAD_CLIP,
            muon_params=f"{len(muon_names)} tensors "
                        f"({sum(p.numel() for p in muon_params)} params)",
            adamw_params=f"{len(adamw_names)} tensors "
                         f"({sum(p.numel() for p in adamw_params)} params)"),
        symmetry_rule="same stream, same data order, same matched-exposure "
                      "stopping rule, same honest clock, same grad clip, "
                      "same calibration eval as the banked rows; the ONLY "
                      "difference is the optimizer ROUTING (and, on the "
                      "bracket arms, the Muon LR point)",
        optimizer_knobs_held_at_incumbent=dict(
            betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
            warmup_steps=warmup, microbatch=MICROBATCH),
        checkpoint_steps=list(args.checkpoint_steps),
        checkpoint_note=("state snapshots taken inside the training loop at "
                         "each checkpoint step (the bake-off's own plan); "
                         "every eval runs on restored snapshots after "
                         "training, so the training path is unperturbed by "
                         "construction"),
        summary=summary, identity=identity, versions=_versions(),
        bindings=dict(matched_arms.registration_bindings(),
                      banked_rows=BANKED,
                      banked_bakeoff_summary=str(
                          RECEIPTS / "optimizer_bakeoff_summary.json"),
                      muon_control_summary=str(
                          RECEIPTS / "muon_control_summary.json")),
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
                          routing=arm["routing"],
                          muon_params=len(muon_names),
                          adamw_params=len(adamw_names))), flush=True)

    run = _train_with_snapshots(
        model, optimizer, set_lr, stream, int(args.bytes), args.label,
        out / "training.jsonl", tuple(args.checkpoint_steps))
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]

    # the protocol-identical start receipt: the optimizer cannot touch the
    # first forward, so step 1 must reproduce the BANKED muon3x arm's step-1
    # train loss at the same seed, bit-for-bit
    if not args.smoke:
        banked_step1 = _banked_step1_loss(args.seed)
        if run["rows"][0]["loss"] != banked_step1:
            raise ValueError(
                f"step-1 train loss {run['rows'][0]['loss']!r} != the banked "
                f"muon3x row's {banked_step1!r} at seed {args.seed} (the "
                f"protocol start is not identical)")

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

    # ---- the mechanism-unchanged receipt at the final checkpoint (the
    # bake-off's own diagnostics, imported unchanged)
    diagnostics = None
    if run["status"] == "completed" and run["snapshots"]:
        model.load_state_dict(run["snapshots"][max(run["snapshots"])])
        diagnostics = probe_diagnostics(model, stream, windows=args.probe_windows)

    receipt = dict(
        probe="the symmetric routing extension", label=args.label,
        family="hybrid-muon", routing=arm["routing"],
        routing_inventory=dict(
            muon=f"{len(muon_names)} tensors "
                 f"({sum(p.numel() for p in muon_params)} params)",
            adamw=f"{len(adamw_names)} tensors "
                  f"({sum(p.numel() for p in adamw_params)} params)",
            muon_names=muon_names, adamw_names=adamw_names),
        routing_note=_arm_routing_note(arm["routing"]),
        conventions_note=arm["note"], predictions=PREDICTIONS,
        claim_rule=CLAIM_RULE, status=run["status"], seed=args.seed,
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
            args.smoke or run["rows"][0]["loss"] == _banked_step1_loss(args.seed)),
        banked=BANKED, checkpoints=checkpoint_rows, diagnostics=diagnostics,
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
    """The symmetric table (both machines x both routings, two seeds where
    run), the pre-stated prediction verdicts, the LR-bracket read, and the
    symmetric best-vs-best verdict with every margin read against the
    endpoint-variance and seed bands."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    def arm_curve(path: Path):
        if not (path / "checkpoints.jsonl").exists():
            return {}
        return {r["step"]: r for r in _load_checkpoints(
            path / "checkpoints.jsonl")}

    # the banked curves (cited, unmodified) + this lane's arms
    curves = dict(
        winner_muon3x_seed3401=arm_curve(BAKEOFF_RUNS / "muon3x_seed3401"),
        winner_muon3x_seed3402=arm_curve(BAKEOFF_RUNS / f"muon3x_seed{SECOND_SEED}"),
        gdn2_adamw=arm_curve(SIDE_BY_SIDE_RUNS / "gdn2_armed_seed3401"),
        winner_adamw=arm_curve(SIDE_BY_SIDE_RUNS / "k10_tuned_armed_seed3401"),
        gdn2_muon3x_max=arm_curve(MUON_CONTROL_RUNS / "gdn2_muon3x_seed3401"),
        gdn2_muon3x_gates_adamw=arm_curve(
            MUON_CONTROL_RUNS / "gdn2_muon3x_gates_adamw_seed3401"),
    )
    for run_dir in sorted(RUNS.glob("*_seed*")):
        label = run_dir.name[:-len("_seed3401")] \
            if run_dir.name.endswith("_seed3401") else run_dir.name
        curves[label] = arm_curve(run_dir)

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
            "winner_muon3x_seed3401": BAKEOFF_RUNS / "muon3x_seed3401",
            "winner_muon3x_seed3402": BAKEOFF_RUNS
            / f"muon3x_seed{SECOND_SEED}",
            "gdn2_adamw": SIDE_BY_SIDE_RUNS / "gdn2_armed_seed3401",
            "winner_adamw": SIDE_BY_SIDE_RUNS / "k10_tuned_armed_seed3401",
            "gdn2_muon3x_max": MUON_CONTROL_RUNS / "gdn2_muon3x_seed3401",
            "gdn2_muon3x_gates_adamw":
                MUON_CONTROL_RUNS / "gdn2_muon3x_gates_adamw_seed3401",
        }
        if label in paths:
            return _final_of(paths[label])
        run_dir = RUNS / f"{label}_seed3401"
        if not run_dir.exists():
            run_dir = RUNS / f"{label}_seed{SECOND_SEED}"
        if not run_dir.exists():
            return None
        return _final_of(run_dir)

    finals = {label: final_of(label) for label in curves}

    # ---- the pre-stated prediction verdicts
    prediction_verdicts = {}
    for key, pred in PREDICTIONS.items():
        arm_final = (finals.get(pred["arm"]) or {}).get("final_eval_nll")
        if arm_final is None:
            prediction_verdicts[key] = dict(
                prediction=pred, verdict="UNMEASURED (the arm has not landed)")
            continue
        delta = arm_final - pred["reference"]
        if key == "routing_spread":
            hit = abs(delta) <= pred["band"]
            verdict = ("CONFIRMED: the max-Muon arm lands inside the 0.02 "
                       f"band (delta {delta:+.4f}) -- our machine has no "
                       "meaningful routing surface to exploit" if hit else
                       "OUTSIDE THE BAND: a genuine decay-gate "
                       "optimization-sensitivity finding (a 27K-param gate "
                       f"steering a 150M machine; delta {delta:+.4f}) -- "
                       "reported as the finding, not smoothed")
        else:
            hit = arm_final >= pred["reference"] - pred["band"]
            verdict = ("CONFIRMED: neutral-to-worse, addressing prefers Muon "
                       f"(delta vs the banked B-analog {delta:+.4f})" if hit
                       else "REFUTED: the broad-mapping arm BEATS the banked "
                            "B-analog by more than the band -- the address "
                            "projections train better under AdamW on this "
                            "machine")
        prediction_verdicts[key] = dict(prediction=pred, arm_final=arm_final,
                                        delta=delta, verdict=verdict)

    # ---- the symmetric best-vs-best read
    our_candidates = {label: f["final_eval_nll"] for label, f in finals.items()
                      if f and f.get("final_eval_nll") is not None
                      and (label.startswith("winner_muon3x")
                           or label.startswith("muon"))}
    gdn2_candidates = {label: f["final_eval_nll"]
                       for label, f in finals.items()
                       if f and f.get("final_eval_nll") is not None
                       and label.startswith("gdn2")}
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
                       f"the winner's own seed band is {WINNER_SEED_DELTA:.4f})"
                       " -- no leads/behind claim is bankable at this margin")
        symmetric = dict(rule=CLAIM_RULE, our_best=dict(
            label=our_best_label, final=our_best),
            gdn2_best=dict(label=gdn2_best_label, final=gdn2_best),
            margin=margin, verdict=verdict,
            endpoint_variance_band=ENDPOINT_VARIANCE_BAND,
            winner_seed_band=WINNER_SEED_DELTA)

    summary = dict(probe="the symmetric routing extension assemble",
                   banked=BANKED, claim_rule=CLAIM_RULE,
                   predictions=PREDICTIONS,
                   prediction_verdicts=prediction_verdicts,
                   finals=finals, overlay=overlay,
                   symmetric_best_vs_best=symmetric)
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(
        predictions={k: v.get("verdict") for k, v in
                     prediction_verdicts.items()},
        symmetric=(symmetric or {}).get("verdict"),
        finals={k: (v or {}).get("final_eval_nll")
                for k, v in finals.items()}), indent=1), flush=True)
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
                                        / "routing_symmetry_summary.json"))
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
