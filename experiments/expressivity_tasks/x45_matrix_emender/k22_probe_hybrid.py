"""K22 -- THE HYBRID probe: the arms on GDN2's committed 150M stack.

The registered arms (``PREREGISTRATION_K22_HYBRID.md``, committed before
any arm):

``hybrid``         the hybrid machine (GDN2's committed arm + the composed
                   slot mixer at layers {6, 12, 18} through the zero-born
                   read gate), LIVE: the gates learn to open.  Seed 3401
                   first; seed 3402 owed on any >= 0.01 win per the
                   standing rule.
``hybrid_frozen``  THE FALSIFIER: the same machine with the whole slot
                   branch frozen at the zero-read birth.  It must be
                   bit-identical to the banked GDN2 row's own trajectory
                   (``protocol_correction_runs/gdn2_speedrun_rescored``,
                   the 1.3522 machine's own 204-step log) at EVERY step
                   and land at 1.3521734476089478 post-cooldown bit for
                   bit -- the strongest plumbing receipt, the K18
                   perfect-receipt precedent the expectation to beat.

Protocol: the matched-exposure probe (train until 20,000,000 targets,
landing 20,022,866 on the registered flagship stream at the standing
seed), the corrected cooldown-primary scoring (the 24-step linear
cooldown, EMA-0.99 secondary), GDN2's banked row's own recipe VERBATIM
(HybridMuonX45 at the 5x speedrun point, option B routing, warmup 69,
microbatch 8, grad clip 1.0) -- the ONLY difference from the banked row is
the machine (the slot branch, born inert).

The instruments: the C5 mechanism rows on the slot layers (identity
share, effective rows, occupancy, the op mix), THE DECORATIVE SWEEP AS THE
HEADLINE (the slot-read mask per slot layer post-cooldown: do the slot
layers become load-bearing -- does the uniformity finding invert?), the
gate trajectory (which slot layers open their read gates, how far), and
the at-birth open-signal receipt (dL/dgate nonzero, every other slot
parameter's gradient exactly zero at birth).

Usage (one leased card, cards 2/3; see ``run_k22_hybrid_leased.sh``):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k22_probe_hybrid \\
        run_arm --arm hybrid --seed 3401 --output k22_runs/hybrid_seed3401
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

# The probe subprocess discipline (configure_production fail-closes on a
# missing allocator knob if these are not set before CUDA initialization).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
    import (BETA2, GRAD_CLIP, MICROBATCH)
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction \
    import (COOLDOWN_STEPS, CORRECTED_SCORING, EMA_DECAY,
            ENDPOINT_VARIANCE_BAND, WINNER_SEED_DELTA)
from experiments.expressivity_tasks.x45_matrix_emender.\
    k15_probe_composition import (BANKED, CHECKPOINT_STEPS, EXPOSURE_TARGETS,
                                  PRIMARY_SEED, PROBE_WINDOWS, SECOND_SEED,
                                  _banked_mechanism_anchors)
from experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
    import composed_4x_banked
from experiments.expressivity_tasks.x45_matrix_emender.muon_control import (
    GDN2_BANKED_ADAMW)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k22_runs"

# ---- the banked rows this lane cites (never re-run, never rewritten) ----
# GDN2's committed arm's own corrected-scoring row: the 1.3522 machine
# (protocol_correction_runs/gdn2_speedrun_rescored, seed 3401, option B,
# 5x).  OURS: the composed machine's banked 4x rows (the K15 assemble
# receipt; 1.3811/1.3832 post-cooldown).
GDN2_POST_COOLDOWN = BANKED["gdn2"]["post_cooldown"]       # 1.3521734476089478
GDN2_CONSTANT_LR = BANKED["gdn2"]["constant_lr"]           # 1.3950541530336653
GDN2_EMA180 = BANKED["gdn2"]["ema180"]                     # 1.3803383282252721
GDN2_RESCORED_RUN = ROOT / "protocol_correction_runs" / "gdn2_speedrun_rescored"
GDN2_LR_BRACKET_RUNS = ROOT / "gdn2_lr_bracket_runs"
# GDN2's committed arm's parameters (the banked row's own start.json).
GDN2_MACHINE_PARAMETERS = 149_855_670
BANDS = dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
             winner_seed_delta=WINNER_SEED_DELTA)
WIN_BAR = 0.01                     # the standing second-seed / signal bar
# GDN2's banked row's own recipe: the 5x speedrun point (the constant-LR
# probe's selection, the 1.3522 row's own config) -- the hybrid inherits
# it VERBATIM; the ONLY difference from the banked row is the machine.
GDN2_LR_MULTIPLIER = 5
PEAK_LR = GDN2_LR_MULTIPLIER * GDN2_BANKED_ADAMW["lr"]

ARMS = {
    "hybrid": dict(
        kind="hybrid", frozen=False,
        note="the hybrid machine: GDN2's committed arm (the banked 1.3522 "
             "machine) with the composed slot mixer at layers {6, 12, 18} "
             "added to the dense mixer output through the ZERO-BORN read "
             "gate -- born as PURE GDN2 bit for bit, then the gates LEARN "
             "to open (the gate's gradient is the open signal)"),
    "hybrid_frozen": dict(
        kind="hybrid", frozen=True,
        note="THE FALSIFIER: the same machine with the whole slot branch "
             "frozen at the zero-read birth (requires_grad=False on every "
             "slot parameter and the read gates; the branch sits outside "
             "the optimizer, the clip and the EMA by construction) -- it "
             "must be bit-identical to the banked GDN2 row's own "
             "trajectory at every step and land at the banked "
             "1.3521734476089478 post-cooldown bit for bit"),
}

RULES = dict(
    H1=dict(
        text="H-1: the hybrid's post-cooldown row lands BELOW GDN2's "
             "1.3521734476089478 -- the hybrid beats pure GDN2 outright at "
             "matched exposure (the Jamba/Samba/Meta matched-param "
             "minority-interleave pattern).  A margin >= 0.01 is a BANKABLE "
             "win; a sub-band delta gets the parity phrasing (no "
             "band-worship).",
        bar=WIN_BAR, reference=GDN2_POST_COOLDOWN),
    H2=dict(
        text="H-2: the hybrid's post-cooldown row lands at or below the "
             "composed machine's own same-seed row (1.3811104467936925 at "
             "3401 / 1.383232252938407 at 3402) -- the slot store pays "
             "inside a dense stack even though the pure slot machine lost "
             "(the Meta complementarity receipt).",
        bar=0.0),
    REFUTATION=dict(
        text="REFUTATION: if the hybrid lands >= pure GDN2's row, the "
             "literature's minority-interleave gain did NOT transfer to "
             "byte-LM at 150M/20M -- receipt it plainly; the uniformity "
             "finding would then extend to the hybrid regime."),
    second_seed=dict(
        text="the standing rule: any win >= 0.01 vs GDN2's banked row at "
             "seed 3401 owes the second seed 3402.",
        bar=WIN_BAR),
    band_discipline=dict(
        text="the bands on every margin: the 0.075 endpoint-variance band "
             "(cross-machine) and the 0.0119 winner seed delta "
             "(within-machine).  A delta inside the seed band is PARITY, "
             "phrased as parity; only a >= 0.01 win vs GDN2's "
             "1.3521734476089478 is bankable."),
    falsifier=dict(
        text="the frozen falsifier's own bar: bit-identical to the banked "
             "GDN2 trajectory at every one of its 204 steps (max per-step "
             "delta 0.0) and bit-identical landing at "
             "1.3521734476089478 post-cooldown.  ANY divergence is a "
             "birth-parity conflict: STOP and escalate (the falsifier is "
             "non-negotiable)."),
    birth_parity=dict(
        text="both arms are BORN AS GDN2: the step-1 train loss must be "
             "bit-identical to the banked GDN2 row's own same-seed step-1 "
             "(the optimizer cannot touch the first forward); the slot "
             "contribution is exactly zero at birth (the read gate born at "
             "0.0), every other slot parameter's gradient exactly zero, "
             "the gate's own gradient nonzero (the open signal)."),
)


# ------------------------------------------------------------- plumbing


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def k22_bindings() -> dict:
    """Hash-bind every authority and engine this lane runs under."""
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    identity = L.flagship_identity()
    return dict(
        registration=str(ROOT / "PREREGISTRATION_K22_HYBRID.md"),
        registration_sha256=digest(ROOT / "PREREGISTRATION_K22_HYBRID.md"),
        k22_hybrid_sha256=digest(ROOT / "k22_hybrid.py"),
        k22_gate_receipt_sha256=digest(ROOT / "k22_gate_receipt.py"),
        k15_composition_sha256=digest(ROOT / "k15_composition.py"),
        k18_short_conv_sha256=digest(ROOT / "k18_short_conv.py"),
        matched_arms_sha256=digest(ROOT / "matched_arms.py"),
        muon_control_sha256=digest(ROOT / "muon_control.py"),
        protocol_correction_sha256=digest(ROOT / "protocol_correction.py"),
        external_gdn2_sha256=digest(REPO / "ndm/models/external_gdn2.py"),
        lattice_registration_sha256=identity["lattice_sha256"],
        banked_gdn2_row=dict(
            run_dir=str(GDN2_RESCORED_RUN),
            start_sha256=digest(GDN2_RESCORED_RUN / "start.json"),
            result_sha256=digest(GDN2_RESCORED_RUN / "result.json"),
            training_log_sha256=digest(GDN2_RESCORED_RUN / "training.jsonl"),
            post_cooldown=GDN2_POST_COOLDOWN),
    )


def gdn2_banked_trajectory(seed: int) -> list:
    """The banked GDN2 same-seed training log: the rescored row's own
    204-step log at seed 3401; the lr-bracket row's own 180-step
    constant-phase log at seed 3402 (the rescored instrument exists at
    3401 only -- the second-seed falsifier compares the constant phase)."""
    if seed == PRIMARY_SEED:
        path = GDN2_RESCORED_RUN / "training.jsonl"
    else:
        path = GDN2_LR_BRACKET_RUNS / \
            f"gdn2_muon5x_gates_adamw_seed{seed}" / "training.jsonl"
    if not path.exists():
        raise ValueError(f"the banked GDN2 same-seed row is missing: {path}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def gdn2_banked_step1(seed: int) -> float:
    return gdn2_banked_trajectory(seed)[0]["loss"]


def _slot_read_gates(model) -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    return {str(i): float(model.model.blocks[i].layer.gdn2.read_gate.detach())
            for i in hybrid_slot_layers(model)}


# ------------------------------------------------- the at-birth receipts


def hybrid_birth_receipts(model, stream, seed: int, microbatch: int) -> dict:
    """The at-probe-scale birth receipts, BEFORE any training:

    * **born-as-GDN2**: the step-1-parity twin receipt -- one real
      microbatch's loss on the hybrid vs the banked GDN2 row's own
      step-1 train loss is deferred to the training log itself; here the
      receipt is the OPEN SIGNAL and the INERT receipt: the read gates'
      gradients are NONZERO at birth (the open signal exists), while every
      OTHER slot parameter's gradient is EXACTLY ZERO (the slot branch is
      inert at birth, by construction -- the gate's zero cuts the
      gradient), and every read gate is exactly 0.0.
    * **the store runs at birth**: the slot stores fill (APPEND/CORRECT
      fire) on the same microbatch -- the machinery runs, nothing reads
      out.
    """
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch, _k15_walk_stats
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import short_conv1d_causal
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    slot_layers = hybrid_slot_layers(model)
    rows = next(iter(stream.batches("train")))
    x, y, _valid = next(iter(grouped(rows, microbatch)))
    loss = model.loss(x, y)
    loss.backward()
    gates = {}
    inert = {}
    for i in slot_layers:
        wrapper = model.model.blocks[i].layer.gdn2
        gate = wrapper.read_gate
        gates[str(i)] = dict(
            value=float(gate.detach()),
            born_at_exact_zero=bool(gate.detach() == 0.0),
            grad=float(gate.grad.detach()) if gate.grad is not None else None)
        slot_grads = {name: param.grad
                     for name, param in wrapper.slot.named_parameters()
                     if param.grad is not None}
        inert[str(i)] = dict(
            tensors_with_grad=len(slot_grads),
            all_exactly_zero=bool(
                all(bool((g == 0).all()) for g in slot_grads.values())),
            max_abs_grad=float(max(
                (g.abs().max().item() for g in slot_grads.values()),
                default=0.0)))
    model.zero_grad(set_to_none=True)
    # the store-run receipt: the ops fire on the slot layers at birth
    op_rows = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in slot_layers:
            mixer = model.model.blocks[i].layer.gdn2.slot
            xin = model.model.blocks[i].norm(
                model.model.embed(x).detach())
            batch, tokens, _ = xin.shape
            _payload, write_keys, query_keys, _scalar = \
                _project_keyed(mixer, xin)
            payload = short_conv1d_causal(
                mixer.value(xin), mixer.v_conv.weight).view_as(_payload)
            payload, write_keys = payload.float(), write_keys.float()
            query_keys = query_keys.float()
            lam_vec, lam_inc = mixer.channel_decay(xin)
            pol32 = (_round_f32(mixer.policy.tau_match),
                     _round_f32(mixer.policy.tau_merge),
                     _round_f32(mixer.policy.tau_pop),
                     _round_f32(mixer.policy.tie_break),
                     _round_f32(mixer.policy.read_scale))
            state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                mixer.value_dim, dtype=torch.float32,
                                device=xin.device)
            live = torch.zeros(batch, mixer.heads, mixer.rank_max,
                               dtype=torch.bool, device=xin.device)
            keys = torch.zeros(batch, mixer.heads, mixer.rank_max,
                               mixer.key_dim, dtype=torch.float32,
                               device=xin.device)
            out = _k15_forward_launch(
                payload, write_keys, query_keys, lam_vec.float(),
                (1.0 - mixer.write_scalar.view(1, 1, -1, 1)
                 * torch.sigmoid(mixer.b_proj(xin)).view(
                     batch, tokens, mixer.heads, mixer.value_dim)).float(),
                state, live, keys, pol32, mixer.rank, mixer.segment_tokens,
                True, collect=True, warps=8, stages=3)
            stats, _schedule = _k15_walk_stats(
                out, mixer.rank, mixer, payload=payload,
                decay_vec=lam_vec.float())
            op_rows.append((i, stats))
    walk = _agg_walk(op_rows)
    ops_fill = bool(all(
        entry["op_rate_per_token"].get("append", 0.0)
        + entry["op_rate_per_token"].get("correct", 0.0) > 0.0
        for _layer, entry in walk["layers"].items())) \
        if walk.get("per_layer") else False
    return dict(
        probe="k22_birth_receipts",
        seed=seed,
        read_gates=gates,
        open_signal=dict(
            note="dLoss/dgate x slot content at birth: the gate's own "
                 "gradient (the open signal) is nonzero at birth while "
                 "every other slot parameter receives an exactly-zero "
                 "gradient (inert at birth)",
            all_gates_born_at_exact_zero=all(
                g["born_at_exact_zero"] for g in gates.values()),
            any_gate_gradient_nonzero=any(
                (g["grad"] or 0.0) != 0.0 for g in gates.values())),
        inert_at_birth=dict(
            per_layer=inert,
            all_inert=bool(all(v["all_exactly_zero"]
                              for v in inert.values()))),
        store_runs_at_birth=dict(
            note="the slot stores fill at birth (the machinery runs; "
                 "nothing reads out until a gate opens)",
            ops_fill=ops_fill,
            per_layer_op_rate={str(layer): entry["op_rate_per_token"]
                               for layer, entry in walk.get("per_layer",
                                                           {}).items()},
            occupancy={str(layer): entry["occupancy_mean_live_slots"]
                       for layer, entry in walk.get("per_layer", {}).items()}),
    )


# ------------------------------------------------------ the C5 instrument


def k22_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The hybrid's mechanism receipts on the SLOT LAYERS: the K15/K18
    instrument (effective rows, occupancy, the op mix, the identity
    share, the p3 gate telemetry, the conv telemetry) run on the slot
    mixers at their own inputs, PLUS the K22 read-gate rows (the trained
    gates' values -- the mechanism's direct read).  The decorative
    slot-read sweep runs separately (the probe's own sweep receipt)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _probe_windows, _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch, _k15_walk_stats
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import short_conv1d_causal
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    rows = _probe_windows(stream, windows)
    model.eval()
    slot_layers = hybrid_slot_layers(model)
    per_layer = []
    gate_rows = []
    conv_rows = []

    def make_hook(layer_index, mixer):
        pol32 = (_round_f32(mixer.policy.tau_match),
                 _round_f32(mixer.policy.tau_merge),
                 _round_f32(mixer.policy.tau_pop),
                 _round_f32(mixer.policy.tie_break),
                 _round_f32(mixer.policy.read_scale))

        def hook(_module, args, _output):
            xin = args[0]
            batch, tokens, _ = xin.shape
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                _payload, write_keys, query_keys, _scalar = \
                    _project_keyed(mixer, xin)
                payload = short_conv1d_causal(
                    mixer.value(xin), mixer.v_conv.weight).view_as(_payload)
                payload, write_keys = payload.float(), write_keys.float()
                query_keys = query_keys.float()
                lam_vec, lam_inc = mixer.channel_decay(xin)
                drift = (lam_vec.float() - lam_inc.float()
                         .unsqueeze(-1)).abs()
                gate_vec = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.b_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gw_dist = (1.0 - gate_vec.float()).abs()
                r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.g_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gr_dist = (r_vec.float() - 1.0).abs()
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out = _k15_forward_launch(
                    payload, write_keys, query_keys, lam_vec.float(),
                    gate_vec.float(), state, live, keys, pol32, mixer.rank,
                    mixer.segment_tokens, True, collect=True, warps=8,
                    stages=3)
            with torch.no_grad():
                stats, _schedule = _k15_walk_stats(
                    out, mixer.rank, mixer, payload=payload,
                    decay_vec=lam_vec.float())
                per_layer.append((layer_index, stats))
                gate_rows.append(dict(
                    layer=layer_index,
                    decay_gate=dict(
                        mean_distance_from_registered_scalar=float(drift.mean()),
                        max=float(drift.max())),
                    write_gate=dict(
                        mean_distance_from_1=float(gw_dist.float().mean()),
                        max=float(gw_dist.float().max())),
                    read_gate=dict(
                        mean_distance_from_1=float(gr_dist.float().mean()),
                        max=float(gr_dist.float().max()))))
                weight = mixer.v_conv.weight.detach()
                conv_rows.append(dict(
                    layer=layer_index,
                    tap_means=[float(weight[:, j].float().mean())
                               for j in range(weight.shape[1])],
                    tap_stds=[float(weight[:, j].float().std())
                              for j in range(weight.shape[1])],
                    kernel_l2=float(weight.float().norm()),
                    distance_from_identity_birth=float(
                        (weight
                         - torch.tensor(
                             [0.0, 0.0, 0.0, 1.0], device=weight.device)
                         .view(1, -1)).abs().mean()),
                    distance_from_identity_birth_max=float(
                        (weight
                         - torch.tensor(
                             [0.0, 0.0, 0.0, 1.0], device=weight.device)
                         .view(1, -1)).abs().max())))
        return hook

    handles = [model.model.blocks[i].layer.gdn2.slot.register_forward_hook(
        make_hook(i, model.model.blocks[i].layer.gdn2.slot))
        for i in slot_layers]
    started = time.monotonic()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1))
                model.loss(x, _y)
    finally:
        for handle in handles:
            handle.remove()
    wall = time.monotonic() - started
    summary = _agg_walk(per_layer)
    summary["probe"] = dict(
        split="calibration", windows=len(rows), mixers=len(slot_layers),
        slot_layers=list(slot_layers),
        regime="eval, bf16 autocast, the k15 composed persistent walk on "
               "the slot layers' own inputs (collect=True)",
        probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-slot-layer means over the probe windows: the p3 channels' "
             "born-identity drift on the slot mixers",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in gate_rows})
    summary["conv_telemetry"] = dict(
        note="the slot mixers' learned conv kernels (per tap, the distance "
             "from the identity birth)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in conv_rows})
    summary["k22_read_gates"] = _slot_read_gates(model)
    return summary


def k22_slot_read_mask_sweep(model, stream) -> dict:
    """THE HEADLINE INSTRUMENT (the hybrid's decorative sweep): each slot
    layer's READ GATE zeroed for the eval pass, one slot layer at a time,
    then all three -- the load-bearing test.  In our pure machine the
    banked sweep found 4-5 of 24 layers >= 0.02 dNLL (the uniformity
    finding); the hybrid's question is whether the 3 slot layers become
    load-bearing inside the dense stack (does the uniformity finding
    invert?).  The all-slots cell is the trained dense-stack floor (the
    slot reads removed entirely).  Ablation effects, not marginals."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid \
        import hybrid_slot_layers
    os.environ["X40_BATCHED_READS"] = "0"
    slot_layers = hybrid_slot_layers(model)
    base = L.evaluate(model, stream, "gdn2", "calibration")["score"]
    per_layer = {}
    for i in slot_layers:
        wrapper = model.model.blocks[i].layer.gdn2
        saved = float(wrapper.read_gate.detach())
        with torch.no_grad():
            wrapper.read_gate.zero_()
        try:
            nll = L.evaluate(model, stream, "gdn2", "calibration")["score"]
        finally:
            with torch.no_grad():
                wrapper.read_gate.fill_(saved)
        per_layer[str(i)] = nll
        print(f"  slot-read-mask layer {i}: nll {nll:.4f} "
              f"(base {base:.4f}, gate {saved:.6f})", flush=True)
    saved_all = {i: float(model.model.blocks[i].layer.gdn2.read_gate.detach())
                 for i in slot_layers}
    try:
        with torch.no_grad():
            for i in slot_layers:
                model.model.blocks[i].layer.gdn2.read_gate.zero_()
        all_masked = L.evaluate(model, stream, "gdn2", "calibration")["score"]
    finally:
        with torch.no_grad():
            for i, value in saved_all.items():
                model.model.blocks[i].layer.gdn2.read_gate.fill_(value)
    return dict(
        note=("the hybrid's decorative sweep: one slot layer's read gate "
              "zeroed for the eval pass (its slot read removed), one at a "
              "time, plus the all-slots cell (the trained dense-stack "
              "floor).  The uniformity question's direct test: the banked "
              "pure-machine sweep found 4-5 of 24 layers >= 0.02 dNLL -- do "
              "the hybrid's 3 slot layers clear that bar?"),
        base_nll=base, masked_nll_per_slot_layer=per_layer,
        all_slots_masked_nll=all_masked,
        delta_per_slot_layer={k: v - base for k, v in per_layer.items()},
        all_slots_delta=all_masked - base,
        load_bearing_bar=0.02,
        slot_layers_ge_bar=sum(1 for v in per_layer.values()
                               if (v - base) >= 0.02))


# ------------------------------------------------------------ the clause read


def evaluate_clauses(receipt: dict) -> dict:
    """The registration's pre-stated rules, evaluated per seed exactly as
    stated: H-1 (the bankable-win bar vs GDN2), H-2 (vs the composed
    machine's own same-seed row), the REFUTATION clause, the falsifier's
    own bar, the second-seed rule, and the band discipline on every
    margin."""
    post = receipt.get("final_eval_nll_post_cooldown")
    seed = receipt["seed"]
    arm = receipt["arm"]
    composed = composed_4x_banked()["post_cooldown"][seed]
    out = dict(rules=RULES, bands=BANDS,
               same_seed_composed_post_cooldown=composed,
               vs_gdn2_post_cooldown=None, gdn2_contact_flag=None)
    if post is None:
        return out
    out["vs_gdn2_post_cooldown"] = post - GDN2_POST_COOLDOWN
    out["gdn2_contact_flag"] = bool(post <= GDN2_POST_COOLDOWN)
    out["improvement_vs_gdn2"] = GDN2_POST_COOLDOWN - post
    out["improvement_vs_composed_ours"] = composed - post
    out["within_winner_seed_band_of_gdn2"] = bool(
        abs(post - GDN2_POST_COOLDOWN) < WINNER_SEED_DELTA)
    if arm == "hybrid_frozen":
        div = receipt.get("falsifier_divergence") or {}
        landed = bool(receipt.get("falsifier_landing_bit_identical"))
        out["clause_at_this_seed"] = (
            "FALSIFIER_PASSED" if (div.get("bit_identical_steps")
                                   == div.get("steps_compared")
                                   and div.get("steps_compared", 0) > 0
                                   and landed)
            else "FALSIFIER_DIVERGED -- STOP AND ESCALATE")
        out["falsifier"] = dict(
            steps_compared=div.get("steps_compared"),
            bit_identical_steps=div.get("bit_identical_steps"),
            max_abs_loss_delta=div.get("max_abs_loss_delta"),
            landing_bit_identical=landed,
            expected_post_cooldown=GDN2_POST_COOLDOWN)
        return out
    margin = GDN2_POST_COOLDOWN - post
    if margin >= WIN_BAR:
        out["clause_at_this_seed"] = "H-1_WIN_at_this_seed (>= 0.01 vs GDN2)"
        out["second_seed_owed"] = True
    elif post >= GDN2_POST_COOLDOWN:
        out["clause_at_this_seed"] = (
            "REFUTED_at_this_seed: the hybrid lands >= pure GDN2's row -- "
            "the literature's minority-interleave gain did NOT transfer to "
            "byte-LM at 150M/20M; receipt it plainly; the uniformity "
            "finding extends to the hybrid regime")
        out["second_seed_owed"] = False
    else:
        # unreachable by ordering (any margin < the win bar is inside the
        # seed band), kept explicit so the branch cannot silently change
        out["clause_at_this_seed"] = (
            "H-1_PARITY_at_this_seed (inside the 0.0119 seed band of "
            "GDN2's row: contact without a bankable margin; the parity "
            "phrasing -- no band-worship)")
        out["second_seed_owed"] = False
    out["H2_held"] = bool(post <= composed)
    out["H2_margin"] = composed - post
    return out


# --------------------------------------------------------------- the arm


def run_arm(args) -> dict:
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import HybridMuonX45
    from experiments.expressivity_tasks.x45_matrix_emender.k22_hybrid import (
        SLOT_LAYERS, _routing_cross_check_k22, convert_gdn2_to_hybrid,
        freeze_slot_branch, hybrid_slot_parameter_delta, hybrid_slot_layers,
        slot_mixer_parameter_count, split_parameters_k22)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        protocol_correction import _train_corrected

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r} (expected one of "
                         f"{sorted(ARMS)})")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed!r} (the arms run "
                         f"seed {PRIMARY_SEED} first; the >= 0.01 win-bar "
                         f"rule owes {SECOND_SEED})")
    arm = ARMS[args.arm]
    seed = args.seed
    frozen = bool(arm["frozen"])
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()

    # ---- the machine: GDN2's committed build VERBATIM + the K22 graft.
    # The GDN2 build's RNG consumption order is the banked row's own
    # (build first, graft after; the slot mixers build under a forked RNG).
    model, optimizer_config, summary = matched_arms.build_gdn2(seed)
    for key, want in GDN2_BANKED_ADAMW.items():
        actual = (tuple(optimizer_config[key]) if key == "betas"
                  else optimizer_config[key])
        if actual != want:
            raise ValueError(f"the GDN2 machine's config drifted on {key}: "
                             f"{actual!r} != the banked {want!r}")
    width = int(model.model.width)
    parameters_dense = sum(p.numel() for p in model.parameters())
    if parameters_dense != GDN2_MACHINE_PARAMETERS:
        raise ValueError(f"the GDN2 machine's parameters must be the banked "
                         f"row's {GDN2_MACHINE_PARAMETERS}; got "
                         f"{parameters_dense}")
    model = convert_gdn2_to_hybrid(model)
    model = model.train()
    slot_delta = hybrid_slot_parameter_delta(width, SLOT_LAYERS)
    parameters_total = sum(p.numel() for p in model.parameters())
    if parameters_total != GDN2_MACHINE_PARAMETERS + slot_delta:
        raise ValueError(f"the hybrid's parameters must be the banked "
                         f"{GDN2_MACHINE_PARAMETERS} + the registration's "
                         f"{slot_delta}; got {parameters_total}")
    if tuple(hybrid_slot_layers(model)) != tuple(SLOT_LAYERS):
        raise ValueError("the slot layers drifted from the registration's")
    # the falsifier's freeze: the whole slot branch leaves the optimizer,
    # the clip and the EMA by construction
    frozen_tensors = freeze_slot_branch(model) if frozen else 0
    if frozen:
        assert all(not p.requires_grad
                   for i in SLOT_LAYERS
                   for p in model.model.blocks[i].layer.gdn2.slot.parameters())
        assert all(not model.model.blocks[i].layer.gdn2.read_gate.requires_grad
                   for i in SLOT_LAYERS)

    # ---- the optimizer: the banked GDN2 row's own recipe VERBATIM (the
    # 5x speedrun point, option B), the slot branch routed per the K18
    # convention mirrored
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_k22(model, frozen)
    routing_receipt = _routing_cross_check_k22(model, seed)
    optimizer = HybridMuonX45(
        muon_params, adamw_params, lr=PEAK_LR, betas=tuple(
            optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"])
    optimizer_block = dict(
        optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
        peak_lr=PEAK_LR, lr_multiplier=GDN2_LR_MULTIPLIER,
        lr_convention=(f"the banked GDN2 row's own {GDN2_LR_MULTIPLIER}x "
                       f"speedrun point ({GDN2_LR_MULTIPLIER}x the machine's "
                       f"tuned AdamW LR {GDN2_BANKED_ADAMW['lr']!r}) -- the "
                       "1.3522 row's own config, inherited VERBATIM; the "
                       "ONLY difference from the banked row is the machine"),
        betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"],
        warmup_steps=optimizer_config["warmup_steps"],
        microbatch=optimizer_config["microbatch"], grad_clip=GRAD_CLIP,
        muon_params=f"{len(muon_names)} tensors",
        adamw_params=f"{len(adamw_names)} tensors",
        inert_while_closed_note="while a read gate is exactly zero, every "
                                "other slot parameter's gradient is exactly "
                                "zero (inert at birth); decoupled weight "
                                "decay still shrinks the inert slot "
                                "parameters each step (the recipe's own "
                                "dynamics, stated, not patched)")

    machine_note = (
        "the hybrid: GDN2's committed arm (matched_arms.build_gdn2 "
        "verbatim, the banked 1.3522 machine, width "
        f"{width}, 24 dense mixers) with the composed slot mixer (the "
        "K15/K18 engine VERBATIM: the K13 additive read, the K11B p3 "
        "gates, the K10 separated keys at the mimetic birth, the K18 "
        "identity-born conv) at the slot layers "
        f"{tuple(SLOT_LAYERS)}, added to the dense mixer's output "
        "through the ZERO-BORN read gate -- born as PURE GDN2 bit for "
        "bit; the slot store machinery runs from step 1, nothing reads "
        "out until a gate opens"
        + ("; THE FALSIFIER: the whole slot branch frozen at birth"
           if frozen else "; the gates LEARN to open"))

    run_config = dict(
        probe="k22_probe_hybrid", arm=args.arm, seed=seed,
        arm_note=arm["note"], machine=machine_note, frozen=frozen,
        corrected_scoring=CORRECTED_SCORING, rules=RULES,
        exposure_targets=int(args.bytes),
        model_build="matched_arms.build_gdn2 verbatim (the registered "
                    "lattice_gdn2 arm, its searched nomination's knobs, "
                    "its own RNG order) + k22_hybrid.convert_gdn2_to_hybrid "
                    "(the wrapper HOLDS the committed GDN2ExternalLayer; "
                    "the composed slot mixer built under a forked RNG at "
                    "the host width; the read gates born at exact 0.0)",
        placement=dict(slot_layers=list(SLOT_LAYERS),
                       rationale="mid-stack spread (the operator's "
                                "architecture go): Memorizing Transformers' "
                                "mid-stack optimum for a memory layer in a "
                                "dense stack (Wu et al., arXiv:2203.08913), "
                                "even 6-spacing (the Jamba/Samba "
                                "minority-interleave pattern at 1:7), our "
                                "own decorative sweep's load-bearing mass "
                                "in the first half (layers 0-6 hold the "
                                "top deltas) -- the earliest mid-stack "
                                "spread covering the decline"),
        store_geometry=dict(heads=2, value_dim=128, rank=128, rank_max=128,
                            key_dim=16, ctx=4, segment_tokens=64,
                            policy="the K19-confirmed incumbent thresholds "
                                   "(tau_match 0.45, tau_merge 0.85, "
                                   "tau_pop 0.10, read_scale 2.787)"),
        independent_stores=True,
        shared_pool_follow_up="named follow-up, NOT this lane (independent "
                              "stores per slot layer first)",
        optimizer=optimizer_block,
        routing="the banked GDN2 row's own option-B split (verified equal "
                "against a pure-GDN2 twin's own split on the normalized "
                "dense names) + the slot branch per the K18 convention "
                "mirrored: matrices on Muon, the gate channels + the conv "
                "kernel + the K22 read gate on AdamW; the frozen "
                "falsifier's slot branch excluded by requires_grad",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(dense=GDN2_MACHINE_PARAMETERS,
                        slot_added=slot_delta,
                        slot_added_percent=round(
                            100.0 * slot_delta / GDN2_MACHINE_PARAMETERS, 3),
                        slot_mixer_parameters=slot_mixer_parameter_count(width),
                        read_gates=len(SLOT_LAYERS),
                        total=parameters_total,
                        note="purely ADDITIVE (the falsifier's own "
                             "construction: the dense mixers stay, "
                             "bit-identical at birth); nothing removed; "
                             "the exact delta quoted per the registration"),
        checkpoint_steps=list(args.checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear cooldown "
                       "(the corrected-scoring extension, same stream); the "
                       "step-180 constant-LR eval is retained for banked "
                       "comparability"),
        velocity_note="the K20/K21 receipts: the hybrid step is projected "
                      "1.37x pure-GDN2 (3-of-24 at the measured 7.28x "
                      "per-mixer ratio; the checkpointed block re-runs the "
                      "slot scan in backward, so the honest wall is "
                      "reported first-class) -- matched-EXPOSURE means the "
                      "arms eat the SAME bytes in different wall-clocks",
        both_bands=BANDS,
        anchors=dict(banked_gdn2=dict(
                        post_cooldown=GDN2_POST_COOLDOWN,
                        constant_lr=GDN2_CONSTANT_LR, ema180=GDN2_EMA180,
                        run_dir=str(GDN2_RESCORED_RUN)),
                     composed_ours=composed_4x_banked(),
                     banked_mechanism_ours=_banked_mechanism_anchors()),
        bindings=k22_bindings(),
        summary=summary, identity=identity, versions=_versions(),
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
    print(json.dumps(dict(arm=args.arm, card=card, seed=seed, frozen=frozen,
                          slot_layers=list(SLOT_LAYERS), width=width,
                          parameters=parameters_total,
                          slot_parameters_added=slot_delta,
                          peak_lr=PEAK_LR)), flush=True)

    # ---- the at-birth receipts, BEFORE any training
    birth_receipt = hybrid_birth_receipts(
        model, stream, seed, optimizer_config["microbatch"])
    run_config["birth_receipts"] = birth_receipt
    if not birth_receipt["open_signal"]["all_gates_born_at_exact_zero"] \
            or not birth_receipt["inert_at_birth"]["all_inert"]:
        raise ValueError("the BORN-AS-GDN2 receipts failed at probe scale "
                         "-- escalate, not excuse (the birth discipline is "
                         "non-negotiable)")
    print("  born-as-GDN2: gates exact-zero "
          f"{birth_receipt['open_signal']['all_gates_born_at_exact_zero']}; "
          "open signal "
          f"{birth_receipt['open_signal']['any_gate_gradient_nonzero']}; "
          "slot inert "
          f"{birth_receipt['inert_at_birth']['all_inert']}; store fills "
          f"{birth_receipt['store_runs_at_birth']['ops_fill']}", flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, PEAK_LR, optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"{args.arm}_seed{seed}", out / "training.jsonl",
        tuple(args.checkpoint_steps), cooldown_steps, EMA_DECAY)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # ---- the step-1 parity receipt (the protocol-identical start: the
    # optimizer cannot touch the first forward, so step 1 must reproduce
    # the banked GDN2 row's own step-1 train loss bit for bit)
    step1 = run["rows"][0]["loss"]
    banked_step1 = gdn2_banked_step1(seed)
    print(f"  step-1 train loss {step1!r} (the banked GDN2 row's own "
          f"{banked_step1!r})", flush=True)
    if not args.smoke and step1 != banked_step1:
        raise ValueError(
            f"step-1 train loss {step1!r} != the banked GDN2 same-seed "
            f"row's {banked_step1!r} -- the arm's machine is not the "
            f"committed GDN2 machine byte for byte at birth (escalate, "
            f"not excuse)")

    # ---- the falsifier's honest divergence receipt: the per-step loss
    # divergence vs the banked GDN2 row's own training log (bit-identity
    # is the expectation to beat: the K18 perfect full-trajectory receipt)
    falsifier_divergence = None
    if frozen:
        banked_rows = gdn2_banked_trajectory(seed) if not args.smoke else []
        divs = []
        for r, b in zip(run["rows"], banked_rows):
            if r["step"] != b["step"]:
                break
            divs.append(dict(step=r["step"],
                             loss_delta=r["loss"] - b["loss"],
                             bit_identical=bool(r["loss"] == b["loss"])))
        if divs:
            first_div = next((d for d in divs
                              if not d["bit_identical"]), None)
            falsifier_divergence = dict(
                steps_compared=len(divs),
                bit_identical_steps=sum(1 for d in divs if d["bit_identical"]),
                first_diverging_step=(first_div or {}).get("step"),
                max_abs_loss_delta=max(abs(d["loss_delta"]) for d in divs),
                reference=str(GDN2_RESCORED_RUN / "training.jsonl")
                         if seed == PRIMARY_SEED else
                         str(GDN2_LR_BRACKET_RUNS /
                             f"gdn2_muon5x_gates_adamw_seed{seed}" /
                             "training.jsonl"),
                note="the frozen hybrid vs the banked GDN2 row's own "
                     "training log: the K18 perfect full-trajectory "
                     "receipt is the expectation to beat; ANY divergence "
                     "is a birth-parity conflict (STOP and escalate)")

    # ---- the scoring pass (restored snapshots; training is over)
    scores = []
    gate_trajectory = {}
    for step in args.checkpoint_steps:            # the constant-LR curve
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        gate_trajectory[str(step)] = _slot_read_gates(model)
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "gdn2", args.eval_split)
        row = dict(kind="constant", step=step,
                   bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"],
                   eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        scores.append(row)
        print(f"  constant step {step}: eval {row['eval_nll']:.4f} "
              f"gates {gate_trajectory[str(step)]}", flush=True)
    if run["snapshots"] and run["ema"] is not None and \
            run["constant_steps"] in run["snapshots"]:
        model.load_state_dict(run["snapshots"][run["constant_steps"]])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(run["ema"][name].to(param.dtype))
        gate_trajectory["ema_180"] = _slot_read_gates(model)
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "gdn2", args.eval_split)
        scores.append(dict(kind="ema_180", step=run["constant_steps"],
                           bytes=rows_by_step[run["constant_steps"]]["bytes"],
                           eval_nll=result["score"],
                           eval_windows=result["eval_windows"],
                           eval_targets=result["eval_targets"],
                           ema_decay=EMA_DECAY,
                           note="ranking-only: biased toward mid-run weights"))
        print(f"  EMA at step {run['constant_steps']}: "
              f"eval {result['score']:.4f}", flush=True)
    falsifier_landing_bit_identical = None
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        gate_trajectory["final"] = _slot_read_gates(model)
        gate_trajectory["init"] = {str(i): 0.0 for i in SLOT_LAYERS}
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "gdn2", args.eval_split)
        scores.append(dict(kind="post_cooldown", step=run["step"],
                           bytes=run["seen"], eval_nll=result["score"],
                           eval_windows=result["eval_windows"],
                           eval_targets=result["eval_targets"]))
        print(f"  post-cooldown final (step {run['step']}): "
              f"eval {result['score']:.4f}", flush=True)
        if frozen and not args.smoke:
            falsifier_landing_bit_identical = bool(
                result["score"] == GDN2_POST_COOLDOWN)
            if seed == PRIMARY_SEED and not falsifier_landing_bit_identical:
                raise ValueError(
                    f"the FROZEN falsifier landed at {result['score']!r} != "
                    f"the banked {GDN2_POST_COOLDOWN!r} -- a birth-parity "
                    f"conflict (STOP and escalate, the falsifier is "
                    f"non-negotiable)")
    with (out / "scores.jsonl").open("w", buffering=1) as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    # ---- the falsifier's trajectory bar (non-smoke): every compared step
    # must be bit-identical, or the lane stops
    if frozen and falsifier_divergence is not None:
        if falsifier_divergence["bit_identical_steps"] \
                != falsifier_divergence["steps_compared"]:
            raise ValueError(
                "the FROZEN falsifier diverged from the banked GDN2 "
                f"trajectory at step "
                f"{falsifier_divergence['first_diverging_step']} -- a "
                f"birth-parity conflict (STOP and escalate, the falsifier "
                f"is non-negotiable)")

    # ---- the mechanism receipts at BOTH states + the decorative sweep
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
    if run["status"] == "completed":
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            diagnostics_180 = k22_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_180 = k22_slot_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = k22_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_final = k22_slot_read_mask_sweep(model, stream)

    constant_final = next((r["eval_nll"] for r in scores
                           if r["kind"] == "constant"
                           and r["step"] == run["constant_steps"]), None)
    post_cooldown = next((r["eval_nll"] for r in scores
                          if r["kind"] == "post_cooldown"), None)
    ema_final = next((r["eval_nll"] for r in scores if r["kind"] == "ema_180"),
                     None)
    receipt = dict(
        probe="k22_probe_hybrid", arm=args.arm, seed=seed,
        arm_note=arm["note"], machine=machine_note, frozen=frozen,
        corrected_scoring=CORRECTED_SCORING, status=run["status"],
        slot_layers=list(SLOT_LAYERS), width=width,
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
        parameters=parameters_total,
        parameters_dense=GDN2_MACHINE_PARAMETERS,
        parameters_slot_added=slot_delta,
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=step1,
        step1_receipt=dict(
            arm_step1=step1, banked_gdn2_same_seed_step1=banked_step1,
            parity=bool(args.smoke or step1 == banked_step1),
            rule=RULES["birth_parity"]["text"]),
        birth_receipts=birth_receipt,
        falsifier_divergence=falsifier_divergence,
        falsifier_landing_bit_identical=falsifier_landing_bit_identical,
        gate_trajectory=gate_trajectory,
        scores=scores,
        anchors=dict(banked_gdn2=dict(
                        post_cooldown=GDN2_POST_COOLDOWN,
                        constant_lr=GDN2_CONSTANT_LR, ema180=GDN2_EMA180),
                     composed_ours=composed_4x_banked(),
                     banked_mechanism_ours=_banked_mechanism_anchors()),
        diagnostics_step180=diagnostics_180,
        diagnostics_post_cooldown=diagnostics_final,
        slot_read_mask_sweep_step180=sweep_180,
        slot_read_mask_sweep_post_cooldown=sweep_final,
        routing_receipt=routing_receipt,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        config=run_config, session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            run["rows"], keep_first=min(5, max(1, len(run["rows"])))))
    receipt["clauses"] = evaluate_clauses(receipt)
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
    sweep = r.get("slot_read_mask_sweep_post_cooldown") or {}
    gates = r.get("gate_trajectory") or {}
    return dict(
        arm=r["arm"], seed=r["seed"], machine=r["machine"], frozen=r["frozen"],
        status=r["status"], post_cooldown=r["final_eval_nll_post_cooldown"],
        constant_lr=r["final_eval_nll_constant_lr"],
        ema_180=r["ema_eval_nll"], step1_train_loss=r["step1_train_loss"],
        step1_parity=r["step1_receipt"]["parity"],
        train_seconds_constant=r["train_seconds_constant"],
        parameters=r["parameters"],
        parameters_slot_added=r["parameters_slot_added"],
        clause=r["clauses"].get("clause_at_this_seed"),
        vs_gdn2_post_cooldown=r["clauses"].get("vs_gdn2_post_cooldown"),
        gdn2_contact_flag=r["clauses"].get("gdn2_contact_flag"),
        second_seed_owed=r["clauses"].get("second_seed_owed"),
        effective_rows=diag.get("effective_read_rows"),
        occupancy=(diag.get("occupancy") or {}).get("mean_live_slots"),
        identity_share=diag.get("identity_rewrite_share"),
        op_mix={str(layer): entry.get("op_rate_per_token")
                for layer, entry in (diag.get("per_layer") or {}).items()},
        gates=dict(init=gates.get("init"), final=gates.get("final"),
                   ema_180=gates.get("ema_180")),
        slot_read_mask_sweep=dict(
            base_nll=sweep.get("base_nll"),
            delta_per_slot_layer=sweep.get("delta_per_slot_layer"),
            all_slots_delta=sweep.get("all_slots_delta"),
            slot_layers_ge_bar=sweep.get("slot_layers_ge_bar")),
        falsifier=dict(
            divergence=r.get("falsifier_divergence"),
            landing_bit_identical=r.get("falsifier_landing_bit_identical")))


def assemble(args) -> dict:
    """The arms' table, the pre-stated clauses applied exactly as stated,
    both parents' banked rows quoted, the mechanism comparison (the
    headline: the slot layers' load-bearing sweep), and the escalation
    flags.  The final gap-quote phrasing is the supervisor's to adjudicate
    (the K13 clause-conflict precedent): this assemble states the measured
    rows and the flags, not the final wording."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    rows = {}
    for label in ("hybrid", "hybrid_frozen"):
        for seed in (PRIMARY_SEED, SECOND_SEED):
            row = _arm_row(Path(args.runs) / f"{label}_seed{seed}")
            if row is not None:
                rows[f"{label}_seed{seed}"] = row
    hybrid_1 = rows.get(f"hybrid_seed{PRIMARY_SEED}")
    frozen_1 = rows.get(f"hybrid_frozen_seed{PRIMARY_SEED}")
    verdict = dict(
        pre_state=RULES,
        banked_parents=dict(
            gdn2=dict(post_cooldown=GDN2_POST_COOLDOWN,
                      constant_lr=GDN2_CONSTANT_LR, ema180=GDN2_EMA180,
                      parameters=GDN2_MACHINE_PARAMETERS,
                      run_dir=str(GDN2_RESCORED_RUN)),
            composed_ours=dict(
                post_cooldown={
                    str(PRIMARY_SEED):
                        composed_4x_banked()["post_cooldown"][PRIMARY_SEED],
                    str(SECOND_SEED):
                        composed_4x_banked()["post_cooldown"][SECOND_SEED]},
                note="the composed machine's banked 4x rows (K15 assemble "
                     "receipt)")),
        arms=rows)
    escalation_flags = []
    if frozen_1 is not None:
        div = (frozen_1["falsifier"]["divergence"] or {})
        landed = frozen_1["falsifier"]["landing_bit_identical"]
        if not (div.get("bit_identical_steps") == div.get("steps_compared")
                and div.get("steps_compared", 0) > 0 and landed):
            escalation_flags.append(
                "FALSIFIER_DIVERGED: the frozen hybrid is not bit-identical "
                "to the banked GDN2 trajectory -- a birth-parity conflict "
                "(the falsifier is non-negotiable)")
        else:
            verdict["falsifier"] = dict(
                clause="PASSED PERFECTLY" if landed else "TRAJECTORY-ONLY",
                steps_compared=div.get("steps_compared"),
                max_abs_loss_delta=div.get("max_abs_loss_delta"),
                landing_bit_identical=landed)
    if hybrid_1 is not None:
        post = hybrid_1["post_cooldown"]
        verdict["H1_margin_vs_gdn2"] = GDN2_POST_COOLDOWN - post
        verdict["H2_margin_vs_composed_ours"] = \
            composed_4x_banked()["post_cooldown"][PRIMARY_SEED] - post
        verdict["H1"] = ("WIN (bankable)" if GDN2_POST_COOLDOWN - post
                         >= WIN_BAR else
                         "REFUTED (the minority-interleave gain did not "
                         "transfer; the uniformity finding extends to the "
                         "hybrid regime)" if post >= GDN2_POST_COOLDOWN else
                         "CONTACT/PARITY (sub-band: the parity phrasing)")
        verdict["H2"] = ("HELD" if post
                         <= composed_4x_banked()["post_cooldown"][PRIMARY_SEED]
                         else "NOT HELD")
        verdict["second_seed_owed"] = bool(
            GDN2_POST_COOLDOWN - post >= WIN_BAR)
        sweep = hybrid_1["slot_read_mask_sweep"]
        verdict["headline_instrument"] = dict(
            note="the decorative sweep, the headline: do the slot layers "
                 "become load-bearing in the hybrid (does the uniformity "
                 "finding invert -- the pure machine's own banked regime "
                 "is 4-5 of 24 layers >= 0.02 dNLL)?",
            slot_layers_ge_bar=sweep.get("slot_layers_ge_bar"),
            delta_per_slot_layer=sweep.get("delta_per_slot_layer"),
            all_slots_delta=sweep.get("all_slots_delta"),
            gates_final=(hybrid_1["gates"] or {}).get("final"))
    verdict["escalation_flags"] = escalation_flags
    verdict["bands"] = BANDS
    path = Path(args.runs) / "assemble.json"
    atomic_json(path, verdict)
    print(json.dumps({k: verdict[k] for k in
                      ("arms", "escalation_flags")} | {
                          k: verdict[k] for k in
                          ("H1", "H2", "falsifier", "headline_instrument")
                          if k in verdict}, indent=1, default=str), flush=True)
    return verdict


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run_arm")
    r.add_argument("--arm", required=True, choices=sorted(ARMS))
    r.add_argument("--seed", type=int, default=PRIMARY_SEED)
    r.add_argument("--bytes", type=float, default=EXPOSURE_TARGETS)
    r.add_argument("--output", required=True)
    r.add_argument("--eval-split", default="calibration")
    r.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    r.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    r.add_argument("--smoke", action="store_true",
                   help="the pipeline shakedown: tiny exposure, 3-step "
                        "cooldown, no parity assertions")
    a = sub.add_parser("assemble")
    a.add_argument("--runs", default=str(RUNS))
    args = ap.parse_args()
    if args.cmd == "run_arm":
        run_arm(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
