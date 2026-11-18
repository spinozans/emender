"""K11 -- the learned-gates quality probe (gate C): the G1/G2/G3 arms at the
matched-exposure protocol under the CORRECTED scoring, vs the banked winner
config (hybrid Muon 4x lineage).

THE QUESTION THIS STAGE EXISTS FOR (``PREREGISTRATION_LEARNED_GATES.md``
claims S1/S2): at the matched-exposure probe protocol (the 20,022,866-target
landing, 180 constant-LR steps), do the learned-gate arms beat the banked
winner config, and do the pre-registered MECHANISM predictions M-1..M-5
(committed before any probe measurement) land against the measured anchors?

STANDING AMENDMENTS applied (they landed after the K11 registration was
written; see ``RESULT_PROTOCOL_CORRECTION.md`` / ``protocol_correction.py``):
every arm is scored under the corrected protocol -- the 24-step linear
cooldown PRIMARY, the EMA-0.99 iterate at step 180 SECONDARY (ranking-only),
and the constant-LR step-180 eval RETAINED for banked comparability.  The
comparison anchors, both quoted: the constant-LR banked best
**1.4899048124040877** (muon4x_banked_seed3402; the same config's seed-3401
row is 1.4907409463609969) and the corrected post-cooldown banked best
**1.4338248457227434** (winner_speedrun_rescored, seed 3402).  The arms run
the banked winner config VERBATIM (hybrid Muon at 4x the tuned AdamW LR, the
banked bake-off routing with the gate projections on the AdamW side -- the
DECAY gate's own convention, receipted) plus the arm's gate channels at the
mimetic-neutral birth, so an arm's step-1 train loss must be BIT-IDENTICAL
to the banked same-seed row's (the 0-flip-at-birth receipt at probe scale).

The arms (the registration's 1): ``g1`` (write-strength + read-out gates),
``g2`` (the dense gated write channel), ``g3`` (combined), primary seed 3401,
plus the second seed (3402) on the best arm (the post-cooldown primary score
decides "best").  The mechanism instrument (the registration's 4): the K10
probe's own walk stats + identity-rewrite share, measured at BOTH the
step-180 constant snapshot (the banked-comparable state) and the
post-cooldown final (the primary state), plus the read-mask sweep (the
side-by-side's C4 diagnostic, imported verbatim) and the gate telemetry
(per-layer write-gate distance from 1, dense-deposit magnitude, read-out
gate -- the learned admission controller's own readout).

Provenance (the lane's own rules): the engine is committed and read-only
(``k11_learned_gates.py``); ``velocity_ladder._build_k10``,
``protocol_correction._train_corrected``/``CORRECTED_SCORING``,
``routing_symmetry.split_parameters_winner``, ``k10_probe_separated_keys``
instruments, ``side_by_side.k10_read_mask_sweep`` and ``matched_arms``
helpers are imported verbatim; no ``k10_*``/``k12_*`` source, no
PREREGISTRATION, no ``lattice.py`` is edited.

Deliverables: per-arm receipts under ``k11_runs/<arm>_seed<seed>/`` and the
assemble summary ``receipts/k11_learned_gates_summary.json`` (the
corrected-ladder row for the best arm included).

Usage (see ``run_k11_learned_gates_leased.sh`` for the detached leased
driver, cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k11_probe_learned_gates \\
        arm --arm g1 --seed 3401 --output k11_runs/g1_seed3401
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
RUNS = ROOT / "k11_runs"
ROUTING_SYMMETRY_RUNS = ROOT / "routing_symmetry_runs"
PROTOCOL_CORRECTION_RUNS = ROOT / "protocol_correction_runs"

EXPOSURE_TARGETS = 20_000_000          # the 20,022,866 landing at step 180
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
PROBE_WINDOWS = 4
LR_MULTIPLIER = 4                      # the banked winner lineage (muon4x)
PRIMARY_SEED = 3401
SECOND_SEED = 3402

# The corrected scoring, imported verbatim (the standing amendment).
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction \
    import (COOLDOWN_STEPS, CORRECTED_SCORING, EMA_DECAY,
            ENDPOINT_VARIANCE_BAND, WINNER_SEED_DELTA, _train_corrected)  # noqa: E402

# The banked comparison anchors (cited, never re-run here; every one read
# from the banked run's own receipts, not retyped from memory where a file
# exists -- the step-1 losses and the mechanism anchors are loaded from the
# banked result.json files at run time).
BANKED = dict(
    constant_lr=dict(
        seed3401=1.4907409463609969, seed3402=1.4899048124040877,
        note="the banked winner config's own constant-LR step-180 rows "
             "(routing_symmetry_runs/muon4x_banked_seed{3401,3402})"),
    post_cooldown=dict(
        seed3402=1.4338248457227434,
        note="the corrected post-cooldown banked best "
             "(protocol_correction_runs/winner_speedrun_rescored, seed "
             "3402 -- the banked best row's own config re-scored)"),
    ema180=dict(
        seed3402=1.5408323151724679,
        note="the secondary EMA-0.99 iterate at step 180, banked "
             "(ranking-only instrument)"),
    gdn2=dict(constant_lr=1.3950541530336653, post_cooldown=1.3521734476089478,
              ema180=1.3803383282252721,
              note="the banked GDN2 best rows (the cross-machine context; "
                   "the ~0.09 residual this stage dissects)"),
)
# The registration's own pre-stated mechanism anchors (section 4, the
# bake-off muon3x-era rows) and the side-by-side's AdamW-era sweep anchors,
# both quoted WITH their era caveats; the fresher same-config anchors are
# loaded from the banked result.json files at run time.
REGISTRATION_ANCHORS = dict(
    identity_rewrite_share=0.123, read_entropy_nats=1.664,
    effective_rows_side_by_side=5.6,       # e^1.73, RESULT_SIDE_BY_SIDE_0239 5
    sweep_layers_ge_threshold=7, sweep_near_decorative=17, sweep_total=24,
    sweep_threshold=0.02, sweep_near_band=0.0125,
    m2_drop_bar=0.03, m3_entropy_bar=1.8,
    caveat="M-2/M-3 anchored one surface back (the bake-off muon3x rows); "
           "M-4 anchored TWO surfaces back (the side-by-side's AdamW-era "
           "read-mask sweep -- the muon4x-era sweep was never banked). The "
           "same-config seed-matched anchors are receipted alongside.",
)

ARMS = {
    "g1": dict(channels=dict(write=True, readout=True, dense=False),
               note="the minimal learned-gate arm: the write-strength gate "
                    "(APPEND admits g*p, CORRECT blends "
                    "v+g*(p-v) on the PRE-decay row) plus the read-out gate "
                    "(reads' *= g_out before the out projection)"),
    "g2": dict(channels=dict(write=False, readout=False, dense=True),
               note="the dense-write arm: every token deposits "
                    "m_d*sigmoid(w_d*x+b_d)*p on its BEST-MATCH LIVE row "
                    "(P3), as schedule lane 2; isolates the dense channel"),
    "g3": dict(channels=dict(write=True, readout=True, dense=True),
               note="combined: write-strength gate + read-out gate + the "
                    "dense channel"),
}

# ------------------------------------------------------------------ the anchors


def _banked_result(seed: int) -> dict:
    path = ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{seed}" / "result.json"
    if not path.exists():
        raise ValueError(f"the banked same-seed row is missing: {path}")
    return json.loads(path.read_text())


def _banked_step1_loss(seed: int) -> float:
    path = ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{seed}" \
        / "training.jsonl"
    if not path.exists():
        raise ValueError(f"the banked same-seed training log is missing: "
                         f"{path}")
    return json.loads(path.read_text().splitlines()[0])["loss"]


def _banked_mechanism_anchors() -> dict:
    """The freshest same-config mechanism anchors, loaded from the banked
    receipts (both seeds' step-180 state, plus the banked post-cooldown
    state from the protocol-correction re-score)."""
    out = {}
    for seed in (PRIMARY_SEED, SECOND_SEED):
        r = _banked_result(seed)
        d = r["diagnostics"]
        out[f"seed{seed}_step180"] = dict(
            identity_rewrite_share=d["identity_rewrite_share"],
            read_entropy_nats=d["read_weight_entropy"]["mean"],
            occupancy_mean_live_slots=d["occupancy"]["mean_live_slots"],
            op_rate_per_token=d["op_rate_per_token_lane0"],
            constant_lr_nll=r["final_eval_nll"])
    rescored = json.loads((PROTOCOL_CORRECTION_RUNS
                          / "winner_speedrun_rescored" / "result.json")
                          .read_text())
    d = rescored["diagnostics"]
    out["seed3402_post_cooldown"] = dict(
        identity_rewrite_share=d["identity_rewrite_share"],
        read_entropy_nats=d["read_weight_entropy"]["mean"],
        occupancy_mean_live_slots=d["occupancy"]["mean_live_slots"],
        op_rate_per_token=d["op_rate_per_token_lane0"],
        post_cooldown_nll=rescored["final_eval_nll_post_cooldown"])
    return out

# ----------------------------------------------------------- the routing split


def split_parameters_gated(model, arm: str):
    """The banked winner's own routing (``banked_gates_adamw``) with the k11
    gate projections on the hybrid's AdamW side -- the incumbent's own
    convention (the DECAY gate's side: "a gate, not a hidden map").  The
    incumbent's classification is routing_symmetry's own regex inventory;
    the k11 gate names are the only addition, and the caller cross-checks
    the result against ``split_parameters_winner`` on the un-converted twin
    (the banked split must be reproduced exactly, gate names aside)."""
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import (_DECAY_BIAS, _DECAY_WEIGHT, _EMBED_HEAD, _MIXER_MATRIX, _MLP,
                _NORMS)
    channels = ARMS[arm]["channels"]
    # the per-head zero-born scalars' own names (the engine's own attrs:
    # write_gate_scalar / readout_scalar / dense_scalar)
    scalar_name = {"write": "write_gate_scalar", "readout": "readout_scalar",
                   "dense": "dense_scalar"}
    gate_names = []
    for channel in ("write", "readout", "dense"):
        if channels[channel]:
            gate_names += [f"{channel}_gate.weight", f"{channel}_gate.bias",
                           scalar_name[channel]]
    gate_re = re.compile(
        r"blocks\.\d+\.mixer\.(" + "|".join(
            re.escape(n) for n in gate_names) + r")$")
    muon, adamw = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if gate_re.fullmatch(name):
            adamw.append(name)
            continue
        if any(r.fullmatch(name) for r in _EMBED_HEAD):
            kind = "embed_head"
        elif _MIXER_MATRIX.fullmatch(name) is not None:
            kind = "mixer_core"
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
                f"unclassified parameter: {name} (the inventory is "
                f"enumerated; a drift must fail loudly, not silently route)")
        if kind in ("decay_weight", "decay_bias", "norm"):
            adamw.append(name)
        else:
            if param.ndim != 2:
                raise ValueError(f"Muon-owned parameter is not a full 2D "
                                 f"matrix: {name}")
            muon.append(name)
    if not muon or not adamw:
        raise ValueError("the hybrid requires both semantic groups non-empty")
    params = dict(model.named_parameters())
    return ([params[n] for n in muon], [params[n] for n in adamw],
            muon, adamw)


def _routing_cross_check(model, arm: str, seed: int) -> dict:
    """The no-drift receipt: an un-converted twin of the SAME build (same
    seed, same build order) must route EXACTLY as the banked winner's own
    split routes it; the gated model's split must equal that plus exactly
    the arm's gate tensors on the AdamW side."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import split_parameters_winner
    muon, adamw, muon_names, adamw_names = split_parameters_gated(model, arm)
    twin, _cfg, _summary = _build_k10(dict(_tuned_winner()), seed)
    _t_muon, _t_adamw, twin_muon, twin_adamw = split_parameters_winner(
        twin, "banked_gates_adamw")
    del twin, _t_muon, _t_adamw
    torch.cuda.empty_cache()
    channels = ARMS[arm]["channels"]
    scalar_name = {"write": "write_gate_scalar", "readout": "readout_scalar",
                   "dense": "dense_scalar"}
    per_layer_gate_tensors = sum(
        3 for channel in ("write", "readout", "dense") if channels[channel])
    expected_gate_names = sorted(
        f"blocks.{i}.mixer.{name}"
        for i in range(len(model.blocks))
        for c in ("write", "readout", "dense") if channels[c]
        for name in (f"{c}_gate.weight", f"{c}_gate.bias", scalar_name[c]))
    gained = sorted(set(adamw_names) - set(twin_adamw))
    if sorted(muon_names) != sorted(twin_muon):
        raise ValueError("the gated model's Muon group drifted from the "
                         "banked winner's own split")
    if gained != expected_gate_names:
        raise ValueError(
            f"the AdamW side must gain EXACTLY the arm's "
            f"{per_layer_gate_tensors * len(model.blocks)} gate tensors "
            f"over the banked split; got {len(gained)}: {gained[:6]}...")
    return dict(
        muon_tensors=len(muon_names),
        muon_params=sum(p.numel() for p in muon),
        adamw_tensors=len(adamw_names),
        adamw_params=sum(p.numel() for p in adamw),
        gate_tensors_on_adamw=len(gained),
        cross_check="the un-converted twin's banked_gates_adamw split "
                    "reproduced exactly (name-for-name); the AdamW side "
                    "gains exactly the arm's gate tensors (asserted "
                    "in-process)")


def _tuned_winner() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    return dict(TUNED_WINNER)

# --------------------------------------------------- the mechanism instrument


def k11_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED gated arm's mechanism receipts: the K10 probe's own walk
    stats (op rates, occupancy, read entropy, the identity-rewrite share --
    the raw-payload cosine instrument, anchor-continuous) on the k11 collect
    launch, PLUS the gate telemetry (per-layer write-gate distance from 1,
    dense-deposit magnitude, read-out gate -- the learned admission
    controller's own readout)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _agg_walk, _probe_windows, _walk_stats
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11_learned_gates import _k11_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        key_separation_probe import causal_context
    from experiments.expressivity_tasks.x45_matrix_emender.model import (
        DECAY_FLOOR)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    rows = _probe_windows(stream, windows)
    model.eval()
    per_layer = []
    gate_rows = []

    def make_hook(layer_index, mixer):
        pol32 = (_round_f32(mixer.policy.tau_match),
                 _round_f32(mixer.policy.tau_merge),
                 _round_f32(mixer.policy.tau_pop),
                 _round_f32(mixer.policy.tie_break),
                 _round_f32(mixer.policy.read_scale))

        def hook(_module, args, _output):
            x = args[0]
            batch, tokens, _ = x.shape
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                context = causal_context(x, mixer.ctx).reshape(
                    batch, tokens, mixer.ctx * x.shape[-1])
                payload = mixer.value(x).view(batch, tokens, mixer.heads,
                                              mixer.value_dim)
                write_keys = mixer.key(context).view(batch, tokens,
                                                     mixer.heads,
                                                     mixer.key_dim)
                query_keys = mixer.query_key(context).view(
                    batch, tokens, mixer.heads, mixer.key_dim)
                decay = torch.sigmoid(mixer.decay(x)).clamp(
                    min=DECAY_FLOOR, max=1.0 - 1e-6)
                gated = mixer.channels["write"]
                use_dense = mixer.channels["dense"]
                if gated:
                    gates = 1.0 - mixer.write_gate_scalar * torch.sigmoid(
                        mixer.write_gate(x))
                    gw_dist = (1.0 - gates).abs()
                else:
                    gates = payload.new_ones((1, 1, 1))
                    gw_dist = None
                if use_dense:
                    dense_scale = mixer.dense_scalar.view(1, 1, -1, 1) \
                        * torch.sigmoid(mixer.dense_gate(x)).unsqueeze(-1)
                    dense = dense_scale * payload
                else:
                    dense_scale = None
                    dense = payload.new_zeros((1, 1, 1, 1))
                if mixer.channels["readout"]:
                    g_out = 1.0 + mixer.readout_scalar * torch.sigmoid(
                        mixer.readout_gate(x))
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                keys0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.key_dim, dtype=torch.float32,
                                    device=x.device)
                out = _k11_forward_launch(
                    payload.float(), dense.float(), write_keys.float(),
                    query_keys.float(), decay.float(),
                    gates.float() if gated else gates, state, live0, keys0,
                    pol32, mixer.rank, mixer.segment_tokens, gated, use_dense,
                    collect=True, warps=8, stages=3)
            with torch.no_grad():
                stats, _schedule = _walk_stats(
                    out, mixer.rank, mixer, payload=payload.float(),
                    decay=decay.float())
                per_layer.append((layer_index, stats))
                row = dict(layer=layer_index)
                if gw_dist is not None:
                    gates32 = gates.float()
                    row["write_gate"] = dict(
                        mean=float(gates32.mean()),
                        mean_distance_from_1=float(gw_dist.float().mean()),
                        min=float(gates32.min()), max=float(gates32.max()))
                if mixer.channels["readout"]:
                    go32 = g_out.float()
                    row["readout_gate"] = dict(
                        mean=float(go32.mean()),
                        mean_distance_from_1=float((go32 - 1).abs().mean()),
                        min=float(go32.min()), max=float(go32.max()))
                if dense_scale is not None:
                    ds = dense_scale.float()
                    row["dense_deposit_scale"] = dict(
                        mean=float(ds.abs().mean()),
                        max=float(ds.abs().max()))
                if row:
                    gate_rows.append(row)
        return hook

    handles = [block.mixer.register_forward_hook(make_hook(i, block.mixer))
               for i, block in enumerate(model.blocks)]
    started = time.monotonic()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1))
                model.features(x)
    finally:
        for handle in handles:
            handle.remove()
    wall = time.monotonic() - started
    summary = _agg_walk(per_layer, with_identity=True)
    summary["probe"] = dict(split="calibration", windows=len(rows),
                            mixers=len(model.blocks),
                            regime="eval, bf16 autocast, the k11 persistent "
                                   "walk the arm trains on (collect=True)",
                            probe_wall_seconds=round(wall, 1))
    summary["effective_read_rows"] = float(torch.exp(torch.tensor(
        summary["read_weight_entropy"]["mean"])))
    layers = sorted({r["layer"] for r in gate_rows})
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the write gate's "
             "distance from 1 (how far the learned admission controller "
             "opened), the dense-deposit scale (m_d*sigmoid(w_d*x+b_d)), "
             "the read-out gate -- born at exact identity/zero (P5)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in gate_rows},
        write_gate_mean_distance_from_1=float(sum(
            r["write_gate"]["mean_distance_from_1"] for r in gate_rows
            if "write_gate" in r) / max(1, len(gate_rows))) if any(
            "write_gate" in r for r in gate_rows) else None,
        readout_gate_mean_distance_from_1=float(sum(
            r["readout_gate"]["mean_distance_from_1"] for r in gate_rows
            if "readout_gate" in r) / max(1, len(gate_rows))) if any(
            "readout_gate" in r for r in gate_rows) else None,
        dense_deposit_scale_mean=float(sum(
            r["dense_deposit_scale"]["mean"] for r in gate_rows
            if "dense_deposit_scale" in r) / max(1, len(gate_rows))) if any(
            "dense_deposit_scale" in r for r in gate_rows) else None,
        n_layers=len(layers))
    return summary

# ------------------------------------------------------------- the predictions


def evaluate_predictions(receipt: dict) -> dict:
    """M-1..M-5 (the registration's 4) against the amended anchors -- every
    read stated with its anchor's era, a null reported as a result."""
    anchors = receipt["anchors"]
    reg = REGISTRATION_ANCHORS
    constant_final = receipt["final_eval_nll_constant_lr"]
    post_cooldown = receipt["final_eval_nll_post_cooldown"]
    ema_final = receipt["ema_eval_nll"]
    seed = receipt["seed"]
    banked_constant = (BANKED["constant_lr"]["seed3401"] if seed == 3401
                       else BANKED["constant_lr"]["seed3402"])
    corrected_anchor = BANKED["post_cooldown"]["seed3402"]
    seed_matched = seed == SECOND_SEED
    diag180 = receipt["diagnostics_step180"]
    identity = diag180["identity_rewrite_share"]
    entropy = diag180["read_weight_entropy"]["mean"]
    occupancy = diag180["occupancy"]["mean_live_slots"]
    sweep = receipt["read_mask_sweep_step180"]
    deltas = [abs(v) for v in sweep["delta_per_layer"].values()]
    layers_ge = sum(1 for v in sweep["delta_per_layer"].values()
                    if v >= reg["sweep_threshold"])
    near_decorative = sum(1 for v in deltas if v <= reg["sweep_near_band"])
    m2_met = (identity
              <= anchors["seed_matched_step180"]["identity_rewrite_share"]
              - reg["m2_drop_bar"])
    m3_met = entropy > reg["m3_entropy_bar"]
    return dict(
        M1_nll=dict(
            prediction="each gate arm's final eval NLL improves on the "
                       "winner's matched-surface row (the registration's "
                       "band -0.02 to -0.25 vs its 1.5146 anchor; the "
                       "standing amendments re-anchor the comparison to "
                       "the banked winner config, hybrid Muon 4x lineage)",
            constant_lr=dict(measured=constant_final, anchor=banked_constant,
                             improvement=constant_final - banked_constant
                             if constant_final is not None else None,
                             met=bool(constant_final is not None
                                      and constant_final < banked_constant)),
            post_cooldown_primary=dict(
                measured=post_cooldown, anchor=corrected_anchor,
                seed_matched=seed_matched,
                improvement=post_cooldown - corrected_anchor
                if post_cooldown is not None else None,
                met=bool(post_cooldown is not None
                         and post_cooldown < corrected_anchor),
                note=("the PRIMARY corrected score vs the corrected banked "
                      "best; " + ("seed-matched (3402)" if seed_matched else
                                  "cross-seed (arm 3401 vs anchor 3402): "
                                  "read against the "
                                  f"{ENDPOINT_VARIANCE_BAND}-nat endpoint "
                                  f"band and the {WINNER_SEED_DELTA:.4f} "
                                  "winner seed delta"))),
            ema180_secondary=dict(measured=ema_final,
                                  anchor=BANKED["ema180"]["seed3402"],
                                  ranking_only=True),
            refuted_if_worse="the refutation clause: worse than the banked "
                             "row under every scoring -> the learned-gate "
                             "lever does not own the residual at this "
                             "granularity; reported as refuted, not excused"),
        M2_identity=dict(
            prediction="the identity-rewrite share falls at least "
                       f"{reg['m2_drop_bar']} absolute below the "
                       f"{reg['identity_rewrite_share']} anchor (the "
                       "registration's C5 headline)",
            measured=identity,
            anchor_registration=reg["identity_rewrite_share"],
            anchor_registration_bar=(reg["identity_rewrite_share"]
                                     - reg["m2_drop_bar"]),
            anchor_seed_matched_step180=anchors[
                "seed_matched_step180"]["identity_rewrite_share"],
            met=bool(m2_met),
            unchanged=bool(abs(identity - anchors[
                "seed_matched_step180"]["identity_rewrite_share"]) <= 0.02)),
        M3_entropy=dict(
            prediction=f"read entropy rises above {reg['m3_entropy_bar']} "
                       "nats (the registration's bar vs its 1.664 anchor)",
            measured=entropy,
            anchor_registration=reg["read_entropy_nats"],
            anchor_seed_matched_step180=anchors[
                "seed_matched_step180"]["read_entropy_nats"],
            effective_read_rows=diag180["effective_read_rows"],
            anchor_effective_rows_side_by_side=reg[
                "effective_rows_side_by_side"],
            met=bool(m3_met)),
        M4_decorative=dict(
            prediction=f"the read-mask sweep moves >= 10 of 24 layers to "
                       f"dNLL >= {reg['sweep_threshold']} (the banked "
                       "winner: 7 layers >= 0.02, 17 near-decorative)",
            measured=dict(layers_ge_threshold=layers_ge,
                          near_decorative=near_decorative,
                          total=len(sweep["delta_per_layer"]),
                          base_nll=sweep["base_nll"]),
            anchor=dict(layers_ge_threshold=reg["sweep_layers_ge_threshold"],
                        near_decorative=reg["sweep_near_decorative"],
                        caveat="the AdamW-era side-by-side sweep anchor "
                               "(the muon4x-era sweep was never banked); "
                               "the arm's sweep runs at BOTH the step-180 "
                               "and post-cooldown states, quoted together"),
            met=bool(layers_ge >= 10)),
        M5_op_mix=dict(
            prediction="NO PREDICTION (the op mix is a decision-channel "
                       "property the gates cannot move at fixed weights, "
                       "P2); reported, not predicted",
            measured=diag180["op_rate_per_token_lane0"],
            anchor_seed_matched_step180=anchors[
                "seed_matched_step180"]["op_rate_per_token"]),
        gate_telemetry=diag180["gate_telemetry"],
        refutation=dict(
            clause="if the mechanism numbers do not move while NLL does "
                   "(or vice versa), the dissociation is reported honestly "
                   "-- it re-scopes C5; if NO mechanism number moves and "
                   "NLL does not improve, the learned-gate lever does not "
                   "own the residual and that is reported as refuted, not "
                   "excused"))

# ------------------------------------------------------------------- the arm


def run_arm(args) -> dict:
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import HybridMuonX45
    from experiments.expressivity_tasks.x45_matrix_emender.\
        optimizer_bakeoff import MICROBATCH, _require_lease, _versions
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11_learned_gates import (arm_parameter_delta,
                                  convert_model_to_gated_writes)

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r}")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed} (the primary "
                         f"arms run seed {PRIMARY_SEED}; the second seed "
                         f"{SECOND_SEED} runs on the best arm only)")
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    arm = ARMS[args.arm]
    seed = args.seed

    # ---- the machine: the banked winner build verbatim, then the arm's
    # gate channels at the mimetic-neutral birth (P5)
    model, optimizer_config, summary = _build_k10(dict(_tuned_winner()), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_gated_writes(model, arm=args.arm)
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = arm_parameter_delta(args.arm, depth, mixer0.width, mixer0.heads)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta:
        raise ValueError(f"the registration's 1.5 parameter accounting must "
                         f"hold: {parameters_before} + {delta} != "
                         f"{parameters_after}")
    expected_delta = {"g1": 55_488, "g2": 27_744, "g3": 83_232}[args.arm]
    if delta != expected_delta or parameters_after != 149_981_808 + expected_delta:
        raise ValueError(f"the registered deltas: {args.arm} must add "
                         f"{expected_delta}, got {delta}")

    # ---- the optimizer: the banked winner lineage (hybrid Muon 4x), the
    # gate projections on the AdamW side, cross-checked against the banked
    # split on the un-converted twin
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_gated(model, args.arm)
    routing_receipt = _routing_cross_check(model, args.arm, seed)
    peak_lr = LR_MULTIPLIER * optimizer_config["lr"]
    optimizer = HybridMuonX45(muon_params, adamw_params, lr=peak_lr,
                              betas=tuple(optimizer_config["betas"]),
                              weight_decay=optimizer_config["weight_decay"])

    anchors = dict(
        banked=BANKED, registration=REGISTRATION_ANCHORS,
        banked_mechanism=_banked_mechanism_anchors(),
        seed_matched_step180=None,
        bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                   winner_seed_delta=WINNER_SEED_DELTA))
    anchors["seed_matched_step180"] = anchors["banked_mechanism"][
        f"seed{seed}_step180"]
    run_config = dict(
        probe="k11_learned_gates (gate C)", arm=args.arm, seed=seed,
        channels=arm["channels"], arm_note=arm["note"],
        corrected_scoring=CORRECTED_SCORING,
        exposure_targets=int(args.bytes),
        model_build="velocity_ladder._build_k10 verbatim (the tuned "
                    "winner's config, the banked bake-off build) + "
                    "convert_model_to_gated_writes (the arm's gate "
                    "channels at the mimetic-neutral birth, P5)",
        optimizer=dict(
            optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
            peak_lr=peak_lr, lr_multiplier=LR_MULTIPLIER,
            adamw_tuned_lr=optimizer_config["lr"],
            lr_convention=f"{LR_MULTIPLIER}x the machine's own tuned AdamW "
                          "LR -- the banked winner lineage (the speedrun "
                          "point the constant-LR probe selected)",
            betas=tuple(optimizer_config["betas"]),
            weight_decay=optimizer_config["weight_decay"],
            warmup_steps=optimizer_config["warmup_steps"],
            microbatch=optimizer_config["microbatch"], grad_clip=1.0),
        routing="banked_gates_adamw + the k11 gate projections on AdamW",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(before=parameters_before, added=delta,
                        total=parameters_after,
                        registration_delta=expected_delta),
        checkpoint_steps=list(args.checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear "
                       "cooldown (the corrected-scoring extension, same "
                       "stream); the step-180 constant-LR eval is retained "
                       "for banked comparability"),
        birth_rule="P5: the arm is born as the banked winner (gates at "
                   "identity/zero); the step-1 train loss must be "
                   "BIT-IDENTICAL to the banked same-seed row's",
        second_seed_rule=f"seed {SECOND_SEED} runs on the BEST arm only "
                         "(the post-cooldown primary score decides)",
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                     k11_preregistration=str(
                         ROOT / "PREREGISTRATION_LEARNED_GATES.md"),
                     k11_gates_receipt=str(RECEIPTS / "k11_gates.json")),
        summary=summary, identity=identity, versions=_versions(),
        git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns()))
    print(json.dumps(dict(arm=args.arm, card=card, seed=seed,
                          parameters=parameters_after,
                          muon=len(muon_names), adamw=len(adamw_names),
                          peak_lr=peak_lr)), flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, peak_lr, optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"{args.arm}_seed{seed}", out / "training.jsonl",
        tuple(args.checkpoint_steps), cooldown_steps, EMA_DECAY)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # the 0-flip-at-birth receipt at probe scale: the optimizer cannot
    # touch the first forward, and the gates are born at identity/zero, so
    # step 1 must reproduce the BANKED same-seed row bit-for-bit
    banked_step1 = _banked_step1_loss(seed)
    if not args.smoke and run["rows"][0]["loss"] != banked_step1:
        raise ValueError(
            f"step-1 train loss {run['rows'][0]['loss']!r} != the banked "
            f"same-seed row's {banked_step1!r} (the birth is not "
            f"mimetic-neutral -- an engine-level parity break)")

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

    # ---- the mechanism receipts: the walk stats + identity share + gate
    # telemetry at BOTH the step-180 constant state (banked-comparable) and
    # the post-cooldown final (the primary state); the read-mask sweep at
    # both states (side_by_side's instrument, imported verbatim)
    from experiments.expressivity_tasks.x45_matrix_emender.side_by_side \
        import k10_read_mask_sweep
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
    if run["status"] == "completed":
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            diagnostics_180 = k11_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_180 = k10_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = k11_probe_diagnostics(
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
        probe="k11_learned_gates (gate C)", arm=args.arm, seed=seed,
        channels=arm["channels"], arm_note=arm["note"],
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
        parameters=parameters_after,
        parameters_added=delta,
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=run["rows"][0]["loss"],
        step1_bit_identical_to_banked=bool(
            args.smoke or run["rows"][0]["loss"] == banked_step1),
        banked_constant_same_seed=(BANKED["constant_lr"]["seed3401"]
                                   if seed == 3401
                                   else BANKED["constant_lr"]["seed3402"]),
        scores=scores,
        anchors=anchors,
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
    if run["status"] == "completed" and diagnostics_180 is not None:
        receipt["predictions"] = evaluate_predictions(receipt)
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("arm", "seed", "status", "exposure_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll",
                       "step1_bit_identical_to_banked")}), flush=True)
    return receipt

# ------------------------------------------------------------------- assemble


def assemble(args) -> dict:
    """The corrected ladder: every arm x every scoring vs both quoted
    anchors, the mechanism adjudication M-1..M-5, the best arm's row (the
    corrected-ladder row), and the second-seed read -- nulls reported as
    results."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    arms = {}
    for arm in ARMS:
        for seed in (PRIMARY_SEED, SECOND_SEED):
            path = RUNS / f"{arm}_seed{seed}"
            if not (path / "result.json").exists():
                continue
            r = json.loads((path / "result.json").read_text())
            arms[f"{arm}_seed{seed}"] = dict(
                arm=arm, seed=seed, status=r["status"],
                constant_lr=r["final_eval_nll_constant_lr"],
                post_cooldown=r["final_eval_nll_post_cooldown"],
                ema_180=r["ema_eval_nll"],
                cooldown_drop=r["cooldown_drop"],
                step1_bit_identical=r["step1_bit_identical_to_banked"],
                identity_share=(r["diagnostics_step180"] or {}).get(
                    "identity_rewrite_share"),
                read_entropy=(r["diagnostics_step180"] or {}).get(
                    "read_weight_entropy", {}).get("mean"),
                effective_rows=(r["diagnostics_step180"] or {}).get(
                    "effective_read_rows"),
                train_seconds_constant=r["train_seconds_constant"],
                parameters=r["parameters"])
    if not arms:
        raise ValueError("no completed arm receipts to assemble")

    def best(label_field):
        live = {k: v for k, v in arms.items() if v["status"] == "completed"}
        if not live:
            return None
        return min(live, key=lambda k: live[k][label_field]
                    if live[k][label_field] is not None else float("inf"))

    best_post_cooldown = best("post_cooldown")
    summary = dict(
        probe="k11_learned_gates assemble (gate C)",
        corrected_scoring=CORRECTED_SCORING,
        anchors=dict(banked=BANKED, registration=REGISTRATION_ANCHORS,
                     banked_mechanism=_banked_mechanism_anchors(),
                     bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                                winner_seed_delta=WINNER_SEED_DELTA)),
        arms=arms,
        best_arm_post_cooldown=best_post_cooldown,
        corrected_ladder_row=None,
        second_seed_read=None,
        notes=dict(
            scoring="PRIMARY: post-cooldown (24-step linear); SECONDARY: "
                    "EMA-0.99 at step 180 (ranking-only); constant-180 "
                    "retained for banked comparability",
            best_rule="the best arm is decided by the post-cooldown primary "
                      "score at the primary seed; the second seed (3402) "
                      "runs on it",
            null_rule="a null is a result: it would confine the residual to "
                      "GDN2's denser write geometry (per-channel gate "
                      "matrices), not our per-head scalar gate surface"))
    if best_post_cooldown is not None:
        row = arms[best_post_cooldown]
        summary["corrected_ladder_row"] = dict(
            label=best_post_cooldown, arm=row["arm"], seed=row["seed"],
            constant_lr=row["constant_lr"],
            post_cooldown_primary=row["post_cooldown"],
            ema_180_secondary=row["ema_180"],
            vs_banked_constant=(row["constant_lr"]
                                - (BANKED["constant_lr"]["seed3401"]
                                   if row["seed"] == 3401
                                   else BANKED["constant_lr"]["seed3402"])),
            vs_corrected_anchor=(row["post_cooldown"]
                                 - BANKED["post_cooldown"]["seed3402"]),
            vs_gdn2_post_cooldown=(row["post_cooldown"]
                                   - BANKED["gdn2"]["post_cooldown"]
                                   if row["post_cooldown"] is not None
                                   else None))
        second = f"{row['arm']}_seed{SECOND_SEED}"
        if second in arms:
            s = arms[second]
            summary["second_seed_read"] = dict(
                label=second, constant_lr=s["constant_lr"],
                post_cooldown_primary=s["post_cooldown"],
                ema_180_secondary=s["ema_180"],
                seed_matched_vs_banked_constant=(
                    s["constant_lr"] - BANKED["constant_lr"]["seed3402"]
                    if s["constant_lr"] is not None else None),
                seed_matched_vs_corrected_anchor=(
                    s["post_cooldown"] - BANKED["post_cooldown"]["seed3402"]
                    if s["post_cooldown"] is not None else None),
                seed_delta_primary=(s["post_cooldown"] - row["post_cooldown"]
                                    if s["post_cooldown"] is not None
                                    and row["post_cooldown"] is not None
                                    else None))
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(arms={k: dict(constant=v["constant_lr"],
                                        post_cooldown=v["post_cooldown"],
                                        ema=v["ema_180"])
                               for k, v in arms.items()},
                         best_post_cooldown=best_post_cooldown), indent=1),
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
                        "steps, 3 cooldown steps, no banked assertions")
    s = sub.add_parser("assemble")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "k11_learned_gates_summary.json"))
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
