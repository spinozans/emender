"""K11B -- the per-channel-gates quality probe (gate C): the P1/P2/P3 arms
at the matched-exposure protocol under the CORRECTED scoring, vs the banked
winner config (hybrid Muon 4x lineage), plus the FROZEN-GATES falsifier
control.

THE QUESTION THIS STAGE EXISTS FOR (``PREREGISTRATION_CHANNEL_GATES.md``
2.3, committed before any probe measurement): is the "write geometry"
residual actually GATE GRANULARITY?  (i) per-channel decay closes >= 0.02
of the gap => the C5 attribution refines to "gate granularity", the quality
program has one more real lever, and the leaner-op-set registration gains
the per-channel decay gate as a required component; (ii) null at parity =>
the residual is the DENSE SUPERPOSITION ITSELF (values written every token
vs events) -- a form our machine cannot adopt without becoming GDN2 -- the
honest TERMINAL reading of the quality program, receipted as such.  Both
outcomes are results; NO directional prediction is pre-stated for arm 1's
magnitude.

The arms (the registration's 1, cumulative): ``p1`` per-channel
input-dependent DECAY (the f_proj analog, the audit's "strictly poorer"
row, the priority), ``p2`` + per-channel write admission (the b_proj
analog), ``p3`` + per-channel read gates (the g_proj analog).  Primary
seed 3401; the second seed (3402) runs on any arm whose primary score
beats the seed-matched comparison by >= 0.01 (the registration's 2.3 win
bar).  The FROZEN-GATES falsifier control (the registration's 2.4,
MANDATORY): the priority arm's gate parameters frozen at birth -- step-1
loss bit-identical to the banked row, landing at parity, the per-step
divergence measured and quoted (the per-channel DECAY adjoint's channel-sum
reduction order differs from the incumbent's fused scalar reduction, so
beyond step 1 the frozen trajectory diverges in ULPs by construction --
stated, not hidden).

Every arm is the banked winner config VERBATIM plus the arm's per-channel
channels at the mimetic-neutral birth, so a live arm's step-1 train loss
must be BIT-IDENTICAL to the banked same-seed row's (the 0-flip-at-birth
receipt at probe scale).  Standing amendments applied (the corrected
scoring, imported verbatim): the 24-step linear cooldown PRIMARY, EMA-0.99
at step 180 SECONDARY (ranking-only), constant-180 RETAINED for banked
comparability.  Both anchors quoted everywhere: constant-LR banked best
**1.4899048124040877** (seed 3402; the same config's seed-3401 row
**1.4907409463609969**) and the corrected post-cooldown banked best
**1.4338248457227434** (seed 3402).  GDN2's post-cooldown **1.3522** is the
gap reference.

The mechanism instrument (the registration's 2.5): the K11 mechanism
instrument (identity-rewrite share via the PER-CHANNEL trajectory replay,
read entropy / effective rows, occupancy, op mix) at BOTH the step-180
constant snapshot and the post-cooldown final, the read-mask sweep (the
side-by-side's C4 diagnostic, imported verbatim), and the gate telemetry --
the born-identity drift (per-layer mean |lam_vec - lam_inc|, |1-g_vec|,
|1-r_vec|) plus the two-step wake-up receipt.  No directional mechanism
prediction is pre-stated for arm 1 (the task's own rule).

Provenance (the lane's own rules): the engines are committed and read-only
(``k10_separated_keys.py``, ``k11_learned_gates.py``); the lane's NEW
engine is ``k11b_channel_gates.py``;
``velocity_ladder._build_k10``,
``protocol_correction._train_corrected``/``CORRECTED_SCORING``,
``routing_symmetry.split_parameters_winner``, the K10 probe instruments,
``side_by_side.k10_read_mask_sweep`` and the ``matched_arms`` helpers are
imported verbatim; no ``k10_*``/``k11_*``/``k12_*`` source, no committed
PREREGISTRATION, no ``lattice.py`` is edited.

Deliverables: per-arm receipts under ``k11b_runs/<arm>_seed<seed>/`` (the
frozen control under ``k11b_runs/p1_frozen_seed3401/``) and the assemble
summary ``receipts/k11b_channel_gates_summary.json``.

Usage (see ``run_k11b_channel_gates_leased.sh`` for the detached leased
driver, cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k11b_probe_channel_gates \\
        arm --arm p1 --seed 3401 --output k11b_runs/p1_seed3401
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
RUNS = ROOT / "k11b_runs"
ROUTING_SYMMETRY_RUNS = ROOT / "routing_symmetry_runs"
PROTOCOL_CORRECTION_RUNS = ROOT / "protocol_correction_runs"

EXPOSURE_TARGETS = 20_000_000          # the 20,022,866 landing at step 180
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 80, 120, 160, 180)
PROBE_WINDOWS = 4
LR_MULTIPLIER = 4                      # the banked winner lineage (muon4x)
PRIMARY_SEED = 3401
SECOND_SEED = 3402
# The win bar (the registration's 2.3): the second seed runs on any arm whose
# primary score beats the seed-matched comparison by >= this much.
WIN_BAR = 0.01
# The gap-closure bar for reading (i): per-channel decay closes >= this much
# of the ~0.08-nat machine gap.
GAP_CLOSURE_BAR = 0.02

# The corrected scoring, imported verbatim (the standing amendment).
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction \
    import (COOLDOWN_STEPS, CORRECTED_SCORING, EMA_DECAY,
            ENDPOINT_VARIANCE_BAND, WINNER_SEED_DELTA, _train_corrected)  # noqa: E402
from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates \
    import channel_persistent_trajectory  # noqa: E402

# The banked comparison anchors (cited, never re-run here; the step-1 losses
# and the mechanism anchors are loaded from the banked result.json files at
# run time).
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
                   "the ~0.08 residual this stage dissects)"),
)
# The parity references for the mechanism reads (the K11 lane's own
# same-config seed-matched anchors -- no directional prediction is pre-stated
# for arm 1; these are quoted as the references any granularity effect would
# have to move).
MECHANISM_ANCHORS = dict(
    identity_rewrite_share=(0.118, 0.122),
    read_entropy_nats=(1.515, 1.600),
    effective_read_rows=(4.55, 4.95),
    occupancy_mean_live_slots=(33.5, 35.2),
    sweep_layers_ge_threshold=dict(k11_arms=(5, 6), side_by_side_adamw=7),
    sweep_near_decorative=dict(k11_arms=(16, 17), side_by_side_adamw=17),
    sweep_threshold=0.02, sweep_near_band=0.0125, sweep_total=24,
    caveat="the K11 lane's same-config seed-matched step-180 anchors "
           "(both seeds) -- the references any granularity effect would "
           "have to move; the side-by-side's AdamW-era sweep anchors are "
           "quoted alongside (the muon4x-era banked sweep was never run)")

ARMS = {
    "p1": dict(channels=dict(decay=True, write=False, read=False),
               note="the priority arm: PER-CHANNEL INPUT-DEPENDENT DECAY "
                    "(the f_proj analog, the audit's 'strictly poorer "
                    "instance' row) -- lam_vec = clamp(lam_inc*(1 + "
                    "m_f*sigmoid(f_proj(x))), floor, 1-1e-6) per (token, "
                    "head, channel), born bit-equal to the registered "
                    "scalar on every channel"),
    "p2": dict(channels=dict(decay=True, write=True, read=False),
               note="+ PER-CHANNEL WRITE ADMISSION (the b_proj analog): "
                    "g_vec = 1 - m_w*sigmoid(b_proj(x)) per channel; APPEND "
                    "admits g_vec*p, CORRECT blends "
                    "g_vec*p + (1-g_vec)*(lam_vec*v_pre)"),
    "p3": dict(channels=dict(decay=True, write=True, read=True),
               note="+ PER-CHANNEL READ GATES (the g_proj analog), "
                    "combined: r_vec = 1 + m_o*sigmoid(g_proj(x)) per "
                    "channel, reads *= r_vec before the out projection"),
}

# The gate tensors' names in the converted model, per channel (the engine's
# own attrs; the routing places every one on the hybrid's AdamW side).
GATE_TENSOR_NAMES = {
    "decay": ("f_proj.0.weight", "f_proj.1.weight", "decay_scalar"),
    "write": ("b_proj.weight", "write_scalar"),
    "read": ("g_proj.0.weight", "g_proj.1.weight", "g_proj.1.bias",
             "read_scalar"),
}


def gate_tensor_names(arm: str):
    channels = ARMS[arm]["channels"]
    names = []
    for channel in ("decay", "write", "read"):
        if channels[channel]:
            names += list(GATE_TENSOR_NAMES[channel])
    return names


# ------------------------------------------------------------------ the anchors


def _banked_result(seed: int) -> dict:
    path = ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{seed}" / "result.json"
    if not path.exists():
        raise ValueError(f"the banked same-seed row is missing: {path}")
    return json.loads(path.read_text())


def _banked_training_rows(seed: int) -> list:
    path = ROUTING_SYMMETRY_RUNS / f"muon4x_banked_seed{seed}" \
        / "training.jsonl"
    if not path.exists():
        raise ValueError(f"the banked same-seed training log is missing: "
                        f"{path}")
    return [json.loads(line) for line in path.read_text().splitlines()]


def _banked_step1_loss(seed: int) -> float:
    return _banked_training_rows(seed)[0]["loss"]


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


def split_parameters_gated(model, arm: str, frozen: bool = False):
    """The banked winner's own routing (``banked_gates_adamw``) with the
    k11b per-channel gate projections on the hybrid's AdamW side -- the
    incumbent's own convention (the DECAY gate's side: "a gate, not a
    hidden map").  The incumbent's classification is routing_symmetry's own
    regex inventory; the k11b gate names are the only addition, and the
    caller cross-checks the result against ``split_parameters_winner`` on
    the un-converted twin (the banked split must be reproduced exactly,
    gate names aside).  A frozen control's gate tensors carry
    requires_grad=False and are excluded by the requires_grad filter -- the
    split then equals the banked twin's exactly (the falsifier's own
    plumbing receipt)."""
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import (_DECAY_BIAS, _DECAY_WEIGHT, _EMBED_HEAD, _MIXER_MATRIX, _MLP,
                _NORMS)
    gate_names = gate_tensor_names(arm)
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


def _routing_cross_check(model, arm: str, seed: int, frozen: bool) -> dict:
    """The no-drift receipt: an un-converted twin of the SAME build (same
    seed, same build order) must route EXACTLY as the banked winner's own
    split routes it; the gated model's split must equal that plus exactly
    the arm's gate tensors on the AdamW side (or, for the frozen control,
    equal the banked twin's split exactly)."""
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
    expected_gate_names = sorted(
        f"blocks.{i}.mixer.{name}"
        for i in range(len(model.blocks))
        for name in gate_tensor_names(arm))
    gained = sorted(set(adamw_names) - set(twin_adamw))
    if sorted(muon_names) != sorted(twin_muon):
        raise ValueError("the gated model's Muon group drifted from the "
                         "banked winner's own split")
    if frozen:
        if gained != []:
            raise ValueError("the frozen control's AdamW side must gain "
                             f"NOTHING over the banked split; got {gained}")
    elif gained != expected_gate_names:
        raise ValueError(
            f"the AdamW side must gain EXACTLY the arm's "
            f"{len(expected_gate_names)} gate tensors over the banked split; "
            f"got {len(gained)}: {gained[:6]}...")
    return dict(
        muon_tensors=len(muon_names),
        muon_params=sum(p.numel() for p in muon),
        adamw_tensors=len(adamw_names),
        adamw_params=sum(p.numel() for p in adamw),
        gate_tensors_on_adamw=len(gained),
        frozen=frozen,
        cross_check="the un-converted twin's banked_gates_adamw split "
                    "reproduced exactly (name-for-name); the AdamW side "
                    "gains exactly the arm's gate tensors (asserted "
                    "in-process; the frozen control gains NOTHING)")


def _tuned_winner() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    return dict(TUNED_WINNER)

# --------------------------------------------------- the mechanism instrument


def _identity_share_channel(schedule, payload, decay_vec, rank):
    """The K10 identity-share instrument at the PER-CHANNEL decay: the share
    of CORRECT decisions whose write payload's fp32 value-channel cosine to
    the targeted row's PRE-WRITE content is >= 0.999, over the walk's own
    recorded schedule and the lane-specific per-channel trajectory replay
    (K10's ``_identity_share`` verbatim except the replay)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys \
        import IDENTITY_COS
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import \
        CORRECT
    import torch
    batch, tokens, heads, value_dim = payload.shape
    device = payload.device
    state0 = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                          device=device)
    live0 = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
    decay32 = decay_vec.float().permute(0, 2, 1, 3).contiguous()
    _r, _s, _l, traj = channel_persistent_trajectory(state0, live0, decay32,
                                                      schedule)
    op0 = schedule.op[..., 0]                        # (b, T, h)
    slot0 = schedule.slot[..., 0]                    # (b, T, h)
    correct = op0 == CORRECT
    if not bool(correct.any()):
        return 0.0, 0
    payload32 = payload.float()                      # (b, T, h, V)
    slot_bht = slot0.permute(0, 2, 1)                # (b, h, T)
    gather_index = slot_bht.unsqueeze(-1).unsqueeze(-1).expand(
        batch, heads, tokens, 1, value_dim)
    rows = traj.gather(3, gather_index).squeeze(3)          # (b, h, T, V)
    rows = rows.permute(0, 2, 1, 3)                         # (b, T, h, V)
    pv = payload32 / payload32.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    rv = rows / rows.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    cos = (pv * rv).sum(-1)                                 # (b, T, h)
    identity = correct & (cos >= IDENTITY_COS)
    return (float(identity.sum()) / float(correct.sum()),
            int(correct.sum()))


def k11b_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED per-channel arm's mechanism receipts: the K10 probe's own
    walk stats (op rates, occupancy, read entropy) on the k11b collect
    launch, the identity-rewrite share through the PER-CHANNEL trajectory
    replay, PLUS the gate telemetry -- the born-identity drift (per-layer
    mean |lam_vec - lam_inc|, |1-g_vec|, |1-r_vec| -- the learned
    per-channel controller's own readout)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _agg_walk, _probe_windows, _walk_stats
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_channel_gates import _k11b_forward_launch
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
                lam_vec, lam_inc = mixer.channel_decay(x)
                drift = (lam_vec.float() - lam_inc.float()
                         .unsqueeze(-1)).abs()
                gated = mixer.channels["write"]
                if gated:
                    gates = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                        * torch.sigmoid(mixer.b_proj(x)).view(
                            batch, tokens, mixer.heads, mixer.value_dim)
                    gw_dist = (1.0 - gates).abs()
                else:
                    gates = payload.new_ones((1, 1, 1, 1))
                    gw_dist = None
                if mixer.channels["read"]:
                    r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                        * torch.sigmoid(mixer.g_proj(x)).view(
                            batch, tokens, mixer.heads, mixer.value_dim)
                    gr_dist = (r_vec - 1.0).abs()
                else:
                    gr_dist = None
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                keys0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.key_dim, dtype=torch.float32,
                                    device=x.device)
                out = _k11b_forward_launch(
                    payload.float(), write_keys.float(), query_keys.float(),
                    lam_vec.float(), gates.float() if gated else gates,
                    state, live0, keys0, pol32, mixer.rank,
                    mixer.segment_tokens, gated, collect=True, warps=8,
                    stages=3)
            with torch.no_grad():
                stats, schedule = _walk_stats(
                    out, mixer.rank, mixer, payload=payload.float(),
                    decay=None)
                share, n_correct = _identity_share_channel(
                    schedule, payload.float(), lam_vec.float(), mixer.rank)
                stats["identity_rewrite_share"] = share
                stats["identity_n_correct"] = n_correct
                per_layer.append((layer_index, stats))
                row = dict(layer=layer_index)
                row["decay_gate"] = dict(
                    mean_distance_from_registered_scalar=float(drift.mean()),
                    max=float(drift.max()),
                    lam_vec_mean=float(lam_vec.float().mean()),
                    lam_vec_min=float(lam_vec.float().min()),
                    lam_vec_max=float(lam_vec.float().max()))
                if gw_dist is not None:
                    row["write_gate"] = dict(
                        mean_distance_from_1=float(gw_dist.float().mean()),
                        max=float(gw_dist.float().max()))
                if gr_dist is not None:
                    row["read_gate"] = dict(
                        mean_distance_from_1=float(gr_dist.float().mean()),
                        max=float(gr_dist.float().max()))
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
                            regime="eval, bf16 autocast, the k11b "
                                   "persistent walk the arm trains on "
                                   "(collect=True)",
                            probe_wall_seconds=round(wall, 1))
    summary["effective_read_rows"] = float(torch.exp(torch.tensor(
        summary["read_weight_entropy"]["mean"])))
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the born-identity "
             "drift -- the per-channel decay gate's distance from the "
             "REGISTERED SCALAR (how far the per-channel timescales "
             "opened), the write gate's distance from 1, the read gate's "
             "distance from 1 (born at exact identity per channel, P5)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in gate_rows},
        decay_gate_mean_distance_from_registered_scalar=float(sum(
            r["decay_gate"]["mean_distance_from_registered_scalar"]
            for r in gate_rows) / max(1, len(gate_rows))),
        write_gate_mean_distance_from_1=float(sum(
            r["write_gate"]["mean_distance_from_1"] for r in gate_rows
            if "write_gate" in r) / max(1, len(gate_rows))) if any(
            "write_gate" in r for r in gate_rows) else None,
        read_gate_mean_distance_from_1=float(sum(
            r["read_gate"]["mean_distance_from_1"] for r in gate_rows
            if "read_gate" in r) / max(1, len(gate_rows))) if any(
            "read_gate" in r for r in gate_rows) else None,
        n_layers=len(gate_rows))
    return summary

# ------------------------------------------------------------- the predictions


def evaluate_predictions(receipt: dict) -> dict:
    """The registration's 2.3 two-way question against the measured arm --
    every read stated with its anchor, a null reported as a result."""
    anchors = receipt["anchors"]
    constant_final = receipt["final_eval_nll_constant_lr"]
    post_cooldown = receipt["final_eval_nll_post_cooldown"]
    ema_final = receipt["ema_eval_nll"]
    seed = receipt["seed"]
    banked_constant = (BANKED["constant_lr"]["seed3401"] if seed == 3401
                       else BANKED["constant_lr"]["seed3402"])
    corrected_anchor = BANKED["post_cooldown"]["seed3402"]
    seed_matched = seed == SECOND_SEED
    seed_matched_constant = (anchors["banked_mechanism"]
                             [f"seed{seed}_step180"]["constant_lr_nll"])
    gap_reference = BANKED["gdn2"]["post_cooldown"]
    wins_constant = (constant_final is not None
                     and seed_matched_constant - constant_final >= WIN_BAR)
    wins_primary = (post_cooldown is not None
                    and corrected_anchor - post_cooldown >= WIN_BAR)
    closes_gap = (post_cooldown is not None
                  and corrected_anchor - post_cooldown >= GAP_CLOSURE_BAR)
    diag180 = receipt["diagnostics_step180"] or {}
    identity = diag180.get("identity_rewrite_share")
    entropy = (diag180.get("read_weight_entropy") or {}).get("mean")
    occupancy = (diag180.get("occupancy") or {}).get("mean_live_slots")
    sweep = receipt.get("read_mask_sweep_step180") or {}
    deltas = [abs(v) for v in (sweep.get("delta_per_layer") or {}).values()]
    layers_ge = sum(1 for v in (sweep.get("delta_per_layer") or {}).values()
                    if v >= MECHANISM_ANCHORS["sweep_threshold"])
    near_decorative = sum(1 for v in deltas
                          if v <= MECHANISM_ANCHORS["sweep_near_band"])
    telemetry = diag180.get("gate_telemetry") or {}
    return dict(
        C1_nll=dict(
            constant_lr=dict(measured=constant_final, anchor=banked_constant,
                             seed_matched_banked_constant=seed_matched_constant,
                             improvement_vs_seed_matched=(
                                 seed_matched_constant - constant_final
                                 if constant_final is not None else None),
                             met_win_bar=bool(wins_constant)),
            post_cooldown_primary=dict(
                measured=post_cooldown, anchor=corrected_anchor,
                seed_matched=seed_matched,
                improvement=corrected_anchor - post_cooldown
                if post_cooldown is not None else None,
                met_win_bar=bool(wins_primary),
                met_gap_closure_bar=bool(closes_gap),
                note=("the PRIMARY corrected score vs the corrected banked "
                      "best; the win bar 0.01 decides the second seed, the "
                      "gap-closure bar 0.02 decides reading (i); "
                      + ("seed-matched (3402)" if seed_matched else
                         "cross-seed (arm 3401 vs anchor 3402): read "
                         "against the "
                         f"{ENDPOINT_VARIANCE_BAND}-nat endpoint band and "
                         f"the {WINNER_SEED_DELTA:.4f} winner seed delta"))),
            ema180_secondary=dict(measured=ema_final,
                                  anchor=BANKED["ema180"]["seed3402"],
                                  ranking_only=True),
            vs_gdn2_post_cooldown=(post_cooldown - gap_reference
                                   if post_cooldown is not None else None)),
        C2_identity=dict(measured=identity,
                         anchor_range=MECHANISM_ANCHORS[
                             "identity_rewrite_share"],
                         caveat=MECHANISM_ANCHORS["caveat"]),
        C3_entropy=dict(measured=entropy,
                        effective_read_rows=diag180.get(
                            "effective_read_rows"),
                        anchor_range=MECHANISM_ANCHORS["read_entropy_nats"],
                        anchor_effective_rows=MECHANISM_ANCHORS[
                            "effective_read_rows"]),
        C4_occupancy=dict(measured=occupancy,
                          anchor_range=MECHANISM_ANCHORS[
                              "occupancy_mean_live_slots"]),
        C5_decorative=dict(
            measured=dict(layers_ge_threshold=layers_ge,
                          near_decorative=near_decorative,
                          total=len(deltas),
                          base_nll=sweep.get("base_nll")),
            anchor=MECHANISM_ANCHORS["sweep_layers_ge_threshold"],
            caveat="the K11 arms' own sweep rows (5-6/24 at >= 0.02, 16-17/24 "
                   "near-decorative) and the side-by-side's AdamW-era "
                   "anchor (7/17) -- the muon4x-era banked sweep was never "
                   "run; the arm's sweep runs at BOTH the step-180 and "
                   "post-cooldown states, quoted together"),
        gate_telemetry=telemetry,
        the_question=dict(
            reading="(i) gate granularity" if closes_gap else
                    ("(ii) the dense superposition itself -- the honest "
                     "TERMINAL reading" if not wins_primary
                     else "a win below the 0.02 gap-closure bar: reported "
                          "as measured, the reading held for the "
                          "assemble"),
            refutation=dict(
                clause="if the per-channel arms land at parity -- no "
                       "mechanism number moves and the primary score does "
                       "not beat the seed-matched banked rows beyond the "
                       "band -- then GATE GRANULARITY does not own the "
                       "residual at probe scale, and the honest reading is "
                       "(ii): the residual is the dense superposition "
                       "itself. That is reported as the TERMINAL reading, "
                       "not excused -- with the probe-scale caveat stated "
                       "and the named follow-ups held open")))

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
        k11b_channel_gates import (arm_parameter_delta,
                                   convert_model_to_channel_gates)

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r}")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed} (the primary "
                         f"arms run seed {PRIMARY_SEED}; the second seed "
                         f"{SECOND_SEED} runs on a winning arm only)")
    if args.frozen and args.arm != "p1":
        raise ValueError("the frozen-gates falsifier control is registered "
                         "on the priority arm p1 only")
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    arm = ARMS[args.arm]
    seed = args.seed
    frozen = bool(args.frozen)

    # ---- the machine: the banked winner build verbatim, then the arm's
    # per-channel channels at the mimetic-neutral birth (P5)
    model, optimizer_config, summary = _build_k10(dict(_tuned_winner()), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_channel_gates(model, arm=args.arm)
    if frozen:
        gate_re = re.compile(r"blocks\.\d+\.mixer\.(" + "|".join(
            re.escape(n) for n in gate_tensor_names(args.arm)) + r")$")
        frozen_names = []
        for name, param in model.named_parameters():
            if gate_re.fullmatch(name):
                param.requires_grad_(False)
                frozen_names.append(name)
        if not frozen_names:
            raise ValueError("the frozen control found no gate tensors")
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = arm_parameter_delta(args.arm, depth, mixer0.width, mixer0.heads,
                                mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta:
        raise ValueError(f"the registration's 1.5 parameter accounting must "
                         f"hold: {parameters_before} + {delta} != "
                         f"{parameters_after}")
    expected_delta = {"p1": 2_555_952, "p2": 6_094_944,
                      "p3": 8_657_040}[args.arm]
    if delta != expected_delta or \
            parameters_after != 149_981_808 + expected_delta:
        raise ValueError(f"the registered deltas: {args.arm} must add "
                         f"{expected_delta}, got {delta}")

    # ---- the optimizer: the banked winner lineage (hybrid Muon 4x), the
    # per-channel gate projections on the AdamW side, cross-checked against
    # the banked split on the un-converted twin
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_gated(model, args.arm)
    routing_receipt = _routing_cross_check(model, args.arm, seed, frozen)
    peak_lr = LR_MULTIPLIER * optimizer_config["lr"]
    optimizer = HybridMuonX45(muon_params, adamw_params, lr=peak_lr,
                              betas=tuple(optimizer_config["betas"]),
                              weight_decay=optimizer_config["weight_decay"])

    anchors = dict(
        banked=BANKED, mechanism_anchors=MECHANISM_ANCHORS,
        banked_mechanism=_banked_mechanism_anchors(),
        bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                   winner_seed_delta=WINNER_SEED_DELTA, win_bar=WIN_BAR,
                   gap_closure_bar=GAP_CLOSURE_BAR))
    run_config = dict(
        probe="k11b_channel_gates (gate C)", arm=args.arm, seed=seed,
        frozen=frozen, channels=arm["channels"], arm_note=arm["note"],
        corrected_scoring=CORRECTED_SCORING,
        exposure_targets=int(args.bytes),
        model_build="velocity_ladder._build_k10 verbatim (the tuned "
                    "winner's config, the banked bake-off build) + "
                    "convert_model_to_channel_gates (the arm's per-channel "
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
        routing="banked_gates_adamw + the k11b per-channel gate projections "
                "on AdamW",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(before=parameters_before, added=delta,
                        total=parameters_after,
                        registration_delta=expected_delta,
                        added_share=round(delta / parameters_after, 5),
                        gdn2_gate_share_note="GDN2's own gate matrices are "
                                             "18.3% of its 149.9M; even arm "
                                             "p3 (+5.77%) is ~3x leaner -- "
                                             "the asymmetry is the finding"),
        checkpoint_steps=list(args.checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear "
                       "cooldown (the corrected-scoring extension, same "
                       "stream); the step-180 constant-LR eval is retained "
                       "for banked comparability"),
        birth_rule="P5: the arm is born as the banked winner (the per-channel "
                   "decay bit-equal to the registered scalar on every "
                   "channel, the gates at exact identity); the step-1 train "
                   "loss must be BIT-IDENTICAL to the banked same-seed "
                   "row's",
        frozen_rule=("the falsifier control: the gate parameters frozen at "
                    "birth (requires_grad=False, excluded from the "
                    "optimizer); step-1 bit-identity + a parity landing "
                    "with the per-step divergence measured (the per-channel "
                    "DECAY adjoint's channel-sum reduction order differs "
                    "from the incumbent's fused scalar reduction, so "
                    "beyond step 1 the frozen trajectory diverges in ULPs "
                    "by construction -- registration 2.4)") if frozen else
                   "live arm: the gates train",
        second_seed_rule=f"seed {SECOND_SEED} runs on any arm whose primary "
                         f"score beats the seed-matched comparison by "
                         f">= {WIN_BAR}",
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                     k11b_preregistration=str(
                         ROOT / "PREREGISTRATION_CHANNEL_GATES.md"),
                     k11b_gates_receipt=str(RECEIPTS / "k11b_gates.json")),
        summary=summary, identity=identity, versions=_versions(),
        git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns()))
    print(json.dumps(dict(arm=args.arm, card=card, seed=seed, frozen=frozen,
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

    # ---- the 0-flip-at-birth receipt at probe scale: the optimizer cannot
    # touch the first forward, and the channels are born at the registered
    # values, so step 1 must reproduce the BANKED same-seed row bit-for-bit
    banked_step1 = _banked_step1_loss(seed)
    if not args.smoke and run["rows"][0]["loss"] != banked_step1:
        raise ValueError(
            f"step-1 train loss {run['rows'][0]['loss']!r} != the banked "
            f"same-seed row's {banked_step1!r} (the birth is not "
            f"mimetic-neutral -- an engine-level parity break)")
    # ---- the frozen control's divergence receipt (registration 2.4): exact
    # 0 at step 1, ULP-scale growth measured after
    frozen_divergence = None
    if frozen and not args.smoke:
        banked_rows = _banked_training_rows(seed)
        per_step = []
        for row in run["rows"]:
            if row["step"] <= len(banked_rows):
                banked_loss = banked_rows[row["step"] - 1]["loss"]
                per_step.append(dict(
                    step=row["step"],
                    banked_loss=banked_loss, loss=row["loss"],
                    delta=row["loss"] - banked_loss,
                    bit_identical=row["loss"] == banked_loss))
        frozen_divergence = dict(
            per_step=per_step,
            step1_bit_identical=bool(per_step and per_step[0][
                "bit_identical"]),
            steps_bit_identical=sum(1 for r in per_step
                                    if r["bit_identical"]),
            max_abs_delta=max((abs(r["delta"]) for r in per_step),
                              default=None),
            note="the per-channel DECAY adjoint's channel-sum reduction "
                 "order differs from the incumbent's fused scalar "
                 "reduction: beyond step 1 the frozen trajectory diverges "
                 "in ULPs by construction (registration 2.4) -- the control "
                 "measures everything-except-gate-learning at the band's "
                 "scale")

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

    # ---- the mechanism receipts: the walk stats + the per-channel identity
    # share + the gate telemetry at BOTH states; the read-mask sweep at
    # both states (side_by_side's instrument, imported verbatim)
    from experiments.expressivity_tasks.x45_matrix_emender.side_by_side \
        import k10_read_mask_sweep
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
    if run["status"] == "completed" and not frozen:
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            diagnostics_180 = k11b_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_180 = k10_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = k11b_probe_diagnostics(
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
        probe="k11b_channel_gates (gate C)", arm=args.arm, seed=seed,
        frozen=frozen, channels=arm["channels"], arm_note=arm["note"],
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
        parameters_added_share=round(delta / parameters_after, 5),
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=run["rows"][0]["loss"],
        step1_bit_identical_to_banked=bool(
            args.smoke or run["rows"][0]["loss"] == banked_step1),
        frozen_divergence_receipt=frozen_divergence,
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
                      ("arm", "seed", "frozen", "status", "exposure_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll",
                       "step1_bit_identical_to_banked")}), flush=True)
    return receipt

# ------------------------------------------------------------------- assemble


def assemble(args) -> dict:
    """The corrected ladder: every arm x every scoring vs both quoted
    anchors, the two-way question's adjudication, the best arm's row, the
    frozen-control read, and the second-seed read -- nulls reported as
    results."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    arms = {}
    frozen = None
    for path in sorted(RUNS.glob("*_seed*")):
        if path.name.startswith("smoke_"):
            continue
        if not (path / "result.json").exists():
            continue
        r = json.loads((path / "result.json").read_text())
        label = path.name
        entry = dict(
            arm=r["arm"], seed=r["seed"], status=r["status"],
            frozen=r.get("frozen", False),
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
            occupancy=(r["diagnostics_step180"] or {}).get(
                "occupancy", {}).get("mean_live_slots"),
            decay_gate_drift=(((r["diagnostics_step180"] or {}).get(
                "gate_telemetry") or {}).get(
                "decay_gate_mean_distance_from_registered_scalar")),
            parameters=r["parameters"],
            train_seconds_constant=r["train_seconds_constant"])
        if entry["frozen"]:
            frozen = dict(entry, label=label,
                           divergence=r.get("frozen_divergence_receipt"))
        else:
            arms[label] = entry
    if not arms:
        raise ValueError("no completed arm receipts to assemble")

    def best(label_field):
        live = {k: v for k, v in arms.items() if v["status"] == "completed"}
        if not live:
            return None
        return min(live, key=lambda k: live[k][label_field]
                    if live[k][label_field] is not None else float("inf"))

    best_post_cooldown = best("post_cooldown")
    corrected_anchor = BANKED["post_cooldown"]["seed3402"]
    gap_reference = BANKED["gdn2"]["post_cooldown"]
    summary = dict(
        probe="k11b_channel_gates assemble (gate C)",
        corrected_scoring=CORRECTED_SCORING,
        anchors=dict(banked=BANKED, mechanism=MECHANISM_ANCHORS,
                     banked_mechanism=_banked_mechanism_anchors(),
                     bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                               winner_seed_delta=WINNER_SEED_DELTA,
                               win_bar=WIN_BAR,
                               gap_closure_bar=GAP_CLOSURE_BAR)),
        arms=arms,
        frozen_control=frozen,
        best_arm_post_cooldown=best_post_cooldown,
        corrected_ladder_row=None,
        second_seed_read=None,
        the_question=None)
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
            vs_corrected_anchor=(row["post_cooldown"] - corrected_anchor),
            vs_gdn2_post_cooldown=(row["post_cooldown"] - gap_reference
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
                    s["post_cooldown"] - corrected_anchor
                    if s["post_cooldown"] is not None else None),
                seed_delta_primary=(s["post_cooldown"] - row["post_cooldown"]
                                    if s["post_cooldown"] is not None
                                    and row["post_cooldown"] is not None
                                    else None))
    # the two-way question's adjudication (the registration's 2.3)
    best_closure = None
    for label, entry in arms.items():
        if entry["status"] != "completed" or entry["post_cooldown"] is None:
            continue
        if entry["seed"] == 3401:
            closure = corrected_anchor - entry["post_cooldown"]
            if best_closure is None or closure > best_closure[1]:
                best_closure = (label, closure)
    if best_closure is not None:
        label, closure = best_closure
        if closure >= GAP_CLOSURE_BAR:
            reading = ("(i) GATE GRANULARITY: per-channel decay closes "
                       f">= {GAP_CLOSURE_BAR} of the gap ({label}: "
                       f"{closure:.4f}); the C5 attribution refines to "
                       "'gate granularity', the quality program has one "
                       "more real lever, and the leaner-op-set registration "
                       "gains the per-channel decay gate as a required "
                       "component -- the second seed must confirm at the "
                       f"{WIN_BAR} bar before the claim is banked")
        elif closure < WIN_BAR:
            reading = ("(ii) THE DENSE SUPERPOSITION ITSELF: no arm beats "
                       "the corrected anchor beyond the win bar (best "
                       f"closure {closure:.4f} at {label}); gate granularity "
                       "does not own the residual at probe scale -- the "
                       "honest TERMINAL reading of the quality program's "
                       "write-geometry attribution, receipted as such "
                       "(probe-scale caveat stated; the per-slot "
                       "address-axis decay and the per-channel dense "
                       "deposit are the named follow-ups)")
        else:
            reading = (f"a win below the gap-closure bar ({label}: "
                       f"{closure:.4f} vs the 0.01 win bar and the "
                       f"{GAP_CLOSURE_BAR} closure bar): reported as "
                       "measured; the second seed decides whether it "
                       "survives, the reading is held for the operator")
        summary["the_question"] = dict(reading=reading,
                                       best_closure=best_closure)
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(arms={k: dict(constant=v["constant_lr"],
                                        post_cooldown=v["post_cooldown"],
                                        ema=v["ema_180"])
                               for k, v in arms.items()},
                         frozen_post_cooldown=(frozen or {}).get(
                             "post_cooldown"),
                         best_post_cooldown=best_post_cooldown,
                         the_question=(summary["the_question"] or {}).get(
                             "reading")), indent=1), flush=True)
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
    a.add_argument("--frozen", action="store_true",
                   help="the falsifier control: the gate parameters frozen "
                        "at birth (registered on the priority arm p1 "
                        "only)")
    a.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, 3 checkpoint "
                        "steps, 3 cooldown steps, no banked assertions")
    s = sub.add_parser("assemble")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "k11b_channel_gates_summary.json"))
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
