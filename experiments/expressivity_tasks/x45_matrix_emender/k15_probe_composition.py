"""K15 -- THE COMPOSITION quality probe: the ONE arm (`composition`) at the
matched-exposure protocol under the CORRECTED scoring, vs BOTH anchors --
the K13 ADDITIVE rows (the composition's OWN read-form baseline: 1.3928 /
1.3845 post-cooldown -- the composition must beat its own baseline to
count as composing) and the banked/GDN2 ladder -- at BOTH seeds (3401,
3402), both primary.

THE QUESTION THIS STAGE EXISTS FOR
(``PREREGISTRATION_COMPOSITION.md`` 3, committed before any probe
measurement): do the session's two proven-positive levers COMPOSE -- the
K13 additive read (banked score channel verbatim, softmax removed,
strength dropped, the g_proj gate zero-born) under the K11B p3 per-channel
gate stack (the f_proj DECAY + b_proj write + g_proj read channels, born
at the registered scalar equivalents)?  NO directional prediction is
pre-stated; the pre-registered clauses govern: (C-1) beats the additive
baseline by >= 0.01 at BOTH seeds => the levers COMPOSE, the new best
machine is the composition, and the honest rematch row is the
composition's post-cooldown vs GDN2's 1.3522; (C-2) parity => the gates'
+0.010 was READ-FORM-DEPENDENT (the softmax read needed the channel
gates; the additive read does not); (C-3) worse at both seeds => the gates
INTERFERE with the additive read (the per-channel decay gates modulate
exactly the channels the additive read now sums).

STANDING AMENDMENTS applied verbatim (imported): the 24-step linear cooldown
PRIMARY, the EMA-0.99 iterate at step 180 SECONDARY (ranking-only), the
constant-LR step-180 eval RETAINED for banked comparability.

THE BIRTH RECEIPTS (the machine-variant's own, the registration's 2.2 --
no BANKED step-1 parity is possible and none is asserted): (1)
BORN-AS-THE-ADDITIVE-MACHINE -- on the first training batch, layer for
layer, at the arm's own inputs, the composed scan's every channel is
`torch.equal` to the K13 ADDITIVE kernel's own committed launch (the gates
born at the registered scalar equivalents through the production
`channel_decay` expression under autocast); (2) DECISIONS-BIT-IDENTICAL
under the birth values (no gate value, no read form feeds op selection);
(3) the gates born at the registered values (`lam_vec == lam_inc` per
channel bit for bit, `g_vec == 1`, `r_vec == 1`); (4) the step-1 train loss
is asserted BIT-IDENTICAL to the K13 additive arm's own same-seed row
(the composition's nearest-kin parity -- the banked row's step-1 is a
different function's value and is quoted, never asserted).

The mechanism instrument (the registration's 2.3): the K13 instrument
(the |w|-mass effective rows R13-4, the top-row |w|-mass share -- the
collapse detector -- the mean live-row count, the birth-magnitude rows)
UNDER the K11B instrument (the identity-rewrite share through the
per-channel trajectory replay, occupancy, op mix, the gate telemetry for
ALL THREE gate channels), plus the decorative-store sweep
(`side_by_side.k10_read_mask_sweep`, imported verbatim) at BOTH the
step-180 constant snapshot and the post-cooldown final.

Provenance (the lane's own rules): the engines are committed and read-only
(`k13_additive_read.py`, `k11b_channel_gates.py`, `k10_separated_keys.py`);
`velocity_ladder._build_k10`,
`protocol_correction._train_corrected`/`CORRECTED_SCORING`,
`routing_symmetry.split_parameters_winner`,
`k11b_probe_channel_gates.split_parameters_gated`/`_routing_cross_check`/
`gate_tensor_names`/`_identity_share_channel`,
`k13_probe_additive_read._agg_walk`,
`k10_probe_separated_keys` instruments,
`side_by_side.k10_read_mask_sweep` and the `matched_arms` helpers are
imported verbatim; no `k10_*`/`k11*`/`k12_*`/`k13_*`/`x46_*` source, no
committed PREREGISTRATION, no `lattice.py` is edited.

Deliverables: per-arm receipts under `k15_runs/composition_seed<seed>/`
and the assemble summary `receipts/k15_composition_summary.json` (the
C-1/C-2/C-3 adjudication across seeds + the GDN2 rematch headline).

Usage (see `run_k15_composition_leased.sh` for the detached leased driver,
cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k15_probe_composition \\
        arm --arm composition --seed 3401 --output k15_runs/composition_seed3401
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

# The probe subprocess sets these BEFORE importing torch (the matched-arms
# discipline; configure_production() fail-closes on a missing allocator knob).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k15_runs"
K13_RUNS = ROOT / "k13_runs"

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

# The banked comparison anchors (cited, never re-run here).
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
              note="the banked GDN2 best rows (the gap reference -- the "
                   "composition's post-cooldown row vs 1.3522 is the "
                   "headline: the honest rematch)"))
# The composition's OWN read-form baseline: the K13 additive arm's landed
# rows (receipts/k13_additive_read_summary.json; the loader falls back to
# these constants if the receipt file is unavailable).
K13_ADDITIVE = dict(
    post_cooldown=dict(seed3401=1.3928245476314, seed3402=1.384450146130153),
    constant_lr=dict(seed3401=1.439220769064767, seed3402=1.435692889349801),
    ema_180=dict(seed3401=1.4283077546528407, seed3402=1.437227657863072),
    step1_train_loss=dict(seed3401=5.782775391046648,
                          seed3402=5.720172563202926),
    mechanism=dict(
        effective_read_rows=dict(seed3401=16.424976348876953,
                                 seed3402=16.68810272216797),
        top_row_w_mass_share=dict(seed3401=0.15604202003087658,
                                  seed3402=0.15366868533289912),
        identity_rewrite_share=dict(seed3401=0.12543127376869717,
                                    seed3402=0.11708593053841875),
        occupancy=dict(seed3401=26.213897705078125,
                       seed3402=26.533884684244793)),
    note="the K13 additive machine's own landed rows -- the composition "
         "must beat ITS OWN read-form baseline to count as composing")
# The pre-registered adjudication bars (the registration's 3).
REGISTRATION_ANCHORS = dict(
    compose_bar=0.01,                  # C-1: beats the additive baseline by
                                       # >= this at BOTH seeds
    k11b_p3_softmax_rows=dict(
        post_cooldown=dict(seed3401=1.4235332352774483,
                           seed3402=1.4234301022128885),
        note="the K11B p3 rows ON THE SOFTMAX read (+0.0103/+0.0104 vs the "
             "banked anchor) -- the read-form-dependence question's own "
             "reference"),
    bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
               winner_seed_delta=WINNER_SEED_DELTA))

# The one arm (the registration's 2.1): the banked winner build + the K13
# additive read + the K11B p3 gate stack at the mimetic-neutral birth.
ARMS = ("composition",)


# ------------------------------------------------- the banked-anchor loaders


def _banked_result(seed: int) -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11_probe_learned_gates import _banked_result as loader
    return loader(seed)


def _banked_step1_loss(seed: int) -> float:
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11_probe_learned_gates import _banked_step1_loss as loader
    return loader(seed)


def _banked_mechanism_anchors() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11_probe_learned_gates import _banked_mechanism_anchors as loader
    return loader()


def _k13_additive_step1(seed: int) -> float:
    """The K13 additive arm's own step-1 row, loaded from its committed
    receipt (the constant above is the fallback)."""
    path = K13_RUNS / f"additive_seed{seed}" / "result.json"
    if path.exists():
        return json.loads(path.read_text())["step1_train_loss"]
    return K13_ADDITIVE["step1_train_loss"][f"seed{seed}"]


def _tuned_winner() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    return dict(TUNED_WINNER)


# -------------------------------------------------------- the birth receipts


def birth_receipts(model, stream, seed: int, microbatch: int) -> dict:
    """The machine-variant's own birth receipts (the registration's 2.2), run
    BEFORE any training on the FIRST train batch (the batch the step-1 loss
    sees):

    1. BORN-AS-THE-ADDITIVE-MACHINE: layer for layer, at the arm's own
       inputs, the composed scan's EVERY channel is `torch.equal` to the K13
       ADDITIVE kernel's own committed launch -- the gates are born at the
       registered scalar equivalents through the production `channel_decay`
       expression under autocast, so the composition's forward at birth IS
       the additive machine's.
    2. The gates born at the registered values: `lam_vec == lam_inc` per
       channel bit for bit, `g_vec == 1`, `r_vec == 1` per mixer (asserted
       per layer on the batch).
    3. The birth-magnitude rows (the additive machine's own, quoted for
       context): the per-head read RMS, composed vs the BANKED read, on the
       same inputs.
    """
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_separated_keys import _k10_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read \
        import _k13_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    twin, _cfg, _summary = _build_k10(dict(_tuned_winner()), seed)
    # the conversion must have SHARED the banked projections: spot-assert
    # every layer's carried modules bit-equal to the fresh twin's.
    for i, (block, twin_block) in enumerate(zip(model.blocks, twin.blocks)):
        a, t = block.mixer, twin_block.mixer
        for name in ("value", "key", "query_key", "out", "decay"):
            if not torch.equal(getattr(a, name).weight,
                               getattr(t, name).weight):
                raise ValueError(f"layer {i}: the conversion must share the "
                                 f"banked {name} projection bit for bit")
    del twin
    torch.cuda.empty_cache()

    batch_rows = next(iter(stream.batches("train")))
    x, y, _valid = next(grouped(batch_rows, microbatch))

    per_layer = []
    flips_total = lanes_total = 0
    rms_add, rms_bank = [], []
    gate_checks = []

    def make_hook(layer_index, mixer):
        pol32 = (_round_f32(mixer.policy.tau_match),
                 _round_f32(mixer.policy.tau_merge),
                 _round_f32(mixer.policy.tau_pop),
                 _round_f32(mixer.policy.tie_break),
                 _round_f32(mixer.policy.read_scale))

        def hook(_module, args, _output):
            xin = args[0]                      # the mixer's float input
            batch, tokens, _ = xin.shape
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                payload, write_keys, query_keys, _decay_scalar = \
                    _project_keyed(mixer, xin)
                payload, write_keys = payload.float(), write_keys.float()
                query_keys = query_keys.float()
                # the production birth expressions under autocast (the
                # falsifier's standing remedy)
                lam_vec, lam_inc = mixer.channel_decay(xin)
                lam_vec32 = lam_vec.float()
                scalar32 = lam_inc.float().unsqueeze(-1).expand(
                    batch, tokens, mixer.heads, mixer.value_dim
                ).contiguous()[..., 0].contiguous()
                gate_vec = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.b_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.g_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gate_checks.append(dict(
                    lam_vec_bit_equal_registered_scalar=bool(torch.equal(
                        lam_vec, lam_inc.unsqueeze(-1).expand_as(lam_vec)
                        .to(lam_vec.dtype))),
                    g_vec_born_at_exact_1=bool(
                        torch.equal(gate_vec, torch.ones_like(gate_vec))),
                    r_vec_born_at_exact_1=bool(
                        torch.equal(r_vec, torch.ones_like(r_vec)))))
                gates32 = gate_vec.float()
                state = torch.zeros(batch, mixer.heads, mixer.rank,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out_k15 = _k15_forward_launch(
                    payload, write_keys, query_keys, lam_vec32, gates32,
                    state, live, keys, pol32, mixer.rank,
                    mixer.segment_tokens, True, collect=True, warps=8,
                    stages=3)
                state2 = torch.zeros(batch, mixer.heads, mixer.rank,
                                     mixer.value_dim, dtype=torch.float32,
                                     device=xin.device)
                live2 = torch.zeros(batch, mixer.heads, mixer.rank,
                                    dtype=torch.bool, device=xin.device)
                keys2 = torch.zeros(batch, mixer.heads, mixer.rank,
                                    mixer.key_dim, dtype=torch.float32,
                                    device=xin.device)
                out_k13 = _k13_forward_launch(
                    payload, write_keys, query_keys, scalar32, state2, live2,
                    keys2, pol32, mixer.rank, mixer.segment_tokens,
                    collect=True, warps=8, stages=3)
                state3 = torch.zeros(batch, mixer.heads, mixer.rank,
                                     mixer.value_dim, dtype=torch.float32,
                                     device=xin.device)
                live3 = torch.zeros(batch, mixer.heads, mixer.rank,
                                    dtype=torch.bool, device=xin.device)
                keys3 = torch.zeros(batch, mixer.heads, mixer.rank,
                                    mixer.key_dim, dtype=torch.float32,
                                    device=xin.device)
                out_bank = _k10_forward_launch(
                    payload, write_keys, query_keys, scalar32, state3, live3,
                    keys3, pol32, mixer.rank, mixer.segment_tokens,
                    collect=True, warps=8, stages=3)
            flips = lanes = 0
            for name in ("slot", "erase", "op", "live_after", "payload",
                         "live_pre", "state", "live", "keys", "weights",
                         "reads"):
                a, b = out_k13[name], out_k15[name]
                flips += int((a != b).sum())
                lanes += a.numel()
            r_add = out_k15["reads"].float()
            r_bank = out_bank["reads"].float()
            per_layer.append(dict(
                layer=layer_index, flips=flips, lanes=lanes,
                read_rms_composed_per_head=[
                    float(v) for v in
                    r_add.pow(2).mean(dim=(0, 1, 3)).sqrt()],
                read_rms_banked_per_head=[
                    float(v) for v in
                    r_bank.pow(2).mean(dim=(0, 1, 3)).sqrt()],
                mean_live_rows=float(
                    out_k15["live_pre"].float().sum(-1).mean())))
            nonlocal flips_total, lanes_total, rms_add, rms_bank
            flips_total += flips
            lanes_total += lanes
            rms_add.append(r_add.pow(2).mean(dim=(0, 1, 3)).sqrt())
            rms_bank.append(r_bank.pow(2).mean(dim=(0, 1, 3)).sqrt())
        return hook

    handles = [block.mixer.register_forward_hook(make_hook(i, block.mixer))
               for i, block in enumerate(model.blocks)]
    try:
        with torch.no_grad():
            model.features(x)
    finally:
        for handle in handles:
            handle.remove()

    rms_add_t = torch.stack(rms_add)         # (layers, heads)
    rms_bank_t = torch.stack(rms_bank)
    receipt = dict(
        note="the machine-variant's own birth receipts on the FIRST train "
             "batch (the batch the step-1 loss sees): NO banked step-1 "
             "loss parity is possible (a different function) and none is "
             "asserted; the step-1 loss IS asserted bit-identical to the "
             "K13 additive arm's own same-seed row (the nearest kin)",
        born_as_the_additive_machine=dict(
            flips=flips_total, lanes=lanes_total,
            bit_identical=bool(flips_total == 0),
            note="every channel of the composed forward launch vs the K13 "
                 "ADDITIVE kernel's own committed launch at identical "
                 "inputs, every layer, both engines' committed kernels "
                 "(the gates born at the registered scalar equivalents "
                 "through the production channel_decay expression)"),
        gates_born_at_registered_values=dict(
            all_checks=bool(gate_checks)
            and all(c["lam_vec_bit_equal_registered_scalar"]
                    and c["g_vec_born_at_exact_1"]
                    and c["r_vec_born_at_exact_1"] for c in gate_checks),
            n_layers=len(gate_checks)),
        birth_magnitude=dict(
            per_head_read_rms_composed_mean=[
                float(v) for v in rms_add_t.mean(0)],
            per_head_read_rms_banked_mean=[
                float(v) for v in rms_bank_t.mean(0)],
            composed_over_banked_per_head_mean=[
                float(a / max(b, 1e-30))
                for a, b in zip(rms_add_t.mean(0), rms_bank_t.mean(0))],
            note="R13-5's rows quoted for context: at birth the composition "
                 "IS the additive machine (receipt 1), so the honest "
                 "~O(live-rows)x magnitude consequence is the additive "
                 "machine's own; any early-loss divergence is reported "
                 "with this attached"),
        per_layer=per_layer[:1] + per_layer[-1:],   # the first and last layer
        mean_live_rows_birth=float(
            sum(p["mean_live_rows"] for p in per_layer) / len(per_layer)))
    return receipt


# --------------------------------------------------- the mechanism instrument


def _k15_walk_stats(out, rank, mixer, payload=None, decay_vec=None):
    """The K13 |w|-mass instrument (R13-4) with the K11B per-channel
    identity share: op mix, occupancy, effective rows e^H of the |w|-mass
    distribution on live rows, the top-row |w|-mass share (the collapse
    detector), and the identity-rewrite share through the PER-CHANNEL
    trajectory replay."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        MatrixSchedule, op_histogram)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import _identity_share_channel
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    lane0 = torch.bincount(schedule.op[..., 0].reshape(-1).to("cpu"),
                           minlength=6)
    occupancy = out["live_pre"].float().sum(-1)      # (b, T, h)
    w = out["weights"][..., :rank].float().abs()      # the |w|-mass (signed!)
    mass = w.sum(-1)
    has = mass > 0
    probs = torch.where(has.unsqueeze(-1),
                        w / mass.clamp_min(1e-30).unsqueeze(-1),
                        torch.zeros_like(w))
    contrib = torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs))
    entropy = -contrib.sum(-1)                        # (b, T, h)
    top_share = torch.where(has, probs.max(-1).values, torch.zeros_like(mass))
    stats = dict(
        lane0=[int(v) for v in lane0],
        hist=op_histogram(schedule),
        decisions=int(occupancy.numel()),
        occupancy_sum=float(occupancy.sum()),
        occupancy_n=int(occupancy.numel()),
        occupancy_first=float(occupancy[:, 0].float().mean()),
        entropy_sum=float(entropy[has].sum()),
        entropy_n=int(has.sum()),
        top_share_sum=float(top_share[has].sum()),
        rank=rank)
    if payload is not None and decay_vec is not None:
        share, n_correct = _identity_share_channel(
            schedule, payload, decay_vec, rank)
        stats["identity_rewrite_share"] = share
        stats["identity_n_correct"] = n_correct
    return stats, schedule


def k15_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED arm's mechanism receipts: the K13 instrument (the |w|-mass
    effective rows + the collapse detector, occupancy, op mix, the
    identity-rewrite share through the per-channel replay) UNDER the K11B
    gate telemetry (all three gate channels' born-identity drift, per layer)
    on the k15 collect launch."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _probe_windows
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
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
            xin = args[0]
            batch, tokens, _ = xin.shape
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                payload, write_keys, query_keys, _scalar = _project_keyed(
                    mixer, xin)
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
                lam_vec32 = lam_vec.float()
                gates32 = gate_vec.float()
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out = _k15_forward_launch(
                    payload, write_keys, query_keys, lam_vec32, gates32,
                    state, live, keys, pol32, mixer.rank,
                    mixer.segment_tokens, True, collect=True, warps=8,
                    stages=3)
            with torch.no_grad():
                stats, _schedule = _k15_walk_stats(
                    out, mixer.rank, mixer, payload=payload,
                    decay_vec=lam_vec32)
                per_layer.append((layer_index, stats))
                gate_rows.append(dict(
                    layer=layer_index,
                    decay_gate=dict(
                        mean_distance_from_registered_scalar=float(drift.mean()),
                        max=float(drift.max()),
                        lam_vec_mean=float(lam_vec32.mean()),
                        lam_vec_min=float(lam_vec32.min()),
                        lam_vec_max=float(lam_vec32.max())),
                    write_gate=dict(
                        mean_distance_from_1=float(gw_dist.float().mean()),
                        max=float(gw_dist.float().max())),
                    read_gate=dict(
                        mean_distance_from_1=float(gr_dist.float().mean()),
                        max=float(gr_dist.float().max()))))
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
    summary = _agg_walk(per_layer)
    summary["probe"] = dict(split="calibration", windows=len(rows),
                            mixers=len(model.blocks),
                            regime="eval, bf16 autocast, the k15 composed "
                                   "persistent walk the arm trains on "
                                   "(collect=True)",
                            probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the born-identity "
             "drift for ALL THREE gate channels -- the per-channel decay "
             "gate's distance from the REGISTERED SCALAR, the write gate's "
             "and the read gate's distance from 1 (the learned per-channel "
             "controller's own readout UNDER the additive read)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in gate_rows},
        decay_gate_mean_distance_from_registered_scalar=float(sum(
            r["decay_gate"]["mean_distance_from_registered_scalar"]
            for r in gate_rows) / max(1, len(gate_rows))),
        write_gate_mean_distance_from_1=float(sum(
            r["write_gate"]["mean_distance_from_1"] for r in gate_rows)
            / max(1, len(gate_rows))),
        read_gate_mean_distance_from_1=float(sum(
            r["read_gate"]["mean_distance_from_1"] for r in gate_rows)
            / max(1, len(gate_rows))),
        n_layers=len(gate_rows))
    return summary


# ------------------------------------------------------------- the predictions


def evaluate_predictions(receipt: dict) -> dict:
    """The registration's 3 pre-stated clauses, evaluated per seed verbatim
    against the composition's OWN read-form baseline (the assemble
    adjudicates across seeds; no directional prediction was pre-stated)."""
    seed = receipt["seed"]
    post = receipt.get("final_eval_nll_post_cooldown")
    constant = receipt.get("final_eval_nll_constant_lr")
    ema = receipt.get("ema_eval_nll")
    additive_post = K13_ADDITIVE["post_cooldown"][f"seed{seed}"]
    additive_constant = K13_ADDITIVE["constant_lr"][f"seed{seed}"]
    additive_ema = K13_ADDITIVE["ema_180"][f"seed{seed}"]
    bar = REGISTRATION_ANCHORS["compose_bar"]
    out = {}
    if post is not None:
        improvement = additive_post - post
        if improvement >= bar:
            clause = "C-1_at_this_seed (COMPOSE: beats the additive baseline)"
        elif improvement <= -bar:
            clause = "C-3_at_this_seed (INTERFERE: worse than the baseline)"
        else:
            clause = ("C-2_at_this_seed (parity inside the +/-0.01 band: "
                      "read-form-dependence)")
        out["the_composition_clause"] = dict(
            post_cooldown_primary=post,
            additive_baseline_post_cooldown=additive_post,
            improvement=improvement, compose_bar=bar, clause=clause,
            constant_lr=dict(measured=constant,
                             additive_baseline=additive_constant,
                             improvement=additive_constant - constant
                             if constant is not None else None),
            ema_180=dict(measured=ema, additive_baseline=additive_ema),
            note="the seed's own clause; MET for C-1 needs BOTH seeds; "
                 "mixed seeds adjudicate to the CONSERVATIVE clause")
        # the headline rows: the honest rematch vs GDN2 + the ladder
        out["the_headline_rows"] = dict(
            vs_gdn2_post_cooldown=post - BANKED["gdn2"]["post_cooldown"],
            vs_banked_corrected_anchor=post
            - BANKED["post_cooldown"]["seed3402"],
            fraction_of_corrected_gap_closed=max(
                0.0, BANKED["post_cooldown"]["seed3402"] - post)
            / (BANKED["post_cooldown"]["seed3402"]
               - BANKED["gdn2"]["post_cooldown"]))
    # M-1: the mechanism instrument, reported per arm (no win/lose bar).
    diag = receipt.get("diagnostics_post_cooldown") or \
        receipt.get("diagnostics_step180") or {}
    mech = K13_ADDITIVE["mechanism"]
    out["M_1_mechanism_reported"] = dict(
        effective_rows_vs_additive_baseline=dict(
            measured=diag.get("effective_read_rows"),
            additive_baseline=mech["effective_read_rows"][f"seed{seed}"],
            note="the interesting instrument: did the per-channel decay "
                 "gates change the additive read's |w|-mass distribution?"),
        top_row_w_mass_share_vs_additive_baseline=dict(
            measured=(diag.get("top_row_w_mass_share") or {}).get("mean"),
            additive_baseline=mech["top_row_w_mass_share"][f"seed{seed}"],
            note="the collapse detector"),
        identity_share_vs_additive_baseline=dict(
            measured=diag.get("identity_rewrite_share"),
            additive_baseline=mech["identity_rewrite_share"][f"seed{seed}"]),
        occupancy_vs_additive_baseline=dict(
            measured=(diag.get("occupancy") or {}).get("mean_live_slots"),
            additive_baseline=mech["occupancy"][f"seed{seed}"]),
        op_mix=diag.get("op_rate_per_token_lane0"),
        gate_telemetry=dict(
            decay=(diag.get("gate_telemetry") or {}).get(
                "decay_gate_mean_distance_from_registered_scalar"),
            write=(diag.get("gate_telemetry") or {}).get(
                "write_gate_mean_distance_from_1"),
            read=(diag.get("gate_telemetry") or {}).get(
                "read_gate_mean_distance_from_1")),
        read_mask_sweep=dict(
            load_bearing_layers_ge_threshold=sum(
                1 for v in ((receipt.get("read_mask_sweep_post_cooldown")
                             or {}).get("delta_per_layer") or {}).values()
                if v >= 0.02),
            total=24))
    out["pre_state"] = ("NO directional prediction on the headline was "
                        "pre-stated (the registration's 3); the C-1/C-2/C-3 "
                        "clauses are the pre-committed readings")
    return out


# ----------------------------------------------------------------- the runner


def run_arm(args) -> dict:
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import HybridMuonX45
    from experiments.expressivity_tasks.x45_matrix_emender.\
        optimizer_bakeoff import _require_lease, _versions
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import (_routing_cross_check,
                                         split_parameters_gated)
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import (composed_parameter_delta,
                convert_model_to_composition)

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r} (expected one of "
                         f"{ARMS})")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed} (the composition "
                         f"arm runs both {PRIMARY_SEED} and {SECOND_SEED}, "
                         f"both primary -- the K13 additive rows are the "
                         f"baseline at both)")
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    seed = args.seed

    # ---- the machine: the banked winner build verbatim, then the K13
    # additive read + the K11B p3 gate stack at the mimetic-neutral birth
    # (the registration's 1)
    model, optimizer_config, summary = _build_k10(dict(_tuned_winner()), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_composition(model)
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = composed_parameter_delta(
        depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta:
        raise ValueError(f"the registration's 0.2 parameter accounting must "
                         f"hold: {parameters_before} + {delta} != "
                         f"{parameters_after}")
    if delta != 8_657_040 or parameters_after != 149_981_808 + 8_657_040:
        raise ValueError(f"the registered delta must be 8,657,040 (p3's own "
                         f"gate stack at depth 24; the additive machine's "
                         f"own 152,543,904 + p3's decay+write channels "
                         f"6,094,944); got {delta}")

    # ---- the optimizer: the banked winner lineage (hybrid Muon 4x), the
    # gate tensors on the AdamW side, cross-checked against the banked split
    # (the K11B probe's own generic plumbing, imported verbatim, arm p3)
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_gated(model, "p3")
    routing_receipt = _routing_cross_check(model, "p3", seed, False)
    peak_lr = LR_MULTIPLIER * optimizer_config["lr"]
    optimizer = HybridMuonX45(muon_params, adamw_params, lr=peak_lr,
                              betas=tuple(optimizer_config["betas"]),
                              weight_decay=optimizer_config["weight_decay"])

    anchors = dict(
        banked=BANKED, k13_additive=K13_ADDITIVE,
        registration=REGISTRATION_ANCHORS,
        banked_mechanism=_banked_mechanism_anchors(),
        bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                   winner_seed_delta=WINNER_SEED_DELTA))
    run_config = dict(
        probe="k15_composition", arm=args.arm, seed=seed,
        arm_note="the banked winner + the K13 additive read (the banked "
                 "score channel raw, live-gated; softmax and ||S_j|| "
                 "dropped) + the K11B p3 per-channel gate stack (the "
                 "f_proj decay, b_proj write, g_proj read channels) at "
                 "the mimetic-neutral birth",
        corrected_scoring=CORRECTED_SCORING,
        exposure_targets=int(args.bytes),
        model_build="velocity_ladder._build_k10 verbatim (the tuned "
                    "winner's config, the banked bake-off build) + "
                    "convert_model_to_composition (the K15 conversion; "
                    "the gates born at the registered scalar equivalents, "
                    "the g_proj read gate zero-born at exact 1)",
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
        routing="banked_gates_adamw + the p3 gate tensors on AdamW (the "
                "k11b probe's own generic split, imported verbatim)",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(before=parameters_before, added=delta,
                        total=parameters_after,
                        note="p3's own gate stack (the read channel is K13's "
                             "own committed construction, instantiated once, "
                             "never double-added); the additive read itself "
                             "is parameter-free"),
        checkpoint_steps=list(args.checkpoint_steps),
        score_form="R13-1/R13-2 under the K11B p3 channels: w_j = the banked "
                   "score channel raw, live-gated; lam_vec per channel, "
                   "g_vec write admission, r_vec read gate",
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear "
                       "cooldown (the corrected-scoring extension, same "
                       "stream); the step-180 constant-LR eval is retained "
                       "for banked comparability"),
        birth_rule="the machine-variant's own: NO banked step-1 parity (a "
                   "different function); the receipts are "
                   "BORN-AS-THE-ADDITIVE-MACHINE (every channel vs the K13 "
                   "kernel at fixed inputs, through the production "
                   "channel_decay expression) + the gates born at the "
                   "registered values + the step-1 loss asserted "
                   "bit-identical to the K13 additive arm's own same-seed "
                   "row (the nearest kin)",
        both_seeds_primary=True,
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                     k15_preregistration=str(
                         ROOT / "PREREGISTRATION_COMPOSITION.md"),
                     k15_gates_receipt=str(RECEIPTS / "k15_gates.json"),
                     k13_additive_summary=str(
                         RECEIPTS / "k13_additive_read_summary.json")),
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

    # ---- the birth receipts, BEFORE any training (the registration's 2.2)
    birth = birth_receipts(model, stream, seed, optimizer_config["microbatch"])
    if not args.smoke:
        if not birth["born_as_the_additive_machine"]["bit_identical"]:
            raise ValueError("the BORN-AS-THE-ADDITIVE-MACHINE birth receipt "
                             "failed: the composed forward must reproduce "
                             "the K13 additive kernel bit for bit at the "
                             "birth values")
        if not birth["gates_born_at_registered_values"]["all_checks"]:
            raise ValueError("the gates must be born at the registered "
                             "scalar equivalents (lam_vec == lam_inc per "
                             "channel, g_vec == 1, r_vec == 1)")
    banked_step1 = _banked_step1_loss(seed)
    k13_step1 = _k13_additive_step1(seed)
    run_config["banked_step1_loss_reference"] = banked_step1
    run_config["k13_additive_step1_loss_reference"] = k13_step1
    print(json.dumps(dict(birth_receipts=birth)), flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, peak_lr, optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"{args.arm}_seed{seed}", out / "training.jsonl",
        tuple(args.checkpoint_steps), cooldown_steps, EMA_DECAY)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # ---- the nearest-kin step-1 parity receipt (the registration's 2.2):
    # the composition's step-1 train loss must be BIT-IDENTICAL to the K13
    # additive arm's own same-seed row (the same function at birth); the
    # BANKED row's step-1 is a different function's value, quoted only.
    step1 = run["rows"][0]["loss"]
    print(f"  step-1 train loss {step1!r} (the K13 additive arm's own row's "
          f"is {k13_step1!r} -- the nearest-kin parity, asserted; the "
          f"banked same-seed row's is {banked_step1!r} -- a DIFFERENT "
          f"function's value, disclosed)", flush=True)
    if not args.smoke and step1 != k13_step1:
        raise ValueError(
            f"step-1 train loss {step1!r} != the K13 additive arm's own "
            f"same-seed row's {k13_step1!r} -- the composition's birth is "
            f"not the additive machine's birth (an engine-level parity "
            f"break between the two composed engines; escalate, not excuse)")

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
        probe="k15_composition", arm=args.arm, seed=seed,
        arm_note="the banked winner + the K13 additive read + the K11B p3 "
                 "per-channel gate stack at the mimetic-neutral birth",
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
        step1_train_loss=step1,
        step1_receipt=dict(
            arm_step1=step1, k13_additive_same_seed_step1=k13_step1,
            k13_parity=bool(step1 == k13_step1),
            banked_same_seed_step1=banked_step1,
            note="the nearest-kin parity vs the K13 additive arm's own row "
                 "is ASSERTED; the banked row's step-1 is a different "
                 "function's value and none is asserted"),
        birth_receipts=birth,
        additive_baseline=dict(
            post_cooldown=K13_ADDITIVE["post_cooldown"][f"seed{seed}"],
            constant_lr=K13_ADDITIVE["constant_lr"][f"seed{seed}"],
            ema_180=K13_ADDITIVE["ema_180"][f"seed{seed}"]),
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
    if run["status"] == "completed" and diagnostics_final is not None:
        receipt["predictions"] = evaluate_predictions(receipt)
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("arm", "seed", "status", "exposure_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll")}),
          flush=True)
    return receipt


# ------------------------------------------------------------------- assemble


def assemble(args) -> dict:
    """The across-seed adjudication: the C-1/C-2/C-3 clauses (the compose
    bar needs BOTH seeds; mixed seeds adjudicate to the CONSERVATIVE
    clause), the GDN2 rematch headline, and the mechanism rows vs the K13
    additive baseline."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    arms = {}
    for seed in (PRIMARY_SEED, SECOND_SEED):
        path = RUNS / f"composition_seed{seed}"
        if not (path / "result.json").exists():
            continue
        r = json.loads((path / "result.json").read_text())
        arms[f"composition_seed{seed}"] = dict(
            arm="composition", seed=seed, status=r["status"],
            constant_lr=r["final_eval_nll_constant_lr"],
            post_cooldown=r["final_eval_nll_post_cooldown"],
            ema_180=r["ema_eval_nll"],
            cooldown_drop=r["cooldown_drop"],
            step1_train_loss=r["step1_train_loss"],
            step1_k13_parity=r["step1_receipt"]["k13_parity"],
            additive_baseline_post_cooldown=r["additive_baseline"][
                "post_cooldown"],
            born_as_additive_machine=r["birth_receipts"][
                "born_as_the_additive_machine"]["bit_identical"],
            identity_share=(r["diagnostics_post_cooldown"] or {}).get(
                "identity_rewrite_share"),
            effective_rows=(r["diagnostics_post_cooldown"] or {}).get(
                "effective_read_rows"),
            top_row_share=(r["diagnostics_post_cooldown"] or {}).get(
                "top_row_w_mass_share", {}).get("mean"),
            occupancy=(r["diagnostics_post_cooldown"] or {}).get(
                "occupancy", {}).get("mean_live_slots"),
            gate_drifts={k: ((r["diagnostics_post_cooldown"] or {}).get(
                "gate_telemetry") or {}).get(k) for k in (
                "decay_gate_mean_distance_from_registered_scalar",
                "write_gate_mean_distance_from_1",
                "read_gate_mean_distance_from_1")},
            train_seconds_constant=r["train_seconds_constant"],
            parameters=r["parameters"])
    if not arms:
        raise ValueError("no completed arm receipts to assemble")

    bar = REGISTRATION_ANCHORS["compose_bar"]
    improvements = {}
    for seed in (PRIMARY_SEED, SECOND_SEED):
        row = arms.get(f"composition_seed{seed}")
        if row and row["post_cooldown"] is not None:
            improvements[seed] = (row["additive_baseline_post_cooldown"]
                                 - row["post_cooldown"])
    clause = None
    if len(improvements) == 2:
        if all(v >= bar for v in improvements.values()):
            clause = "C-1 COMPOSE"
        elif all(v <= -bar for v in improvements.values()):
            clause = "C-3 INTERFERE"
        else:
            clause = "C-2 PARITY (read-form-dependence)"
    # the headline: the composition's best post-cooldown row vs GDN2
    best_seed = min((s for s in improvements),
                    key=lambda s: arms[f"composition_seed{s}"]["post_cooldown"],
                    default=None)
    headline = None
    if best_seed is not None:
        post = arms[f"composition_seed{best_seed}"]["post_cooldown"]
        gdn2 = BANKED["gdn2"]["post_cooldown"]
        banked = BANKED["post_cooldown"]["seed3402"]
        headline = dict(
            best_post_cooldown=post, seed=best_seed,
            vs_gdn2=post - gdn2,
            fraction_of_corrected_gap_closed=max(0.0, banked - post)
            / (banked - gdn2),
            note="the honest rematch row: the composed machine vs GDN2's "
                 "post-cooldown 1.3522 (the remaining gap)")
    verdicts = {
        "C-1 COMPOSE": ("the two levers COMPOSE: the composition beats its "
                        "own additive baseline by >= 0.01 at both seeds; "
                        "the new best machine of the program is the "
                        "composition; the honest rematch row vs GDN2 "
                        "1.3522 is the headline"),
        "C-2 PARITY (read-form-dependence)": (
            "the gates' +0.010 was READ-FORM-DEPENDENT: the softmax read "
            "needed the channel gates, the additive read does not -- the "
            "levers do NOT compose and that is the interaction finding, "
            "receipted with the mechanism instrument attached; the "
            "additive read alone remains the best machine"),
        "C-3 INTERFERE": ("the gates INTERFERE with the additive read: the "
                          "per-channel decay gates modulate exactly the "
                          "channels the additive read now sums, and the "
                          "composition is worse than its own baseline at "
                          "both seeds -- receipted as the interference, "
                          "the mechanism instrument the inspection entry"),
    }
    summary = dict(
        probe="k15_composition assemble",
        corrected_scoring=CORRECTED_SCORING,
        score_form="K13's additive read under K11B p3's per-channel "
                   "decay/write/read gate stack (the gates born at the "
                   "registered scalar equivalents)",
        anchors=dict(banked=BANKED, k13_additive=K13_ADDITIVE,
                     registration=REGISTRATION_ANCHORS),
        arms=arms,
        improvements_vs_additive_baseline=improvements,
        compose_bar=bar,
        clause=clause,
        verdict=verdicts.get(clause),
        headline_rematch_vs_gdn2=headline,
        notes=dict(
            scoring="PRIMARY: post-cooldown (24-step linear); SECONDARY: "
                    "EMA-0.99 at step 180 (ranking-only); constant-180 "
                    "retained for banked comparability",
            seeds="BOTH seeds primary (the composition's own baseline "
                  "exists at both seeds); the compose bar needs BOTH",
            adjudication="the specific pre-stated clause governs the "
                         "verdict (the K13 clause-conflict precedent); "
                         "mixed seeds adjudicate to the CONSERVATIVE "
                         "clause -- if the clauses disagree across seeds, "
                         "the weaker reading governs and both rows are "
                         "quoted next to each other",
            refutation_clause="if the composition lands at parity or worse "
                              "vs its own additive baseline, the levers do "
                              "not compose and that is reported as the "
                              "finding, not excused"))
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(arms={k: dict(post_cooldown=v["post_cooldown"],
                                        additive_baseline=v[
                                            "additive_baseline_post_cooldown"],
                                        constant=v["constant_lr"],
                                        ema=v["ema_180"],
                                        effective_rows=v["effective_rows"],
                                        step1_k13_parity=v["step1_k13_parity"])
                               for k, v in arms.items()},
                         improvements=improvements, clause=clause,
                         headline=headline), indent=1), flush=True)
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
                        "steps, 3 cooldown steps, no birth assertions")
    s = sub.add_parser("assemble")
    s.add_argument("--out", default=str(RECEIPTS
                                        / "k15_composition_summary.json"))
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
