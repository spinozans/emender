"""K13 -- the additive-read quality probe: the ONE arm (`additive`) at the
matched-exposure protocol under the CORRECTED scoring, vs the banked winner
config (hybrid Muon 4x lineage), at BOTH seeds (3401, 3402) -- no birth
parity is possible (a different function), so both seeds are primary
evidence.

THE QUESTION THIS STAGE EXISTS FOR
(``PREREGISTRATION_ADDITIVE_READ.md`` 3, committed before any probe
measurement): does GDN2's ADDITIVE read form -- the row-softmax replaced by
the additive sum of the banked score channel's raw similarities, gated by
the committed g_proj read gate -- move the read-dilution instrument
(M-R1), and does it close any of the ~0.08-nat corrected gap the C5
write/read form owns (M-R2)?  NO directional prediction is pre-stated on
the headline; the mechanism instrument is reported per arm (M-R3).

STANDING AMENDMENTS applied verbatim (imported): the 24-step linear cooldown
PRIMARY, the EMA-0.99 iterate at step 180 SECONDARY (ranking-only), the
constant-LR step-180 eval RETAINED for banked comparability.  Both
comparison anchors are quoted everywhere: the constant-LR banked best
**1.4899048124040877** (seed 3402; the same config's seed-3401 row is
**1.4907409463609969**) and the corrected post-cooldown banked best
**1.4338248457227434** (seed 3402); GDN2's post-cooldown
**1.3521734476089478** is the gap reference (the corrected gap is 0.0816).

THE BIRTH RECEIPTS (the machine-variant's own, the registration's 2.2 -- no
step-1 loss parity is possible and none is asserted): (1)
DECISIONS-BIT-IDENTICAL -- on the first training batch, layer for layer, at
the arm's own inputs, the additive scan's recorded schedule is `torch.equal`
to the BANKED read's own kernel launch's (the read form cannot feed op
selection); (2) THE BIRTH-MAGNITUDE DIAGNOSTIC (ruling R13-5) -- the
per-head read-output RMS, additive vs the banked read, on the same inputs;
(3) the gate is born at exact 1 (`r_vec == 1`), so the arm is the PURE
additive read.  The arm's step-1 train loss is a DIFFERENT function's value
and is logged next to the banked row's with the disclosure attached.

The mechanism instrument (the registration's 2.3): the K10/K11 walk stats +
identity-rewrite share + occupancy + op mix at BOTH the step-180 constant
snapshot and the post-cooldown final, PLUS the additive read's own channels
(ruling R13-4: the |w|-mass effective rows, the top-row |w|-mass share, the
mean live-row count), the gate telemetry (the g_proj read gate's distance
from 1, per layer), and the decorative-store sweep
(`side_by_side.k10_read_mask_sweep`, imported verbatim) at both states.

Provenance (the lane's own rules): the engine is committed and read-only
(`k13_additive_read.py`); `velocity_ladder._build_k10`,
`protocol_correction._train_corrected`/`CORRECTED_SCORING`,
`routing_symmetry.split_parameters_winner`, `k10_probe_separated_keys`
instruments, `k11_probe_learned_gates`' banked-anchor loaders,
`side_by_side.k10_read_mask_sweep` and `matched_arms` helpers are imported
verbatim; no `k10_*`/`k11*`/`k12_*`/`x46_*` source, no PREREGISTRATION, no
`lattice.py` is edited.

Deliverables: per-arm receipts under `k13_runs/additive_seed<seed>/` and
the assemble summary `receipts/k13_additive_read_summary.json` (the
corrected ladder + the M-R1..M-R3 adjudication).

Usage (see `run_k13_additive_read_leased.sh` for the detached leased driver,
cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k13_probe_additive_read \\
        arm --arm additive --seed 3401 --output k13_runs/additive_seed3401
"""
from __future__ import annotations

import argparse
import json
import math
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
RUNS = ROOT / "k13_runs"

from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    OP_NAMES)  # noqa: E402

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

# The banked comparison anchors (cited, never re-run here; the step-1 losses
# and the mechanism anchors are loaded from the banked result.json files at
# run time via the committed k11 probe's own loaders).
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
                   "the ~0.08 residual this stage dissects)"))
# The registration's 3 pre-stated bars, verbatim.
REGISTRATION_ANCHORS = dict(
    mr1_factor_bar=2.0,                # effective rows >= 2x the seed-matched
                                        # banked anchor, at BOTH seeds
    mr2_close_bar=0.02,                 # closes >= 0.02 of the corrected gap
    sweep_layers_ge_threshold=7,        # the AdamW-era load-bearing anchor
    sweep_threshold=0.02, sweep_near_band=0.0125, sweep_total=24,
    effective_rows_side_by_side=5.6,     # the read-dilution finding's own row
    bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
               winner_seed_delta=WINNER_SEED_DELTA))

# The one arm (the registration's 2.1): the banked winner build + the
# additive read + the zero-born g_proj gate.
ARMS = ("additive",)


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


def _tuned_winner() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import TUNED_WINNER
    return dict(TUNED_WINNER)


# ----------------------------------------------------------- the routing split


def split_parameters_additive(model):
    """The banked winner's own routing (``banked_gates_adamw``) with the k13
    gate tensors (``g_proj`` + ``read_scalar``, the committed k11b read
    channel's own names) on the hybrid's AdamW side -- the incumbent's own
    convention (the DECAY gate's side: "a gate, not a hidden map").  The
    caller cross-checks the result against ``split_parameters_winner`` on the
    un-converted twin (the banked split must be reproduced exactly, gate
    names aside)."""
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import (_DECAY_BIAS, _DECAY_WEIGHT, _EMBED_HEAD, _MIXER_MATRIX, _MLP,
                _NORMS)
    gate_names = ["g_proj.0.weight", "g_proj.1.weight", "g_proj.1.bias",
                  "read_scalar"]
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


def _routing_cross_check(model, seed: int) -> dict:
    """The no-drift receipt: an un-converted twin of the SAME build (same
    seed, same build order) must route EXACTLY as the banked winner's own
    split routes it; the additive model's split must equal that plus exactly
    the arm's gate tensors on the AdamW side."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import split_parameters_winner
    muon, adamw, muon_names, adamw_names = split_parameters_additive(model)
    twin, _cfg, _summary = _build_k10(dict(_tuned_winner()), seed)
    _t_muon, _t_adamw, twin_muon, twin_adamw = split_parameters_winner(
        twin, "banked_gates_adamw")
    del twin, _t_muon, _t_adamw
    torch.cuda.empty_cache()
    expected_gate_names = sorted(
        f"blocks.{i}.mixer.{name}"
        for i in range(len(model.blocks))
        for name in ("g_proj.0.weight", "g_proj.1.weight", "g_proj.1.bias",
                     "read_scalar"))
    gained = sorted(set(adamw_names) - set(twin_adamw))
    if sorted(muon_names) != sorted(twin_muon):
        raise ValueError("the additive model's Muon group drifted from the "
                         "banked winner's own split")
    if gained != expected_gate_names:
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
        cross_check="the un-converted twin's banked_gates_adamw split "
                    "reproduced exactly (name-for-name); the AdamW side "
                    "gains exactly the arm's gate tensors (asserted "
                    "in-process)")


# -------------------------------------------------------- the birth receipts


def birth_receipts(model, stream, seed: int, microbatch: int) -> dict:
    """The machine-variant's own birth receipts (the registration's 2.2), run
    BEFORE any training on the FIRST train batch (the batch the step-1 loss
    sees):

    1. DECISIONS-BIT-IDENTICAL: layer for layer, at the arm's own inputs,
       the additive scan's recorded schedule (slot/op/erase/live_after,
       payload lanes, pre-decision liveness) is `torch.equal` to the BANKED
       read's own committed kernel launch's on the same inputs -- the read
       form cannot feed op selection.  The banked twin's projections are
       asserted bit-equal to the arm's first (the conversion shares them).
    2. THE BIRTH-MAGNITUDE DIAGNOSTIC (R13-5): the per-head read-output RMS,
       additive vs the banked read, on the same inputs -- the honest
       ~O(live-rows)x consequence of removing the simplex.
    3. The gate is born at exact 1: `r_vec == 1` bit for bit, so the arm is
       the PURE additive read (asserted per mixer on the batch).
    """
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys \
        import _k10_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read \
        import _k13_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
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
                payload, write_keys, query_keys, decay = _project_keyed(
                    mixer, xin)
                payload, write_keys = payload.float(), write_keys.float()
                query_keys, decay = query_keys.float(), decay.float()
                state = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.value_dim, dtype=torch.float32,
                                   device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out_add = _k13_forward_launch(
                    payload, write_keys, query_keys, decay, state, live,
                    keys, pol32, mixer.rank, mixer.segment_tokens,
                    collect=True, warps=8, stages=3)
                out_bank = _k10_forward_launch(
                    payload, write_keys, query_keys, decay, state, live,
                    keys, pol32, mixer.rank, mixer.segment_tokens,
                    collect=True, warps=8, stages=3)
            flips = lanes = 0
            for name in ("slot", "erase", "op", "live_after"):
                a, b = out_bank[name], out_add[name]
                flips += int((a != b).sum())
                lanes += a.numel()
            for name in ("payload", "live_pre", "state", "live", "keys"):
                a, b = out_bank[name], out_add[name]
                flips += int((a != b).sum())
                lanes += a.numel()
            r_add = out_add["reads"].float()
            r_bank = out_bank["reads"].float()
            per_layer.append(dict(
                layer=layer_index, flips=flips, lanes=lanes,
                read_rms_additive_per_head=[
                    float(v) for v in
                    r_add.pow(2).mean(dim=(0, 1, 3)).sqrt()],
                read_rms_banked_per_head=[
                    float(v) for v in
                    r_bank.pow(2).mean(dim=(0, 1, 3)).sqrt()],
                mean_live_rows=float(
                    out_add["live_pre"].float().sum(-1).mean())))
            nonlocal flips_total, lanes_total, rms_add, rms_bank
            flips_total += flips
            lanes_total += lanes
            rms_add.append(r_add.pow(2).mean(dim=(0, 1, 3)).sqrt())
            rms_bank.append(r_bank.pow(2).mean(dim=(0, 1, 3)).sqrt())
        return hook

    # receipt 3: the gate is born at exact 1 on this batch (r_vec == 1)
    r_vec_checks = []
    with torch.no_grad():
        for block in model.blocks:
            mixer = block.mixer
            r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                * torch.sigmoid(mixer.g_proj(x))
            r_vec_checks.append(bool(
                torch.equal(r_vec, torch.ones_like(r_vec))))

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
        note="the machine-variant's birth receipts on the FIRST train batch "
             "(the batch the step-1 loss sees): NO step-1 loss parity is "
             "possible (a different function) and none is asserted",
        decisions_bit_identical=dict(
            flips=flips_total, lanes=lanes_total,
            bit_identical=bool(flips_total == 0),
            note="the additive scan's recorded schedule vs the BANKED read's "
                 "own committed kernel launch at identical inputs, every "
                 "layer (the read form cannot feed op selection)"),
        birth_magnitude=dict(
            per_head_read_rms_additive_mean=[
                float(v) for v in rms_add_t.mean(0)],
            per_head_read_rms_banked_mean=[
                float(v) for v in rms_bank_t.mean(0)],
            additive_over_banked_per_head_mean=[
                float(a / max(b, 1e-30))
                for a, b in zip(rms_add_t.mean(0), rms_bank_t.mean(0))],
            note="R13-5: the honest ~O(live-rows)x magnitude consequence of "
                 "removing the simplex, stated not guessed; any early-loss "
                 "divergence is reported with this attached"),
        gate_born_at_exact_1=all(r_vec_checks),
        per_layer=per_layer[:1] + per_layer[-1:],   # the first and last layer
        mean_live_rows_birth=float(
            sum(p["mean_live_rows"] for p in per_layer) / len(per_layer)))
    return receipt


# --------------------------------------------------- the mechanism instrument


def _k13_walk_stats(out, rank, mixer, payload=None, decay=None):
    """The K10/K11 mechanism metrics from a collect launch's recorded
    channels, with the additive read's own |w|-mass instrument (R13-4):
    effective rows = e^H of the |w|-mass distribution over live rows, plus
    the top-row |w|-mass share (the collapse detector)."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        MatrixSchedule, op_histogram)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _identity_share
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    lane0 = torch.bincount(schedule.op[..., 0].reshape(-1).to("cpu"),
                           minlength=len(OP_NAMES))
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
    if payload is not None and decay is not None:
        share, n_correct = _identity_share(schedule, payload, decay, rank)
        stats["identity_rewrite_share"] = share
        stats["identity_n_correct"] = n_correct
    return stats, schedule


def _agg_walk(per_layer):
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        OP_NAMES)
    decisions = sum(s["decisions"] for _, s in per_layer)
    by_layer = {}
    for layer, stats in per_layer:
        by_layer.setdefault(layer, []).append(stats)
    layers = {}
    for layer, stats_list in sorted(by_layer.items()):
        layer_decisions = sum(s["decisions"] for s in stats_list)
        entry = dict(
            op_rate_per_token={OP_NAMES[i]:
                               sum(s["lane0"][i] for s in stats_list)
                               / max(1, layer_decisions)
                               for i in range(len(OP_NAMES))},
            occupancy_mean_live_slots=float(
                sum(s["occupancy_sum"] for s in stats_list)
                / max(1, sum(s["occupancy_n"] for s in stats_list))),
            read_weight_entropy_nats=float(
                sum(s["entropy_sum"] for s in stats_list)
                / max(1, sum(s["entropy_n"] for s in stats_list))),
            top_row_share=float(
                sum(s["top_share_sum"] for s in stats_list)
                / max(1, sum(s["entropy_n"] for s in stats_list))))
        if "identity_rewrite_share" in stats_list[0]:
            entry["identity_rewrite_share"] = float(
                sum(s["identity_rewrite_share"] * s["identity_n_correct"]
                    for s in stats_list)
                / max(1, sum(s["identity_n_correct"] for s in stats_list)))
        layers[layer] = entry
    hist = {name: sum(s["hist"][name] for _, s in per_layer)
            for name in OP_NAMES}
    summary = dict(
        op_rate_per_token_lane0={OP_NAMES[i]:
                                 sum(s["lane0"][i] for _, s in per_layer)
                                 / decisions
                                 for i in range(len(OP_NAMES))},
        op_histogram_all_lanes=hist,
        occupancy=dict(
            note="live slots at every token, from the walk's recorded "
                 "pre-decision liveness channel",
            mean_live_slots=float(sum(s["occupancy_sum"] for _, s in per_layer)
                                  / max(1, decisions)),
            rank=int(per_layer[0][1]["rank"])),
        read_weight_entropy=dict(
            units="nats over the |w|-mass distribution on live rows (R13-4: "
                  "the additive weights are signed)",
            mean=float(sum(s["entropy_sum"] for _, s in per_layer)
                       / max(1, sum(s["entropy_n"] for _, s in per_layer)))),
        top_row_w_mass_share=dict(
            units="the max |w|-mass share per token (the collapse detector)",
            mean=float(sum(s["top_share_sum"] for _, s in per_layer)
                       / max(1, sum(s["entropy_n"] for _, s in per_layer)))),
        per_layer=layers)
    if "identity_rewrite_share" in per_layer[0][1]:
        summary["identity_rewrite_share"] = float(
            sum(s["identity_rewrite_share"] * s["identity_n_correct"]
                for _, s in per_layer)
            / max(1, sum(s["identity_n_correct"] for _, s in per_layer)))
        summary["identity_n_correct"] = int(
            sum(s["identity_n_correct"] for _, s in per_layer))
    summary["effective_read_rows"] = float(torch.exp(torch.tensor(
        summary["read_weight_entropy"]["mean"])))
    return summary


def k13_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED arm's mechanism receipts: the walk stats (op rates,
    occupancy, the |w|-mass effective rows + the collapse detector, the
    identity-rewrite share) on the k13 collect launch, PLUS the g_proj gate
    telemetry (the learned read gate's distance from 1, per layer)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _probe_windows
    from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read \
        import _k13_forward_launch
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
                payload, write_keys, query_keys, decay = _project_keyed(
                    mixer, xin)
                payload, write_keys = payload.float(), write_keys.float()
                query_keys, decay = query_keys.float(), decay.float()
                state = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.value_dim, dtype=torch.float32,
                                   device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out = _k13_forward_launch(
                    payload, write_keys, query_keys, decay, state, live,
                    keys, pol32, mixer.rank, mixer.segment_tokens,
                    collect=True, warps=8, stages=3)
                r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.g_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
            with torch.no_grad():
                stats, _schedule = _k13_walk_stats(
                    out, mixer.rank, mixer, payload=payload, decay=decay)
                per_layer.append((layer_index, stats))
                r32 = r_vec.float()
                gate_rows.append(dict(
                    layer=layer_index,
                    read_gate=dict(
                        mean=float(r32.mean()),
                        mean_distance_from_1=float((r32 - 1).abs().mean()),
                        min=float(r32.min()), max=float(r32.max()))))
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
                            regime="eval, bf16 autocast, the k13 persistent "
                                   "walk the arm trains on (collect=True)",
                            probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the g_proj read gate's "
             "distance from 1 (did training pull r_vec off the born "
             "identity), the k11b readout convention",
        per_layer={str(r["layer"]): r["read_gate"] for r in gate_rows},
        read_gate_mean_distance_from_1=float(sum(
            r["read_gate"]["mean_distance_from_1"] for r in gate_rows)
            / max(1, len(gate_rows))),
        n_layers=len(gate_rows))
    return summary


# ------------------------------------------------------------- the predictions


def evaluate_predictions(receipt: dict) -> dict:
    """The registration's 3 pre-stated predictions, evaluated verbatim
    against the matched anchors (no directional prediction on the headline
    was pre-stated)."""
    seed = receipt["seed"]
    anchors = receipt["anchors"]
    banked_mech = anchors["banked_mechanism"]
    band = ENDPOINT_VARIANCE_BAND
    out = {}

    # ---- M-R1: the read-dilution instrument MOVES (|w|-mass effective
    # rows, R13-4) -- bar: >= 2x the seed-matched banked anchor at BOTH
    # seeds (the assemble evaluates both; the single-seed row states its
    # own half).
    diag = receipt.get("diagnostics_post_cooldown") or \
        receipt.get("diagnostics_step180") or {}
    effective = diag.get("effective_read_rows")
    banked_rows = float(math.exp(
        banked_mech[f"seed{seed}_step180"]["read_entropy_nats"]))
    if effective is not None:
        out["M-R1_read_dilution_moves"] = dict(
            measured_effective_rows=effective,
            seed_matched_banked_effective_rows=banked_rows,
            factor=effective / banked_rows,
            bar_factor=REGISTRATION_ANCHORS["mr1_factor_bar"],
            met_at_this_seed=bool(
                effective >= REGISTRATION_ANCHORS["mr1_factor_bar"]
                * banked_rows),
            top_row_w_mass_share=diag.get(
                "top_row_w_mass_share", {}).get("mean"),
            side_by_side_anchor=REGISTRATION_ANCHORS[
                "effective_rows_side_by_side"],
            note="MET needs BOTH seeds; the collapse detector is the top-row "
                 "|w|-mass share -- a collapse below the bar REFINES the C5 "
                 "read to 'dilution is not the softmax's doing'")

    # ---- M-R2: the NLL question, the attribution this stage exists for.
    post = receipt.get("final_eval_nll_post_cooldown")
    constant = receipt.get("final_eval_nll_constant_lr")
    ema = receipt.get("ema_eval_nll")
    corrected_anchor = BANKED["post_cooldown"]["seed3402"]
    banked_constant = (BANKED["constant_lr"]["seed3401"] if seed == 3401
                       else BANKED["constant_lr"]["seed3402"])
    verdict = None
    if post is not None:
        delta = post - corrected_anchor
        if delta <= -REGISTRATION_ANCHORS["mr2_close_bar"]:
            verdict = ("CLOSES >= 0.02 of the corrected gap: the C5 "
                       "attribution refines to THE READ FORM (competitive "
                       "softmax dilution); the terminal reading is amended "
                       "and the read form joins the leaner registration")
        elif abs(delta) <= band:
            verdict = ("PARITY inside the 0.075 endpoint-variance band: the "
                       "read form is NOT the residual's owner and the "
                       "superposition story stands with one more receipt")
        else:
            verdict = ("WORSE, outside the band on the losing side: "
                       "reported with the birth-magnitude diagnostic "
                       "attached (R13-5) -- the magnitude-blowup-vs-form "
                       "attribution is then an inspectable question for a "
                       "named follow-up (a gain-tuned read), not a post-hoc "
                       "rescue")
        out["M-R2_nll_attribution"] = dict(
            post_cooldown_primary=post,
            corrected_anchor=corrected_anchor, delta=delta,
            close_bar=REGISTRATION_ANCHORS["mr2_close_bar"],
            band=band, verdict=verdict,
            seed_matched_constant_delta=(constant - banked_constant
                                         if constant is not None else None),
            ema_seed3402_crosscheck=(ema if ema is not None else None),
            gdn2_gap_reference=dict(
                post_cooldown=BANKED["gdn2"]["post_cooldown"],
                banked_gap=corrected_anchor - BANKED["gdn2"]["post_cooldown"]))

    # ---- M-R3: the mechanism instrument, reported per arm (P2, no bar).
    sweep = receipt.get("read_mask_sweep_post_cooldown") or \
        receipt.get("read_mask_sweep_step180") or {}
    load_bearing = None
    if sweep:
        threshold = REGISTRATION_ANCHORS["sweep_threshold"]
        load_bearing = sum(
            1 for v in sweep["delta_per_layer"].values() if v >= threshold)
        out["M-R3_mechanism_reported"] = dict(
            identity_share_vs_banked=dict(
                measured=diag.get("identity_rewrite_share"),
                seed_matched_banked=banked_mech[f"seed{seed}_step180"][
                    "identity_rewrite_share"]),
            occupancy_vs_banked=dict(
                measured=diag.get("occupancy", {}).get("mean_live_slots"),
                seed_matched_banked=banked_mech[f"seed{seed}_step180"][
                    "occupancy_mean_live_slots"]),
            op_mix_vs_banked=dict(
                measured=diag.get("op_rate_per_token_lane0"),
                seed_matched_banked=banked_mech[f"seed{seed}_step180"][
                    "op_rate_per_token"]),
            decorative_store_sweep=dict(
                load_bearing_layers_ge_threshold=load_bearing,
                threshold=threshold,
                anchor_load_bearing=REGISTRATION_ANCHORS[
                    "sweep_layers_ge_threshold"],
                anchor_total=REGISTRATION_ANCHORS["sweep_total"],
                more_stores_load_bearing=bool(
                    load_bearing > REGISTRATION_ANCHORS[
                        "sweep_layers_ge_threshold"]),
                all_layers_masked_delta=sweep.get("all_layers_delta")),
            gate_telemetry=diag.get("gate_telemetry", {}).get(
                "read_gate_mean_distance_from_1"))
    out["pre_state"] = ("NO directional prediction on the headline was "
                        "pre-stated (the registration's 3); M-R2's "
                        "two-sided clause is the pre-committed reading")
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
        optimizer_bakeoff import MICROBATCH, _require_lease, _versions
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.k13_additive_read \
        import (convert_model_to_additive_read,
                read_gate_parameter_delta)

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r} (expected one of "
                         f"{ARMS})")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed} (the additive arm "
                         f"runs both {PRIMARY_SEED} and {SECOND_SEED}, both "
                         f"primary -- no birth parity is possible)")
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    seed = args.seed

    # ---- the machine: the banked winner build verbatim, then the additive
    # read + the zero-born g_proj gate (the registration's 1)
    model, optimizer_config, summary = _build_k10(dict(_tuned_winner()), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_additive_read(model)
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = read_gate_parameter_delta(
        depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta:
        raise ValueError(f"the registration's 1.5 parameter accounting must "
                         f"hold: {parameters_before} + {delta} != "
                         f"{parameters_after}")
    if delta != 2_562_096 or parameters_after != 149_981_808 + 2_562_096:
        raise ValueError(f"the registered delta must be 2,562,096 (the "
                         f"k11b read channel at depth 24); got {delta}")

    # ---- the optimizer: the banked winner lineage (hybrid Muon 4x), the
    # gate tensors on the AdamW side, cross-checked against the banked split
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_additive(model)
    routing_receipt = _routing_cross_check(model, seed)
    peak_lr = LR_MULTIPLIER * optimizer_config["lr"]
    optimizer = HybridMuonX45(muon_params, adamw_params, lr=peak_lr,
                              betas=tuple(optimizer_config["betas"]),
                              weight_decay=optimizer_config["weight_decay"])

    anchors = dict(
        banked=BANKED, registration=REGISTRATION_ANCHORS,
        banked_mechanism=_banked_mechanism_anchors(),
        bands=dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
                   winner_seed_delta=WINNER_SEED_DELTA))
    anchors["seed_matched_step180"] = anchors["banked_mechanism"][
        f"seed{seed}_step180"]
    run_config = dict(
        probe="k13_additive_read", arm=args.arm, seed=seed,
        arm_note="the banked winner + the additive read (w_j = the banked "
                 "score channel raw, live-gated; softmax and ||S_j|| "
                 "dropped) + the zero-born g_proj read gate",
        corrected_scoring=CORRECTED_SCORING,
        exposure_targets=int(args.bytes),
        model_build="velocity_ladder._build_k10 verbatim (the tuned "
                    "winner's config, the banked bake-off build) + "
                    "convert_model_to_additive_read (the K13 conversion; "
                    "the k11b g_proj read gate zero-born at exact 1)",
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
        routing="banked_gates_adamw + the k13 gate tensors on AdamW",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(before=parameters_before, added=delta,
                        total=parameters_after,
                        note="the k11b R4 precedent: the 0.1% capacity "
                             "tolerance is dropped for a real GDN2-granular "
                             "gate transplant (+1.70%); the read-form change "
                             "itself is parameter-free"),
        checkpoint_steps=list(args.checkpoint_steps),
        score_form="R13-1: w_j = read_scale*cos(q_key, K_j) VERBATIM "
                   "(inherited-from-banked, a gain now, not re-tuned); "
                   "R13-2: no ||S_j|| factor (magnitude rides S[j])",
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear "
                       "cooldown (the corrected-scoring extension, same "
                       "stream); the step-180 constant-LR eval is retained "
                       "for banked comparability"),
        birth_rule="the machine-variant's own: NO step-1 loss parity (a "
                   "different function); the receipts are "
                   "DECISIONS-BIT-IDENTICAL vs the banked read's kernel at "
                   "fixed inputs + the birth-magnitude diagnostic (R13-5) "
                   "+ the gate born at exact 1",
        both_seeds_primary=True,
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                     k13_preregistration=str(
                         ROOT / "PREREGISTRATION_ADDITIVE_READ.md"),
                     k13_gates_receipt=str(RECEIPTS / "k13_gates.json")),
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
        if not birth["decisions_bit_identical"]["bit_identical"]:
            raise ValueError("the DECISIONS-BIT-IDENTICAL birth receipt "
                             "failed: the read form must not touch the "
                             "recorded schedule at fixed inputs")
        if not birth["gate_born_at_exact_1"]:
            raise ValueError("the g_proj gate must be born at exact 1")
    banked_step1 = _banked_step1_loss(seed)
    run_config["banked_step1_loss_reference"] = banked_step1
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

    # the disclosed no-parity note: the step-1 loss is a different
    # function's value; the banked row's is quoted next to it, never
    # asserted equal.
    step1 = run["rows"][0]["loss"]
    print(f"  step-1 train loss {step1!r} (the banked same-seed row's is "
          f"{banked_step1!r} -- a DIFFERENT function's value, disclosed per "
          f"the registration)", flush=True)

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
            diagnostics_180 = k13_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_180 = k10_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = k13_probe_diagnostics(
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
        probe="k13_additive_read", arm=args.arm, seed=seed,
        arm_note="the banked winner + the additive read + the zero-born "
                 "g_proj gate",
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
        step1_disclosure=dict(
            arm_step1=step1, banked_same_seed_step1=banked_step1,
            note="NO birth parity (a different function, the registration's "
                 "2.2); the birth receipts below carry the honesty instead"),
        birth_receipts=birth,
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
    """The corrected ladder: the one arm at both seeds vs both quoted
    anchors, the M-R1..M-R3 adjudication across seeds (M-R1's bar needs
    BOTH seeds), and the pre-stated verdicts applied verbatim."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    arms = {}
    for seed in (PRIMARY_SEED, SECOND_SEED):
        path = RUNS / f"additive_seed{seed}"
        if not (path / "result.json").exists():
            continue
        r = json.loads((path / "result.json").read_text())
        arms[f"additive_seed{seed}"] = dict(
            arm="additive", seed=seed, status=r["status"],
            constant_lr=r["final_eval_nll_constant_lr"],
            post_cooldown=r["final_eval_nll_post_cooldown"],
            ema_180=r["ema_eval_nll"],
            cooldown_drop=r["cooldown_drop"],
            step1_train_loss=r["step1_train_loss"],
            birth_decisions_bit_identical=r["birth_receipts"][
                "decisions_bit_identical"]["bit_identical"],
            birth_read_rms_additive_over_banked=r["birth_receipts"][
                "birth_magnitude"]["additive_over_banked_per_head_mean"],
            identity_share=(r["diagnostics_post_cooldown"] or {}).get(
                "identity_rewrite_share"),
            effective_rows=(r["diagnostics_post_cooldown"] or {}).get(
                "effective_read_rows"),
            top_row_share=(r["diagnostics_post_cooldown"] or {}).get(
                "top_row_w_mass_share", {}).get("mean"),
            occupancy=(r["diagnostics_post_cooldown"] or {}).get(
                "occupancy", {}).get("mean_live_slots"),
            train_seconds_constant=r["train_seconds_constant"],
            parameters=r["parameters"])
    if not arms:
        raise ValueError("no completed arm receipts to assemble")

    banked_mech = _banked_mechanism_anchors()
    # M-R1 across seeds: the bar needs BOTH seeds.
    mr1 = None
    if len(arms) == 2:
        factors = {}
        for seed in (PRIMARY_SEED, SECOND_SEED):
            row = arms[f"additive_seed{seed}"]
            banked_rows = math.exp(
                banked_mech[f"seed{seed}_step180"]["read_entropy_nats"])
            factors[seed] = (row["effective_rows"] / banked_rows
                             if row["effective_rows"] is not None else None)
        both = all(f is not None and f >= REGISTRATION_ANCHORS["mr1_factor_bar"]
                   for f in factors.values())
        collapsed = all(f is not None and f < 1.0 for f in factors.values())
        mr1 = dict(
            factors_vs_seed_matched_banked=factors,
            bar_factor=REGISTRATION_ANCHORS["mr1_factor_bar"],
            met_at_both_seeds=bool(both),
            refuted_by_collapse=bool(collapsed and not both),
            verdict=("MET -- the competitive->additive signature: the "
                     "instrument moved toward the live-row count"
                     if both else
                     ("REFUTED -- the additive read collapses to few "
                      "dominant rows anyway; the C5 read refines to "
                      "'dilution is not the softmax's doing'"
                      if collapsed else
                      "NOT MET at both seeds (between the bar and "
                      "collapse) -- reported as measured")))

    summary = dict(
        probe="k13_additive_read assemble",
        corrected_scoring=CORRECTED_SCORING,
        score_form="R13-1/R13-2: the banked score channel verbatim, raw and "
                   "live-gated; softmax and ||S_j|| gone; the k11b g_proj "
                   "gate zero-born",
        anchors=dict(banked=BANKED, registration=REGISTRATION_ANCHORS,
                     banked_mechanism=banked_mech),
        arms=arms,
        M_R1=mr1,
        notes=dict(
            scoring="PRIMARY: post-cooldown (24-step linear); SECONDARY: "
                    "EMA-0.99 at step 180 (ranking-only); constant-180 "
                    "retained for banked comparability",
            seeds="BOTH seeds primary (no birth parity possible); the "
                  "corrected anchor exists at 3402, so M-R2's primary "
                  "reading is the 3402 post-cooldown row with the 3401 "
                  "constant/EMA rows as cross-checks",
            refutation_clause="if NO mechanism number moves AND the NLL "
                              "does not improve beyond parity, the "
                              "read-form lever does not own the residual "
                              "and that is reported as refuted, not "
                              "excused"))
    # M-R2's across-seed reading: the primary seed-3402 post-cooldown row
    # against the corrected anchor, the seed-3401 rows quoted.
    a3402 = arms.get(f"additive_seed{SECOND_SEED}")
    if a3402 and a3402["post_cooldown"] is not None:
        delta = a3402["post_cooldown"] - BANKED["post_cooldown"]["seed3402"]
        gap = (BANKED["post_cooldown"]["seed3402"]
               - BANKED["gdn2"]["post_cooldown"])
        closed = min(0.0, -delta) if delta < 0 else 0.0
        if delta <= -REGISTRATION_ANCHORS["mr2_close_bar"]:
            verdict = ("CLOSES >= 0.02: the C5 attribution refines to THE "
                       "READ FORM (competitive softmax dilution); the "
                       "terminal reading is amended and the read form joins "
                       "the leaner registration")
        elif abs(delta) <= ENDPOINT_VARIANCE_BAND:
            verdict = ("PARITY inside the 0.075 band: the read form is NOT "
                       "the residual's owner; the superposition story "
                       "stands with one more receipt")
        else:
            verdict = ("WORSE outside the band: the magnitude-blowup-vs-form "
                       "attribution is inspectable via the birth-magnitude "
                       "diagnostic (R13-5); a gain-tuned read is the "
                       "nameable follow-up, not a post-hoc rescue")
        summary["M_R2"] = dict(
            post_cooldown_primary_seed3402=a3402["post_cooldown"],
            corrected_anchor=BANKED["post_cooldown"]["seed3402"],
            delta=delta, gap_reference=gap,
            fraction_of_gap_closed=closed / gap,
            close_bar=REGISTRATION_ANCHORS["mr2_close_bar"],
            band=ENDPOINT_VARIANCE_BAND,
            seed3401_rows={k: arms[f"additive_seed{PRIMARY_SEED}"][k]
                           for k in ("constant_lr", "post_cooldown", "ema_180")
                           if arms.get(f"additive_seed{PRIMARY_SEED}")},
            verdict=verdict)
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(arms={k: dict(constant=v["constant_lr"],
                                        post_cooldown=v["post_cooldown"],
                                        ema=v["ema_180"],
                                        effective_rows=v["effective_rows"])
                               for k, v in arms.items()},
                         M_R1=mr1, M_R2=summary.get("M_R2", {})), indent=1),
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
                                        / "k13_additive_read_summary.json"))
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
