"""K10 -- the separated-key quality probe: does the receipted key fix transfer
to real text at 150 M scale?

The question this stage exists for (``PREREGISTRATION_SEPARATED_KEYS.md``
claims S1/S2): at the matched-exposure probe protocol (20,022,866 targets, seed
3401, the ONE transferred LR, the K6 protocol verbatim), does the separated-key
arm's held-out NLL beat the banked K6 fused per-token row (**2.0278
nats/byte**), and do the pre-registered MECHANISM predictions (K-2/K-3/K-4,
committed before this run) land against the MEASURED fused-path anchors?

The protocol is the K6 probe's verbatim (`k6_probe_sequentiality.py`), with the
arm swapped: the SAME ``matrix_ops`` model (149,981,808 parameters, the
parameter-NEUTRAL key swap asserted), built by the same build order (seed 3401),
then every mixer converted in place to the separated one
(``convert_model_to_separated_keys``: the content query out, the key/query-key
projections in, mimetic-initialized at the fixed orthonormal family), the SAME
registered flagship stream, the SAME transferred optimizer, the SAME
matched-exposure stopping rule and honest clock, the SAME document-disjoint
calibration eval.

The mechanism instrument (the registration's 4.0, the first precisely-defined
measurement of the diagnosed identity-rewrite regime): *identity-rewrite = a
CORRECT decision whose write payload's fp32 value-channel cosine to the
targeted row's pre-write content is >= 0.999; the share is over CORRECT
decisions* -- computed from the recorded schedule and K4's
``persistent_scan_trajectory`` replay of the same walk.  Its anchors, both
measured by this probe on the same windows:

* **(a) the fused path at INIT** (the unconverted model's registered
  projections, K6's own fused launch -- the embedding-dominated regime the
  CMA-ES diagnosis receipts);
* **(b) the paired ablation**: K6's fused content-addressing launch over the
  SAME trained key-arm parameters' payload/decay channels with the content
  query taken as the payload itself (the key arm has no content query to
  share; the registered fused machine's own query was trained -- which is why
  anchor (a) runs the unconverted model) -- isolating the addressing channel
  at fixed payloads.

Deliverables: ``receipts/k10_probe.json`` and the comparison table in
``K10_SEPARATED_KEYS.md``.

Usage (see ``run_k10_probe_leased.sh`` for the detached leased driver):

    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys arm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_state import atomic_json
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
    CORRECT, OP_NAMES)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS, MIMETIC_SEED,
    _k10_forward_launch, convert_model_to_separated_keys, span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k4_fused_backward import (
    persistent_scan_trajectory)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32, _seq_forward_launch)
from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR
from experiments.expressivity_tasks.x45_matrix_emender.schedule import RoutingPolicy
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import \
    separated_sequential_oracle_scan  # noqa: F401  (the parity fixed point)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k10_runs"
SEED = 3401
EXPOSURE_TARGETS = 20_000_000
PROBE_WINDOWS = 4
SEGMENT_TOKENS = DEFAULT_SEGMENT_TOKENS
KEY_DIM = DEFAULT_KEY_DIM
CTX = DEFAULT_CONTEXT
# The instrument's threshold: a CORRECT whose write payload's cosine to the
# targeted row's pre-write content is at least this is an identity rewrite.
IDENTITY_COS = 0.999

# The banked K6 per-token row (the arm to beat; never rewritten).
K6_BANKED = dict(eval_nll_nats_per_byte=2.0278, exposure_bytes=20_022_866,
                 occupancy_mean_live_slots=73.68, read_entropy_nats=1.0846,
                 correct_rate=0.8843, merge_rate=0.00017)

# The pre-registered mechanism predictions (PREREGISTRATION_SEPARATED_KEYS.md
# 4.1, committed before this run).
PREDICTIONS = dict(
    K1_nll=dict(direction="improves on the banked 2.0278",
                band="improvement in [0.02, 0.35] nats/byte (honestly wide)",
                met=lambda r: r < K6_BANKED["eval_nll_nats_per_byte"],
                band_met=lambda r: 0.02 <= (K6_BANKED["eval_nll_nats_per_byte"]
                                            - r) <= 0.35),
    K2_identity=dict(
        direction="falls materially below BOTH fused-path anchors",
        band="at least 0.10 below the lower anchor; unchanged if within 0.02",
        refuted_if="within 0.02 of either anchor (share within noise)"),
    K3_occupancy=dict(direction="falls from the banked 73.7/128",
                      band="<= 65 mean live slots; refuted if > 73.7"),
    K4_entropy=dict(direction="rises from the banked 1.085 nats",
                    band="> 1.2 nats"),
    K5_merge=dict(direction="NO PREDICTION (its recovery prediction was "
                            "falsified once, at K6 4; reported, not re-predicted)"),
    KEY_COLLAPSE=dict(guard="span_key_std >= 0.05 in every layer (the Wheeler "
                            "receipt's guard)"),
    REFUTATION=dict(clause="identity-rewrite share within 0.02 of the fused "
                           "anchors OR NLL worse than 2.0278 -> the fix does "
                           "not transfer to real text and the residual needs "
                           "a different owner"))


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------- the instrument


def _identity_share(schedule, payload, decay, rank):
    """The registration's 4.0 instrument: the share of CORRECT decisions whose
    write payload's fp32 value-channel cosine to the targeted row's PRE-WRITE
    content is >= 0.999, over the walk's own recorded schedule and K4's value
    trajectory replay of that schedule.  Returns (share, n_correct)."""
    batch, tokens, heads, value_dim = payload.shape
    device = payload.device
    zeros_w = torch.zeros(batch, heads, tokens, rank, dtype=torch.float32,
                          device=device)
    decay32 = decay.float().permute(0, 2, 1).contiguous()
    state0 = torch.zeros(batch, heads, rank, value_dim, dtype=torch.float32,
                          device=device)
    live0 = torch.zeros(batch, heads, rank, dtype=torch.bool, device=device)
    _r, _s, _l, traj = persistent_scan_trajectory(
        state0, live0, zeros_w, decay32,
        schedule)          # (b, h, T, rank, value_dim): the pre-decision states
    op0 = schedule.op[..., 0]                        # (b, T, h)
    slot0 = schedule.slot[..., 0]                    # (b, T, h)
    correct = op0 == CORRECT
    if not bool(correct.any()):
        return 0.0, 0
    payload32 = payload.float()                      # (b, T, h, V)
    # gather the pre-write row content at the CORRECT slots: traj is (b, h, T,
    # rank, V); index per (b, h, T) with slot0 permuted to (b, h, T).
    slot_bht = slot0.permute(0, 2, 1)               # (b, h, T)
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


def _walk_stats(out, rank, mixer, payload=None, decay=None):
    """The K6 mechanism metrics from a collect launch's recorded channels."""
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        MatrixSchedule, op_histogram)
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    lane0 = torch.bincount(schedule.op[..., 0].reshape(-1).to("cpu"),
                           minlength=len(OP_NAMES))
    occupancy = out["live_pre"].float().sum(-1)      # (b, T, h)
    w = out["weights"][..., :rank].float()
    mass = w.sum(-1)
    has = mass > 0
    probs = torch.where(has.unsqueeze(-1),
                        w / mass.clamp_min(1e-30).unsqueeze(-1),
                        torch.zeros_like(w))
    contrib = torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs))
    entropy = -contrib.sum(-1)                        # (b, T, h)
    stats = dict(
        lane0=[int(v) for v in lane0],
        hist=op_histogram(schedule),
        decisions=int(occupancy.numel()),
        occupancy_sum=float(occupancy.sum()),
        occupancy_n=int(occupancy.numel()),
        occupancy_first=float(occupancy[:, 0].float().mean()),
        entropy_sum=float(entropy[has].sum()),
        entropy_n=int(has.sum()),
        rank=rank)
    if payload is not None and decay is not None:
        share, n_correct = _identity_share(schedule, payload, decay, rank)
        stats["identity_rewrite_share"] = share
        stats["identity_n_correct"] = n_correct
    return stats, schedule


def _agg_walk(per_layer, with_identity=False):
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
                / max(1, sum(s["entropy_n"] for s in stats_list))))
        if with_identity:
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
            units="nats over the live-slot weight distribution, per token",
            mean=float(sum(s["entropy_sum"] for _, s in per_layer)
                       / max(1, sum(s["entropy_n"] for _, s in per_layer)))),
        per_layer=layers)
    if with_identity:
        summary["identity_rewrite_share"] = float(
            sum(s["identity_rewrite_share"] * s["identity_n_correct"]
                for _, s in per_layer)
            / max(1, sum(s["identity_n_correct"] for _, s in per_layer)))
        summary["identity_n_correct"] = int(
            sum(s["identity_n_correct"] for _, s in per_layer))
    return summary


def _probe_windows(stream, windows: int):
    rows = [row for row in stream.windows("calibration")
            if len(row[2]) == L.CONTEXT]
    if len(rows) < windows:
        raise ValueError(f"calibration split has {len(rows)} full windows; "
                         f"the probe needs {windows}")
    return rows[:windows]


def _project_keyed(mixer, x):
    """The K10 mixer's own projections from its input (the trained walk's)."""
    from experiments.expressivity_tasks.x45_matrix_emender.key_separation_probe \
        import causal_context
    batch, tokens, _ = x.shape
    context = causal_context(x, mixer.ctx).reshape(batch, tokens,
                                                   mixer.ctx * x.shape[-1])
    payload = mixer.value(x).view(batch, tokens, mixer.heads, mixer.value_dim)
    write_keys = mixer.key(context).view(batch, tokens, mixer.heads,
                                         mixer.key_dim)
    query_keys = mixer.query_key(context).view(batch, tokens, mixer.heads,
                                               mixer.key_dim)
    decay = torch.sigmoid(mixer.decay(x)).clamp(min=DECAY_FLOOR,
                                                 max=1.0 - 1e-6)
    return payload, write_keys, query_keys, decay


# --------------------------------------------------- the arm's own diagnostics


def probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED key arm's mechanism receipts: op rates, per-token occupancy,
    read entropy (the K6 metrics, the same walk the arm trains on), plus the
    key channel's -- the identity-rewrite share (the instrument), span_key_std
    (the collapse watch) and the emitted-key spread, per layer."""
    rows = _probe_windows(stream, windows)
    model.eval()
    per_layer = []
    key_stats = []

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
                payload, write_keys, query_keys, decay = _project_keyed(mixer, x)
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.value_dim, dtype=torch.float32,
                                   device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                keys0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=x.device)
                out = _k10_forward_launch(payload.float(), write_keys.float(),
                                          query_keys.float(), decay.float(),
                                          state, live0, keys0, pol32,
                                          mixer.rank, mixer.segment_tokens,
                                          collect=True, warps=8, stages=3)
            with torch.no_grad():
                stats, _schedule = _walk_stats(out, mixer.rank, mixer,
                                               payload=payload.float(),
                                               decay=decay.float())
                spread = span_key_std(write_keys.float())
                emitted = write_keys.float().reshape(-1, mixer.key_dim)
                unit = emitted / emitted.norm(dim=-1, keepdim=True) \
                    .clamp_min(1e-30)
                n = unit.shape[0]
                gram = unit @ unit.t()
                key_stats.append(dict(
                    layer=layer_index,
                    span_key_std_min=float(spread.min()),
                    span_key_std_mean=float(spread.mean()),
                    mean_pairwise_key_cos=float((gram.sum() - n)
                                                / (n * (n - 1)))
                    if n > 1 else 0.0,
                    n_emitted=int(n)))
                per_layer.append((layer_index, stats))
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
    layers = sorted({s["layer"] for s in key_stats})
    key_summary = dict(
        span_key_std_min=float(min(s["span_key_std_min"] for s in key_stats)),
        span_key_std_mean=float(sum(s["span_key_std_mean"] for s in key_stats)
                                / len(key_stats)),
        mean_pairwise_key_cos=float(sum(s["mean_pairwise_key_cos"]
                                        for s in key_stats)
                                    / len(key_stats)),
        per_layer_span_key_std={str(l): float(
            sum(s["span_key_std_mean"] for s in key_stats if s["layer"] == l)
            / max(1, sum(1 for s in key_stats if s["layer"] == l)))
            for l in layers},
        guard_floor=0.05,
        within_guard=bool(all(s["span_key_std_min"] >= 0.05
                              for s in key_stats)))
    return dict(
        probe=dict(split="calibration", windows=len(rows),
                   mixers=len(model.blocks),
                   regime="eval, bf16 autocast, the persistent walk the arm "
                          "trains on (collect=True)",
                   probe_wall_seconds=round(wall, 1)),
        key_collapse_watch=key_summary,
        **summary)


# ------------------------------------------------------------- the anchors


def fused_at_init_anchor(stream, windows: int = PROBE_WINDOWS) -> dict:
    """Anchor (a): the FUSED path at INIT -- a second model built by the same
    order (seed 3401) and left UNCONVERTED, its registered query/value/decay
    projections driving K6's own fused launch on the same probe windows.  The
    embedding-dominated regime the CMA-ES diagnosis receipts; the operator's
    quoted 97.66 % CORRECT / ~48 live slots (K8_GROWTH.md 5) is this anchor's
    own first-mixer snapshot."""
    model, _optimizer_config, _summary = matched_arms.build_matrix(
        "matrix_ops", 128, SEED, k5b=False)
    model = model.cuda().eval()
    rows = _probe_windows(stream, windows)
    per_layer = []

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
                query = mixer.query(x).view(batch, tokens, mixer.heads,
                                            mixer.value_dim)
                payload = mixer.value(x).view(batch, tokens, mixer.heads,
                                              mixer.value_dim)
                decay = torch.sigmoid(mixer.decay(x)).clamp(
                    min=DECAY_FLOOR, max=1.0 - 1e-6)
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.value_dim, dtype=torch.float32,
                                   device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                out = _seq_forward_launch(query.float(), payload.float(),
                                           decay.float(), state, live0, pol32,
                                           mixer.rank, SEGMENT_TOKENS,
                                           collect=True, warps=8, stages=3)
            with torch.no_grad():
                stats, _schedule = _walk_stats(out, mixer.rank, mixer,
                                                payload=payload.float(),
                                                decay=decay.float())
                per_layer.append((layer_index, stats))
        return hook

    handles = [block.mixer.register_forward_hook(make_hook(i, block.mixer))
               for i, block in enumerate(model.blocks)]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1))
                model.features(x)
    finally:
        for handle in handles:
            handle.remove()
    del model
    torch.cuda.empty_cache()
    summary = _agg_walk(per_layer, with_identity=True)
    summary["probe"] = dict(split="calibration", windows=len(rows),
                            regime="the UNCONVERTED model at init (seed 3401), "
                                   "K6's fused launch, the registered "
                                   "projections")
    return summary


def paired_ablation_anchor(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """Anchor (b): the paired ablation -- K6's fused content-addressing launch
    over the SAME trained key-arm parameters' payload/decay channels, the
    content query taken as the payload itself (the pure content-addressing
    machine: the key arm has no content query to share).  Isolates the
    addressing channel at fixed payloads."""
    rows = _probe_windows(stream, windows)
    model.eval()
    per_layer = []

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
                payload = mixer.value(x).view(batch, tokens, mixer.heads,
                                              mixer.value_dim)
                decay = torch.sigmoid(mixer.decay(x)).clamp(
                    min=DECAY_FLOOR, max=1.0 - 1e-6)
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.value_dim, dtype=torch.float32,
                                   device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                out = _seq_forward_launch(payload.float(), payload.float(),
                                          decay.float(), state, live0, pol32,
                                          mixer.rank, SEGMENT_TOKENS,
                                          collect=True, warps=8, stages=3)
            with torch.no_grad():
                stats, _schedule = _walk_stats(out, mixer.rank, mixer,
                                               payload=payload.float(),
                                               decay=decay.float())
                per_layer.append((layer_index, stats))
        return hook

    handles = [block.mixer.register_forward_hook(make_hook(i, block.mixer))
               for i, block in enumerate(model.blocks)]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for row in rows:
                x, _y, _valid = next(grouped([row], 1))
                model.features(x)
    finally:
        for handle in handles:
            handle.remove()
    summary = _agg_walk(per_layer, with_identity=True)
    summary["probe"] = dict(split="calibration", windows=len(rows),
                            regime="the TRAINED key arm's value/decay channels "
                                   "under fused content addressing "
                                   "(query := payload; no content query "
                                   "exists on this arm)")
    return summary


# ------------------------------------------------------------- the predictions


def evaluate_predictions(receipt: dict) -> dict:
    """K-1..K-5, the guard and the refutation clause, against the
    pre-stated bands (PREREGISTRATION_SEPARATED_KEYS.md 4.1)."""
    nll = receipt["eval_nll_nats_per_byte"]
    identity = receipt["diagnostics"]["identity_rewrite_share"]
    anchor_init = receipt["anchors"]["fused_at_init"]["identity_rewrite_share"]
    anchor_abl = receipt["anchors"]["paired_ablation"]["identity_rewrite_share"]
    occupancy = receipt["diagnostics"]["occupancy"]["mean_live_slots"]
    entropy = receipt["diagnostics"]["read_weight_entropy"]["mean"]
    merge = receipt["diagnostics"]["op_rate_per_token_lane0"]["merge"]
    guard_ok = receipt["diagnostics"]["key_collapse_watch"]["within_guard"]
    lower_anchor = min(anchor_init, anchor_abl)
    k1_met = PREDICTIONS["K1_nll"]["met"](nll)
    k2_met = identity <= lower_anchor - 0.10
    k2_unchanged = (abs(identity - anchor_init) <= 0.02
                    or abs(identity - anchor_abl) <= 0.02)
    k3_met = occupancy <= 65.0
    k3_refuted = occupancy > K6_BANKED["occupancy_mean_live_slots"]
    k4_met = entropy > 1.2
    refuted = (not k1_met) or k2_unchanged
    return dict(
        K1_nll=dict(prediction=PREDICTIONS["K1_nll"]["direction"],
                    band=PREDICTIONS["K1_nll"]["band"],
                    measured=nll,
                    improvement=K6_BANKED["eval_nll_nats_per_byte"] - nll,
                    met=bool(k1_met),
                    band_met=bool(PREDICTIONS["K1_nll"]["band_met"](nll))),
        K2_identity=dict(prediction=PREDICTIONS["K2_identity"]["direction"],
                          band=PREDICTIONS["K2_identity"]["band"],
                          measured=identity,
                          anchor_fused_at_init=anchor_init,
                          anchor_paired_ablation=anchor_abl,
                          met=bool(k2_met),
                          unchanged=bool(k2_unchanged)),
        K3_occupancy=dict(prediction=PREDICTIONS["K3_occupancy"]["direction"],
                          band=PREDICTIONS["K3_occupancy"]["band"],
                          banked=K6_BANKED["occupancy_mean_live_slots"],
                          measured=occupancy, met=bool(k3_met),
                          refuted=bool(k3_refuted)),
        K4_entropy=dict(prediction=PREDICTIONS["K4_entropy"]["direction"],
                        band=PREDICTIONS["K4_entropy"]["band"],
                        banked=K6_BANKED["read_entropy_nats"],
                        measured=entropy, met=bool(k4_met)),
        K5_merge=dict(prediction=PREDICTIONS["K5_merge"]["direction"],
                      measured=merge,
                      banked=K6_BANKED["merge_rate"]),
        KEY_COLLAPSE=dict(prediction=PREDICTIONS["KEY_COLLAPSE"]["guard"],
                         measured=receipt["diagnostics"]["key_collapse_watch"],
                         met=bool(guard_ok)),
        refutation=dict(clause=PREDICTIONS["REFUTATION"]["clause"],
                        refuted=bool(refuted)))


# ------------------------------------------------------------------- the arm


def train_arm(args) -> dict:
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None:
        raise ValueError("no GPU lease: set GPU_LEASE_HELD before training")
    if visible is None:
        raise ValueError("CUDA_VISIBLE_DEVICES must be set before launch")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: recorded card {card} vs "
                         f"CUDA_VISIBLE_DEVICES {visible}")

    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", 128, args.seed, k5b=False)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_separated_keys(model, key_dim=args.keydim,
                                            ctx=args.ctx,
                                            segment_tokens=SEGMENT_TOKENS,
                                            seed=MIMETIC_SEED)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != 149_981_808 or parameters_after != parameters_before:
        raise ValueError(f"the parameter-neutral swap must hold: "
                         f"{parameters_before} -> {parameters_after} "
                         f"(the audited count is 149,981,808)")
    summary = dict(summary)
    summary["scan_path"] = "k10_separated_keys (per-token, the key channel)"
    summary["segment_tokens"] = SEGMENT_TOKENS
    summary["chunk_tokens"] = None
    summary["key_dim"] = args.keydim
    summary["key_context"] = args.ctx
    summary["key_init"] = ("mimetic at the fixed orthonormal family (QR, seed "
                           f"{MIMETIC_SEED} + head), free drift")
    microbatch = optimizer_config["microbatch"]
    model = model.cuda().train()
    out = Path(args.runs_dir) / f"separated_keys_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=False)
    exposure = int(args.bytes)
    stopping_rule = (f"matched-EXPOSURE: train until {exposure} targets seen "
                     "(~20M, the probe exposure); the honest elapsed wall is "
                     "reported first-class")
    config = dict(probe="k10_separated_keys", arm="matrix_ops",
                  addressing="separated write-context keys (key_dim 16)",
                  seed=args.seed, exposure_targets=exposure,
                  key_dim=args.keydim, key_context=args.ctx,
                  segment_tokens=SEGMENT_TOKENS, optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT,
                  windows_per_step=64, summary=summary, identity=identity,
                  bindings=dict(
                      matched_arms.registration_bindings(),
                      k10_preregistration=str(
                          ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
                      k10_preregistration_sha256=digest(
                          ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
                      k6_preregistration=str(
                          ROOT / "PREREGISTRATION_SEQUENTIAL_MATRIX.md"),
                      k10_gates_receipt=str(RECEIPTS / "k10_gates.json"),
                      k10_gates_sha256=digest(RECEIPTS / "k10_gates.json")),
                  git=matched_arms.git_state(), stopping_rule=stopping_rule)
    atomic_json(out / "start.json", dict(
        config, config_sha256=matched_arms.config_sha256(config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        parameters=dict(total=parameters_after,
                        trainable=sum(p.numel() for p in model.parameters()
                                      if p.requires_grad))))
    print(json.dumps(dict(addressing="separated-keys", card=card,
                          seed=args.seed, exposure=exposure,
                          parameters=parameters_after)), flush=True)

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets = 0
    log_rows = []
    with (out / "training.jsonl").open("w", buffering=1) as log:
        while targets < exposure:
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                rows = next(batches)
            except StopIteration:
                raise RuntimeError("train split exhausted before the budget: "
                                   "refusing to wrap")
            n = sum(len(r[3]) for r in rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config["lr"] * min(
                    1., (step + 1) / max(1, optimizer_config["warmup_steps"]))
            loss_sum = 0.0
            for x, y, valid in grouped(rows, microbatch):
                loss = model.loss(x, y)
                if not torch.isfinite(loss):
                    raise FloatingPointError("nonfinite loss")
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            step += 1
            targets += n
            row = dict(step=step, active_seconds=elapsed, bytes=targets,
                       loss=loss_sum, time_ns=time.time_ns())
            log.write(json.dumps(row) + "\n")
            log_rows.append(row)
            if step % 10 == 0:
                print(f"  separated-keys step {step} {elapsed:.0f}s "
                      f"{targets / 1e6:.1f}MB loss {loss_sum:.4f}", flush=True)

    # the trained arm's mechanism receipts, then the anchors, then the eval.
    diagnostics = probe_diagnostics(model, stream, windows=args.probe_windows)
    anchor_init = fused_at_init_anchor(stream, windows=args.probe_windows)
    anchor_abl = paired_ablation_anchor(model, stream,
                                        windows=args.probe_windows)
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation path is recorded-read only
    eval_started = time.monotonic()
    result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
    eval_wall = time.monotonic() - eval_started

    receipt = dict(
        probe="k10_separated_keys", arm="matrix_ops",
        addressing="separated write-context keys (key_dim 16, mimetic birth, "
                   "free drift, keys undecayed)",
        seed=args.seed, stopping_rule=stopping_rule,
        exposure_bytes=targets, exposure_steps=step,
        eval_nll_nats_per_byte=result["score"], eval_split=result["eval_split"],
        eval_windows=result["eval_windows"], eval_targets=result["eval_targets"],
        walls=dict(train_seconds=round(elapsed, 1),
                   probe_seconds=diagnostics["probe"]["probe_wall_seconds"],
                   anchor_seconds=None, eval_seconds=round(eval_wall, 1)),
        bytes_per_s=targets / max(1e-9, elapsed),
        parameters=parameters_after,
        diagnostics=diagnostics,
        anchors=dict(fused_at_init=anchor_init,
                    paired_ablation=anchor_abl),
        k6_banked=K6_BANKED,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            log_rows, keep_first=min(5, max(1, len(log_rows)))),
    )
    receipt["predictions"] = evaluate_predictions(receipt)
    receipt_path = Path(args.receipts_dir) / "k10_probe.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("eval_nll_nats_per_byte", "exposure_bytes",
                       "exposure_steps", "bytes_per_s")}, indent=1),
          flush=True)
    print(json.dumps(receipt["predictions"], indent=1), flush=True)
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--keydim", type=int, default=KEY_DIM)
    a.add_argument("--ctx", type=int, default=CTX)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    a.add_argument("--runs-dir", default=str(RUNS))
    a.add_argument("--receipts-dir", default=str(RECEIPTS))
    a.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, smoke paths")
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.smoke:
            args.runs_dir = str(Path(args.runs_dir) / "smoke")
            args.receipts_dir = str(Path(args.receipts_dir) / "smoke")
            args.bytes = 200_000
        train_arm(args)
        return


if __name__ == "__main__":
    main()
