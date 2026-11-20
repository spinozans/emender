"""K14 -- the learned-management-routing quality probe: does a LEARNED op
router move real text at 150 M scale, under the corrected scoring?

The question this stage exists for (``PREREGISTRATION_LEARNED_MANAGEMENT.md``
claims S1/S2, committed before this run; the birth construction was
supervisor-approved before the engine was locked): at the matched-exposure
probe protocol (20,022,866 targets, seeds 3401/3402, the banked K10-keys
config, the banked optimizer, the corrected scoring -- 24-step linear
cooldown PRIMARY, EMA-0.99 secondary, constant-180 retained for banked
comparability), does the learned-router arm improve on the re-scored
baseline, and do the pre-registered MECHANISM predictions land?  THE OP
MIX IS THE HEADLINE INSTRUMENT: the audit's degeneration diagnosis (CORRECT
identity-dominated, MERGE ~dead, POP idle) is what a learned router should
move.

The arms (the registration's 1):

* ``baseline_seed3401``       the banked K10-keys config itself (the
                              FROZEN-THRESHOLDS control): the K12 lane's
                              own re-scored baseline receipt is REUSED after
                              the in-code config+protocol identity assert
                              (the supervisor's ruling; its constant-180
                              eval is the banked 1.841756020273481 and its
                              post-cooldown the verified 1.6832871096474784).
* ``routing_learned_seed3401``   the same config + the K14 module, the
                              per-head router born at exact zeros and
                              trainable through the admitted estimator (the
                              step-1 loss must equal the banked step-1 loss
                              bit-identically -- the 0-flip-at-birth bar at
                              probe scale).
* ``routing_frozen_seed3401``    the FROZEN-ROUTER falsifier control: the
                              router pinned at birth; step-1 bit-identical
                              (the K11B 2.4-class bar) with the per-step
                              deltas against the banked row receipted
                              honestly (full-trajectory bit-identity
                              expected).
* ``baseline_seed3402`` / ``routing_learned_seed3402`` /
  ``routing_frozen_seed3402``    the seed-matched twins (the seed-3402
                              baseline runs FIRST; the 3402 arms'
                              step-1/falsifier bars read against its own
                              log).

Deliverables: per-arm receipts under ``k14_runs/<label>/`` (start.json,
training.jsonl, scores.jsonl, result.json -- the mechanism diagnostics, the
routing instruments, the read-mask sweep and the banked-equivalence asserts
included) and the assemble summary
``receipts/k14_routing_summary.json``.

Usage (see ``run_k14_routing_leased.sh`` for the detached leased driver,
cards 5/6):

    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k14_probe_routing \\
        arm --label routing_learned_seed3401
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k14_probe_routing assemble
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

# The probe subprocess sets these BEFORE importing torch (the matched-arms
# discipline; configure_production() fail-closes on a missing allocator knob).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
from experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys import (
    PROBE_WINDOWS as DIAG_WINDOWS, _agg_walk, _probe_windows, _project_keyed,
    _walk_stats, probe_diagnostics as k10_probe_diagnostics)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS, MIMETIC_SEED,
    _k10_forward_launch, convert_model_to_separated_keys, span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k14_learned_routing import (
    RoutingSeparatedMixer, _k14_forward_launch, convert_model_to_routing,
    routing_available, routing_parameter_delta)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction import (
    COOLDOWN_STEPS, EMA_DECAY, GRAD_CLIP, _train_corrected, cooldown_factor)
from experiments.expressivity_tasks.x45_matrix_emender.side_by_side import (
    k10_read_mask_sweep)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k14_runs"
BANKED_RUN = ROOT / "k10_runs" / "separated_keys_seed3401"
K12_BASELINE_RECEIPT = RECEIPTS / "k12_baseline_seed3401.json"
K12_BASELINE_RUN = ROOT / "k12_runs" / "baseline_seed3401"
SEED = 3401
EXPOSURE_TARGETS = 20_000_000
SEGMENT_TOKENS = DEFAULT_SEGMENT_TOKENS
KEY_DIM = DEFAULT_KEY_DIM
CTX = DEFAULT_CONTEXT

# The banked K10-keys row (the arm to beat; never rewritten).
K10_BANKED = dict(eval_nll_nats_per_byte=1.841756020273481,
                  exposure_bytes=20_022_866, exposure_steps=180,
                  occupancy_mean_live_slots=24.0287348429362,
                  read_entropy_nats=1.6541064706071544,
                  correct_rate=0.97939, append_rate=0.01748,
                  merge_rate=0.00313, pop_rate=0.0,
                  identity_share=0.10321439792683149,
                  banked_log=str(BANKED_RUN / "training.jsonl"))
# The verified re-scored baseline anchors (the supervisor's required
# correction: the 1.6892253501074654 row is K12's forward-degenerate fixed-w
# bracket ARM, NOT the baseline).
BASELINE_3401 = dict(post_cooldown=1.6832871096474784,
                     constant_180=1.841756020273481,
                     ema_180=1.8054006099700928,
                     receipt=str(K12_BASELINE_RECEIPT))
# The C5 gap references, quoted as CONTEXT (the registration's 1): GDN2's
# own post-cooldown at its protocol, and the best-vs-best structural
# residual (~0.08) the M-M4 read measures against.
GDN2_REFERENCES = dict(gdn2_post_cooldown=1.3522,
                       our_speedrun_post_cooldown=1.4338,
                       best_vs_best_residual=0.0816,
                       source="RESULT_PROTOCOL_CORRECTION.md / "
                              "RESULT_LEARNED_GATES.md")
# The audit's degeneration anchors (M-M1/M-M2/M-M3's quoted context).
AUDIT_ANCHORS = dict(correct_rate=0.97939, merge_rate=0.00313,
                     pop_rate=0.0, identity_share=0.10321439792683149,
                     source="RESULT_OP_SET_AUDIT.md / receipts/k10_probe.json")
OP_NAMES = ("none", "append", "correct", "pop", "merge")

ARMS = {
    "baseline_seed3401": dict(kind="baseline", seed=3401, trainable=None,
                              note="reused from the K12 lane's own re-scored "
                                   "baseline receipt after the in-code "
                                   "config+protocol identity assert (the "
                                   "supervisor's ruling; runnable but not "
                                   "re-run -- it would reproduce the banked "
                                   "row bit for bit)"),
    "routing_learned_seed3401": dict(kind="routing", seed=3401,
                                     trainable=True),
    "routing_frozen_seed3401": dict(kind="routing", seed=3401,
                                    trainable=False),
    "baseline_seed3402": dict(kind="baseline", seed=3402, trainable=None),
    "routing_learned_seed3402": dict(kind="routing", seed=3402,
                                     trainable=True),
    "routing_frozen_seed3402": dict(kind="routing", seed=3402,
                                    trainable=False),
}

CORRECTED_SCORING = dict(
    primary="post-cooldown eval: a 24-step linear cooldown (13.3% of the "
            "180-step budget, factor (25-k)/24, never zero) continues the "
            "same registered stream after the constant-phase matched-"
            "exposure landing, then the document-disjoint calibration eval",
    secondary="EMA-0.99 iterate at step 180 (constant phase only), "
              "ranking-only",
    banked_comparability="the constant-LR step-180 eval, retained",
    cooldown_steps=COOLDOWN_STEPS, ema_decay=EMA_DECAY)

# The mechanism bands (the registration's 3, pre-stated).
BANDS = dict(
    mm1_material_tv=0.02, mm1_material_flip_rate=0.01,
    mm1_unchanged_tv=0.005, mm1_unchanged_flip_rate=0.001,
    mm2_pop_rate=0.001, mm2_merge_rate=0.01,
    mm3_identity_fall=0.02, mm3_identity_unchanged=0.005,
    mm4_lever_nats=0.02, endpoint_variance_band=0.075)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require_lease() -> str:
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("no GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES "
                         "must be set before launch")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: recorded card {card} vs "
                         f"CUDA_VISIBLE_DEVICES {visible}")
    return card


def _banked_log_rows() -> list:
    banked = [json.loads(line) for line
              in Path(K10_BANKED["banked_log"]).read_text().splitlines()
              if line.strip()]
    return banked


def _baseline_log_rows(seed: int) -> tuple:
    """The seed-matched baseline's own training rows: the banked log at 3401
    (bit-identical to the K12 baseline's own re-score), the fresh
    baseline_seed3402 run's log at 3402."""
    if seed == 3401:
        return _banked_log_rows(), str(K10_BANKED["banked_log"])
    path = RUNS / "baseline_seed3402" / "training.jsonl"
    if not path.exists():
        raise ValueError("the seed-3402 arms read their step-1/falsifier bars "
                         "against baseline_seed3402's own log -- run "
                         "baseline_seed3402 first")
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.strip()]
    return rows, str(path)


def _assert_k12_baseline_identity() -> dict:
    """The supervisor's reuse condition, asserted in-code before the assemble
    quotes the K12 re-scored baseline as the seed-3401 baseline row: the
    config+protocol identity of that receipt against THIS lane's own arms."""
    receipt = json.loads(K12_BASELINE_RECEIPT.read_text())
    config = receipt["config"]
    checks = dict(
        arm_is_baseline=receipt["arm"] == "baseline",
        probe_seed=config["seed"] == 3401,
        key_dim=config["key_dim"] == KEY_DIM,
        key_context=config["key_context"] == CTX,
        segment_tokens=config["segment_tokens"] == SEGMENT_TOKENS,
        exposure=config["exposure_targets"] == EXPOSURE_TARGETS,
        optimizer=config["optimizer"]["microbatch"] is not None,
        cooldown=config["corrected_scoring"]["cooldown_steps"]
        == COOLDOWN_STEPS,
        ema=config["corrected_scoring"]["ema_decay"] == EMA_DECAY,
        parameters=receipt["parameters"] == 149_981_808,
        constant_180=receipt["final_eval_nll_constant_lr"]
        == K10_BANKED["eval_nll_nats_per_byte"],
        post_cooldown=receipt["final_eval_nll_post_cooldown"]
        == BASELINE_3401["post_cooldown"],
        banked_equivalence=bool(receipt["banked_equivalence"]
                                .get("bit_identical")))
    if not all(checks.values()):
        raise ValueError(f"the K12 baseline receipt fails the config+protocol "
                         f"identity assert: {checks}")
    return dict(receipt=str(K12_BASELINE_RECEIPT), checks=checks,
                reused_row=dict(
                    constant_180=receipt["final_eval_nll_constant_lr"],
                    post_cooldown=receipt["final_eval_nll_post_cooldown"],
                    ema_180=receipt["ema_eval_nll"]))


# ------------------------------------------------------- the K14 instruments


def _op_mix_from_diagnostics(diag: dict) -> dict:
    rates = (diag.get("op_rate_per_token_lane0") or {})
    return {name: float(rates.get(name, 0.0)) for name in OP_NAMES}


def tv_distance(mix_a: dict, mix_b: dict) -> float:
    return 0.5 * sum(abs(mix_a.get(n, 0.0) - mix_b.get(n, 0.0))
                     for n in OP_NAMES)


def routing_probe_diagnostics(model, stream, windows: int = DIAG_WINDOWS):
    """The TRAINED routing arm's mechanism receipts: the K10 probe's own
    instruments (op rates, occupancy, read entropy, identity share, the
    KEY-COLLAPSE watch -- imported) computed on the k14 launch with the
    TRAINED router, PLUS this registration's own: the FLIP-RATE instrument
    (the learned op vs the chain's own op at the same pre-token state, from
    the kernel's recorded chain_op channel), the learned op mix, the
    recorded score margins, and the trained router telemetry per layer."""
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        OP_NAMES as OPS)
    rows = _probe_windows(stream, windows)
    model.eval()
    per_layer = []
    key_stats = []
    routing_stats = []

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
                g = mixer.router_scores(x)     # the production expression
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=x.device)
                live0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    dtype=torch.bool, device=x.device)
                keys0 = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.key_dim, dtype=torch.float32,
                                    device=x.device)
                out = _k14_forward_launch(
                    payload.float(), write_keys.float(), query_keys.float(),
                    decay.float(), g.float(), state, live0, keys0, pol32,
                    mixer.rank, mixer.segment_tokens, collect=True, warps=8,
                    stages=3)
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
                op0 = out["op"][..., 0]                     # (b, T, h)
                chain = out["chain_op"]
                flips = (op0 != chain)
                tokens_n = op0.numel()
                s = out["op_scores"]                        # (b, T, h, 5)
                top2 = s.topk(2, dim=-1).values
                margins = top2[..., 0] - top2[..., 1]
                finite = torch.isfinite(margins)
                routing_stats.append(dict(
                    layer=layer_index,
                    tokens=int(tokens_n),
                    ops_flipped_off_chain=int(flips.sum()),
                    flip_rate=float(flips.sum()) / max(1, tokens_n),
                    op_rates={OPS[i]: float((op0 == i).sum())
                              / max(1, tokens_n) for i in range(5)},
                    chain_op_rates={OPS[i]: float((chain == i).sum())
                                    / max(1, tokens_n) for i in range(5)},
                    argmax_margin_mean=float(margins[finite].mean())
                    if bool(finite.any()) else None,
                    argmax_margin_min=float(margins[finite].min())
                    if bool(finite.any()) else None,
                    router_weight_norm=float(
                        mixer.router.weight.detach().float().norm()),
                    router_bias_norm=float(
                        mixer.router.bias.detach().float().norm()),
                    router_abs_mean=float(
                        mixer.router.weight.detach().float().abs().mean()),
                    g_abs_mean_per_op=[float(v) for v in
                                      g.float().abs().mean(dim=(0, 1, 2))]))
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
    by_layer = {}
    for s in routing_stats:
        by_layer.setdefault(s["layer"], []).append(s)
    tokens_all = sum(s["tokens"] for s in routing_stats)
    routing_summary = dict(
        note="THE OP MIX IS THE HEADLINE INSTRUMENT: the learned op rates, "
             "the flip rate (the learned op vs the chain's own threshold op "
             "recorded at the same pre-token state), the recorded argmax "
             "margins, and the trained router telemetry",
        op_rate_per_token={OPS[i]: float(
            sum(s["op_rates"][OPS[i]] for s in routing_stats)
            / max(1, len(routing_stats))) for i in range(5)},
        chain_op_rate_per_token={OPS[i]: float(
            sum(s["chain_op_rates"][OPS[i]] for s in routing_stats)
            / max(1, len(routing_stats))) for i in range(5)},
        ops_flipped_off_chain=int(
            sum(s["ops_flipped_off_chain"] for s in routing_stats)),
        flip_rate_all_layers=float(
            sum(s["ops_flipped_off_chain"] for s in routing_stats)
            / max(1, tokens_all)),
        argmax_margin_mean=float(sum(
            s["argmax_margin_mean"] for s in routing_stats
            if s["argmax_margin_mean"] is not None)
            / max(1, sum(1 for s in routing_stats
                         if s["argmax_margin_mean"] is not None))),
        router_weight_norm_mean=float(
            sum(s["router_weight_norm"] for s in routing_stats)
            / max(1, len(routing_stats))),
        router_abs_mean_mean=float(
            sum(s["router_abs_mean"] for s in routing_stats)
            / max(1, len(routing_stats))),
        g_abs_mean_per_op_all_layers=[
            float(sum(s["g_abs_mean_per_op"][k] for s in routing_stats)
                  / max(1, len(routing_stats))) for k in range(5)],
        per_layer={str(l): dict(
            flip_rate=float(sum(s["flip_rate"] for s in ss) / len(ss)),
            op_rates={OPS[i]: float(sum(s["op_rates"][OPS[i]] for s in ss)
                                    / len(ss)) for i in range(5)},
            router_weight_norm=float(ss[0]["router_weight_norm"]),
            n_windows=len(ss))
            for l, ss in sorted(by_layer.items())})
    return dict(
        probe=dict(split="calibration", windows=len(rows),
                   mixers=len(model.blocks),
                   regime="eval, bf16 autocast, the persistent walk the arm "
                          "trains on (collect=True), the TRAINED router "
                          "through the production expression",
                   probe_wall_seconds=round(wall, 1)),
        key_collapse_watch=key_summary,
        routing=routing_summary,
        **summary)


# ------------------------------------------------------------------- the arm


def train_arm(args) -> dict:
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    if args.label not in ARMS:
        raise ValueError(f"unregistered arm label: {args.label}")
    arm = ARMS[args.label]
    if args.seed != arm["seed"]:
        raise ValueError(f"the {args.label} arm's seed is {arm['seed']}, "
                         f"got {args.seed}")

    # ---- the machine: the banked K10-keys build order VERBATIM, then the
    # routing conversion (the ONLY difference; its module creation is
    # RNG-state-guarded and born at exact zeros, so the birth weights are
    # the banked row's own bit for bit)
    model, optimizer_config, summary = matched_arms.build_matrix(
        "matrix_ops", 128, args.seed, k5b=False)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_separated_keys(model, key_dim=args.keydim,
                                             ctx=args.ctx,
                                             segment_tokens=SEGMENT_TOKENS,
                                             seed=MIMETIC_SEED)
    separated_parameters = sum(p.numel() for p in model.parameters())
    if separated_parameters != 149_981_808:
        raise ValueError(f"the separated conversion must land the audited "
                         f"149,981,808; got {separated_parameters}")
    if arm["kind"] == "routing":
        model = convert_model_to_routing(model,
                                         router_trainable=arm["trainable"],
                                         segment_tokens=SEGMENT_TOKENS)
        parameters_after = sum(p.numel() for p in model.parameters())
        expected_delta = routing_parameter_delta(
            len(model.blocks), model.blocks[0].mixer.heads,
            model.blocks[0].mixer.ctx, model.blocks[0].mixer.width)
        if parameters_after - separated_parameters != expected_delta:
            raise ValueError(f"the routing delta must be exactly "
                             f"{expected_delta}; got "
                             f"{parameters_after - separated_parameters}")
        for block in model.blocks:
            if not isinstance(block.mixer, RoutingSeparatedMixer):
                raise ValueError("the routing conversion must rewrap "
                                 "every mixer")
            w0 = block.mixer.router.weight.detach()
            b0 = block.mixer.router.bias.detach()
            if not torch.equal(w0, torch.zeros_like(w0)) \
                    or not torch.equal(b0, torch.zeros_like(b0)):
                raise ValueError("the router must be born at EXACT zeros")
            if bool(w0.requires_grad) != bool(arm["trainable"]) \
                    or bool(b0.requires_grad) != bool(arm["trainable"]):
                raise ValueError(f"the router's requires_grad must be "
                                 f"{arm['trainable']}")
        from experiments.expressivity_tasks.x45_matrix_emender. \
            k14_learned_routing import routing_available as available
        if not available(128, 128, args.keydim):
            raise ValueError("the registered shape must take the kernel path")
    else:
        parameters_after = separated_parameters
    summary = dict(summary)
    summary["scan_path"] = ("k14_learned_routing (per-token, the key channel "
                            "+ the LEARNED op routing)"
                            if arm["kind"] == "routing" else
                            "k10_separated_keys (per-token, the key channel)")
    summary["segment_tokens"] = SEGMENT_TOKENS
    summary["chunk_tokens"] = None
    summary["key_dim"] = args.keydim
    summary["key_context"] = args.ctx
    summary["key_init"] = ("mimetic at the fixed orthonormal family (QR, seed "
                           f"{MIMETIC_SEED} + head), free drift")
    microbatch = optimizer_config["microbatch"]
    model = model.cuda().train()
    out = Path(args.runs_dir) / args.label
    out.mkdir(parents=True, exist_ok=False)
    exposure = int(args.bytes)
    stopping_rule = (f"matched-EXPOSURE: train until {exposure} targets seen "
                     "(the banked probe exposure, constant LR), THEN the "
                     f"{COOLDOWN_STEPS}-step linear cooldown (the corrected "
                     "scoring's extension, the same stream)")
    config = dict(probe="k14_learned_routing", label=args.label,
                  arm=arm["kind"],
                  routing=(None if arm["kind"] != "routing" else dict(
                      module="per-head Linear(ctx*width=2304 -> 5 op scores, "
                             "bias) over the DETACHED causal context",
                      birth="born at EXACT zeros (+0.0 normalisation); the "
                            "branch-active margin encoding reproduces the "
                            "threshold chain bit-exactly",
                      trainable=arm["trainable"],
                      estimator=("the supervisor-approved straight-through "
                                 "softmax surrogate on the recorded per-op "
                                 "scores, c_t = <dpayload_t, p_t>, only the "
                                 "router learns; c_t is zero on "
                                 "MERGE/POP/NONE tokens (the disclosed "
                                 "rescope)"
                                 if arm["trainable"] else
                                 "none (pinned: the estimator-free falsifier "
                                 "control)"),
                      parameters=553_200)),
                  seed=args.seed, exposure_targets=exposure,
                  key_dim=args.keydim, key_context=args.ctx,
                  segment_tokens=SEGMENT_TOKENS, optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT,
                  corrected_scoring=CORRECTED_SCORING, bands=BANDS,
                  parameters=dict(total=parameters_after,
                                  routing_delta=(parameters_after
                                                 - separated_parameters)),
                  windows_per_step=64, summary=summary, identity=identity,
                  bindings=dict(
                      matched_arms.registration_bindings(),
                      k14_preregistration=str(
                          ROOT / "PREREGISTRATION_LEARNED_MANAGEMENT.md"),
                      k14_preregistration_sha256=digest(
                          ROOT / "PREREGISTRATION_LEARNED_MANAGEMENT.md"),
                      k10_preregistration=str(
                          ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
                      k14_gates_receipt=str(RECEIPTS / "k14_gates.json"),
                      k14_gates_sha256=digest(RECEIPTS / "k14_gates.json"),
                      banked_k10_row=K10_BANKED,
                      baseline_3401_reused=BASELINE_3401,
                      gdn2_references=GDN2_REFERENCES,
                      audit_anchors=AUDIT_ANCHORS),
                  git=matched_arms.git_state(), stopping_rule=stopping_rule)
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    atomic_json(out / "start.json", dict(
        config, config_sha256=matched_arms.config_sha256(config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        trainable=sum(p.numel() for p in model.parameters()
                      if p.requires_grad)))
    print(json.dumps(dict(label=args.label, card=card, seed=args.seed,
                          exposure=exposure,
                          parameters=parameters_after)), flush=True)

    # ---- the optimizer: the banked row's own construction VERBATIM (the
    # frozen arm's router is excluded by requires_grad; the learned arm's
    # is the ONLY addition)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, optimizer_config["lr"],
        optimizer_config["warmup_steps"], microbatch, stream, exposure,
        args.label, out / "training.jsonl", tuple(args.checkpoint_steps),
        cooldown_steps, EMA_DECAY)
    if run["status"] != "completed":
        raise RuntimeError(f"the {args.label} arm diverged at step "
                           f"{run['step']}")

    # ---- the falsifier/birth bars (the K11B 2.4-class discipline; the
    # smoke shakedown is exempt: tiny exposure, no log comparisons)
    trajectory_receipt = None
    if args.smoke:
        step1_receipt = dict(step1_loss=run["rows"][0]["loss"],
                            bit_identical=None,
                            note="smoke: no banked/log asserts")
    else:
        baseline_rows, baseline_log = _baseline_log_rows(args.seed)
        step1_receipt = dict(
            step1_loss=run["rows"][0]["loss"],
            baseline_step1_loss=baseline_rows[0]["loss"],
            baseline_log=baseline_log,
            bit_identical=bool(run["rows"][0]["loss"]
                               == baseline_rows[0]["loss"]))
        if not step1_receipt["bit_identical"]:
            raise ValueError(
                f"{args.label}: the step-1 loss must equal the seed-matched "
                f"baseline's bit-identically (the 0-flip-at-birth bar): "
                f"{step1_receipt['step1_loss']!r} vs "
                f"{step1_receipt['baseline_step1_loss']!r}")
    if arm["kind"] == "routing" and not arm["trainable"] \
            and not args.smoke:
        # the FROZEN-ROUTER falsifier: full-trajectory bit-identity is
        # EXPECTED (the forward is the chain bit for bit and the six
        # gradients are K10's own body on an identical recorded schedule);
        # any reduction-order growth is disclosed honestly, never hidden
        deltas = []
        for a, b in zip(run["rows"], baseline_rows):
            if a["loss"] != b["loss"]:
                deltas.append(abs(a["loss"] - b["loss"]))
        trajectory_receipt = dict(
            note="the frozen-router falsifier's per-step receipt against the "
                 "seed-matched baseline's own log (the K11B 2.4-class bar, "
                 "with full-trajectory bit-identity expected for this lane "
                 "by construction)",
            rows_compared=len(baseline_rows),
            rows_bit_identical=len(run["rows"]) - len(deltas) > 0,
            exact_bit_equal_every_step=len(deltas) == 0,
            max_abs_delta=max(deltas) if deltas else 0.0)

    # ---- the scoring pass (restored snapshots; training is over)
    rows_by_step = {r["step"]: r for r in run["rows"]}
    scores = []
    for step in args.checkpoint_steps:            # the constant-LR curve
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        row = dict(kind="constant", step=step, bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"],
                   eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        scores.append(row)
        print(f"  constant step {step}: eval {row['eval_nll']:.4f}", flush=True)
    constant_final = next((r["eval_nll"] for r in scores
                           if r["kind"] == "constant"
                           and r["step"] == run["constant_steps"]), None)
    if not args.smoke and args.seed == 3401 and arm["kind"] == "baseline" \
            and constant_final is not None \
            and constant_final != K10_BANKED["eval_nll_nats_per_byte"]:
        raise ValueError(
            f"the baseline arm's constant-180 eval must reproduce the banked "
            f"row bit-identically: {constant_final!r} vs "
            f"{K10_BANKED['eval_nll_nats_per_byte']!r}")
    # the SECONDARY: the EMA iterate at the constant-phase landing
    if run["snapshots"] and run["ema"] is not None \
            and run["constant_steps"] in run["snapshots"]:
        model.load_state_dict(run["snapshots"][run["constant_steps"]])
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    param.copy_(run["ema"][name].to(param.dtype))
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        ema_row = dict(kind="ema_180", step=run["constant_steps"],
                       bytes=rows_by_step[run["constant_steps"]]["bytes"],
                       eval_nll=result["score"],
                       eval_windows=result["eval_windows"],
                       eval_targets=result["eval_targets"],
                       ema_decay=EMA_DECAY,
                       note="ranking-only: biased toward mid-run weights")
        scores.append(ema_row)
        print(f"  EMA at step {run['constant_steps']}: "
              f"eval {ema_row['eval_nll']:.4f}", flush=True)
    # the PRIMARY: the post-cooldown final state
    post_cooldown = None
    if "final" in run["snapshots"]:
        model.load_state_dict(run["snapshots"]["final"])
        os.environ["X40_BATCHED_READS"] = "0"
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        post_cooldown = result["score"]
        final_row = dict(kind="post_cooldown", step=run["step"],
                         bytes=run["seen"], eval_nll=post_cooldown,
                         eval_windows=result["eval_windows"],
                         eval_targets=result["eval_targets"])
        scores.append(final_row)
        print(f"  post-cooldown final (step {run['step']}): "
              f"eval {post_cooldown:.4f}", flush=True)
    with (out / "scores.jsonl").open("w", buffering=1) as f:
        for row in scores:
            f.write(json.dumps(row) + "\n")

    # ---- the mechanism receipts at the post-cooldown final state
    diagnostics = None
    sweep = None
    if not args.smoke and "final" in run["snapshots"]:
        if arm["kind"] == "routing":
            diagnostics = routing_probe_diagnostics(
                model, stream, windows=args.probe_windows)
        else:
            diagnostics = k10_probe_diagnostics(
                model, stream, windows=args.probe_windows)
        sweep = k10_read_mask_sweep(model, stream)

    constant_steps = run["constant_steps"]
    receipt = dict(
        probe="k14_learned_routing", label=args.label, arm=arm["kind"],
        router_trainable=arm["trainable"],
        corrected_scoring=CORRECTED_SCORING, seed=args.seed,
        stopping_rule=stopping_rule, bands=BANDS,
        exposure_bytes=rows_by_step[constant_steps]["bytes"],
        exposure_steps=constant_steps,
        total_bytes=run["seen"], total_steps=run["step"],
        cooldown_steps=cooldown_steps,
        train_seconds_constant=round(run.get("constant_elapsed", 0.0), 1),
        train_seconds_total=round(run["elapsed"], 1),
        bytes_per_s=(rows_by_step[constant_steps]["bytes"]
                     / max(1e-9, run.get("constant_elapsed", 1.0))),
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=next((r["eval_nll"] for r in scores
                           if r["kind"] == "ema_180"), None),
        cooldown_drop=(constant_final - post_cooldown
                        if constant_final is not None
                        and post_cooldown is not None else None),
        step1_train_loss=run["rows"][0]["loss"],
        step1_birth_parity=step1_receipt,
        frozen_router_trajectory=trajectory_receipt,
        parameters=parameters_after,
        scores=scores, diagnostics=diagnostics, read_mask_sweep=sweep,
        k10_banked=K10_BANKED, baseline_3401_reused=BASELINE_3401,
        gdn2_references=GDN2_REFERENCES, audit_anchors=AUDIT_ANCHORS,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card))
    receipt_path = Path(args.receipts_dir) / f"k14_{args.label}.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "exposure_steps", "total_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll")}),
          flush=True)
    return receipt


# ------------------------------------------------------------------ assemble


def evaluate_predictions(summary_rows: dict) -> dict:
    """M-M1..M-M4, the three-way NLL read and the refutation clauses, against
    the pre-stated bands (the registration's 3, committed before the probe).
    Every mechanism anchor is the SEED-MATCHED arm's own re-measured value;
    the banked era's numbers are quoted alongside."""
    base = summary_rows.get("baseline_seed3401")
    if base is None:
        raise ValueError("the assemble needs the baseline arm's row")
    learned = {s: summary_rows.get(f"routing_learned_seed{s}")
               for s in (3401, 3402)}
    frozen = {s: summary_rows.get(f"routing_frozen_seed{s}")
              for s in (3401, 3402)}
    base2 = summary_rows.get("baseline_seed3402")

    def post(row):
        return None if row is None else row["final_eval_nll_post_cooldown"]

    # ---- M-M1: the op mix (THE HEADLINE INSTRUMENT)
    def mix(row):
        if row is None:
            return None
        diag = row.get("diagnostics") or {}
        rates = diag.get("op_rate_per_token_lane0") or {}
        return {n: float(rates.get(n, 0.0)) for n in OP_NAMES}

    def flip_rate(row):
        if row is None:
            return None
        return ((row.get("diagnostics", {}).get("routing", {}) or {})
                .get("flip_rate_all_layers"))

    mm1 = {}
    for s in (3401, 3402):
        anchor = base if s == 3401 else base2
        m_learned, m_anchor = mix(learned[s]), mix(anchor)
        tv = (tv_distance(m_learned, m_anchor)
              if m_learned is not None and m_anchor is not None else None)
        fr = flip_rate(learned[s])
        if tv is None:
            status = "not-run"
        elif tv >= BANDS["mm1_material_tv"] or (fr or 0.0) \
                >= BANDS["mm1_material_flip_rate"]:
            status = "met"
        elif tv <= BANDS["mm1_unchanged_tv"] and (fr or 0.0) \
                <= BANDS["mm1_unchanged_flip_rate"]:
            status = ("unchanged (the thresholds were already optimal -- "
                      "itself a first-class receipt)")
        else:
            status = "directional"
        mm1[f"seed{s}"] = dict(op_mix_learned=m_learned,
                               op_mix_baseline=m_anchor, tv_distance=tv,
                               flip_rate=fr, status=status)
    # ---- M-M2: POP/MERGE revival (with the rescope verbatim)
    mm2 = {}
    for s in (3401, 3402):
        m = mix(learned[s])
        if m is None:
            mm2[f"seed{s}"] = dict(status="not-run")
            continue
        pop, merge = m["pop"], m["merge"]
        met = pop >= BANDS["mm2_pop_rate"] or merge >= BANDS["mm2_merge_rate"]
        mm2[f"seed{s}"] = dict(
            pop_rate_per_token=pop, merge_rate_per_token=merge,
            banked_pop_rate=AUDIT_ANCHORS["pop_rate"],
            banked_merge_rate=AUDIT_ANCHORS["merge_rate"],
            met=bool(met),
            rescope="A null is UNINTERPRETABLE as a learned-management "
                    "verdict for those ops -- it receipts only that the "
                    "write-content surrogate does not indirectly flip "
                    "content-carrying tokens into them; the read-path "
                    "POP/MERGE estimator is the named follow-up "
                    "(the registration's 0.6, verbatim)")
    # ---- M-M3: the identity-CORRECT share falls
    def identity(row):
        if row is None:
            return None
        return (row.get("diagnostics") or {}).get("identity_rewrite_share")

    mm3 = {}
    for s in (3401, 3402):
        anchor = base if s == 3401 else base2
        i_learned, i_anchor = identity(learned[s]), identity(anchor)
        if i_learned is None or i_anchor is None:
            mm3[f"seed{s}"] = dict(status="not-run")
            continue
        delta = i_anchor - i_learned
        if delta >= BANDS["mm3_identity_fall"]:
            status = "met"
        elif abs(delta) <= BANDS["mm3_identity_unchanged"]:
            status = "unchanged"
        else:
            status = "refuted (the share rose)" if delta < 0 else "directional"
        mm3[f"seed{s}"] = dict(identity_learned=i_learned,
                               identity_baseline=i_anchor,
                               fall=delta, status=status,
                               banked_identity_share=AUDIT_ANCHORS[
                                   "identity_share"])
    # ---- M-M4: the NLL question (no directional prediction)
    mm4 = {}
    for s in (3401, 3402):
        anchor = base if s == 3401 else base2
        p_learned, p_anchor, p_frozen = (post(learned[s]), post(anchor),
                                         post(frozen[s]))
        if p_learned is None or p_anchor is None:
            mm4[f"seed{s}"] = dict(status="not-run")
            continue
        delta = p_anchor - p_learned           # positive = the router improves
        moved = mm1[f"seed{s}"]["tv_distance"] is not None \
            and (mm1[f"seed{s}"]["tv_distance"] >= BANDS["mm1_material_tv"]
                 or (mm1[f"seed{s}"]["flip_rate"] or 0.0)
                 >= BANDS["mm1_material_flip_rate"])
        if delta >= BANDS["mm4_lever_nats"]:
            reading = ("learned management is a REAL LEVER: the router "
                      "closes >= 0.02 of the ~0.07 best-vs-best residual, "
                      "and the leaner registration gains the router")
        elif abs(delta) < BANDS["endpoint_variance_band"] and moved:
            reading = ("parity with a MOVED op mix: the ops are "
                       "threshold-insensitive bookkeeping at probe scale -- "
                       "a first-class receipt either way")
        elif abs(delta) < BANDS["endpoint_variance_band"]:
            reading = ("parity with an UNMOVED mix: the thresholds were "
                       "optimal (also first-class)")
        else:
            reading = ("the learned router HURTS: the write-content "
                       "surrogate's bias is binding -- disclosed, with the "
                       "context+margin module and the read-path "
                       "POP/MERGE estimator as the named follow-ups")
        mm4[f"seed{s}"] = dict(
            post_cooldown_learned=p_learned,
            post_cooldown_baseline=p_anchor,
            post_cooldown_frozen=p_frozen,
            delta_vs_baseline=delta,
            endpoint_variance_band=BANDS["endpoint_variance_band"],
            gdn2_reference=GDN2_REFERENCES, reading=reading)
    return dict(
        MM1_op_mix=dict(
            prediction="the learned op mix shifts materially from the "
                       "threshold mix (refuted-with-receipt if the router "
                       "converges back to threshold behaviour)",
            bands=BANDS, seeds=mm1),
        MM2_pop_merge_revival=dict(
            prediction="POP/MERGE usage rises where the audit found them "
                       "idle (POP 0.0, MERGE 0.00313)",
            bands=BANDS, seeds=mm2),
        MM3_identity_correct_share=dict(
            prediction="the identity-CORRECT share falls (the audit's "
                       "degeneration diagnosis is what a learned router "
                       "should thin)",
            bands=BANDS, seeds=mm3),
        MM4_nll_question=dict(
            prediction="no directional prediction; the three-way read is "
                       "pre-stated (real lever / threshold-insensitive "
                       "bookkeeping / the thresholds were optimal)",
            bands=BANDS, seeds=mm4),
        refutation=dict(
            clause="if the mechanism numbers do not move while NLL does (or "
                   "vice versa) the dissociation is reported honestly; a "
                   "null is a result.  The M-M2 rescope binds verbatim: a "
                   "POP/MERGE null is uninterpretable as a "
                   "learned-management verdict for those ops -- the "
                   "write-content surrogate gives them no direct signal "
                   "(the read-path POP/MERGE estimator is the named "
                   "follow-up)"))


def assemble(args) -> dict:
    rows = {}
    for label in ARMS:
        path = Path(args.receipts_dir) / f"k14_{label}.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        rows[label] = receipt
    # the seed-3401 baseline row: the K12 lane's own re-scored baseline,
    # REUSED only after the config+protocol identity assert
    if "baseline_seed3401" not in rows:
        identity = _assert_k12_baseline_identity()
        k12 = json.loads(Path(K12_BASELINE_RECEIPT).read_text())
        rows["baseline_seed3401"] = dict(
            probe="k14_learned_routing", label="baseline_seed3401",
            arm="baseline", router_trainable=None, seed=3401,
            final_eval_nll_constant_lr=k12["final_eval_nll_constant_lr"],
            final_eval_nll_post_cooldown=k12["final_eval_nll_post_cooldown"],
            ema_eval_nll=k12["ema_eval_nll"],
            step1_birth_parity=dict(
                bit_identical=True,
                note="the reused row's own step-1: the K12 baseline asserted "
                     "every constant-phase row bit-identical to the banked "
                     "run's own training.jsonl"),
            exposure_bytes=k12["exposure_bytes"],
            exposure_steps=k12["exposure_steps"],
            train_seconds_constant=k12["train_seconds_constant"],
            train_seconds_total=k12["train_seconds_total"],
            bytes_per_s=k12["bytes_per_s"],
            parameters=k12["parameters"], scores=k12["scores"],
            diagnostics=k12["diagnostics"], read_mask_sweep=k12[
                "read_mask_sweep"],
            k10_banked=K10_BANKED, reused_from=dict(
                receipt=str(K12_BASELINE_RECEIPT),
                identity_assert=identity,
                note="the K12 lane's own re-scored baseline (the banked "
                     "winner re-scored under the corrected protocol; its "
                     "constant-180 eval is asserted bit-identical to the "
                     "banked row and its post-cooldown is the verified "
                     "1.6832871096474784)"))
    else:
        rows["baseline_seed3401"]["reused_from"] = None
    out = dict(probe="k14_learned_routing",
               arms=sorted(rows),
               corrected_scoring=CORRECTED_SCORING, bands=BANDS,
               baseline_3401_reused=BASELINE_3401,
               gdn2_references=GDN2_REFERENCES,
               audit_anchors=AUDIT_ANCHORS,
               scores={label: dict(
                   constant_180=rows[label]["final_eval_nll_constant_lr"],
                   post_cooldown=rows[label]["final_eval_nll_post_cooldown"],
                   ema_180=rows[label]["ema_eval_nll"],
                   walls=dict(constant=rows[label]["train_seconds_constant"],
                              total=rows[label]["train_seconds_total"]),
                   bytes_per_s=rows[label]["bytes_per_s"],
                   step1_bit_identical=rows[label]["step1_birth_parity"][
                       "bit_identical"],
                   frozen_trajectory=(rows[label].get(
                       "frozen_router_trajectory")
                       if "routing" in label else None))
                   for label in sorted(rows)})
    out["predictions"] = evaluate_predictions(rows)
    summary_path = Path(args.receipts_dir) / "k14_routing_summary.json"
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    atomic_json(summary_path, out)
    print(json.dumps(out["predictions"]["MM1_op_mix"], indent=1))
    print(json.dumps(out["predictions"]["MM4_nll_question"], indent=1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("arm")
    a.add_argument("--label", required=True)
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    a.add_argument("--keydim", type=int, default=KEY_DIM)
    a.add_argument("--ctx", type=int, default=CTX)
    a.add_argument("--eval-split", default="calibration")
    a.add_argument("--probe-windows", type=int, default=DIAG_WINDOWS)
    a.add_argument("--checkpoint-steps", type=int, nargs="+", default=[180])
    a.add_argument("--runs-dir", default=str(RUNS))
    a.add_argument("--receipts-dir", default=str(RECEIPTS))
    a.add_argument("--smoke", action="store_true",
                   help="a 3-step cooldown smoke of the plumbing only")
    b = sub.add_parser("assemble")
    b.add_argument("--receipts-dir", default=str(RECEIPTS))
    args = ap.parse_args()
    if args.cmd == "arm":
        if args.smoke:
            args.runs_dir = str(Path(args.runs_dir) / "smoke")
            args.receipts_dir = str(Path(args.receipts_dir) / "smoke")
            args.bytes = 200_000
            args.checkpoint_steps = [2]
        train_arm(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
