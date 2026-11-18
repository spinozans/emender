"""K12 -- the learned-placement quality probe: does the learned free-slot
policy move real text at 150 M scale, under the corrected scoring?

The question this stage exists for (``PREREGISTRATION_PLACEMENT.md`` claims
S1/S2, committed before this run): at the matched-exposure probe protocol
(20,022,866 targets, seed 3401, the banked K10-keys config, the banked
optimizer, the corrected scoring -- 24-step linear cooldown PRIMARY, EMA-0.99
secondary, constant-180 retained for banked comparability), does the learned
placement arm (and the supervisor-ruled FIXED-w bracket) improve on the
re-scored baseline, and do the pre-registered MECHANISM predictions (P-2
address locality, P-3 effective read rows, P-4 decorative stores, P-5 MERGE
revival) land?  The three-way read is pre-stated: (i) the estimator works,
(ii) the surrogate's bias is binding (option (c) becomes the named follow-up),
(iii) THE HONEST NULL -- free-slot placement is decorative on natural text.

The arms (the registration's 2):

* ``baseline_seed3401``       the banked K10-keys config itself (the 1.8418
                              lineage, AdamW, seed 3401) re-scored under the
                              corrected protocol -- the banked comparability
                              row: its 180 constant-phase step losses must be
                              bit-identical to the banked run's own
                              ``training.jsonl`` and its constant-180 eval
                              must reproduce the banked 1.841756020273481.
* ``placement_learned_seed3401``   the same config + the K12 module, the
                              per-head placement weight born at exact +0.0
                              and trainable through the admitted estimator
                              (the step-1 loss must equal the banked step-1
                              loss bit-identically -- the 0-flip-at-birth bar
                              at probe scale).
* ``placement_fixed{0p25,0p5,1p0}_seed3401``   the estimator-free controls:
                              the weight pinned at the ruled bracket values.
* second seeds (3402) on any arm meeting the pre-stated win bar (>= 0.01
  nats post-cooldown over the baseline).

Deliverables: per-arm receipts under ``k12_runs/<label>/`` (start.json,
training.jsonl, scores.jsonl, result.json -- the mechanism diagnostics, the
placement receipts, the address-locality instrument, the read-mask sweep and
the banked-equivalence asserts included) and the assemble summary
``receipts/k12_placement_summary.json``.

Usage (see ``run_k12_placement_leased.sh`` for the detached leased driver,
cards 2/3):

    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k12_probe_placement \\
        arm --label baseline_seed3401
    .venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k12_probe_placement assemble
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
    PROBE_WINDOWS as DIAG_WINDOWS, _agg_walk, _identity_share, _probe_windows,
    _project_keyed, _walk_stats, probe_diagnostics as k10_probe_diagnostics)
from experiments.expressivity_tasks.x45_matrix_emender.k10_separated_keys import (
    DEFAULT_CONTEXT, DEFAULT_KEY_DIM, DEFAULT_SEGMENT_TOKENS, MIMETIC_SEED,
    _k10_forward_launch, convert_model_to_separated_keys, span_key_std)
from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix import (
    _round_f32)
from experiments.expressivity_tasks.x45_matrix_emender.k12_placement import (
    PlacementSeparatedMixer, _k12_forward_launch,
    convert_model_to_placement, placement_parameter_delta)
from experiments.expressivity_tasks.x45_matrix_emender.protocol_correction import (
    COOLDOWN_STEPS, EMA_DECAY, GRAD_CLIP, _train_corrected, cooldown_factor)
from experiments.expressivity_tasks.x45_matrix_emender.side_by_side import (
    k10_read_mask_sweep)

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k12_runs"
BANKED_RUN = ROOT / "k10_runs" / "separated_keys_seed3401"
SEED = 3401
EXPOSURE_TARGETS = 20_000_000
SEGMENT_TOKENS = DEFAULT_SEGMENT_TOKENS
KEY_DIM = DEFAULT_KEY_DIM
CTX = DEFAULT_CONTEXT
WIN_BAR_NATS = 0.01          # the pre-stated second-seed win bar

# The banked K10-keys row (the arm to beat; never rewritten).
K10_BANKED = dict(eval_nll_nats_per_byte=1.841756020273481,
                  exposure_bytes=20_022_866, exposure_steps=180,
                  occupancy_mean_live_slots=24.0287348429362,
                  read_entropy_nats=1.6541064706071544,
                  correct_rate=0.97939, append_rate=0.01748,
                  merge_rate=0.00313, pop_rate=0.0,
                  identity_share=0.10321439792683149,
                  banked_log=str(BANKED_RUN / "training.jsonl"))

ARMS = {
    "baseline_seed3401": dict(kind="baseline", seed=3401,
                              placement_init=None, trainable=None),
    "placement_learned_seed3401": dict(kind="placement", seed=3401,
                                       placement_init=0.0, trainable=True),
    "placement_fixed0p25_seed3401": dict(kind="placement", seed=3401,
                                         placement_init=0.25, trainable=False),
    "placement_fixed0p5_seed3401": dict(kind="placement", seed=3401,
                                        placement_init=0.5, trainable=False),
    "placement_fixed1p0_seed3401": dict(kind="placement", seed=3401,
                                        placement_init=1.0, trainable=False),
    # second seeds: allowed for any placement arm once the win bar is met
    "placement_learned_seed3402": dict(kind="placement", seed=3402,
                                      placement_init=0.0, trainable=True),
    "placement_fixed0p25_seed3402": dict(kind="placement", seed=3402,
                                         placement_init=0.25, trainable=False),
    "placement_fixed0p5_seed3402": dict(kind="placement", seed=3402,
                                        placement_init=0.5, trainable=False),
    "placement_fixed1p0_seed3402": dict(kind="placement", seed=3402,
                                        placement_init=1.0, trainable=False),
    "baseline_seed3402": dict(kind="baseline", seed=3402,
                              placement_init=None, trainable=None),
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


def _assert_banked_equivalence(rows: list, banked: list, label: str) -> dict:
    """The protocol-equivalence proof (the protocol-correction lane's own
    receipt): every constant-phase training-log row bit-identical (bytes AND
    loss) to the banked run's own row.  Fails loudly."""
    mine = [r for r in rows if r.get("phase") == "constant"]
    if len(mine) != len(banked):
        raise ValueError(f"{label}: constant-phase row count {len(mine)} vs "
                         f"banked {len(banked)}")
    for a, b in zip(mine, banked):
        if (a["step"] != b["step"] or a["bytes"] != b["bytes"]
                or a["loss"] != b["loss"]):
            raise ValueError(
                f"{label}: protocol equivalence broken at step {a['step']}: "
                f"mine(bytes={a['bytes']}, loss={a['loss']!r}) vs "
                f"banked(bytes={b['bytes']}, loss={b['loss']!r})")
    return dict(banked_log=K10_BANKED["banked_log"], rows=len(mine),
                bit_identical=True)


# ------------------------------------------------------- the K12 instruments


def _address_locality(keys: torch.Tensor, live: torch.Tensor) -> dict:
    """P-2's instrument: the Pearson correlation between slot distance
    |j - j'| and key cosine cos(K_j, K_j') over LIVE-row pairs, pooled over
    (batch, head) -- the exit key tables of the probe windows.  Locality
    forms = materially NEGATIVE r (near slots hold similar keys)."""
    samples = []
    unit = keys / keys.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    for b in range(keys.shape[0]):
        for h in range(keys.shape[1]):
            idx = torch.nonzero(live[b, h]).flatten()
            if idx.numel() < 2:
                continue
            ii, jj = torch.triu_indices(idx.numel(), idx.numel(), offset=1)
            d = (idx[ii] - idx[jj]).abs().to(torch.float64)
            c = (unit[b, h, idx[ii]] * unit[b, h, idx[jj]]).sum(-1) \
                .to(torch.float64)
            samples.append(torch.stack([d, c], dim=1))
    if not samples:
        return dict(pearson_r=None, n_pairs=0,
                    note="fewer than two live rows")
    pool = torch.cat(samples)
    if pool.shape[0] < 8 or float(pool[:, 0].std()) == 0 \
            or float(pool[:, 1].std()) == 0:
        return dict(pearson_r=None, n_pairs=int(pool.shape[0]),
                    note="degenerate spread")
    # torch.corrcoef uses numpy semantics (each ROW is a variable): the
    # pooled sample matrix is (n_pairs, 2) -> transpose to (2, n_pairs)
    r = float(torch.corrcoef(pool.t())[0, 1])
    return dict(pearson_r=r, n_pairs=int(pool.shape[0]))


def placement_probe_diagnostics(model, stream, windows: int = DIAG_WINDOWS):
    """The TRAINED placement arm's mechanism receipts: the K10 probe's own
    instruments (op rates, occupancy, read entropy, identity share, the
    KEY-COLLAPSE watch -- imported) computed on the K12 launch with the
    TRAINED placement weight, PLUS the placement channel's own: the fire
    rate (APPENDs whose slot differs from K10's first-free rule), the
    chosen-slot residue similarity vs the free-slot mean (the margin), the
    trained w per layer, and the address-locality instrument."""
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        APPEND, OP_NAMES)
    from experiments.expressivity_tasks.x45_matrix_emender.k6_sequential_matrix \
        import _round_f32
    rows = _probe_windows(stream, windows)
    model.eval()
    per_layer = []
    key_stats = []
    locality = []
    placement_stats = []
    trained_w = {}

    def make_hook(layer_index, mixer):
        pol32 = (_round_f32(mixer.policy.tau_match),
                 _round_f32(mixer.policy.tau_merge),
                 _round_f32(mixer.policy.tau_pop),
                 _round_f32(mixer.policy.tie_break),
                 _round_f32(mixer.policy.read_scale))
        w = mixer.placement_weight.detach()

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
                out = _k12_forward_launch(
                    payload.float(), write_keys.float(), query_keys.float(),
                    decay.float(), w.float(), state, live0, keys0,
                    torch.zeros_like(keys0), pol32, mixer.rank,
                    mixer.segment_tokens, collect=True, warps=8, stages=3)
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
                loc = _address_locality(out["keys"].float(), out["live"])
                loc["layer"] = layer_index
                locality.append(loc)
                is_append = out["op"][..., 0] == APPEND
                first_free = (~out["live_pre"]).to(torch.int64).argmax(-1)
                fired = (out["slot"][..., 0] != first_free) & is_append
                n_app = int(is_append.sum())
                placement_stats.append(dict(
                    layer=layer_index,
                    append_tokens=n_app,
                    fired_vs_first_free=int(fired.sum()),
                    fire_rate=(float(fired.sum()) / n_app) if n_app else 0.0,
                    chosen_residue_sim_mean=(
                        float(out["place_sim"][is_append].mean())
                        if n_app else None),
                    free_mean_residue_sim_mean=(
                        float(out["place_free_mean"][is_append].mean())
                        if n_app else None),
                    placement_w=[float(v) for v in w.float().cpu()]))
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
    for s in placement_stats:
        by_layer.setdefault(s["layer"], []).append(s)
    placement_summary = dict(
        note="the placement channel's own receipts on the probe windows: "
             "the fire rate (APPENDs landing off the first-free rule), the "
             "chosen slot's residue similarity vs the free-slot mean, and "
             "the trained per-head weight",
        fire_rate_all_layers=float(
            sum(s["fired_vs_first_free"] for s in placement_stats)
            / max(1, sum(s["append_tokens"] for s in placement_stats))),
        append_tokens_all_layers=int(
            sum(s["append_tokens"] for s in placement_stats)),
        fired_all_layers=int(
            sum(s["fired_vs_first_free"] for s in placement_stats)),
        chosen_residue_sim_mean=float(sum(
            s["chosen_residue_sim_mean"] for s in placement_stats
            if s["chosen_residue_sim_mean"] is not None)
            / max(1, sum(1 for s in placement_stats
                         if s["chosen_residue_sim_mean"] is not None))),
        free_mean_residue_sim_mean=float(sum(
            s["free_mean_residue_sim_mean"] for s in placement_stats
            if s["free_mean_residue_sim_mean"] is not None)
            / max(1, sum(1 for s in placement_stats
                         if s["free_mean_residue_sim_mean"] is not None))),
        per_layer={str(l): dict(
            fire_rate=float(sum(s["fire_rate"] for s in ss) / len(ss)),
            placement_w=[float(v) for v in ss[0]["placement_w"]],
            n_windows=len(ss))
            for l, ss in sorted(by_layer.items())})
    locality_pairs = [s for s in locality if s["pearson_r"] is not None]
    locality_summary = dict(
        note="P-2's instrument: Pearson r between slot distance and key "
             "cosine over live-row pairs, per layer (the exit key tables of "
             "the probe windows); locality forms = materially negative r",
        mean_pearson_r=(float(sum(s["pearson_r"] for s in locality_pairs)
                              / len(locality_pairs))
                        if locality_pairs else None),
        per_layer_pearson_r={str(s["layer"]): s["pearson_r"]
                             for s in locality if s["layer"] is not None},
        n_pairs_total=int(sum(s["n_pairs"] for s in locality)))
    trained_w = {str(l): placement_summary["per_layer"][str(l)]["placement_w"]
                 for l in sorted(by_layer)}
    return dict(
        probe=dict(split="calibration", windows=len(rows),
                   mixers=len(model.blocks),
                   regime="eval, bf16 autocast, the persistent walk the arm "
                          "trains on (collect=True), the TRAINED placement "
                          "weight",
                   probe_wall_seconds=round(wall, 1)),
        key_collapse_watch=key_summary,
        placement=placement_summary,
        address_locality=locality_summary,
        trained_placement_w=trained_w,
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
    # placement conversion (the ONLY difference; it draws no RNG, so the
    # birth weights are the banked row's own bit for bit plus w = +0.0)
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
    if arm["kind"] == "placement":
        model = convert_model_to_placement(
            model, placement_init=arm["placement_init"],
            placement_trainable=arm["trainable"],
            segment_tokens=SEGMENT_TOKENS)
        parameters_after = sum(p.numel() for p in model.parameters())
        expected_delta = placement_parameter_delta(len(model.blocks),
                                                  model.blocks[0].mixer.heads)
        if parameters_after - separated_parameters != expected_delta:
            raise ValueError(f"the placement delta must be exactly "
                             f"{expected_delta}; got "
                             f"{parameters_after - separated_parameters}")
        for block in model.blocks:
            if not isinstance(block.mixer, PlacementSeparatedMixer):
                raise ValueError("the placement conversion must rewrap "
                                 "every mixer")
        w0 = model.blocks[0].mixer.placement_weight.detach()
        if arm["trainable"]:
            if not torch.equal(w0, torch.zeros_like(w0)):
                raise ValueError("the learned arm's weight must be born at "
                                 "EXACT +0.0")
        else:
            want = torch.full_like(w0, float(arm["placement_init"]))
            if not torch.equal(w0, want):
                raise ValueError(f"the fixed arm's weight must be pinned at "
                                 f"{arm['placement_init']}")
        # the registered kernel path must be the path taken at the registered
        # shape (the estimator lives there; the oracle is the honest zero)
        from experiments.expressivity_tasks.x45_matrix_emender.k12_placement \
            import placement_available
        if not placement_available(128, 128, args.keydim):
            raise ValueError("the registered shape must take the kernel path")
    else:
        parameters_after = separated_parameters
    summary = dict(summary)
    summary["scan_path"] = ("k12_placement (per-token, the key channel + the "
                            "learned free-slot placement)"
                            if arm["kind"] == "placement" else
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
    config = dict(probe="k12_placement", label=args.label, arm=arm["kind"],
                  addressing=("separated write-context keys (key_dim 16) + "
                              "the learned free-slot placement on the "
                              "dead-key residue"
                              if arm["kind"] == "placement" else
                              "separated write-context keys (key_dim 16)"),
                  placement=(None if arm["kind"] != "placement" else dict(
                      init=arm["placement_init"], trainable=arm["trainable"],
                      feature="dead-key residue: Res[j] := the dying row's "
                              "last key at POP/MERGE deaths, pre-clear",
                      birth="w born at EXACT +0.0; the +0.0 argmax tie-break "
                            "reproduces first-free bit-exactly",
                      estimator=("the supervisor-approved straight-through "
                                 "softmax surrogate, score inputs detached, "
                                 "only the per-head w learns"
                                 if arm["trainable"] else
                                 "none (pinned: the estimator-free control)"))),
                  seed=args.seed, exposure_targets=exposure,
                  key_dim=args.keydim, key_context=args.ctx,
                  segment_tokens=SEGMENT_TOKENS, optimizer=optimizer_config,
                  microbatch=microbatch, context=L.CONTEXT,
                  corrected_scoring=CORRECTED_SCORING,
                  win_bar_nats=WIN_BAR_NATS,
                  parameters=dict(total=parameters_after,
                                  placement_delta=(parameters_after
                                                   - separated_parameters)),
                  windows_per_step=64, summary=summary, identity=identity,
                  bindings=dict(
                      matched_arms.registration_bindings(),
                      k12_preregistration=str(
                          ROOT / "PREREGISTRATION_PLACEMENT.md"),
                      k12_preregistration_sha256=digest(
                          ROOT / "PREREGISTRATION_PLACEMENT.md"),
                      k10_preregistration=str(
                          ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
                      k10_gates_receipt=str(RECEIPTS / "k10_gates.json"),
                      k12_gates_receipt=str(RECEIPTS / "k12_gates.json"),
                      k12_gates_sha256=digest(RECEIPTS / "k12_gates.json"),
                      banked_k10_row=K10_BANKED),
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

    # ---- the optimizer: the banked row's own construction VERBATIM
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

    # ---- the banked-equivalence asserts (the protocol-identical receipts)
    banked_rows = _banked_log_rows()
    banked_equivalence = None
    step1_receipt = None
    if not args.smoke and args.seed == 3401:
        if arm["kind"] == "baseline":
            banked_equivalence = _assert_banked_equivalence(
                run["rows"], banked_rows, args.label)
        else:
            step1_receipt = dict(
                step1_loss=run["rows"][0]["loss"],
                banked_step1_loss=banked_rows[0]["loss"],
                bit_identical=bool(run["rows"][0]["loss"]
                                   == banked_rows[0]["loss"]))
            if arm["trainable"] and not step1_receipt["bit_identical"]:
                raise ValueError(
                    f"{args.label}: the step-1 loss must equal the banked "
                    f"row's bit-identically (the 0-flip-at-birth bar): "
                    f"{step1_receipt['step1_loss']!r} vs "
                    f"{step1_receipt['banked_step1_loss']!r}")

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
        if arm["kind"] == "placement":
            diagnostics = placement_probe_diagnostics(
                model, stream, windows=args.probe_windows)
        else:
            diagnostics = k10_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            # the address-locality instrument on the baseline too (the
            # matched anchor for P-2): the exit key tables of the probe
            # windows per layer, via the K10 collect launch
            from experiments.expressivity_tasks.x42_pile_scale.byte_velocity \
                import grouped
            rows = _probe_windows(stream, args.probe_windows)
            per_layer_loc = {}
            for layer_index, block in enumerate(model.blocks):
                mixer = block.mixer
                pol32 = (_round_f32(mixer.policy.tau_match),
                         _round_f32(mixer.policy.tau_merge),
                         _round_f32(mixer.policy.tau_pop),
                         _round_f32(mixer.policy.tie_break),
                         _round_f32(mixer.policy.read_scale))
                for row in rows:
                    x, _y, _v = next(grouped([row], 1))
                    with torch.no_grad(), torch.autocast(
                            "cuda", dtype=torch.bfloat16):
                        payload, write_keys, query_keys, decay = \
                            _project_keyed(mixer, x)
                        batch, tokens, _ = x.shape
                        out10 = _k10_forward_launch(
                            payload.float(), write_keys.float(),
                            query_keys.float(), decay.float(),
                            torch.zeros(batch, mixer.heads, mixer.rank_max,
                                        mixer.value_dim,
                                        dtype=torch.float32,
                                        device=x.device),
                            torch.zeros(batch, mixer.heads, mixer.rank_max,
                                        dtype=torch.bool, device=x.device),
                            torch.zeros(batch, mixer.heads, mixer.rank_max,
                                        mixer.key_dim,
                                        dtype=torch.float32,
                                        device=x.device), pol32, mixer.rank,
                            mixer.segment_tokens, collect=True, warps=8,
                            stages=3)
                    loc = _address_locality(out10["keys"].float(),
                                            out10["live"])
                    per_layer_loc.setdefault(layer_index, []).append(loc)
            pooled = [dict(layer=l,
                           pearson_r=(float(sum(s["pearson_r"] for s in ss)
                                            / len(ss))
                                       if all(s["pearson_r"] is not None
                                              for s in ss) else None))
                      for l, ss in sorted(per_layer_loc.items())]
            vals = [p["pearson_r"] for p in pooled
                    if p["pearson_r"] is not None]
            diagnostics["address_locality"] = dict(
                note="P-2's instrument on the baseline (the matched anchor)",
                mean_pearson_r=(float(sum(vals) / len(vals))
                                if vals else None),
                per_layer_pearson_r={str(p["layer"]): p["pearson_r"]
                                     for p in pooled})
        sweep = k10_read_mask_sweep(model, stream)

    constant_steps = run["constant_steps"]
    receipt = dict(
        probe="k12_placement", label=args.label, arm=arm["kind"],
        placement_init=arm["placement_init"],
        placement_trainable=arm["trainable"],
        corrected_scoring=CORRECTED_SCORING, seed=args.seed,
        stopping_rule=stopping_rule,
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
        banked_equivalence=banked_equivalence,
        step1_birth_parity=step1_receipt,
        parameters=parameters_after,
        scores=scores, diagnostics=diagnostics, read_mask_sweep=sweep,
        k10_banked=K10_BANKED,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(config), config=config,
        session=matched_arms.session_meta(card))
    receipt_path = Path(args.receipts_dir) / f"k12_{args.label}.json"
    atomic_json(receipt_path, receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "exposure_steps", "total_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll")}),
          flush=True)
    return receipt



# ------------------------------------------------------------------ assemble


def evaluate_predictions(summary_rows: dict) -> dict:
    """P-1..P-5, the three-way read and the refutation clause, against the
    pre-stated bands (PREREGISTRATION_PLACEMENT.md 3, committed before the
    probe).  All mechanism anchors are the arms' OWN re-scored values (the
    matched instrument); the banked era's numbers are cited alongside."""
    import math
    base = summary_rows.get("baseline_seed3401")
    if base is None:
        raise ValueError("the assemble needs the baseline arm's receipt")
    arms = {k: v for k, v in summary_rows.items()
            if k != "baseline_seed3401" and v is not None}
    evals = {}
    best_label, best_delta = None, None
    for label, row in sorted(arms.items()):
        delta = (base["final_eval_nll_post_cooldown"]
                 - row["final_eval_nll_post_cooldown"])
        evals[label] = dict(post_cooldown=row["final_eval_nll_post_cooldown"],
                           delta_vs_baseline=delta)
        if best_delta is None or delta > best_delta:
            best_label, best_delta = label, delta
    p1_met = best_delta is not None and best_delta > 0
    p1_band = best_delta is not None and 0.0 <= best_delta <= 0.03
    win_arms = [l for l, e in evals.items() if e["delta_vs_baseline"]
                >= WIN_BAR_NATS]
    # P-2: address locality (the mean Pearson r per arm vs the baseline's)
    base_r = (base.get("diagnostics", {}).get("address_locality", {})
              .get("mean_pearson_r"))
    p2 = {}
    for label, row in sorted(arms.items()):
        r = (row.get("diagnostics", {}).get("address_locality", {})
             .get("mean_pearson_r"))
        p2[label] = dict(mean_pearson_r=r,
                         delta_vs_baseline=(None if r is None
                                            or base_r is None else r - base_r))
    p2_met = any(v["delta_vs_baseline"] is not None
                 and v["delta_vs_baseline"] <= -0.02
                 for v in p2.values())
    # P-3: effective rows e^H from the read entropy
    def eff_rows(row):
        h = (row.get("diagnostics", {})
             .get("read_weight_entropy", {}).get("mean"))
        return math.exp(h) if h else None
    base_eff = eff_rows(base)
    p3 = {label: dict(effective_rows=eff_rows(row),
                     delta_vs_baseline=(None if eff_rows(row) is None
                                        or base_eff is None
                                        else eff_rows(row) - base_eff))
          for label, row in sorted(arms.items())}
    p3_met = any(v["delta_vs_baseline"] is not None
                 and v["delta_vs_baseline"] >= 0.5 for v in p3.values())
    # P-4: the read-mask sweep (layers with deltaNLL >= 0.02)
    def loadbearing(row):
        sweep = row.get("read_mask_sweep") or {}
        deltas = sweep.get("delta_per_layer") or {}
        return sum(1 for v in deltas.values() if v >= 0.02)
    base_lb = loadbearing(base)
    p4 = {label: dict(layers_deltanll_ge_0p02=loadbearing(row),
                     delta_vs_baseline=loadbearing(row) - base_lb)
          for label, row in sorted(arms.items())}
    p4_met = any(v["layers_deltanll_ge_0p02"] >= 10 for v in p4.values())
    # P-5: MERGE revival
    def merge_rate(row):
        return ((row.get("diagnostics", {})
                 .get("op_rate_per_token_lane0", {}) or {}).get("merge"))
    p5 = {label: dict(merge_rate_per_token=merge_rate(row))
          for label, row in sorted(arms.items())
          if merge_rate(row) is not None}
    p5_met = any(v["merge_rate_per_token"] >= 0.01 for v in p5.values())
    # the three-way read (supervisor-ruled)
    learned = summary_rows.get("placement_learned_seed3401") or {}
    learned_w = ((learned.get("diagnostics", {})
                  .get("placement", {}) or {}).get("fire_rate_all_layers"))
    learned_w_move = ((learned.get("diagnostics", {})
                       .get("trained_placement_w", {}) or {}))
    fixed_helped = [l for l in evals
                    if l.startswith("placement_fixed")
                    and evals[l]["delta_vs_baseline"] > 0]
    learned_helped = ("placement_learned_seed3401" in evals
                      and evals["placement_learned_seed3401"]
                      ["delta_vs_baseline"] > 0)
    if learned_helped and (not fixed_helped
                           or evals["placement_learned_seed3401"]
                           ["delta_vs_baseline"]
                           >= max(evals[l]["delta_vs_baseline"]
                                  for l in fixed_helped)):
        three_way = dict(outcome="i", reading="the estimator works: learned-w "
                         "matches or beats the best fixed point")
    elif fixed_helped and not learned_helped:
        three_way = dict(outcome="ii",
                         reading="the surrogate's bias is binding: a fixed w "
                         "helps while the learned w does not -- option (c) "
                         "(attached score inputs) becomes the named "
                         "follow-up")
    elif not fixed_helped and not learned_helped:
        three_way = dict(outcome="iii", reading="THE HONEST NULL: free-slot "
                         "placement is decorative on natural text; C5's "
                         "write/read form owns the residual")
    else:
        three_way = dict(outcome="i-partial",
                         reading="both the learned and a fixed w help; the "
                         "estimator and the lever both stand, best arm "
                         f"{best_label}")
    return dict(
        P1_nll=dict(prediction="at least one placement arm improves on the "
                     "baseline's post-cooldown NLL (seed 3401)",
                    band="improvement in [0.0, 0.03] nats (F1-sized, "
                         "honestly wide); the >= 0.01 win bar fires the "
                         "second seed",
                    baseline_post_cooldown=base["final_eval_nll_post_cooldown"],
                    arms=evals, met=bool(p1_met), band_met=bool(p1_band),
                    win_arms=win_arms,
                    best_arm=best_label, best_delta=best_delta),
        P2_address_locality=dict(
            prediction="the live-pair slot-distance/key-cosine Pearson r "
                       "goes materially negative vs the baseline",
            band="delta r <= -0.02; unchanged if |delta r| <= 0.01",
            baseline_mean_pearson_r=base_r, arms=p2, met=bool(p2_met)),
        P3_effective_read_rows=dict(
            prediction="effective rows e^H rise from the baseline's own",
            band=">= +0.5 rows; unchanged within +/-0.25; refuted if lower",
            baseline_effective_rows=base_eff, arms=p3, met=bool(p3_met)),
        P4_decorative_stores=dict(
            prediction="layers with read-mask deltaNLL >= 0.02 rise",
            band=">= 10 of 24; refuted if not above the baseline's own",
            baseline_layers_ge_0p02=base_lb, arms=p4, met=bool(p4_met)),
        P5_merge_revival=dict(
            prediction="MERGE rate rises to >= 0.01/token -- the first "
                       "positive per-token MERGE receipt",
            band=">= 0.01/token; reported honestly either way",
            banked_merge_rate=K10_BANKED["merge_rate"], arms=p5,
            met=bool(p5_met)),
        three_way_read=three_way,
        refutation=dict(
            clause="if the mechanism numbers do not move while NLL does (or "
                   "vice versa) the dissociation is reported honestly; a "
                   "null is a result -- it would say free-slot placement is "
                   "not load-bearing on natural text"))


def assemble(args) -> dict:
    rows = {}
    for label in ARMS:
        path = Path(args.runs_dir) / label / "result.json"
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        rows[label] = receipt
    out = dict(probe="k12_placement", arms=sorted(rows),
               corrected_scoring=CORRECTED_SCORING,
               win_bar_nats=WIN_BAR_NATS,
               scores={label: dict(
                   constant_180=rows[label]["final_eval_nll_constant_lr"],
                   post_cooldown=rows[label]["final_eval_nll_post_cooldown"],
                   ema_180=rows[label]["ema_eval_nll"],
                   walls=dict(constant=rows[label]["train_seconds_constant"],
                              total=rows[label]["train_seconds_total"]),
                   bytes_per_s=rows[label]["bytes_per_s"])
                   for label in sorted(rows)})
    out["predictions"] = evaluate_predictions(rows)
    summary_path = Path(args.receipts_dir) / "k12_placement_summary.json"
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    atomic_json(summary_path, out)
    print(json.dumps(out["predictions"]["P1_nll"], indent=1))
    print(json.dumps(out["predictions"]["three_way_read"], indent=1))
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
                   help="harness shakedown: tiny exposure, 3 cooldown steps, "
                        "no banked asserts, no diagnostics")
    b = sub.add_parser("assemble")
    b.add_argument("--runs-dir", default=str(RUNS))
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
    if args.cmd == "assemble":
        assemble(args)
        return


if __name__ == "__main__":
    main()
