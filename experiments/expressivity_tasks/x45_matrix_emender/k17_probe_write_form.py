"""K17 -- THE WRITE FORM UNDER THE ADDITIVE READ: the probe (the arms, the
mechanism instruments, the assemble) -- the dense gated write (the
b_proj-delta form) and the read-gain bracket on the COMMITTED composed
machine.

THE QUESTION THIS LANE EXISTS FOR (the K15 adjudication's named follow-up,
the LAST structural candidate owner of the composed machine's remaining
~0.03 vs GDN2): the morning's G2 arm HURT on the SOFTMAX machine (+0.007)
-- the per-head dense-write channel did not compose with the COMPETITIVE
read.  The machine has since changed fundamentally: the K13 ADDITIVE read
(the load-bearing lever, ~60% of the corrected gap closed) under the K11B
p3 per-channel gate stack (the composed machine, the program's best
absolute rows 1.3811104467936925 / 1.383232252938407 post-cooldown,
+0.0289/+0.0311 remaining vs GDN2's 1.3521734476089478, parity-within-band
under the supervisor's ruling of 2026-11-25).  The read/write interface is
different now: the read law and the write law may compose the way the
gates and the read law did not.

THE ARMS (``PREREGISTRATION_K17_WRITE_FORM.md``, committed BEFORE any
probe measurement; every arm at the EXACT probe protocol -- the
20,022,866-target matched-exposure landing at 180 constant-LR steps on
the registered flagship stream, the document-disjoint calibration eval,
the honest clock, the protocol's own grad-clip, the banked knobs asserted,
the inherited 4x hybrid-Muon point, the p3 gate tensors AND the dense
channel's tensors on the AdamW side -- under the CORRECTED scoring: the
24-step linear cooldown PRIMARY, EMA-0.99 secondary, constant-180
retained), seed 3401 first:

* ``dense``        the composed machine + the DENSE GATED WRITE (the
                   b_proj-delta form, born at exact zero: the machine is
                   BORN AS THE COMPOSED MACHINE bit for bit) -- G2's form
                   reborn on the additive interface.
* ``dense_frozen`` the FROZEN-WRITE FALSIFIER CONTROL (the K11B R4
                   precedent, mandatory): the dense channel's parameters
                   frozen at birth -- the bar: step-1 bit-identity to the
                   composed machine's own row, and a landing at parity
                   with the composed machine's same-seed banked row.
* ``gain1x`` / ``gain2x`` / ``gain4x``   the READ-GAIN bracket (config
                   arms, not machine variants): the composed machine byte
                   for byte with ``read_scale`` in {1.0, 2.0, 4.0}.  The
                   machine's ACTUAL inherited gain is the tuned
                   softmax-machine point 2.7870458737509636 (the factual
                   correction, stated in the registration: the receipts
                   quote it, not the registration-text folklore) -- the
                   inherited cell's rows are the K15 composition's own at
                   both seeds (cited, never re-run).
* ``combined``     the dense write + the best gain cell, ONLY per the
                   pre-stated rule (either lever's seed-3401 improvement
                   >= 0.01).

THE PRE-STATE: NO DIRECTIONAL PREDICTION.  The clauses W-1/W-2/W-3 (the
compose bar / parity / interference), G-1 (the gain), the conditional
combined arm, the standing 0.01 win-bar second-seed rule, and the band
discipline (at-or-past GDN2's 1.3521734476089478 -> both scorings, both
seeds, ESCALATED) are receipted in every arm and applied exactly as
stated by the assemble; the final gap-quote phrasing is the supervisor's
to adjudicate (the K13 clause-conflict precedent).

PROVENANCE: the K15 ENGINE IS COMMITTED AND READ-ONLY -- the machine is
``velocity_ladder._build_k10`` + ``convert_model_to_composition``
(imported verbatim) [+ ``convert_model_to_dense_write`` -- the K17
conversion, at the zero-born birth]; no ``k10_*``/``k11*``/``k12_*``/
``k13_*``/``k15_*``/``k16_*`` source, no committed PREREGISTRATION, no
``lattice.py`` is edited.

Usage (see ``run_k17_write_form_leased.sh`` for the detached leased
driver, cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k17_probe_write_form \\
        arm --arm dense --seed 3401 --output k17_runs/dense_seed3401
    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k17_probe_write_form \\
        assemble
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
from experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
    import composed_4x_banked

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k17_runs"
K15_RUNS = ROOT / "k15_runs"

# The inherited hybrid-Muon point (K16's own receipt: already the composed
# machine's bracket optimum at both seeds).
INHERITED_LR_MULTIPLIER = 4
TUNED_ADAMW_LR = TUNED_WINNER["lr"]

# The composed machine's ACTUAL inherited read gain -- the tuned
# softmax-machine point (the registration's factual correction: the "2.0"
# of the K13 registration text is the RoutingPolicy CLASS DEFAULT, which
# the banked build overrides; the receipts' read_scale=2.0 rows are the
# base build's stale summary record, not the machine's policy).
INHERITED_READ_SCALE = TUNED_WINNER["read_scale"]

GDN2_POST_COOLDOWN = BANKED["gdn2"]["post_cooldown"]
BANKED_CORRECTED_ANCHOR = BANKED["post_cooldown"]["seed3402"]
BANDS = dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
             winner_seed_delta=WINNER_SEED_DELTA)
WIN_BAR = 0.01                     # the standing second-seed / signal bar

# The read-gain bracket (the task's own numbers; the inherited 2.787 cell's
# rows are the K15 composition's own at both seeds -- cited, never re-run).
GAIN_ARMS = {"gain1x": 1.0, "gain2x": 2.0, "gain4x": 4.0}

ARMS = {
    "dense": dict(
        kind="dense", read_scale=None,
        note="the composed machine + the DENSE GATED WRITE (the b_proj-"
             "delta form): every token deposits m_d*sigmoid(d_proj(x)) (.) "
             "p at its best-match LIVE row (P3), recorded as schedule lane "
             "2, born at exact zero -- the machine is BORN AS THE COMPOSED "
             "MACHINE bit for bit"),
    "dense_frozen": dict(
        kind="dense", read_scale=None, frozen=True,
        note="the FROZEN-WRITE FALSIFIER CONTROL (mandatory, the K11B R4 "
             "precedent): the dense channel's parameters frozen at birth "
             "(requires_grad=False, excluded from the optimizer; the "
             "routing then equals the composed machine's own split "
             "exactly) -- the bar: step-1 bit-identity to the composed "
             "machine's own row, and a landing at parity with the composed "
             "machine's same-seed banked row"),
    "gain1x": dict(kind="gain", read_scale=1.0,
                   note="the read-gain bracket's near side: read_scale 1.0"),
    "gain2x": dict(kind="gain", read_scale=2.0,
                   note="the read-gain bracket's policy-default cell: "
                        "read_scale 2.0"),
    "gain4x": dict(kind="gain", read_scale=4.0,
                   note="the read-gain bracket's far side past the "
                        "inherited 2.787 point: read_scale 4.0"),
    "combined": dict(kind="dense", read_scale="arg",
                     note="the combined arm: the dense write + the best "
                          "read-gain cell -- fires ONLY per the pre-stated "
                          "rule (either lever's seed-3401 improvement "
                          ">= 0.01); pass the gain with --gain"),
}

# The pre-stated rules (the registration's 4; receipted in every arm).
RULES = dict(
    W1=dict(bar=WIN_BAR,
            text="the dense arm's post-cooldown row improves on the "
                 "composed machine's same-seed banked row by >= 0.01 at "
                 "BOTH seeds -> the dense gated write COMPOSES with the "
                 "additive read; the write-form attribution owns that "
                 "much of the residual"),
    W2=dict(text="improvement at one seed inside the 0.0119 seed band and "
                 "no replication at the other -> PARITY-grade evidence "
                 "(the C-2 read-form-dependence verdict's own phrasing), "
                 "reported as measured, NOT banked as composing"),
    W3=dict(text="the dense arm lands WORSE than its same-seed baseline "
                 "beyond the band -> the G2-under-softmax result REPLICATES "
                 "under the additive read (the write form does not "
                 "compose), reported as the refutation with the C5 "
                 "mechanism rows attached"),
    G1=dict(bar=WIN_BAR,
            text="any gain cell improving the same-seed baseline by >= 0.01 "
                 "at 3401 -> the gain owns part of the residual (a config "
                 "yield, the K16 convention); inside the seed band -> the "
                 "inherited point stands, receipted"),
    combined=dict(text="the combined arm fires ONLY if either lever's "
                        "seed-3401 row improves >= 0.01: dense + the best "
                        "gain cell"),
    second_seed=dict(bar=WIN_BAR,
                     text="any arm whose seed-3401 improvement is >= 0.01 "
                          "owes seed 3402 BEFORE any claim (the standing "
                          "win-bar rule)"),
    band_discipline=dict(text="any arm whose post-cooldown row would put "
                              "the machine AT OR PAST GDN2's "
                              "1.3521734476089478 is reported with BOTH "
                              "scorings and BOTH seeds before any claim, "
                              "and ESCALATED, not adjudicated here; any "
                              "inversion of a standing quality conclusion "
                              "escalates likewise"),
    falsifier=dict(text="the frozen-write control's bar: (a) the control's "
                        "step-1 train loss MUST be bit-identical to the "
                        "composed machine's own same-seed row (the "
                        "forward-at-birth plumbing receipt); (b) the "
                        "control's landing MUST be at parity with the "
                        "composed machine's same-seed banked row (inside "
                        "the winner's own measured seed-delta scale on "
                        "the primary scoring); the honest limitation: "
                        "full-trajectory bit-identity is NOT achievable "
                        "by construction (the K11B 2.4 precedent) -- the "
                        "per-step loss divergence is measured and quoted"))


# --------------------------------------------------------------- the receipts


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def _dense_tensor_names():
    return ("d_proj.weight", "dense_scalar")


def split_parameters_k17(model, frozen: bool = False):
    """The banked winner's own routing (``banked_gates_adamw``) with the k11b
    p3 per-channel gate projections AND the K17 dense channel's tensors on
    the hybrid's AdamW side -- the incumbent's own convention (the DECAY
    gate's side: "a gate, not a hidden map").  A frozen control's dense
    tensors carry ``requires_grad=False`` and are excluded by the
    ``requires_grad`` filter -- the split then equals the composed
    machine's own exactly (the falsifier's own plumbing receipt)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_probe_channel_gates \
        import gate_tensor_names
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import (_DECAY_BIAS, _DECAY_WEIGHT, _EMBED_HEAD, _MIXER_MATRIX, _MLP,
                _NORMS)
    gate_names = list(gate_tensor_names("p3"))
    if not frozen:
        gate_names += list(_dense_tensor_names())
    gate_re = re.compile(
        r"blocks\.\d+\.mixer\.(" + "|".join(
            re.escape(n) for n in gate_names) + r")$")
    muon, adamw = [], []
    params = dict(model.named_parameters())
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
                f"unclassified parameter: {name} (the inventory is closed: "
                f"the banked winner's patterns + the p3 gate tensors + the "
                f"dense channel's tensors)")
        if kind in ("decay_weight", "decay_bias", "norm"):
            adamw.append(name)
        else:
            if param.ndim != 2:
                raise ValueError(f"Muon-owned parameter is not a full 2D "
                                 f"matrix: {name}")
            muon.append((name, param))
    muon_names = [entry[0] for entry in muon]
    muon_params = [entry[1] for entry in muon]
    adamw_params = [params[n] for n in adamw]
    return muon_params, adamw_params, muon_names, adamw


def _routing_cross_check_k17(model, seed: int, frozen: bool) -> dict:
    """The no-drift receipt: an un-converted twin of the SAME build must
    route EXACTLY as the banked winner's own split routes it; the K17
    model's split must equal that plus exactly the p3 gate tensors (and,
    on the dense arms with the channel live, the dense channel's tensors)
    on the AdamW side -- or, for the frozen control, equal the composed
    machine's own split exactly)."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import ComposedReadMixer
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import DenseWriteComposedMixer
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import split_parameters_winner
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import gate_tensor_names
    has_dense = isinstance(model.blocks[0].mixer, DenseWriteComposedMixer)
    if not has_dense and not isinstance(model.blocks[0].mixer,
                                        ComposedReadMixer):
        raise ValueError("the K17 arms' mixers must be composed (K15) or "
                         "dense-composed (K17)")
    muon, adamw, muon_names, adamw_names = split_parameters_k17(model, frozen)
    twin, _cfg, _summary = _build_k10(dict(TUNED_WINNER), seed)
    _t_muon, _t_adamw, twin_muon, twin_adamw = split_parameters_winner(
        twin, "banked_gates_adamw")
    del twin, _t_muon, _t_adamw
    torch.cuda.empty_cache()
    if sorted(muon_names) != sorted(twin_muon):
        raise ValueError("the K17 model's Muon group drifted from the "
                         "banked winner's own split")
    expected_gate_names = sorted(
        f"blocks.{i}.mixer.{name}"
        for i in range(len(model.blocks))
        for name in gate_tensor_names("p3"))
    expected = set(twin_adamw) | set(expected_gate_names)
    if has_dense and not frozen:
        expected |= set(
            f"blocks.{i}.mixer.{name}"
            for i in range(len(model.blocks))
            for name in _dense_tensor_names())
    if sorted(adamw_names) != sorted(expected):
        raise ValueError("the K17 model's AdamW group must equal the banked "
                         "winner's own plus the p3 gate tensors (plus the "
                         "dense channel's tensors on the live dense arms)")
    return dict(muon_tensors=len(muon_names), adamw_tensors=len(adamw_names),
                dense_channel_live=bool(has_dense and not frozen),
                note="the banked winner's own split + the p3 gate tensors ("
                     "+ the dense channel's tensors on the live dense arms) "
                     "on the AdamW side; the frozen control: the banked + "
                     "p3 only -- the dense tensors are excluded by "
                     "requires_grad")


# --------------------------------------------------- the birth receipts


def dense_birth_receipts(model, stream, seed: int, microbatch: int) -> dict:
    """The write-form machine's own birth receipts, run BEFORE any training
    on the FIRST train batch (the batch the step-1 loss sees):

    1. BORN AS THE COMPOSED MACHINE: layer for layer, at the arm's own
       inputs, the dense scan's every SHARED channel is `torch.equal` to
       the K15 COMPOSED kernel's own committed launch -- the dense channel
       is born at exact zero (the mixer's own expression
       ``0*sigmoid(0)*p``), so the dense machine's forward at birth IS the
       composed machine's (and hence the K13 additive machine's: the K15
       receipt's own chain).
    2. The dense channel born at exact zero per mixer, and the p3 gates
       born at the registered scalar equivalents (the K15 receipt's own
       checks, re-run on this machine).
    3. The dense lane's own birth telemetry: the fire rate (the live-guard
       coverage -- the lane FIRES at birth, its payload is the exact-zero
       identity) and the -0.0 flip count of the zero deposit (asserted
       zero).
    """
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import _k17_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    batch_rows = next(iter(stream.batches("train")))
    x, y, _valid = next(grouped(batch_rows, microbatch))

    per_layer = []
    flips_total = lanes_total = 0
    fire_sum = fire_count = 0
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
                payload, write_keys, query_keys, _scalar = \
                    _project_keyed(mixer, xin)
                payload, write_keys = payload.float(), write_keys.float()
                query_keys = query_keys.float()
                # the production birth expressions under autocast
                lam_vec, lam_inc = mixer.channel_decay(xin)
                lam_vec32 = lam_vec.float()
                gates = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.b_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gates32 = gates.float()
                # the dense channel's OWN birth expression (born at exact
                # zero, per-element +-0.0 with the payload's sign)
                dense = mixer.dense_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.d_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim) \
                    * payload
                dense32 = dense.float()
                gate_checks.append(dict(
                    lam_vec_bit_equal_registered_scalar=bool(torch.equal(
                        lam_vec, lam_inc.unsqueeze(-1).expand_as(lam_vec)
                        .to(lam_vec.dtype))),
                    g_vec_born_at_exact_1=bool(
                        torch.equal(gates, torch.ones_like(gates))),
                    dense_born_at_exact_zero=bool(
                        torch.equal(dense32.abs().max(),
                                   torch.zeros((), device=dense32.device)))))
                state = torch.zeros(batch, mixer.heads, mixer.rank,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out_k17 = _k17_forward_launch(
                    payload, dense32, write_keys, query_keys, lam_vec32,
                    gates32, state, live, keys, pol32, mixer.rank,
                    mixer.segment_tokens, True, True, collect=True, warps=8,
                    stages=3)
                state2 = torch.zeros(batch, mixer.heads, mixer.rank,
                                     mixer.value_dim, dtype=torch.float32,
                                     device=xin.device)
                live2 = torch.zeros(batch, mixer.heads, mixer.rank,
                                     dtype=torch.bool, device=xin.device)
                keys2 = torch.zeros(batch, mixer.heads, mixer.rank,
                                    mixer.key_dim, dtype=torch.float32,
                                    device=xin.device)
                out_k15 = _k15_forward_launch(
                    payload, write_keys, query_keys, lam_vec32, gates32,
                    state2, live2, keys2, pol32, mixer.rank,
                    mixer.segment_tokens, True, collect=True, warps=8,
                    stages=3)
            flips = lanes = 0
            for name in ("slot", "erase", "op", "live_after"):
                a, b = out_k15[name], out_k17[name][..., :2]
                flips += int((a != b).sum())
                lanes += a.numel()
            for name in ("payload", "live_pre", "reads", "state", "live",
                         "weights", "keys"):
                a = out_k15[name]
                b = out_k17[name] if name != "payload" \
                    else out_k17[name][..., :2, :]
                flips += int((a != b).sum())
                lanes += a.numel()
            fired = out_k17["op"][..., 2] != 0
            nonlocal fire_sum, fire_count
            fire_sum += int(fired.sum())
            fire_count += int(fired.numel())
            per_layer.append(dict(
                layer=layer_index, flips=flips, lanes=lanes,
                dense_fire_rate=float(fired.float().mean()),
                mean_live_rows=float(
                    out_k17["live_pre"].float().sum(-1).mean())))
            nonlocal flips_total, lanes_total
            flips_total += flips
            lanes_total += lanes
        return hook

    handles = [block.mixer.register_forward_hook(make_hook(i, block.mixer))
               for i, block in enumerate(model.blocks)]
    try:
        with torch.no_grad():
            model.features(x)
    finally:
        for handle in handles:
            handle.remove()

    receipt = dict(
        note="the write-form machine's own birth receipts on the FIRST "
             "train batch: the dense channel is born at EXACT ZERO (the "
             "committed zero-born gate conventions), so the machine is "
             "BORN AS THE COMPOSED MACHINE bit for bit -- and hence born "
             "as the K13 additive machine (the K15 receipt's own chain); "
             "the step-1 loss IS asserted bit-identical to the K13 "
             "additive arm's own same-seed row (the nearest kin)",
        born_as_the_composed_machine=dict(
            flips=flips_total, lanes=lanes_total,
            bit_identical=bool(flips_total == 0),
            note="every SHARED channel of the dense forward launch vs the "
                 "K15 composed kernel's own committed launch at identical "
                 "inputs, every layer (lanes 0/1 of the 3-lane schedule); "
                 "the -0.0 IEEE corner of the zero deposit is the only "
                 "legal escape and its count is asserted zero"),
        dense_channel_born_at_exact_zero=dict(
            all_checks=bool(gate_checks)
            and all(c["lam_vec_bit_equal_registered_scalar"]
                    and c["g_vec_born_at_exact_1"]
                    and c["dense_born_at_exact_zero"] for c in gate_checks),
            n_layers=len(gate_checks)),
        dense_lane_birth_telemetry=dict(
            fire_rate=float(fire_sum / max(1, fire_count)),
            note="the dense lane FIRES at birth on the live guards alone "
                 "(the target must be a live best-match row) with an "
                 "exact-zero payload -- the IEEE identity; the gradient "
                 "reaches the zero-born scalar at step 0 (P5)"),
        per_layer=per_layer[:1] + per_layer[-1:])   # the first and last layer
    return receipt


# --------------------------------------------- the mechanism instruments


def _k17_walk_stats(out, rank, mixer, payload=None, decay_vec=None):
    """The K15 |w|-mass instrument (unchanged) + the dense lane's own rows:
    op mix, occupancy, effective rows e^H of the |w|-mass distribution on
    live rows, the top-row |w|-mass share (the collapse detector), the
    identity-rewrite share through the lane-generic per-channel replay,
    and the C5 write-channel telemetry (the dense deposit's mass vs the
    event writes')."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import (
        CORRECT, MERGE, NONE, op_histogram)
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_channel_gates \
        import LANES as LANES_2  # noqa: F401  (the composed machine's 2)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import _identity_share_channel
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import LANES as LANES_3
    from experiments.expressivity_tasks.x45_matrix_emender.algebra import \
        MatrixSchedule
    schedule = MatrixSchedule(slot=out["slot"], erase=out["erase"],
                              payload=out["payload"], op=out["op"],
                              live_after=out["live_after"])
    assert out["slot"].shape[-1] == LANES_3
    lane0 = torch.bincount(schedule.op[..., 0].reshape(-1).to("cpu"),
                           minlength=6)
    occupancy = out["live_pre"].float().sum(-1)      # (b, T, h)
    w = out["weights"][..., :rank].float().abs()      # the |w|-mass
    mass = w.sum(-1)
    has = mass > 0
    probs = torch.where(has.unsqueeze(-1),
                        w / mass.clamp_min(1e-30).unsqueeze(-1),
                        torch.zeros_like(w))
    contrib = torch.where(probs > 0, probs * probs.log(), torch.zeros_like(probs))
    entropy = -contrib.sum(-1)
    top_share = torch.where(has, probs.max(-1).values, torch.zeros_like(mass))
    # the C5 write-channel telemetry: the dense deposit vs the event writes
    dense_payload = out["payload"][..., 2, :].float()       # (b, T, h, V)
    event_payload = out["payload"][..., 0, :].float()
    fired = out["op"][..., 2] != NONE
    carrying = (out["op"][..., 0] != NONE) & (out["op"][..., 0] != MERGE)
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
        rank=rank,
        dense_fire_count=int(fired.sum()),
        dense_fire_rate=float(fired.float().mean()),
        dense_mass_sum=float(dense_payload.abs().sum()),
        dense_mass_mean=float(dense_payload.abs().mean()),
        event_mass_mean=float(event_payload.abs().mean()),
        event_mass_mean_on_carrying_ops=float(event_payload[carrying].mean())
        if bool(carrying.any()) else 0.0,
        dense_over_event_mass_ratio=float(
            dense_payload.abs().mean()
            / max(event_payload.abs().mean(), 1e-30)))
    if payload is not None and decay_vec is not None:
        share, n_correct = _identity_share_channel(
            schedule, payload, decay_vec, rank)
        stats["identity_rewrite_share"] = share
        stats["identity_n_correct"] = n_correct
    return stats, schedule


def k17_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED dense arm's mechanism receipts: the K15 instrument (the
    |w|-mass effective rows + the collapse detector, occupancy, op mix, the
    identity-rewrite share through the lane-generic replay) UNDER the K11B
    gate telemetry (all three p3 channels' born-identity drift) AND the
    dense channel's own telemetry (the C5 instrument: the deposit's mass vs
    the event writes' mass, the gate's distance from the zero birth, the
    per-head scalar drift, the fire rate), on the k17 collect launch."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _probe_windows, _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import _k17_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    rows = _probe_windows(stream, windows)
    model.eval()
    per_layer = []
    gate_rows = []
    dense_rows = []

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
                gates = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.b_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gw_dist = (1.0 - gates.float()).abs()
                r_vec = 1.0 + mixer.read_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.g_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gr_dist = (r_vec.float() - 1.0).abs()
                lam_vec32 = lam_vec.float()
                gates32 = gates.float()
                # the dense channel's TRAINED expression (the C5
                # instrument's own subject)
                dense = mixer.dense_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.d_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim) \
                    * payload
                dense32 = dense.float()
                state = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank_max,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out = _k17_forward_launch(
                    payload, dense32, write_keys, query_keys, lam_vec32,
                    gates32, state, live, keys, pol32, mixer.rank,
                    mixer.segment_tokens, True, True, collect=True, warps=8,
                    stages=3)
            with torch.no_grad():
                stats, _schedule = _k17_walk_stats(
                    out, mixer.rank, mixer, payload=payload,
                    decay_vec=lam_vec32)
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
                dense_rows.append(dict(
                    layer=layer_index,
                    mean_abs_deposit=float(dense32.abs().mean()),
                    max_abs_deposit=float(dense32.abs().max()),
                    deposit_over_payload_scale=float(
                        dense32.abs().mean()
                        / max(payload.float().abs().mean(), 1e-30)),
                    fire_rate=stats["dense_fire_rate"],
                    dense_over_event_mass_ratio=stats[
                        "dense_over_event_mass_ratio"],
                    scalar_drift=[float(v) for v in
                                  mixer.dense_scalar.detach().float().abs()]))
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
                            regime="eval, bf16 autocast, the k17 dense "
                                   "persistent walk the arm trains on "
                                   "(collect=True)",
                            probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the born-identity "
             "drift for ALL THREE p3 gate channels (the K15 instrument, "
             "imported semantics, re-run on the dense machine's launch)",
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
    summary["dense_write_telemetry"] = dict(
        note="the C5 instrument, per layer over the probe windows: how "
             "much mass the DENSE channel carries vs the EVENT writes "
             "(the recorded schedule's own masses), the trained deposit's "
             "scale relative to the payload, the fire rate (the "
             "live-guard coverage), and the per-head zero-born scalar's "
             "drift -- the receipt that says whether the dense channel "
             "CARRIED anything at all (the K11 rows' own reference: "
             "deposit 0.0015-0.024 of payload scale)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in dense_rows},
        mean_abs_deposit=float(sum(r["mean_abs_deposit"] for r in dense_rows)
                               / max(1, len(dense_rows))),
        mean_deposit_over_payload_scale=float(sum(
            r["deposit_over_payload_scale"] for r in dense_rows)
            / max(1, len(dense_rows))),
        mean_fire_rate=float(sum(r["fire_rate"] for r in dense_rows)
                             / max(1, len(dense_rows))),
        mean_dense_over_event_mass_ratio=float(sum(
            r["dense_over_event_mass_ratio"] for r in dense_rows)
            / max(1, len(dense_rows))),
        max_scalar_drift=float(max(
            max(r["scalar_drift"]) for r in dense_rows)),
        n_layers=len(dense_rows))
    return summary


# ------------------------------------------------------------ the clause read


def evaluate_clauses(receipt: dict) -> dict:
    """The registration's pre-stated clauses, evaluated per seed verbatim
    against the composed machine's OWN same-seed banked rows (the baseline;
    no directional prediction was pre-stated)."""
    seed = receipt["seed"]
    post = receipt.get("final_eval_nll_post_cooldown")
    four_x = composed_4x_banked()
    baseline = four_x["post_cooldown"][seed]
    out = {}
    if post is not None:
        improvement = baseline - post
        out["same_seed_baseline_post_cooldown"] = baseline
        out["improvement_vs_same_seed_baseline"] = improvement
        out["win_bar"] = WIN_BAR
        if receipt["arm"] in ("dense", "combined"):
            if improvement >= WIN_BAR:
                out["clause_at_this_seed"] = \
                    "W-1_at_this_seed (COMPOSE: beats the composed baseline " \
                    "by >= 0.01)"
            elif improvement >= -WIN_BAR:
                out["clause_at_this_seed"] = \
                    "W-2_at_this_seed (PARITY-grade: inside the seed band)"
            else:
                out["clause_at_this_seed"] = \
                    "W-3_at_this_seed (INTERFERENCE/REFUTED: worse than " \
                    "the baseline beyond the band -- the G2-under-softmax " \
                    "result replicates under the additive read)"
        elif receipt["arm"] == "dense_frozen":
            out["clause_at_this_seed"] = \
                "the FALSIFIER control (its own bar, see the rules receipt)"
        else:
            if improvement >= WIN_BAR:
                out["clause_at_this_seed"] = \
                    "G-1_at_this_seed (the gain owns part of the residual)"
            else:
                out["clause_at_this_seed"] = \
                    "G-1_null_at_this_seed (inside the seed band: the " \
                    "inherited point stands, receipted)"
        out["vs_gdn2_post_cooldown"] = post - GDN2_POST_COOLDOWN
        out["gdn2_contact_flag"] = bool(post <= GDN2_POST_COOLDOWN)
        out["fraction_of_corrected_gap_closed"] = max(
            0.0, BANKED_CORRECTED_ANCHOR - post) / (
            BANKED_CORRECTED_ANCHOR - GDN2_POST_COOLDOWN)
    out["pre_state"] = ("NO directional prediction was pre-stated for "
                        "either lever; every reading keys on the measured "
                        "rows against the two bands, and any "
                        "quality-conclusion inversion or GDN2 contact is "
                        "escalated, not adjudicated here")
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
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import (composed_parameter_delta, convert_model_to_composition)
    from experiments.expressivity_tasks.x45_matrix_emender.k17_dense_write \
        import (DenseWriteComposedMixer, convert_model_to_dense_write,
                dense_parameter_delta)

    L.check_python()
    if args.arm not in ARMS:
        raise ValueError(f"unregistered arm: {args.arm!r} (expected one of "
                         f"{sorted(ARMS)})")
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed!r} (the arms run "
                         f"seed {PRIMARY_SEED} first; the >= 0.01 win-bar "
                         f"rule owes {SECOND_SEED})")
    arm = ARMS[args.arm]
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    seed = args.seed
    frozen = bool(arm.get("frozen", False))

    # ---- the machine: the K15 composed machine VERBATIM (the banked
    # winner build + the K13 additive read + the K11B p3 gate stack at the
    # mimetic-neutral birth); the dense arms add the K17 dense channel at
    # the zero birth; the gain arms swap ONLY the policy constant.
    build_config = dict(TUNED_WINNER)
    gain = arm["read_scale"]
    if args.arm == "combined":
        if args.gain not in GAIN_ARMS:
            raise ValueError(f"the combined arm needs a bracket gain "
                             f"(one of {sorted(GAIN_ARMS)}); got {args.gain!r}")
        gain = GAIN_ARMS[args.gain]
    if arm["kind"] == "dense":
        machine = ("the composed machine + the K17 dense gated write"
                   if gain is None else
                   f"the composed machine + the K17 dense gated write + "
                   f"read_scale = {gain} (the combined arm)")
    else:
        machine = (f"the composed machine with read_scale = {gain} "
                   f"(a config arm; the inherited point is "
                   f"{INHERITED_READ_SCALE!r}, cited from the K15 rows)")
        build_config["read_scale"] = gain
    model, optimizer_config, summary = _build_k10(build_config, seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_composition(model)
    if arm["kind"] == "dense":
        model = convert_model_to_dense_write(model)
        depth = len(model.blocks)
        mixer0 = model.blocks[0].mixer
        dense_added = dense_parameter_delta(
            depth, mixer0.width, mixer0.heads, mixer0.value_dim) \
            - composed_parameter_delta(
                depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    else:
        dense_added = 0
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = composed_parameter_delta(
        depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta + dense_added:
        raise ValueError(f"the parameter accounting must hold: "
                         f"{parameters_before} + {delta} + {dense_added} != "
                         f"{parameters_after}")
    if arm["kind"] == "dense":
        if args.arm == "combined":
            if abs(mixer0.policy.read_scale - gain) > 1e-12:
                raise ValueError("the combined arm's policy read_scale must "
                                 "be the gain cell's own")
        if dense_added != 3_538_992 or parameters_after != 162_177_840:
            raise ValueError(f"the dense machine's parameters must be the "
                             f"registration's (158,638,848 + 3,538,992); "
                             f"got added {dense_added}, total "
                             f"{parameters_after}")
        if not isinstance(mixer0, DenseWriteComposedMixer):
            raise ValueError("the dense arm's mixers must be the K17 ones")
    else:
        if parameters_after != 158_638_848:
            raise ValueError(f"the gain arms' machine must be the composed "
                             f"machine's own 158,638,848; got "
                             f"{parameters_after}")
        if abs(mixer0.policy.read_scale - gain) > 1e-12:
            raise ValueError("the gain arm's policy read_scale must be the "
                            "bracket cell's own")
    # the banked knobs, asserted (a config drift must fail loudly)
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
    # the frozen control: the dense channel's parameters frozen at birth
    if frozen:
        for block in model.blocks:
            block.mixer.d_proj.weight.requires_grad_(False)
            block.mixer.dense_scalar.requires_grad_(False)
        assert all(not p.requires_grad for b in model.blocks
                   for p in (b.mixer.d_proj.weight, b.mixer.dense_scalar))

    # ---- the optimizer: the inherited 4x hybrid-Muon point, the banked
    # routing with the p3 gate tensors AND the dense channel's tensors on
    # the AdamW side
    peak_lr = INHERITED_LR_MULTIPLIER * TUNED_ADAMW_LR
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_k17(model, frozen)
    routing_receipt = _routing_cross_check_k17(model, seed, frozen)
    optimizer = HybridMuonX45(
        muon_params, adamw_params, lr=peak_lr, betas=(0.9, BETA2),
        weight_decay=TUNED_WINNER["weight_decay"])
    optimizer_block = dict(
        optimizer="bakeoff_optimizers.HybridMuonX45 (imported verbatim)",
        peak_lr=peak_lr,
        lr_convention=(f"the inherited 4x point ({INHERITED_LR_MULTIPLIER}x "
                       f"the tuned AdamW base {TUNED_ADAMW_LR!r}) -- the "
                       "K16 bracket's own receipt: already the composed "
                       "machine's optimum at both seeds"),
        betas=(0.9, BETA2), weight_decay=TUNED_WINNER["weight_decay"],
        warmup_steps=TUNED_WINNER["warmup_steps"],
        microbatch=MICROBATCH, grad_clip=GRAD_CLIP)

    four_x = composed_4x_banked()
    anchors = dict(
        banked=BANKED, k13_additive=K13_ADDITIVE, bands=BANDS,
        composed_4x_banked=four_x,
        inherited_read_scale=INHERITED_READ_SCALE,
        banked_mechanism=_banked_mechanism_anchors())
    run_config = dict(
        probe="k17_probe_write_form", arm=args.arm, seed=seed,
        arm_note=arm["note"], machine=machine, frozen=frozen,
        read_scale=gain if arm["kind"] == "gain" else "inherited "
                  f"{INHERITED_READ_SCALE!r}",
        combined_gain=(args.gain if args.arm == "combined" else None),
        corrected_scoring=CORRECTED_SCORING, rules=RULES,
        exposure_targets=int(args.bytes),
        model_build="the K15 composition VERBATIM: velocity_ladder._build_k10 "
                    "(the tuned winner's config, the banked bake-off build) "
                    "+ convert_model_to_composition (the K15 conversion; "
                    "the gates born at the registered scalar equivalents) "
                    "+ convert_model_to_dense_write on the dense arms (the "
                    "K17 conversion; the dense channel born at EXACT zero "
                    "-- the machine is BORN AS THE COMPOSED MACHINE bit "
                    "for bit); the gain arms swap ONLY the policy's "
                    "read_scale constant",
        optimizer=optimizer_block,
        routing="banked_gates_adamw + the p3 gate tensors + the K17 dense "
                "channel's tensors on AdamW (the frozen control: the "
                "banked + p3 only)",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(before=parameters_before, added=delta,
                        dense_added=dense_added, total=parameters_after,
                        note="p3's own gate stack (the read channel "
                             "instantiated once) + the dense channel's "
                             "b_proj-form count on the dense arms"),
        checkpoint_steps=list(args.checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear cooldown "
                       "(the corrected-scoring extension, same stream); the "
                       "step-180 constant-LR eval is retained for banked "
                       "comparability"),
        step1_parity_rule=("the dense arms: the step-1 train loss is asserted "
                           "BIT-IDENTICAL to the K13 additive arm's own "
                           "same-seed row (== the composed machine's own "
                           "row: the machine at birth IS the composed "
                           "machine; the optimizer cannot touch the first "
                           "forward); the gain arms: NO parity exists or "
                           "is asserted (a different gain constant is a "
                           "different function at birth -- a config arm "
                           "like the LR cells, disclosed)"),
        both_bands=BANDS,
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                      k15_summary=str(RECEIPTS / "k15_composition_summary.json"),
                      k15_gates_receipt=str(RECEIPTS / "k15_gates.json"),
                      k16_summary=str(RECEIPTS / "k16_config_bracket_summary.json"),
                      k17_registration=str(
                          ROOT / "PREREGISTRATION_K17_WRITE_FORM.md")),
        summary=summary, identity=identity, versions=_versions(),
        git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns()))
    print(json.dumps(dict(arm=args.arm, card=card, seed=seed, frozen=frozen,
                          read_scale=run_config["read_scale"],
                          parameters=parameters_after)), flush=True)

    banked_step1 = _banked_step1_loss(seed)
    k13_step1 = _k13_additive_step1(seed)
    run_config["banked_step1_loss_reference"] = banked_step1
    run_config["k13_additive_step1_loss_reference"] = k13_step1

    # ---- the birth receipts, BEFORE any training (the dense arms only)
    birth = None
    if arm["kind"] == "dense":
        birth = dense_birth_receipts(model, stream, seed,
                                     optimizer_config["microbatch"])
        if not birth["born_as_the_composed_machine"]["bit_identical"]:
            raise ValueError(
                "the BORN-AS-THE-COMPOSED-MACHINE receipt failed at probe "
                "scale -- escalate, not excuse (bit-equality conflict with "
                "the committed composed machine)")
        run_config["birth_receipts"] = birth
        print("  born-as-the-composed-machine: "
              f"{birth['born_as_the_composed_machine']['flips']} flips in "
              f"{birth['born_as_the_composed_machine']['lanes']} lanes; "
              "dense fire rate "
              f"{birth['dense_lane_birth_telemetry']['fire_rate']:.3f}",
              flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, peak_lr, optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"{args.arm}_seed{seed}", out / "training.jsonl",
        tuple(args.checkpoint_steps), cooldown_steps, EMA_DECAY)
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # ---- the step-1 parity receipt (the dense arms: the nearest-kin bar;
    # the gain arms: disclosed, no parity)
    step1 = run["rows"][0]["loss"]
    if arm["kind"] == "dense":
        print(f"  step-1 train loss {step1!r} (the K13 additive arm's own "
              f"row's is {k13_step1!r} -- asserted bit-identical; the "
              f"inherited composition row's is "
              f"{four_x['step1_train_loss'][seed]!r})", flush=True)
        if not args.smoke and step1 != k13_step1:
            raise ValueError(
                f"step-1 train loss {step1!r} != the K13 additive arm's own "
                f"same-seed row's {k13_step1!r} -- the arm's machine is not "
                f"the composed machine byte for byte (escalate, not excuse)")
        if not args.smoke and step1 != four_x["step1_train_loss"][seed]:
            raise ValueError(
                f"step-1 train loss {step1!r} != the K15 composition arm's "
                f"own same-seed row's {four_x['step1_train_loss'][seed]!r} "
                f"-- the build order drifted (escalate, not excuse)")
    else:
        print(f"  step-1 train loss {step1!r} (NO parity asserted: a gain "
              f"cell is a different function at birth; the composed "
              f"machine's own row is "
              f"{four_x['step1_train_loss'][seed]!r})", flush=True)

    # ---- the falsifier's honest divergence receipt: the per-step loss
    # divergence vs the composed machine's own banked training log
    falsifier_divergence = None
    if frozen:
        k15_log = K15_RUNS / f"composition_seed{seed}" / "training.jsonl"
        if k15_log.exists() and not args.smoke:
            banked_rows = [json.loads(line) for line in
                           k15_log.read_text().splitlines()]
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
                    bit_identical_steps=sum(
                        1 for d in divs if d["bit_identical"]),
                    first_diverging_step=(first_div or {}).get("step"),
                    max_abs_loss_delta=max(
                        abs(d["loss_delta"]) for d in divs),
                    note="the K11B 2.4 honest limitation, receipted: "
                         "full-trajectory bit-identity is NOT achievable "
                         "by construction (the dense lane's backward emits "
                         "exact-zero payload-gradient contributions whose "
                         "signs are reduction-order artifacts; the "
                         "difference compounds chaotically through the "
                         "optimizer steps even with the channel frozen)")

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
        instrument = (k17_probe_diagnostics if arm["kind"] == "dense"
                      else k15_probe_diagnostics)
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            diagnostics_180 = instrument(model, stream,
                                          windows=args.probe_windows)
            sweep_180 = k10_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = instrument(model, stream,
                                           windows=args.probe_windows)
            sweep_final = k10_read_mask_sweep(model, stream)

    constant_final = next((r["eval_nll"] for r in scores
                           if r["kind"] == "constant"
                           and r["step"] == run["constant_steps"]), None)
    post_cooldown = next((r["eval_nll"] for r in scores
                          if r["kind"] == "post_cooldown"), None)
    ema_final = next((r["eval_nll"] for r in scores if r["kind"] == "ema_180"),
                     None)
    receipt = dict(
        probe="k17_probe_write_form", arm=args.arm, seed=seed,
        arm_note=arm["note"], machine=machine, frozen=frozen,
        read_scale=run_config["read_scale"],
        combined_gain=(args.gain if args.arm == "combined" else None),
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
        parameters_added=delta, parameters_dense_added=dense_added,
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=step1,
        step1_receipt=dict(
            arm_step1=step1, k13_additive_same_seed_step1=k13_step1,
            parity_rule=run_config["step1_parity_rule"],
            k13_parity=bool(args.smoke or arm["kind"] != "dense"
                            or step1 == k13_step1),
            composed_4x_same_seed_step1=four_x["step1_train_loss"][seed],
            composed_4x_parity=bool(
                args.smoke or arm["kind"] != "dense"
                or step1 == four_x["step1_train_loss"][seed]),
            banked_same_seed_step1=banked_step1),
        birth_receipts=birth,
        falsifier_divergence=falsifier_divergence,
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
    dense = diag.get("dense_write_telemetry") or {}
    return dict(
        arm=r["arm"], seed=r["seed"], machine=r["machine"], frozen=r["frozen"],
        read_scale=r["read_scale"],
        combined_gain=r.get("combined_gain"),
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
        dense_telemetry=dict(
            mean_abs_deposit=dense.get("mean_abs_deposit"),
            deposit_over_payload_scale=dense.get(
                "mean_deposit_over_payload_scale"),
            fire_rate=dense.get("mean_fire_rate"),
            dense_over_event_mass_ratio=dense.get(
                "mean_dense_over_event_mass_ratio"),
            max_scalar_drift=dense.get("max_scalar_drift")) if dense else None,
        falsifier_divergence=r.get("falsifier_divergence"),
        muon_update_rms=(r.get("muon_update_rms") or {}).get("muon_update_rms")
        if isinstance(r.get("muon_update_rms"), dict) else None)


def assemble(args) -> dict:
    """The arms' table (the dense lever, the falsifier control, the gain
    bracket, the conditional combined arm), the pre-stated clauses applied
    exactly as stated across seeds, the second-seed/combined-arm rules, and
    the re-quoted gap vs GDN2 with the honest band caveats.  The final
    gap-quote phrasing is the supervisor's to adjudicate (the K13
    clause-conflict precedent): this assemble states the measured rows and
    the flags, not the final wording."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    four_x = composed_4x_banked()
    cells: dict[str, dict] = {
        "composed_inherited": dict(
            cited=True, read_scale=INHERITED_READ_SCALE,
            note="the K15 composition arm's own landed rows at BOTH seeds "
                 "(the inherited 4x point AND the inherited read gain "
                 f"{INHERITED_READ_SCALE!r}) -- cited, never re-run",
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
        cell = dict(kind=ARMS[label]["kind"], cited=False,
                    note=ARMS[label]["note"], seeds={})
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
                parameters=row["parameters"],
                effective_rows=row["effective_rows"],
                top_row_share=row["top_row_share"],
                occupancy=row["occupancy"],
                identity_share=row["identity_share"],
                gate_drifts=row["gate_drifts"],
                dense_telemetry=row["dense_telemetry"],
                falsifier_divergence=row["falsifier_divergence"],
                clauses=(json.loads(
                    (RUNS / f"{label}_seed{seed}" / "result.json").read_text()
                )["clauses"] if (RUNS / f"{label}_seed{seed}" /
                                 "result.json").exists() else None))
            if row["post_cooldown"] is not None \
                    and row["post_cooldown"] <= GDN2_POST_COOLDOWN:
                escalation_flags.append(
                    f"the {label} arm at seed {seed} landed at/past GDN2's "
                    f"{GDN2_POST_COOLDOWN} (post-cooldown "
                    f"{row['post_cooldown']}) -- the band-discipline clause: "
                    f"BOTH scorings and BOTH seeds are reported before any "
                    f"claim, and the result is escalated")
        if cell["seeds"]:
            cells[label] = cell
    if not any(not c["cited"] for c in cells.values()):
        raise ValueError("no completed arm receipts to assemble")

    # the pre-stated rules, applied exactly as stated
    def seed_improvement(label, seed):
        row = cells.get(label, {}).get("seeds", {}).get(seed)
        if row is None or row["post_cooldown"] is None:
            return None
        return four_x["post_cooldown"][seed] - row["post_cooldown"]

    adjudication = {}
    for label in ("dense", "dense_frozen", "gain1x", "gain2x", "gain4x",
                  "combined"):
        if label not in cells:
            continue
        imp1 = seed_improvement(label, PRIMARY_SEED)
        imp2 = seed_improvement(label, SECOND_SEED)
        entry = dict(seed3401_improvement=imp1,
                     seed3402_improvement=imp2)
        if label == "dense" or label == "combined":
            if imp1 is not None and imp2 is not None:
                if imp1 >= WIN_BAR and imp2 >= WIN_BAR:
                    entry["verdict"] = "W-1 (COMPOSES at both seeds)"
                elif imp1 >= WIN_BAR or imp2 >= WIN_BAR:
                    entry["verdict"] = ("W-2 (PARITY-grade: one seed inside "
                                        "the band -- reported as measured, "
                                        "not banked as composing; the "
                                        "supervisor adjudicates the mixed-"
                                        "seed rule, the K15 precedent)")
                else:
                    entry["verdict"] = ("W-3 (does not compose: the "
                                        "G2-under-softmax result replicates "
                                        "under the additive read)")
            elif imp1 is not None:
                entry["verdict"] = ("seed-3401 only: "
                                     + ("W-1-shape signal at 3401 -> seed "
                                        "3402 is OWED (the standing win-bar "
                                        "rule)" if imp1 >= WIN_BAR else
                                        "no >= 0.01 signal at 3401"))
        elif label == "dense_frozen":
            step1_ok = cells[label]["seeds"][PRIMARY_SEED]["step1_parity"]
            parity = (imp1 is not None
                      and abs(imp1) <= WINNER_SEED_DELTA)
            entry["verdict"] = (
                "the falsifier control PASSES" if (step1_ok and parity)
                else "the falsifier control FAILS -- escalated")
            if not (step1_ok and parity):
                escalation_flags.append(
                    f"the frozen-write falsifier control did not hold "
                    f"(step-1 parity {step1_ok}, same-seed delta {imp1}) -- "
                    f"escalated, not excused")
        else:
            if imp1 is not None:
                entry["verdict"] = ("G-1: the gain cell owns >= 0.01 of the "
                                    "residual at 3401 -> seed 3402 is OWED"
                                    if imp1 >= WIN_BAR else
                                    "G-1 null at 3401: inside the seed "
                                    "band, the inherited point stands, "
                                    "receipted")
        adjudication[label] = entry

    # the second-seed and combined-arm duties
    second_seeds_owed = []
    for label, entry in adjudication.items():
        if label == "dense_frozen":
            continue
        imp1 = entry.get("seed3401_improvement")
        has3402 = SECOND_SEED in cells.get(label, {}).get("seeds", {})
        if imp1 is not None and imp1 >= WIN_BAR and not has3402:
            second_seeds_owed.append(label)
    lever_signal = any(
        (cells.get(label, {}).get("seeds", {}).get(PRIMARY_SEED) or {})
        .get("post_cooldown") is not None
        and (seed_improvement(label, PRIMARY_SEED) or 0) >= WIN_BAR
        for label in ("dense", "gain1x", "gain2x", "gain4x"))
    combined_rule = dict(
        fires=bool(lever_signal),
        rule=RULES["combined"]["text"],
        note=("the combined arm is owed (a lever showed >= 0.01 at 3401)"
              if lever_signal else
              "the combined arm does NOT fire (no lever showed >= 0.01 at "
              "seed 3401; the pre-stated rule)"),
        combined_run=("combined" in cells))

    # the falsifier's honest receipt, quoted
    falsifier = None
    if "dense_frozen" in cells:
        falsifier = dict(
            step1_parity=cells["dense_frozen"]["seeds"][PRIMARY_SEED][
                "step1_parity"],
            same_seed_delta=seed_improvement("dense_frozen", PRIMARY_SEED),
            divergence=cells["dense_frozen"]["seeds"][PRIMARY_SEED][
                "falsifier_divergence"],
            note="the frozen-write falsifier control: everything-except-"
                 "dense-learning measured at the band's scale (the K11B "
                 "2.4 honest limitation receipted in the arm's own log)")

    # the best measured row across the lane (ours-best under the levers)
    ours_best = None
    for label, cell in cells.items():
        for seed, row in cell["seeds"].items():
            if row["post_cooldown"] is None:
                continue
            if ours_best is None \
                    or row["post_cooldown"] < ours_best["post_cooldown"]:
                ours_best = dict(arm=label, seed=seed,
                                 post_cooldown=row["post_cooldown"],
                                 constant_lr=row.get("constant_lr"),
                                 ema_180=row.get("ema_180"))
    gap_quote = None
    if ours_best is not None:
        post = ours_best["post_cooldown"]
        gap_quote = dict(
            ours_best_post_cooldown=post, arm=ours_best["arm"],
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
                   "controlled comparison; the final gap-quote phrasing "
                   "is the supervisor's to adjudicate (the K13 "
                   "clause-conflict precedent)")

    summary = dict(
        probe="k17_probe_write_form assemble",
        corrected_scoring=CORRECTED_SCORING,
        score_form="the write-form levers on the COMMITTED composed machine: "
                   "the dense gated write (the b_proj-delta form, born as "
                   "the composed machine) and the read-gain bracket "
                   "{1.0, 2.0, 4.0} (config arms; the inherited 2.787 cell "
                   "cited from the K15 rows)",
        arms=cells, adjudication=adjudication,
        second_seeds_owed=second_seeds_owed,
        combined_rule=combined_rule,
        falsifier_control=falsifier,
        escalation_flags=escalation_flags,
        ours_best=ours_best,
        gap_quote_vs_gdn2=gap_quote,
        notes=dict(
            scoring="PRIMARY: post-cooldown (24-step linear); SECONDARY: "
                    "EMA-0.99 at step 180 (ranking-only); constant-180 "
                    "retained for banked comparability",
            bands="every margin is read against BOTH the "
                  f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance band and "
                  f"the winner's {WINNER_SEED_DELTA:.4f} seed delta",
            pre_state="NO directional prediction was pre-stated for either "
                      "lever; the W-1/W-2/W-3/G-1 clauses, the combined-arm "
                      "rule and the standing 0.01 win-bar second-seed rule "
                      "are applied exactly as stated",
            escalation="the final gap-quote phrasing, the falsifier's "
                       "verdict on a failure, any quality-conclusion "
                       "inversion and any GDN2 contact are escalated to the "
                       "supervisor (the K13 clause-conflict precedent "
                       "governs adjudication)"))
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(cells={k: {s: v["post_cooldown"] for s, v in
                                     c["seeds"].items()}
                                 for k, c in cells.items()},
                          adjudication=adjudication,
                          second_seeds_owed=second_seeds_owed,
                          combined_fires=combined_rule["fires"],
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
    a.add_argument("--gain", default=None, choices=sorted(GAIN_ARMS),
                   help="the combined arm's gain cell (the best gain cell "
                        "per the pre-stated rule)")
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
                                        / "k17_write_form_summary.json"))
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
