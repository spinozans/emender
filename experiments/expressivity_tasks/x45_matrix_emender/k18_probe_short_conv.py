"""K18 -- THE SHORT-CONV TRANSPLANT: the probe (the arms, the mechanism
instruments, the assemble) -- GDN2's registered short depthwise
convolution (kernel 4) on the COMMITTED composed machine's payload path.

THE QUESTION THIS LANE EXISTS FOR (the missing-component candidate with
published backing): GDN2's registered block carries a short depthwise
convolution (conv_size=4, default ON) that our comparison machine has
NEVER had -- no convolution exists anywhere in the K10/K15 mixer lineage
-- and every transplant this session ran machine-without-conv vs
machine-with-conv.  The delta-rule family's own ablations say the short
conv is load-bearing locally (published with/without deltas span
0.05-13.5 ppl across the lineage -- heterogeneous; V-only placement
helped more than Q- or K-only in two independent studies).  The composed
machine (the program's best absolute rows 1.3811104467936925 /
1.383232252938407 post-cooldown) remains +0.0289/+0.0311 vs GDN2's
1.3521734476089478: is the conv the missing component that owns part of
the remaining ~0.03?

THE ESCALATED, APPROVED DESIGN (the supervisor's ruling of 2026-11-26):
the PAYLOAD-side conv only -- GDN2's v_conv1d's exact functional analog,
POST-projection (``payload = v_conv(value(x))``), depthwise over
heads*value_dim channels, kernel 4, bias-free, the fused SiLU NOT
transplanted (ours never silus the payload; GDN2 silus q/k/v even with
the conv OFF, so the conv's own ablated delta in GDN2's family is the
TEMPORAL FILTER and ours isolates the same object).  The q/k side already
has the locality GDN2's q/k convs buy (our key/query projections consume
a 4-token causal context window -- exactly the conv's receptive-field
width) while the payload path had NO temporal receptive field at all:
the payload conv COMPLETES the receptive-field parity.  Born at the
IDENTITY KERNEL ``[0, 0, 0, 1]`` (the STANDARD nn.Conv1d/fla tap layout --
the LAST tap is the current token), so the machine is BORN AS THE
COMPOSED MACHINE bit for bit (the K15 birth discipline, the 0-flip bar),
and the conv then LEARNS to deviate -- a VALUE-CHANNEL change: no conv
value feeds op selection (P2 trivially held).

THE ARMS (``PREREGISTRATION_SHORT_CONV.md``, committed BEFORE any probe
measurement; every arm at the EXACT probe protocol -- the
20,022,866-target matched-exposure landing at 180 constant-LR steps on
the registered flagship stream, the document-disjoint calibration eval,
the honest clock, the protocol's own grad-clip, the banked knobs asserted,
the inherited 4x hybrid-Muon point, the p3 gate tensors AND the live
conv's kernel on the AdamW side -- under the CORRECTED scoring: the
24-step linear cooldown PRIMARY, EMA-0.99 secondary, constant-180
retained), seed 3401 first:

* ``conv``          the composed machine + the SHORT CONV (born identity,
                    learns) -- the transplant arm.
* ``conv_frozen``   the FROZEN-CONV FALSIFIER CONTROL (mandatory, the
                    K11B R4 precedent): the conv frozen at the identity
                    birth -- the bar: step-1 bit-identity to the composed
                    machine's own row, a landing at parity with the
                    composed machine's same-seed banked row, and the
                    full-trajectory bit-identity receipt against the
                    composed machine's own banked training log.
* ``shift_fixed``   the FIXED NON-IDENTITY CONTROL (the published
                    mechanism decomposition's own recommendation): the
                    conv at the FIXED token-shift kernel
                    ``[0, 0, 0.5, 0.5]`` (the average of the current and
                    the previous token), frozen, NEVER learned -- the arm
                    that separates ANY LOCAL MIXING from a LEARNED
                    FILTER.  NO birth parity exists or is asserted (a
                    different function at birth -- disclosed, the gain
                    arms' own convention).

THE PRE-STATE: NO DIRECTIONAL PREDICTION.  The THREE-OUTCOME rule (the
registration's own, pre-stated BEFORE any result): (SC-1) the conv closes
>= 0.01 at seed 3401 vs the composed machine's own row -> local-features
owns a share of the residual, seed 3402 owed, the attribution chain
gains the LOCAL-FEATURES term and the uniformity story shrinks; (SC-2)
parity inside the band -> the conv is NOT the residual's owner on byte-LM
text at probe scale, the uniformity attribution stands with one more
receipt (a real possibility: the published effects are heterogeneous and
ours may be at the small end); (SC-3) the conv HURTS beyond the band ->
the payload path's current-token-only structure was load-bearing for the
slot semantics (write what you read, no temporal blur) -- receipted as a
machine-family difference, not a failure.  The shift control's clause
(SH-1) and the falsifier's own bar are receipted in every arm; the band
discipline and the escalation flags are the standing clauses; the final
gap-quote phrasing is the supervisor's to adjudicate (the K13
clause-conflict precedent).

PROVENANCE: the K15 ENGINE IS COMMITTED AND READ-ONLY -- the machine is
``velocity_ladder._build_k10`` + ``convert_model_to_composition``
(imported verbatim) + ``convert_model_to_short_conv`` (the K18
conversion, at the registered birth); no ``k10_*``/``k11*``/``k12_*``/
``k13_*``/``k15_*``/``k16_*``/``k17_*`` source, no committed
PREREGISTRATION, no ``lattice.py`` is edited.

Usage (see ``run_k18_short_conv_leased.sh`` for the detached leased
driver, cards 0/1):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \\
        arm --arm conv --seed 3401 --output k18_runs/conv_seed3401
    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \\
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
                                  _k15_walk_stats)
from experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
    import composed_4x_banked

ROOT = Path(__file__).resolve().parent
RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "k18_runs"
K15_RUNS = ROOT / "k15_runs"

# The inherited hybrid-Muon point (K16's own receipt: already the composed
# machine's bracket optimum at both seeds).
INHERITED_LR_MULTIPLIER = 4
TUNED_ADAMW_LR = TUNED_WINNER["lr"]

GDN2_POST_COOLDOWN = BANKED["gdn2"]["post_cooldown"]
BANKED_CORRECTED_ANCHOR = BANKED["post_cooldown"]["seed3402"]
BANDS = dict(endpoint_variance=ENDPOINT_VARIANCE_BAND,
             winner_seed_delta=WINNER_SEED_DELTA)
WIN_BAR = 0.01                     # the standing second-seed / signal bar

# The conv's registered accounting at the flagship depth: one depthwise
# kernel of heads*value_dim = 256 channels x 4 taps per mixer, 24 mixers.
CONV_PARAMETER_DELTA = 24 * 2 * 128 * 4      # 24,576
COMPOSED_PARAMETERS = 158_638_848
CONV_MACHINE_PARAMETERS = COMPOSED_PARAMETERS + CONV_PARAMETER_DELTA

ARMS = {
    "conv": dict(
        kind="conv", birth="identity", frozen=False,
        note="the composed machine + the SHORT DEPTHWISE CONVOLUTION "
             "(GDN2's registered conv_size=4, bias-free, the silu not "
             "transplanted) on the payload path at GDN2's own position "
             "(post-projection, pre-recurrence; the v_conv1d analog), born "
             "at the identity kernel [0,0,0,1] (the standard nn.Conv1d "
             "layout: the LAST tap is the current token) so the machine is "
             "BORN AS THE COMPOSED MACHINE bit for bit -- then the conv "
             "LEARNS to deviate (a value-channel change; no conv value "
             "feeds op selection)"),
    "conv_frozen": dict(
        kind="conv", birth="identity", frozen=True,
        note="the FROZEN-CONV FALSIFIER CONTROL (mandatory, the K11B R4 "
             "precedent): the conv's kernel frozen at the identity birth "
             "(requires_grad=False, excluded from the optimizer; the "
             "routing then equals the composed machine's own split "
             "exactly) -- the bar: step-1 bit-identity to the composed "
             "machine's own row, a landing at parity with the composed "
             "machine's same-seed banked row, and the full-trajectory "
             "bit-identity receipt against the composed machine's own "
             "banked training log (the K17 falsifier's perfect receipt is "
             "the expectation to beat; ULP-scale divergence is the honest "
             "escape, the K11B 2.4 precedent)"),
    "shift_fixed": dict(
        kind="shift", birth="shift", frozen=True,
        note="the FIXED NON-IDENTITY CONTROL (the published mechanism "
             "decomposition's own recommendation): the conv at the FIXED "
             "token-shift kernel [0,0,0.5,0.5] (the average of the current "
             "and the previous token, in the standard tap layout), frozen, "
             "NEVER learned -- separates ANY LOCAL MIXING from a LEARNED "
             "FILTER; NO birth parity exists or is asserted (a different "
             "function at birth -- disclosed, the config arms' own "
             "convention)"),
}

# The pre-stated rules (the registration's own; receipted in every arm).
RULES = dict(
    SC1=dict(bar=WIN_BAR,
             text="the conv arm's post-cooldown row improves on the "
                  "composed machine's same-seed banked row by >= 0.01 at "
                  "seed 3401 -> LOCAL-FEATURES owns a share of the "
                  "residual: seed 3402 owed BEFORE any claim (the "
                  "standing win-bar rule); if replicated at both seeds, "
                  "the attribution chain gains the LOCAL-FEATURES term "
                  "and the missing-component story replaces part of the "
                  "uniformity story"),
    SC2=dict(text="improvement inside the 0.0119 seed band (and/or no "
                  "replication at the second seed) -> the conv is NOT the "
                  "residual's owner on byte-LM text at probe scale; the "
                  "uniformity attribution stands with one more receipt (a "
                  "real possibility, stated BEFORE the result: the "
                  "published with/without effects are heterogeneous, "
                  "0.05-13.5 ppl across the lineage, and ours may be at "
                  "the small end)"),
    SC3=dict(text="the conv arm lands WORSE than its same-seed baseline "
                  "beyond the band -> the payload path's current-token-"
                  "only structure was load-bearing for the slot semantics "
                  "(write what you read, no temporal blur) -- receipted "
                  "as a machine-family difference, not a failure"),
    SH1=dict(text="the fixed-shift control decomposes the conv's effect: "
                  "ANY LOCAL MIXING (the shift arm's row) vs a LEARNED "
                  "FILTER (the conv arm's row) -- no directional "
                  "prediction is pre-stated for the decomposition; both "
                  "rows are reported as measured against the composed "
                  "machine's own and against each other"),
    falsifier=dict(text="the frozen-conv control's bar: (a) the control's "
                        "step-1 train loss MUST be bit-identical to the "
                        "composed machine's own same-seed row (the "
                        "forward-at-birth plumbing receipt); (b) the "
                        "control's landing MUST be at parity with the "
                        "composed machine's same-seed banked row; (c) "
                        "the full-trajectory bit-identity against the "
                        "composed machine's own banked training log is "
                        "receipted (the K17 falsifier's perfect receipt "
                        "is the expectation to beat; ULP-scale divergence "
                        "is the honest escape, the K11B 2.4 precedent)"),
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
                              "escalates likewise"))


# --------------------------------------------------------------- the receipts


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def _conv_tensor_names():
    return ("v_conv.weight",)


def split_parameters_k18(model, frozen: bool = False):
    """The banked winner's own routing (``banked_gates_adamw``) with the k11b
    p3 per-channel gate projections AND the LIVE conv's kernel on the
    hybrid's AdamW side -- the GDN2 convention mirrored (a depthwise conv
    kernel is not a hidden-map matrix; Muon's Newton-Schulz does not apply
    to it), the incumbent's own side for every non-matrix tensor.  A
    frozen control's conv kernel carries ``requires_grad=False`` and is
    excluded by the ``requires_grad`` filter -- the split then equals the
    composed machine's own exactly (the falsifier's own plumbing
    receipt)."""
    from experiments.expressivity_tasks.x45_matrix_emender.k11b_probe_channel_gates \
        import gate_tensor_names
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import (_DECAY_BIAS, _DECAY_WEIGHT, _EMBED_HEAD, _MIXER_MATRIX, _MLP,
                _NORMS)
    gate_names = list(gate_tensor_names("p3"))
    if not frozen:
        gate_names += list(_conv_tensor_names())
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
                f"conv's kernel)")
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


def _routing_cross_check_k18(model, seed: int, frozen: bool) -> dict:
    """The no-drift receipt: an un-converted twin of the SAME build must
    route EXACTLY as the banked winner's own split routes it; the K18
    model's split must equal that plus exactly the p3 gate tensors (and,
    on the live conv arm, the conv's kernel) on the AdamW side -- or, for
    the frozen controls, equal the composed machine's own split exactly."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import ComposedReadMixer
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import ShortConvComposedMixer
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.routing_symmetry \
        import split_parameters_winner
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import gate_tensor_names
    has_conv = isinstance(model.blocks[0].mixer, ShortConvComposedMixer)
    if not has_conv and not isinstance(model.blocks[0].mixer,
                                       ComposedReadMixer):
        raise ValueError("the K18 arms' mixers must be composed (K15) or "
                         "short-conv composed (K18)")
    muon, adamw, muon_names, adamw_names = split_parameters_k18(model, frozen)
    twin, _cfg, _summary = _build_k10(dict(TUNED_WINNER), seed)
    _t_muon, _t_adamw, twin_muon, twin_adamw = split_parameters_winner(
        twin, "banked_gates_adamw")
    del twin, _t_muon, _t_adamw
    torch.cuda.empty_cache()
    if sorted(muon_names) != sorted(twin_muon):
        raise ValueError("the K18 model's Muon group drifted from the "
                         "banked winner's own split")
    expected_gate_names = sorted(
        f"blocks.{i}.mixer.{name}"
        for i in range(len(model.blocks))
        for name in gate_tensor_names("p3"))
    expected = set(twin_adamw) | set(expected_gate_names)
    if has_conv and not frozen:
        expected |= set(
            f"blocks.{i}.mixer.{name}"
            for i in range(len(model.blocks))
            for name in _conv_tensor_names())
    if sorted(adamw_names) != sorted(expected):
        raise ValueError("the K18 model's AdamW group must equal the banked "
                         "winner's own plus the p3 gate tensors (plus the "
                         "conv's kernel on the live conv arm)")
    return dict(muon_tensors=len(muon_names), adamw_tensors=len(adamw_names),
                conv_kernel_live=bool(has_conv and not frozen),
                note="the banked winner's own split + the p3 gate tensors ("
                     "+ the live conv's kernel on the AdamW side, the GDN2 "
                     "convention mirrored: a depthwise conv kernel is not "
                     "a hidden-map matrix); the frozen controls: the banked "
                     "+ p3 only -- the conv kernel is excluded by "
                     "requires_grad")


# --------------------------------------------------- the birth receipts


def conv_birth_receipts(model, stream, seed: int, microbatch: int) -> dict:
    """The conv machine's own birth receipts, run BEFORE any training on
    the FIRST train batch (the batch the step-1 loss sees), the IDENTITY
    arms only (the shift control has no birth parity, by registration):

    1. BORN AS THE COMPOSED MACHINE: layer for layer, at the arm's own
       inputs, the conv'd payload is `torch.equal` to the raw value
       projection (the identity kernel is exactly the pass-through), and
       the composed kernel's every channel launched on the conv'd payload
       is bit-equal to its own launch on the raw payload -- the machine's
       forward at birth IS the composed machine's (and hence the K13
       additive machine's: the K15 receipt's own chain).
    2. The p3 gates born at the registered scalar equivalents (the K15
       receipt's own checks, re-run on this machine).
    3. The conv's own birth telemetry: the kernel's taps (the registered
       identity), and the sign-of-zero corner count of the pass-through
       (invisible to every ==-based bar; asserted zero flips outside
       exact-zero entries).
    """
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import short_conv1d_causal
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    batch_rows = next(iter(stream.batches("train")))
    x, y, _valid = next(grouped(batch_rows, microbatch))

    per_layer = []
    flips_total = lanes_total = 0
    corner_total = 0
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
                # the conv machine's own payload at the production birth
                # expression (the registered identity kernel)
                conved = short_conv1d_causal(
                    mixer.value(xin), mixer.v_conv.weight).view_as(payload)
                payload32, conved32 = payload.float(), conved.float()
                write_keys32 = write_keys.float()
                query_keys32 = query_keys.float()
                # the production birth expressions under autocast
                lam_vec, lam_inc = mixer.channel_decay(xin)
                lam_vec32 = lam_vec.float()
                gates = 1.0 - mixer.write_scalar.view(1, 1, -1, 1) \
                    * torch.sigmoid(mixer.b_proj(xin)).view(
                        batch, tokens, mixer.heads, mixer.value_dim)
                gates32 = gates.float()
                gate_checks.append(dict(
                    lam_vec_bit_equal_registered_scalar=bool(torch.equal(
                        lam_vec, lam_inc.unsqueeze(-1).expand_as(lam_vec)
                        .to(lam_vec.dtype))),
                    g_vec_born_at_exact_1=bool(
                        torch.equal(gates, torch.ones_like(gates))),
                    conv_payload_bit_equal_raw=bool(
                        torch.equal(conved, payload))))
                # the sign-of-zero corner of the pass-through (counted;
                # invisible to every ==-based bar)
                a = conved32.view(torch.int32)
                b = payload32.view(torch.int32)
                bitwise = a != b
                zeros = payload32 == 0.0
                nonlocal corner_total
                corner_total += int((bitwise & zeros).sum())
                assert int((bitwise & ~zeros).sum()) == 0, \
                    "the identity birth flipped bits outside exact-zero " \
                    "entries -- a real divergence, escalate"
                state = torch.zeros(batch, mixer.heads, mixer.rank,
                                    mixer.value_dim, dtype=torch.float32,
                                    device=xin.device)
                live = torch.zeros(batch, mixer.heads, mixer.rank,
                                   dtype=torch.bool, device=xin.device)
                keys = torch.zeros(batch, mixer.heads, mixer.rank,
                                   mixer.key_dim, dtype=torch.float32,
                                   device=xin.device)
                out_conv = _k15_forward_launch(
                    conved32, write_keys32, query_keys32, lam_vec32,
                    gates32, state, live, keys, pol32, mixer.rank,
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
                out_raw = _k15_forward_launch(
                    payload32, write_keys32, query_keys32, lam_vec32,
                    gates32, state2, live2, keys2, pol32, mixer.rank,
                    mixer.segment_tokens, True, collect=True, warps=8,
                    stages=3)
            flips = lanes = 0
            for name in ("slot", "erase", "op", "live_after", "payload",
                         "live_pre", "reads", "state", "live", "weights",
                         "keys"):
                a, b = out_raw[name], out_conv[name]
                flips += int((a != b).sum())
                lanes += a.numel()
            per_layer.append(dict(
                layer=layer_index, flips=flips, lanes=lanes,
                mean_live_rows=float(
                    out_conv["live_pre"].float().sum(-1).mean())))
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
        note="the conv machine's own birth receipts on the FIRST train "
             "batch: the conv is born at the identity kernel [0,0,0,1] "
             "(the standard nn.Conv1d layout, the LAST tap the current "
             "token), EXACTLY the pass-through, so the machine is BORN AS "
             "THE COMPOSED MACHINE bit for bit -- and hence born as the "
             "K13 additive machine (the K15 receipt's own chain); the "
             "step-1 loss IS asserted bit-identical to the composed "
             "machine's own same-seed row (the nearest kin)",
        born_as_the_composed_machine=dict(
            flips=flips_total, lanes=lanes_total,
            bit_identical=bool(flips_total == 0),
            note="every channel of the composed launch on the conv'd "
                 "payload vs the composed launch on the raw payload, "
                 "every layer; the sign-of-zero IEEE corner of the "
                 "pass-through is the only legal escape and its "
                 "out-of-zeros count is asserted zero"),
        gates_born_at_registered_scalars=dict(
            all_checks=bool(gate_checks)
            and all(c["lam_vec_bit_equal_registered_scalar"]
                    and c["g_vec_born_at_exact_1"]
                    and c["conv_payload_bit_equal_raw"] for c in gate_checks),
            n_layers=len(gate_checks)),
        conv_birth_telemetry=dict(
            sign_of_zero_corner_flips=corner_total,
            note="the IEEE corner (a -0.0 entry flipping to +0.0 through "
                 "the zero taps' additions), counted over every layer's "
                 "first-batch payload; invisible to every ==-based parity "
                 "bar and to the loss"),
        per_layer=per_layer[:1] + per_layer[-1:])   # the first and last layer
    return receipt


# --------------------------------------------- the mechanism instruments


def _conv_kernel_rows(mixer) -> dict:
    """The LEARNED KERNEL receipts (the supervisor's addition, the
    mechanism's direct read): per layer -- the per-tap means across the
    channels, the kernel L2 norm's mean, the distance from the birth
    kernel, and the per-tap spread across channels (did the conv learn a
    token-shift-like mix of the last few payload vectors, and how
    heterogeneous is it across channels?)."""
    import torch
    w = mixer.v_conv.weight.detach().float()          # (channels, 4)
    identity = torch.zeros_like(w)
    identity[:, -1] = 1.0
    dist = (w - identity).norm(dim=1)
    return dict(
        tap_means=[float(v) for v in w.mean(0)],
        tap_abs_means=[float(v) for v in w.abs().mean(0)],
        tap_stds=[float(v) for v in w.std(0)],
        kernel_l2_mean=float(w.norm(dim=1).mean()),
        distance_from_birth_mean=float(dist.mean()),
        distance_from_birth_max=float(dist.max()),
        channels=int(w.shape[0]))


def k18_probe_diagnostics(model, stream, windows: int = PROBE_WINDOWS) -> dict:
    """The TRAINED conv arm's mechanism receipts: the K15 instrument (the
    |w|-mass effective rows + the collapse detector, occupancy, op mix,
    the identity-rewrite share through the per-channel replay) UNDER the
    K11B gate telemetry (all three p3 channels' born-identity drift)
    computed on the CONVED payload (the machine the arm trains on), AND
    the conv's own receipts (the learned kernels: the mechanism question
    is whether the conv learned a token-shift-like mix of the last few
    payload vectors -- the kernel is the direct read), on the k15 collect
    launch.  The decorative-store sweep runs separately (the probe's own
    sweep receipt, the uniformity hypothesis's direct test)."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import (
        grouped)
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k10_probe_separated_keys import _probe_windows, _project_keyed
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k13_probe_additive_read import _agg_walk
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import _k15_forward_launch
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import short_conv1d_causal
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k6_sequential_matrix import _round_f32

    rows = _probe_windows(stream, windows)
    model.eval()
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
                # the conv machine's own payload (the machine the arm
                # trains on)
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
                        max=float(drift.max())),
                    write_gate=dict(
                        mean_distance_from_1=float(gw_dist.float().mean()),
                        max=float(gw_dist.float().max())),
                    read_gate=dict(
                        mean_distance_from_1=float(gr_dist.float().mean()),
                        max=float(gr_dist.float().max()))))
                conv_rows.append(dict(layer=layer_index,
                                      **_conv_kernel_rows(mixer)))
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
                                   "persistent walk on the CONVED payload "
                                   "(the machine the arm trains on; "
                                   "collect=True)",
                            probe_wall_seconds=round(wall, 1))
    summary["gate_telemetry"] = dict(
        note="per-layer means over the probe windows: the born-identity "
             "drift for ALL THREE p3 gate channels (the K15 instrument, "
             "imported semantics, re-run on the conv machine's launch)",
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
    summary["conv_telemetry"] = dict(
        note="the LEARNED KERNEL receipts (the mechanism's direct read): "
             "per layer over the probe windows -- the per-tap means and "
             "spreads across the channels (did the conv learn a "
             "token-shift-like mix of the last few payload vectors?), the "
             "kernel L2 norm, and the distance from the identity birth "
             "(the frozen controls' rows are the birth kernels' own, "
             "unchanged by construction)",
        per_layer={str(r["layer"]): {k: v for k, v in r.items() if k != "layer"}
                   for r in conv_rows},
        mean_tap_means=[float(sum(r["tap_means"][j] for r in conv_rows)
                              / max(1, len(conv_rows)))
                        for j in range(4)],
        mean_tap_stds=[float(sum(r["tap_stds"][j] for r in conv_rows)
                             / max(1, len(conv_rows))) for j in range(4)],
        mean_distance_from_birth=float(sum(
            r["distance_from_birth_mean"] for r in conv_rows)
            / max(1, len(conv_rows))),
        max_distance_from_birth=float(max(
            r["distance_from_birth_max"] for r in conv_rows)),
        n_layers=len(conv_rows))
    return summary


# ------------------------------------------------------------ the clause read


def evaluate_clauses(receipt: dict) -> dict:
    """The registration's pre-stated THREE-OUTCOME rule, evaluated per seed
    verbatim against the composed machine's OWN same-seed banked rows (the
    baseline; no directional prediction was pre-stated)."""
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
        if receipt["arm"] == "conv":
            if improvement >= WIN_BAR:
                out["clause_at_this_seed"] = \
                    "SC-1_at_this_seed (LOCAL-FEATURES: closes >= 0.01 -- " \
                    "seed 3402 owed before any claim)"
            elif improvement >= -WIN_BAR:
                out["clause_at_this_seed"] = \
                    "SC-2_at_this_seed (parity inside the band: the conv " \
                    "is not the residual's owner at probe scale)"
            else:
                out["clause_at_this_seed"] = \
                    "SC-3_at_this_seed (HURTS beyond the band: the " \
                    "payload's current-token-only structure was " \
                    "load-bearing for the slot semantics)"
        elif receipt["arm"] == "conv_frozen":
            out["clause_at_this_seed"] = \
                "the FALSIFIER control (its own bar, see the rules receipt)"
        else:
            out["clause_at_this_seed"] = \
                "the FIXED-SHIFT control (SH-1: reported as measured " \
                "against the baseline AND against the conv arm)"
        out["vs_gdn2_post_cooldown"] = post - GDN2_POST_COOLDOWN
        out["gdn2_contact_flag"] = bool(post <= GDN2_POST_COOLDOWN)
        out["fraction_of_corrected_gap_closed"] = max(
            0.0, BANKED_CORRECTED_ANCHOR - post) / (
            BANKED_CORRECTED_ANCHOR - GDN2_POST_COOLDOWN)
    out["pre_state"] = ("NO directional prediction was pre-stated; the "
                        "THREE-OUTCOME rule (SC-1/SC-2/SC-3), the shift "
                        "control's SH-1 and the falsifier's own bar are "
                        "the registered decision rule, and any "
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
        import composed_parameter_delta
    from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
        import (SHORT_CONV_BIRTHS, convert_model_to_short_conv,
                short_conv_parameter_delta)

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
    birth = arm["birth"]
    identity_born = birth == "identity"
    # ---- the machine: the K15 composed machine VERBATIM (the banked
    # winner build + the K13 additive read + the K11B p3 gate stack at the
    # mimetic-neutral birth); every arm adds the SHORT CONV at its
    # registered birth (identity or the fixed shift), live or frozen.
    machine = (f"the composed machine + the K18 short conv at the "
               f"{birth} birth"
               + (" (frozen at birth, never learned)" if frozen
                  else " (live, learns to deviate)"))
    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import convert_model_to_composition
    model = convert_model_to_composition(model)
    model = convert_model_to_short_conv(model, birth=birth)
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    conv_added = short_conv_parameter_delta(
        depth, mixer0.width, mixer0.heads, mixer0.value_dim) \
        - composed_parameter_delta(
            depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if conv_added != CONV_PARAMETER_DELTA \
            or parameters_after != CONV_MACHINE_PARAMETERS:
        raise ValueError(f"the conv machine's parameters must be the "
                         f"registration's ({COMPOSED_PARAMETERS} + "
                         f"{CONV_PARAMETER_DELTA}); got added {conv_added}, "
                         f"total {parameters_after}")
    if not identity_born:
        # the shift control's kernel is the registered shift, not identity
        w = mixer0.v_conv.weight
        want = torch.tensor(SHORT_CONV_BIRTHS[birth],
                            device=w.device).unsqueeze(0).expand_as(w)
        if not torch.equal(w, want):
            raise ValueError("the shift control's kernel must be the "
                             "registered [0, 0, 0.5, 0.5] birth")
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
    # the frozen controls: the conv's kernel frozen at its birth
    if frozen:
        for block in model.blocks:
            block.mixer.v_conv.weight.requires_grad_(False)
        assert all(not p.requires_grad for b in model.blocks
                   for p in b.mixer.v_conv.parameters())

    # ---- the optimizer: the inherited 4x hybrid-Muon point, the banked
    # routing with the p3 gate tensors AND the live conv's kernel on the
    # AdamW side
    peak_lr = INHERITED_LR_MULTIPLIER * TUNED_ADAMW_LR
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_k18(model, frozen)
    routing_receipt = _routing_cross_check_k18(model, seed, frozen)
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
        banked_mechanism=_banked_mechanism_anchors())
    run_config = dict(
        probe="k18_probe_short_conv", arm=args.arm, seed=seed,
        arm_note=arm["note"], machine=machine, frozen=frozen, birth=birth,
        corrected_scoring=CORRECTED_SCORING, rules=RULES,
        exposure_targets=int(args.bytes),
        model_build="the K15 composition VERBATIM: velocity_ladder._build_k10 "
                    "(the tuned winner's config, the banked bake-off build) "
                    "+ convert_model_to_composition (the K15 conversion; "
                    "the gates born at the registered scalar equivalents) "
                    "+ convert_model_to_short_conv (the K18 conversion; the "
                    "depthwise short conv -- GDN2's registered conv_size=4, "
                    "bias-free, the silu not transplanted -- on the payload "
                    "path at GDN2's own position, born at the registered "
                    "kernel: identity [0,0,0,1] on the conv/falsifier arms, "
                    "the fixed shift [0,0,0.5,0.5] on the shift control)",
        optimizer=optimizer_block,
        routing="banked_gates_adamw + the p3 gate tensors + the live conv's "
                "kernel on AdamW (the GDN2 convention mirrored: a depthwise "
                "conv kernel is not a hidden-map matrix); the frozen "
                "controls: the banked + p3 only",
        routing_receipt=routing_receipt,
        routing_inventory=dict(muon=muon_names, adamw=adamw_names),
        parameters=dict(before=parameters_before,
                        composed_added=composed_parameter_delta(
                            depth, mixer0.width, mixer0.heads,
                            mixer0.value_dim),
                        conv_added=conv_added, total=parameters_after,
                        note="p3's own gate stack (the read channel "
                             "instantiated once) + the conv kernels' "
                             "24,576 (one depthwise 256x4 kernel per "
                             "mixer, 24 mixers -- 0.015% of the machine, "
                             "quoted in every receipt)"),
        checkpoint_steps=list(args.checkpoint_steps),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear cooldown "
                       "(the corrected-scoring extension, same stream); the "
                       "step-180 constant-LR eval is retained for banked "
                       "comparability"),
        step1_parity_rule=("the identity arms (conv, conv_frozen): the "
                           "step-1 train loss is asserted BIT-IDENTICAL to "
                           "the composed machine's own same-seed row (== the "
                           "K13 additive arm's own: the machine at birth IS "
                           "the composed machine; the optimizer cannot touch "
                           "the first forward); the shift control: NO parity "
                           "exists or is asserted (a different function at "
                           "birth -- a fixed structural control, disclosed, "
                           "the config arms' own convention)"),
        both_bands=BANDS,
        anchors=anchors,
        bindings=dict(matched_arms.registration_bindings(),
                      k15_summary=str(RECEIPTS / "k15_composition_summary.json"),
                      k15_gates_receipt=str(RECEIPTS / "k15_gates.json"),
                      k18_gates_receipt=str(RECEIPTS / "k18_gates.json"),
                      k18_registration=str(
                          ROOT / "PREREGISTRATION_SHORT_CONV.md")),
        summary=summary, identity=identity, versions=_versions(),
        git=matched_arms.git_state())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns()))
    print(json.dumps(dict(arm=args.arm, card=card, seed=seed, frozen=frozen,
                          birth=birth, read_scale="inherited",
                          parameters=parameters_after)), flush=True)

    banked_step1 = _banked_step1_loss(seed)
    k13_step1 = _k13_additive_step1(seed)
    run_config["banked_step1_loss_reference"] = banked_step1
    run_config["k13_additive_step1_loss_reference"] = k13_step1

    # ---- the birth receipts, BEFORE any training (the identity arms only)
    birth_receipt = None
    if identity_born:
        birth_receipt = conv_birth_receipts(model, stream, seed,
                                            optimizer_config["microbatch"])
        if not birth_receipt["born_as_the_composed_machine"]["bit_identical"]:
            raise ValueError(
                "the BORN-AS-THE-COMPOSED-MACHINE receipt failed at probe "
                "scale -- escalate, not excuse (bit-equality conflict with "
                "the committed composed machine)")
        if not birth_receipt["gates_born_at_registered_scalars"]["all_checks"]:
            raise ValueError("the p3 gates' birth checks failed at probe "
                             "scale -- escalate, not excuse")
        run_config["birth_receipts"] = birth_receipt
        print("  born-as-the-composed-machine: "
              f"{birth_receipt['born_as_the_composed_machine']['flips']} "
              "flips in "
              f"{birth_receipt['born_as_the_composed_machine']['lanes']} "
              "lanes; sign-of-zero corner "
              f"{birth_receipt['conv_birth_telemetry']['sign_of_zero_corner_flips']}",
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

    # ---- the step-1 parity receipt (the identity arms: the nearest-kin
    # bar; the shift control: disclosed, no parity)
    step1 = run["rows"][0]["loss"]
    if identity_born:
        print(f"  step-1 train loss {step1!r} (the composed machine's own "
              f"row's is {four_x['step1_train_loss'][seed]!r} -- asserted "
              f"bit-identical; the K13 additive arm's own row's is "
              f"{k13_step1!r}; the banked winner's is {banked_step1!r})",
              flush=True)
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
        print(f"  step-1 train loss {step1!r} (NO parity asserted: the "
              f"fixed shift is a different function at birth; the composed "
              f"machine's own row is "
              f"{four_x['step1_train_loss'][seed]!r})", flush=True)

    # ---- the falsifier's honest divergence receipt: the per-step loss
    # divergence vs the composed machine's own banked training log
    falsifier_divergence = None
    if frozen and identity_born:
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
                    note="the frozen-conv falsifier vs the composed "
                         "machine's own banked training log (the K17 "
                         "falsifier's perfect full-trajectory receipt is "
                         "the expectation to beat; the K11B 2.4 honest "
                         "limitation -- ULP-scale divergence from "
                         "exact-zero gradient contributions whose signs "
                         "are reduction-order artifacts -- is the legal "
                         "escape, quoted if it fires)")

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

    # ---- the mechanism receipts at BOTH states + the decorative sweep
    # (the uniformity hypothesis's direct test: does the local temporal
    # filter make more of the 24 layers load-bearing?)
    from experiments.expressivity_tasks.x45_matrix_emender.side_by_side \
        import k10_read_mask_sweep
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
    if run["status"] == "completed":
        if run["constant_steps"] in run["snapshots"]:
            model.load_state_dict(run["snapshots"][run["constant_steps"]])
            diagnostics_180 = k18_probe_diagnostics(
                model, stream, windows=args.probe_windows)
            sweep_180 = k10_read_mask_sweep(model, stream)
        if "final" in run["snapshots"]:
            model.load_state_dict(run["snapshots"]["final"])
            diagnostics_final = k18_probe_diagnostics(
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
        probe="k18_probe_short_conv", arm=args.arm, seed=seed,
        arm_note=arm["note"], machine=machine, frozen=frozen, birth=birth,
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
        parameters_composed_added=composed_parameter_delta(
            depth, mixer0.width, mixer0.heads, mixer0.value_dim),
        parameters_conv_added=conv_added,
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
            k13_parity=bool(args.smoke or not identity_born
                            or step1 == k13_step1),
            composed_4x_same_seed_step1=four_x["step1_train_loss"][seed],
            composed_4x_parity=bool(
                args.smoke or not identity_born
                or step1 == four_x["step1_train_loss"][seed]),
            banked_same_seed_step1=banked_step1),
        birth_receipts=birth_receipt,
        falsifier_divergence=falsifier_divergence,
        scores=scores,
        anchors=anchors,
        inherited_4x_same_seed=dict(
            post_cooldown=four_x["post_cooldown"][seed],
            constant_lr=four_x["constant_lr"][seed],
            ema_180=four_x["ema_180"][seed],
            step1_train_loss=four_x["step1_train_loss"][seed],
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


def _sweep_layers_ge(sweep: dict | None, threshold: float = 0.02) -> int | None:
    """The decorative-sweep row's own count: the layers whose masked-read
    dNLL clears the banked regime's 0.02 threshold (the uniformity
    hypothesis's own scale -- the K15/K17 receipts' 4-5 of 24)."""
    if not sweep:
        return None
    deltas = sweep.get("delta_per_layer") or {}
    return sum(1 for v in deltas.values() if v >= threshold)


def _arm_row(path: Path) -> dict | None:
    if not (path / "result.json").exists():
        return None
    r = json.loads((path / "result.json").read_text())
    diag = r["diagnostics_post_cooldown"] or {}
    conv = diag.get("conv_telemetry") or {}
    return dict(
        arm=r["arm"], seed=r["seed"], machine=r["machine"], frozen=r["frozen"],
        birth=r["birth"], status=r["status"],
        post_cooldown=r["final_eval_nll_post_cooldown"],
        constant_lr=r["final_eval_nll_constant_lr"],
        ema_180=r["ema_eval_nll"], step1_train_loss=r["step1_train_loss"],
        step1_parity=r["step1_receipt"]["composed_4x_parity"],
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
        conv_telemetry=dict(
            tap_means=conv.get("mean_tap_means"),
            tap_stds=conv.get("mean_tap_stds"),
            distance_from_birth=conv.get("mean_distance_from_birth"),
            max_distance_from_birth=conv.get("max_distance_from_birth")) if conv else None,
        decorative_sweep=dict(
            layers_ge_threshold=_sweep_layers_ge(
                r.get("read_mask_sweep_post_cooldown")),
            base_nll=(r.get("read_mask_sweep_post_cooldown") or {})
            .get("base_nll"),
            raw=r.get("read_mask_sweep_post_cooldown")),
        falsifier_divergence=r.get("falsifier_divergence"),
        clauses=r.get("clauses"),
        muon_update_rms=(r.get("muon_update_rms") or {}).get("muon_update_rms")
        if isinstance(r.get("muon_update_rms"), dict) else None)


def assemble(args) -> dict:
    """The arms' table (the conv transplant, the frozen falsifier, the
    fixed-shift control), the pre-stated THREE-OUTCOME rule applied exactly
    as stated, the falsifier's bar, and the re-quoted gap vs GDN2 with the
    honest band caveats.  The final gap-quote phrasing is the supervisor's
    to adjudicate (the K13 clause-conflict precedent): this assemble states
    the measured rows and the flags, not the final wording."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    four_x = composed_4x_banked()
    cells: dict[str, dict] = {
        "composed_inherited": dict(
            cited=True,
            note="the K15 composition arm's own landed rows at BOTH seeds "
                 "(the inherited 4x point) -- cited, never re-run; the "
                 "BASELINE",
            seeds={PRIMARY_SEED: dict(
                post_cooldown=four_x["post_cooldown"][PRIMARY_SEED],
                constant_lr=four_x["constant_lr"][PRIMARY_SEED],
                ema_180=four_x["ema_180"][PRIMARY_SEED],
                step1_train_loss=four_x["step1_train_loss"][PRIMARY_SEED]),
                SECOND_SEED: dict(
                post_cooldown=four_x["post_cooldown"][SECOND_SEED],
                constant_lr=four_x["constant_lr"][SECOND_SEED],
                ema_180=four_x["ema_180"][SECOND_SEED],
                step1_train_loss=four_x["step1_train_loss"][SECOND_SEED])}),
        "gdn2_reference": dict(
            cited=True, post_cooldown=GDN2_POST_COOLDOWN,
            note="GDN2's own banked row at the same probe protocol, the "
                 "cross-machine gap reference (the conv-HAVING machine: "
                 "the X46 disclosure's own subject)"),
    }
    for arm_label in ARMS:
        for seed in (PRIMARY_SEED, SECOND_SEED):
            row = _arm_row(RUNS / f"{arm_label}_seed{seed}")
            if row is not None:
                cells[f"{arm_label}_seed{seed}"] = row

    # ---- the three-outcome adjudication (the registration's own rule)
    conv_1 = cells.get(f"conv_seed{PRIMARY_SEED}")
    conv_2 = cells.get(f"conv_seed{SECOND_SEED}")
    adjudication = dict(
        rule=RULES, bands=BANDS, win_bar=WIN_BAR,
        second_seed_owed=False, escalation_flags=[])
    if conv_1 is not None and conv_1.get("post_cooldown") is not None:
        base_1 = four_x["post_cooldown"][PRIMARY_SEED]
        improvement_1 = base_1 - conv_1["post_cooldown"]
        adjudication["conv_seed3401_improvement"] = improvement_1
        if improvement_1 >= WIN_BAR:
            adjudication["second_seed_owed"] = True
            if conv_2 is not None and conv_2.get("post_cooldown") is not None:
                improvement_2 = (four_x["post_cooldown"][SECOND_SEED]
                                - conv_2["post_cooldown"])
                adjudication["conv_seed3402_improvement"] = improvement_2
                if improvement_2 >= WIN_BAR:
                    adjudication["verdict"] = \
                        "SC-1 REPLICATED AT BOTH SEEDS: the conv owns a " \
                        "LOCAL-FEATURES share of the residual; the " \
                        "attribution chain gains the term and the " \
                        "missing-component story replaces part of the " \
                        "uniformity story"
                elif improvement_2 >= -WIN_BAR:
                    adjudication["verdict"] = \
                        "MIXED (the K15 conservative mixed-seed rule): " \
                        "seed-3401 win, seed-3402 inside the band -- " \
                        "parity-grade, reported as measured, NOT banked " \
                        "as composing"
                else:
                    adjudication["verdict"] = \
                        "MIXED, seed-3402 WORSE beyond the band -- " \
                        "escalated, not adjudicated here"
                    adjudication["escalation_flags"].append(
                        "conv_seed3402_interference")
            else:
                adjudication["verdict"] = \
                    "SC-1 AT SEED 3401: the second seed is OWED before any " \
                    "claim (the standing win-bar rule)"
        elif improvement_1 >= -WIN_BAR:
            adjudication["verdict"] = \
                "SC-2 PARITY: the conv is NOT the residual's owner on " \
                "byte-LM text at probe scale; the uniformity attribution " \
                "stands with one more receipt (the pre-stated small-end " \
                "possibility, receipted)"
            adjudication["second_seed_owed"] = False
        else:
            adjudication["verdict"] = \
                "SC-3 HURTS: the payload path's current-token-only " \
                "structure was load-bearing for the slot semantics -- " \
                "receipted as a machine-family difference, not a failure"
            adjudication["escalation_flags"].append(
                "conv_hurts_beyond_band")
    # the falsifier's bar
    fals = cells.get(f"conv_frozen_seed{PRIMARY_SEED}")
    if fals is not None:
        div = fals.get("falsifier_divergence") or {}
        adjudication["falsifier"] = dict(
            step1_parity=fals.get("step1_parity"),
            steps_compared=div.get("steps_compared"),
            bit_identical_steps=div.get("bit_identical_steps"),
            first_diverging_step=div.get("first_diverging_step"),
            max_abs_loss_delta=div.get("max_abs_loss_delta"),
            landing_vs_banked=(four_x["post_cooldown"][PRIMARY_SEED]
                               - fals["post_cooldown"])
            if fals.get("post_cooldown") is not None else None,
            note="the frozen-conv control's own bar: step-1 bit-identity, "
                 "a landing at parity, and the full-trajectory receipt "
                 "vs the composed machine's banked training log")
    # the shift control's decomposition (SH-1)
    shift = cells.get(f"shift_fixed_seed{PRIMARY_SEED}")
    if shift is not None and conv_1 is not None:
        adjudication["shift_decomposition"] = dict(
            note="SH-1, reported as measured: the fixed shift separates "
                 "ANY LOCAL MIXING from a LEARNED FILTER",
            shift_vs_baseline=(four_x["post_cooldown"][PRIMARY_SEED]
                               - shift["post_cooldown"])
            if shift.get("post_cooldown") is not None else None,
            shift_vs_conv=(conv_1["post_cooldown"] - shift["post_cooldown"])
            if shift.get("post_cooldown") is not None else None)
    # the band discipline
    for label, row in cells.items():
        if row.get("cited") or row.get("post_cooldown") is None:
            continue
        if row["post_cooldown"] <= GDN2_POST_COOLDOWN:
            adjudication["escalation_flags"].append(
                f"{label}_at_or_past_gdn2")
    summary = dict(
        probe="k18_probe_short_conv", assemble=True,
        note="the missing-component candidate's receipts: GDN2's "
             "registered short depthwise conv (kernel 4), transplanted "
             "onto the composed machine's payload path at the escalated, "
             "supervisor-approved design (born identity [0,0,0,1], the "
             "standard tap layout; the silu not transplanted), the "
             "frozen falsifier, and the fixed-shift control",
        cells=cells, adjudication=adjudication,
        bands=BANDS,
        composed_parameters=COMPOSED_PARAMETERS,
        conv_parameter_delta=CONV_PARAMETER_DELTA,
        disclosure=dict(
            x46=("the X46 growth-separator ceiling refutation (gdn2-fixed "
                 "0.93 vs the Welch-bound 0.80) was achieved by the "
                 "conv-HAVING machine: GDN2's registered block carries "
                 "the short conv ON (conv_size=4, default), so the "
                 "refutation is not conv-attributable-in-doubt -- the "
                 "linear ceiling was derived from the key-space geometry "
                 "regardless of the conv, and the trained machine "
                 "exceeded it with ALL its components, conv included"),
            silu=("the fused SiLU of GDN2's ShortConvolution is NOT "
                  "transplanted: ours never silus the payload (a "
                  "standing, pre-existing cross-machine mismatch -- GDN2 "
                  "silus q/k/v even with the conv OFF, so the conv's own "
                  "ablated delta in GDN2's family is the temporal "
                  "filter, and ours isolates the same object); a "
                  "conv+silu arm would break the 0-flip birth bar by "
                  "construction"),
            qk=("the q/k-conv variant (P2 re-registered as decisions "
                "conv-influenced) is a NAMED FOLLOW-UP, not this lane: "
                "our key/query projections already consume a 4-token "
                "causal context window (exactly the conv's "
                "receptive-field width), so the payload conv COMPLETES "
                "the receptive-field parity GDN2 has on all three "
                "sides"))),
    out = RECEIPTS / "k18_short_conv_summary.json"
    atomic_json(out, summary)
    print(json.dumps(dict(cells={k: {kk: vv for kk, vv in v.items()
                                     if kk in ("arm", "seed", "frozen",
                                               "birth", "post_cooldown",
                                               "step1_parity")}
                                for k, v in cells.items()},
                          adjudication=adjudication), indent=1)[:6000])
    print(f"wrote {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    arm = sub.add_parser("arm", help="run one arm at the probe protocol")
    arm.add_argument("--arm", required=True, choices=sorted(ARMS))
    arm.add_argument("--seed", type=int, default=PRIMARY_SEED)
    arm.add_argument("--output", required=True)
    arm.add_argument("--bytes", type=int, default=EXPOSURE_TARGETS)
    arm.add_argument("--eval-split", default="calibration")
    arm.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    arm.add_argument("--checkpoint-steps", type=int, nargs="*",
                     default=list(CHECKPOINT_STEPS))
    arm.add_argument("--smoke", action="store_true",
                     help="the pipeline smoke (3-step cooldown, no parity "
                          "assertions)")
    sub.add_parser("assemble", help="the arms' table + the clause "
                                     "adjudication")
    args = parser.parse_args()
    if args.command == "arm":
        run_arm(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main()
