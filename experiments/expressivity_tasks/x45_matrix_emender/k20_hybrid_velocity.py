"""K20 -- THE HYBRID VELOCITY MATRIX: the hybrid program's first instrument.

Zero training, timing probes only, one card.  THE QUESTION (asked BEFORE any
hybrid registration): the board comparison is matched-TIME (2700 active
seconds), and a hybrid of our slot layers + GDN2's dense layers pays our
scan's cost on every slot layer -- so which interleave ratios are viable
under the matched-time protocol?  Registered receipt so far: our composed
scan forward = 22.96 ms per (layer, window) at the registered shape
(``receipts/k18_gates.json`` gate D).  Unknown: GDN2's own mixer cost at the
SAME shape, the backward ratios, and the end-to-end wall-clock of mixed
stacks.  MEASURE, do not estimate.

THE ARMS (all at the registered shape, batch 8 x 2048 tokens at width 576,
the probe protocol's batch/window regime):

1. ``per_layer`` -- pure-machine per-(layer, window) costs: OUR composed
   mixer (the committed K15/K18 engine, conv included: the K18 arm build
   verbatim -- ``velocity_ladder._build_k10`` + ``convert_model_to_composition``
   + ``convert_model_to_short_conv``) and GDN2's committed arm's mixer
   (``matched_arms.build_gdn2`` verbatim, the FLA ``GatedDeltaNet2`` core).
   Forward AND backward ms per (layer, window-group), warmed >= 20 iters,
   medians not means, CUDA-synced, stated spread, on each layer's OWN
   captured real activation (one real microbatch group of the flagship
   stream driven through the machine with hooks).  The per-layer MLPs are
   timed the same way (the whole-layer swap's second term).  Both engines
   are imported verbatim from the committed code; nothing is modified.

2. ``steps`` -- the end-to-end training-step cost of the EXISTING pure
   machines at the probe protocol's step shape (64 windows/step, microbatch
   8): ours under the banked protocol's HybridMuonX45 (the K18 arm's own
   optimizer split), GDN2 under its registered AdamW nomination -- the
   honest per-step wall-clock both banked rows were built on, per-step
   optimizer seconds receipted separately (the clock is each committed
   loop's own: sync -> stream read -> zero_grad -> lr -> fwd/bwd -> clip ->
   optimizer.step -> sync; the corrected protocol's EMA sits OUTSIDE that
   clock and is therefore excluded here too).

3. ``projection`` -- the PROJECTED hybrid wall-clock table, per ratio
   bracket {3, 4, 6, 8}-of-24 (plus the 24-of-24 pure-ours calibration row),
   computed from the measured per-layer costs, NOT simulated end-to-end (no
   hybrid build: the projection table IS the deliverable).  Two composition
   assumptions are projected, both from measured numbers: (a) mixer-only
   swap (our composed mixer inside an otherwise-GDN2 stack, GDN2's MLPs
   everywhere) and (b) whole-layer swap (our mixer + our MLP on the slot
   layers).  Optimizer included as a stated term: the banked protocol's
   Muon (measured on our machine, which is what would update a hybrid's
   params) vs GDN2's registered AdamW -- both brackets are projected.

4. ``verdict`` -- per ratio: the projected hybrid step cost vs the measured
   pure-GDN2 step cost, stated as "the hybrid sees X% of GDN2's exposure in
   2700 s" (matched time: exposure is inversely proportional to step cost;
   both machines eat 64-window steps, so the exposure ratio IS the step-cost
   ratio).  The scale is the task's own pre-state: ~1.25x wall-clock viable,
   ~2.1x board-fatal at parity quality.

PRE-STATE: no directional prediction; the measurement IS the result.
Registration-style: this lane measures, it does not fix engines -- an engine
that cannot be timed at the registered shape as-committed is reported, not
patched.

Usage (one leased card):

    eval "$(GPU_LEASE_VISIBLE=0 scripts/gpu_lease.sh acquire 1 --wait)"
    OMP_NUM_THREADS=4 python /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k20_hybrid_velocity \\
        --out experiments/expressivity_tasks/x45_matrix_emender/receipts/k20_hybrid_velocity.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
import traceback
from pathlib import Path

# The probe subprocess discipline (configure_production fail-closes on a
# missing allocator knob if these are not set before CUDA initialization).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from experiments.expressivity_tasks.x42_pile_scale import lattice as L
from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
from experiments.expressivity_tasks.x42_pile_scale.velocity_train import (
    configure_production)
from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
    import HybridMuonX45
from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
    import convert_model_to_composition
from experiments.expressivity_tasks.x45_matrix_emender.k18_probe_short_conv \
    import (CONV_MACHINE_PARAMETERS, INHERITED_LR_MULTIPLIER, TUNED_ADAMW_LR,
             split_parameters_k18)
from experiments.expressivity_tasks.x45_matrix_emender.k18_short_conv \
    import convert_model_to_short_conv
from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
    import (BETA2, GRAD_CLIP, MICROBATCH, TUNED_WINNER, _require_lease,
             _versions)
from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
    import _build_k10

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
SEED = 3401                                  # the standing seed
# The registered shape: one (layer, window-group) = batch 8 x 2048 tokens at
# width 576 -- the K10/K13/K15/K17/K18 gate receipts' own fixture shape, and
# the probe protocol's microbatch group.
UNIT = ("one (layer, window-group): batch 8 x 2048 tokens at each machine's "
        "own committed 150M width (ours 576, GDN2 640), bf16 autocast "
        "compute, fp32 params")
DEPTH = 24
GROUPS_PER_STEP = 8                          # 64 windows / microbatch 8
RATIOS = (3, 4, 6, 8, 24)                    # slot layers of 24 (24 = calibration)
# RECORDED banked anchors this run re-measures (cited, never rewritten).
RECORDED = dict(
    composed_scan_forward_ms_per_window=22.964699504276116,
    composed_scan_receipt=str(RECEIPTS / "k18_gates.json"),
    ours_probe_step180_active_seconds=1932.6842099996284,   # k18 conv arm, seed 3401
    ours_probe_run=str(ROOT / "k18_runs/conv_seed3401/training.jsonl"),
    gdn2_probe_step180_active_seconds=584.4091627011076,    # the gdn2-armed probe, seed 3401
    gdn2_probe_run=str(ROOT / "velocity_runs/gdn2_probe_seed3401/training.jsonl"),
    probe_exposure_bytes=20_022_866,
    probe_steps=180,
    note="the banked honest walls at the matched-TARGET probe landing: the "
         "two machines trained the SAME 20,022,866 bytes in DIFFERENT "
         "wall-clocks (ours 1,932.7 s, GDN2 584.4 s) -- the probe protocol's "
         "own time-exposure asymmetry, stated per the K20 task")


# ------------------------------------------------------------------ plumbing


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def git_state() -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    relevant = ["experiments/expressivity_tasks/x45_matrix_emender/k15_composition.py",
                "experiments/expressivity_tasks/x45_matrix_emender/k18_short_conv.py",
                "experiments/expressivity_tasks/x45_matrix_emender/matched_arms.py",
                "experiments/expressivity_tasks/x45_matrix_emender/velocity_ladder.py",
                "experiments/expressivity_tasks/x42_pile_scale/lattice.py",
                "experiments/expressivity_tasks/x42_pile_scale/byte_velocity.py",
                "ndm/models/external_gdn2.py"]
    return dict(commit=git("rev-parse", "HEAD"),
                branch=git("branch", "--show-current"),
                relevant_paths_dirty=git("status", "--porcelain", "--", *relevant))


def session_meta(card: str) -> dict:
    properties = torch.cuda.get_device_properties(0)
    return dict(host=os.uname().nodename, card=card,
                device=torch.cuda.get_device_name(0),
                card_gib=round(properties.total_memory / 2**30, 2),
                torch=torch.__version__,
                triton=__import__("triton").__version__,
                versions=_versions())


def engine_bindings() -> dict:
    """Hash-bind every engine this lane times, before measuring."""
    return dict(
        k15_composition_sha256=digest(
            ROOT / "k15_composition.py"),
        k18_short_conv_sha256=digest(ROOT / "k18_short_conv.py"),
        matched_arms_sha256=digest(ROOT / "matched_arms.py"),
        velocity_ladder_sha256=digest(ROOT / "velocity_ladder.py"),
        external_gdn2_sha256=digest(REPO / "ndm/models/external_gdn2.py"),
        k18_gates_receipt_sha256=digest(RECEIPTS / "k18_gates.json"),
        lattice_registration_sha256=L.flagship_identity()["lattice_sha256"])


def dump(out_path: str, report: dict) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(report, handle, indent=1)
        handle.write("\n")


def accumulate(out_path: str, key: str, payload: dict) -> None:
    data = {}
    path = Path(out_path)
    if path.exists():
        data = json.loads(path.read_text())
    data[key] = payload
    dump(out_path, data)


def spread(samples_ms: list[float]) -> dict:
    """Median-not-mean timing stats, CUDA-synced per iteration."""
    ordered = sorted(samples_ms)
    quantiles = statistics.quantiles(ordered, n=4) if len(ordered) >= 4 else ordered
    median = statistics.median(ordered)
    return dict(iterations=len(ordered), median_ms=round(median, 3),
                p25_ms=round(quantiles[0], 3), p75_ms=round(quantiles[-1], 3),
                min_ms=round(min(ordered), 3), max_ms=round(max(ordered), 3),
                mean_ms=round(statistics.fmean(ordered), 3),
                std_over_median_pct=round(
                    100.0 * (statistics.pstdev(ordered) / median
                             if median else 0.0), 2))


def make_stream():
    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    return stream


# ------------------------------------------------- arm 1: per-(layer, window)


def capture_inputs(model, machine: str, stream, layers, microbatch: int,
                   width: int) -> dict:
    """Drive ONE real microbatch group through the machine and capture each
    measured layer's own mixer/MLP input activations (hooks only; no engine
    code is touched)."""
    captured: dict[str, dict] = {f"layer_{i}": {} for i in layers}
    index_of = {id(b): i for i, b in enumerate(model.blocks)} \
        if machine == "ours" else \
        {id(b): i for i, b in enumerate(model.model.blocks)}

    def hook(name):
        def register(module, args):
            i = index_of.get(id(_parent_of(module)))
            if i in layers:
                captured[f"layer_{i}"][name] = args[0].detach().clone()
        return register

    def _parent_of(module):
        return parents[id(module)]

    parents = {}
    handles = []
    if machine == "ours":
        for i, block in enumerate(model.blocks):
            parents[id(block.mixer)] = block
            parents[id(block.mlp)] = block
            handles.append(block.mixer.register_forward_pre_hook(hook("mixer")))
            handles.append(block.mlp.register_forward_pre_hook(hook("mlp")))
        forward = model.loss
    else:
        for i, block in enumerate(model.model.blocks):
            parents[id(block.layer.gdn2)] = block
            parents[id(block.layer.mlp)] = block
            handles.append(
                block.layer.gdn2.register_forward_pre_hook(hook("mixer")))
            handles.append(block.layer.mlp.register_forward_pre_hook(hook("mlp")))
        forward = model.loss
    rows = next(iter(stream.batches("train")))
    n = sum(len(r[3]) for r in rows)
    x, y, _ = next(iter(grouped(rows, microbatch)))
    with torch.no_grad():
        loss = forward(x, y)
    for handle in handles:
        handle.remove()
    for i in layers:
        for name in ("mixer", "mlp"):
            tensor = captured[f"layer_{i}"][name]
            want = (microbatch, 2048, width)
            if tuple(tensor.shape) != want:
                raise ValueError(f"captured {machine} {name} input at layer {i} "
                                 f"has shape {tuple(tensor.shape)}, want {want}")
    return captured, dict(batch_rows=len(rows), targets=n,
                          width=width,
                          capture_loss=float(loss))


def time_module(fn, x, warmup: int, iters: int) -> dict:
    """Forward and backward wall of one sublayer at the registered shape.

    Forward: the real training form (autocast bf16, graph built).  Backward:
    a fresh forward outside the timed window, then ``out.pow(2).sum()``
    backward into BOTH the parameters and the input (the input carries
    ``requires_grad`` so the measured adjoint includes the dx the real model
    would owe) -- CUDA-synced per iteration, medians not means.
    """
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fn(x)
    torch.cuda.synchronize()
    fwd = []
    for _ in range(iters):
        torch.cuda.synchronize()
        started = time.monotonic()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(x)
        torch.cuda.synchronize()
        fwd.append((time.monotonic() - started) * 1000.0)
        del out

    xin = x.detach().clone().requires_grad_(True)
    for _ in range(warmup):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(xin)
        out.pow(2).sum().backward()
    torch.cuda.synchronize()
    bwd = []
    for _ in range(iters):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fn(xin)
        torch.cuda.synchronize()
        started = time.monotonic()
        out.pow(2).sum().backward()
        torch.cuda.synchronize()
        bwd.append((time.monotonic() - started) * 1000.0)
        del out
    return dict(forward=spread(fwd), backward=spread(bwd),
                reduction="out.pow(2).sum()", dx_included=True)


def per_layer_ours(stream, warmup: int, iters: int, layers):
    """The committed K18 machine (composed mixer, conv included), verbatim."""
    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER), SEED)
    model = convert_model_to_composition(model)
    model = convert_model_to_short_conv(model, birth="identity")
    model = model.cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != CONV_MACHINE_PARAMETERS:
        raise ValueError(f"the conv machine's parameters must be the "
                         f"registration's {CONV_MACHINE_PARAMETERS}; got "
                         f"{parameters}")
    captured, cap = capture_inputs(model, "ours", stream, layers,
                                   optimizer_config["microbatch"],
                                   model.shape.width)
    rows = {}
    for i in layers:
        mixer = model.blocks[i].mixer
        mlp = model.blocks[i].mlp
        rows[f"layer_{i}"] = dict(
            mixer=time_module(mixer, captured[f"layer_{i}"]["mixer"],
                              warmup, iters),
            mlp=time_module(mlp, captured[f"layer_{i}"]["mlp"],
                            warmup, iters))
        print(f"[per_layer/ours] layer {i}: mixer "
              f"fwd {rows[f'layer_{i}']['mixer']['forward']['median_ms']} ms "
              f"bwd {rows[f'layer_{i}']['mixer']['backward']['median_ms']} ms; "
              f"mlp fwd {rows[f'layer_{i}']['mlp']['forward']['median_ms']} ms "
              f"bwd {rows[f'layer_{i}']['mlp']['backward']['median_ms']} ms",
              flush=True)
    report = dict(
        machine="ours: the committed K18 conv machine (the K15 composed "
                "mixer + the K18 identity-born short conv), the K18 arm "
                "build verbatim; mixers and MLPs imported, never modified",
        width_note="the committed machines do NOT share a width: ours is "
                   "576 (the matrix lane's registered backbone), GDN2's "
                   "committed 150M arm is 640 (part8_surface.gdn_config's "
                   "own capacity solve) -- each mixer is timed at its own "
                   "committed build's width, same batch/window regime; "
                   "engines are never modified to force a match",
        parameters=parameters, summary=summary,
        build="velocity_ladder._build_k10(TUNED_WINNER) + "
              "convert_model_to_composition + convert_model_to_short_conv("
              "birth='identity')",
        optimizer_side_note="per-layer probes carry no optimizer; the "
                            "optimizer is arm 2's own term",
        capture=cap, unit=UNIT, warmup_iters=warmup, measured_iters=iters,
        layers=rows)
    del model, captured
    torch.cuda.empty_cache()
    return report


def per_layer_gdn2(stream, warmup: int, iters: int, layers):
    """GDN2's committed arm's mixer, the registered build verbatim."""
    model, optimizer_config, summary = matched_arms.build_gdn2(SEED)
    model = model.cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    captured, cap = capture_inputs(model, "gdn2", stream, layers,
                                   optimizer_config["microbatch"],
                                   int(model.model.width))
    rows = {}
    for i in layers:
        core = model.model.blocks[i].layer.gdn2
        mlp = model.model.blocks[i].layer.mlp

        def mixer_fn(x, core=core):
            return core(x)[0]

        rows[f"layer_{i}"] = dict(
            mixer=time_module(mixer_fn, captured[f"layer_{i}"]["mixer"],
                              warmup, iters),
            mlp=time_module(mlp, captured[f"layer_{i}"]["mlp"],
                            warmup, iters))
        print(f"[per_layer/gdn2] layer {i}: mixer "
              f"fwd {rows[f'layer_{i}']['mixer']['forward']['median_ms']} ms "
              f"bwd {rows[f'layer_{i}']['mixer']['backward']['median_ms']} ms; "
              f"mlp fwd {rows[f'layer_{i}']['mlp']['forward']['median_ms']} ms "
              f"bwd {rows[f'layer_{i}']['mlp']['backward']['median_ms']} ms",
              flush=True)
    report = dict(
        machine="gdn2: the registered lattice_gdn2 arm's own mixer (the "
                "external FLA GatedDeltaNet2 core, short conv included) + "
                "its committed SwiGLU MLP, the registered build verbatim "
                "(width 640, part8_surface.gdn_config's own solve)",
        parameters=parameters, summary=summary,
        build="matched_arms.build_gdn2 (the lattice_gdn2 nomination verbatim)",
        capture=cap, unit=UNIT, warmup_iters=warmup, measured_iters=iters,
        layers=rows)
    del model, captured
    torch.cuda.empty_cache()
    return report


# ------------------------------------------ arm 2: end-to-end step walls


def steps_ours(stream, warmup_steps: int, measure_steps: int) -> dict:
    """The K18 machine under the banked probe protocol's own step body
    (``protocol_correction._train_corrected``'s do_step: the honest clock,
    HybridMuonX45, the K18 arm's optimizer split).  The corrected protocol's
    EMA update sits OUTSIDE that clock in the committed loop, so it is
    excluded here too."""
    model, optimizer_config, summary = _build_k10(dict(TUNED_WINNER), SEED)
    model = convert_model_to_composition(model)
    model = convert_model_to_short_conv(model, birth="identity")
    model = model.cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != CONV_MACHINE_PARAMETERS:
        raise ValueError(f"the conv machine's parameters must be the "
                         f"registration's {CONV_MACHINE_PARAMETERS}; got "
                         f"{parameters}")
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_k18(model, False)
    peak_lr = INHERITED_LR_MULTIPLIER * TUNED_ADAMW_LR
    optimizer = HybridMuonX45(
        muon_params, adamw_params, lr=peak_lr, betas=(0.9, BETA2),
        weight_decay=TUNED_WINNER["weight_decay"])
    warmup = optimizer_config["warmup_steps"]
    batches = iter(stream.batches("train"))
    rows = []
    for i in range(warmup_steps + measure_steps):
        torch.cuda.synchronize()
        started = time.monotonic()
        batch_rows = next(batches)
        n = sum(len(r[3]) for r in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        lr = peak_lr * min(1., (i + 1) / max(1, warmup))
        for group in optimizer.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for x, y, valid in grouped(batch_rows, MICROBATCH):
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP,
                                       error_if_nonfinite=True)
        torch.cuda.synchronize()
        opt_started = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        opt_s = time.monotonic() - opt_started
        wall = time.monotonic() - started
        rows.append(dict(step=i + 1, wall_s=round(wall, 3),
                         opt_s=round(opt_s, 3), targets=n,
                         loss=round(loss_sum, 6)))
        print(f"[steps/ours] step {i + 1}: {wall:.2f} s "
              f"(opt {opt_s:.2f} s), {n} targets", flush=True)
    measured = rows[warmup_steps:]
    report = dict(
        machine="ours: the committed K18 conv machine", parameters=parameters,
        optimizer="bakeoff_optimizers.HybridMuonX45 (the banked protocol's "
                  "optimizer, the K18 arm's own split: matrices on Muon, the "
                  "p3 gate tensors and the conv kernels on AdamW)",
        peak_lr=peak_lr, warmup_steps=warmup, microbatch=MICROBATCH,
        windows_per_step=64, warmup=warmup_steps, measured=measure_steps,
        step_rows=rows,
        step_wall_s=spread([r["wall_s"] * 1000 for r in measured]),
        opt_s=spread([r["opt_s"] * 1000 for r in measured]),
        targets_per_step=spread([float(r["targets"]) for r in measured]),
        clock="the committed probe loop's honest clock: sync -> stream read "
              "-> zero_grad -> lr -> every microbatch's fwd/bwd -> clip -> "
              "optimizer.step -> sync; EMA excluded (outside that clock in "
              "the committed loop)",
        recorded_reference=RECORDED["ours_probe_step180_active_seconds"] /
        RECORDED["probe_steps"])
    del model, optimizer
    torch.cuda.empty_cache()
    return report


def steps_gdn2(stream, warmup_steps: int, measure_steps: int) -> dict:
    """GDN2's committed arm under its own banked loop (matched_arms.train's
    gdn2 body verbatim: fused AdamW at the registered nomination's knobs)."""
    model, optimizer_config, summary = matched_arms.build_gdn2(SEED)
    model = model.cuda().train()
    parameters = sum(p.numel() for p in model.parameters())
    microbatch = optimizer_config["microbatch"]
    warmup = optimizer_config["warmup_steps"]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    rows = []
    for i in range(warmup_steps + measure_steps):
        torch.cuda.synchronize()
        started = time.monotonic()
        batch_rows = next(batches)
        n = sum(len(r[3]) for r in batch_rows)
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = optimizer_config["lr"] * min(
                1., (i + 1) / max(1, warmup))
        loss_sum = 0.0
        for x, y, valid in grouped(batch_rows, microbatch):
            loss = model.loss(x, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            (loss * (valid / n)).backward()
            loss_sum += float(loss.detach()) * valid / n
        torch.cuda.synchronize()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                       error_if_nonfinite=True)
        opt_started = time.monotonic()
        optimizer.step()
        torch.cuda.synchronize()
        opt_s = time.monotonic() - opt_started
        wall = time.monotonic() - started
        rows.append(dict(step=i + 1, wall_s=round(wall, 3),
                         opt_s=round(opt_s, 3), targets=n,
                         loss=round(loss_sum, 6)))
        print(f"[steps/gdn2] step {i + 1}: {wall:.2f} s "
              f"(opt {opt_s:.2f} s), {n} targets", flush=True)
    measured = rows[warmup_steps:]
    report = dict(
        machine="gdn2: the registered lattice_gdn2 arm",
        parameters=parameters,
        optimizer="fused AdamW at the registered nomination's knobs "
                  "(matched_arms.train's gdn2 branch verbatim)",
        microbatch=microbatch, warmup_steps=warmup,
        windows_per_step=64, warmup=warmup_steps, measured=measure_steps,
        step_rows=rows,
        step_wall_s=spread([r["wall_s"] * 1000 for r in measured]),
        opt_s=spread([r["opt_s"] * 1000 for r in measured]),
        targets_per_step=spread([float(r["targets"]) for r in measured]),
        clock="the committed loop's honest clock (matched_arms.train: sync -> "
              "stream read -> zero_grad -> lr -> fwd/bwd -> clip -> "
              "optimizer.step -> sync)",
        recorded_reference=RECORDED["gdn2_probe_step180_active_seconds"] /
        RECORDED["probe_steps"])
    del model, optimizer
    torch.cuda.empty_cache()
    return report


# ------------------------------------- arms 3 + 4: projection and verdict


def machine_medians(per_layer: dict) -> dict:
    mixer_fwd = [row["mixer"]["forward"]["median_ms"]
                 for row in per_layer["layers"].values()]
    mixer_bwd = [row["mixer"]["backward"]["median_ms"]
                 for row in per_layer["layers"].values()]
    mlp_fwd = [row["mlp"]["forward"]["median_ms"]
               for row in per_layer["layers"].values()]
    mlp_bwd = [row["mlp"]["backward"]["median_ms"]
               for row in per_layer["layers"].values()]
    return dict(
        mixer_forward_ms=round(statistics.median(mixer_fwd), 3),
        mixer_backward_ms=round(statistics.median(mixer_bwd), 3),
        mixer_fwd_bwd_ms=round(statistics.median(
            [f + b for f, b in zip(mixer_fwd, mixer_bwd)]), 3),
        mixer_fwd_bwd_spread_pct=round(100.0 * (
            (max(f + b for f, b in zip(mixer_fwd, mixer_bwd))
             - min(f + b for f, b in zip(mixer_fwd, mixer_bwd)))
            / (2 * statistics.median([f + b for f, b in zip(mixer_fwd, mixer_bwd)])
               )), 2),
        mlp_forward_ms=round(statistics.median(mlp_fwd), 3),
        mlp_backward_ms=round(statistics.median(mlp_bwd), 3),
        mlp_fwd_bwd_ms=round(statistics.median(
            [f + b for f, b in zip(mlp_fwd, mlp_bwd)]), 3),
        layers_measured=len(mixer_fwd))


def projection(steps: dict, per_layer_ours_row: dict,
               per_layer_gdn2_row: dict) -> dict:
    ours = machine_medians(per_layer_ours_row)
    gdn2 = machine_medians(per_layer_gdn2_row)
    t_g = steps["gdn2"]["step_wall_s"]["median_ms"] / 1000.0
    t_e = steps["ours"]["step_wall_s"]["median_ms"] / 1000.0
    o_g = steps["gdn2"]["opt_s"]["median_ms"] / 1000.0
    o_e = steps["ours"]["opt_s"]["median_ms"] / 1000.0
    delta_mixer_ms = ours["mixer_fwd_bwd_ms"] - gdn2["mixer_fwd_bwd_ms"]
    delta_mlp_ms = ours["mlp_fwd_bwd_ms"] - gdn2["mlp_fwd_bwd_ms"]

    def row(slots: int) -> dict:
        mixer_add = slots * GROUPS_PER_STEP * delta_mixer_ms / 1000.0
        layer_add = slots * GROUPS_PER_STEP * (delta_mixer_ms + delta_mlp_ms) / 1000.0
        out = dict(slot_layers=slots, dense_layers=DEPTH - slots)
        for label, add, opt_label, opt_term in (
                ("mixer_swap_gdn2_adamw", mixer_add, "gdn2's registered AdamW", 0.0),
                ("mixer_swap_banked_muon", mixer_add,
                 "the banked protocol's Muon (measured on our machine)",
                 o_e - o_g),
                ("layer_swap_gdn2_adamw", layer_add, "gdn2's registered AdamW", 0.0),
                ("layer_swap_banked_muon", layer_add,
                 "the banked protocol's Muon (measured on our machine)",
                 o_e - o_g)):
            step_s = t_g + add + opt_term
            out[label] = dict(
                step_s=round(step_s, 3),
                wall_ratio_vs_pure_gdn2=round(step_s / t_g, 3),
                exposure_pct_of_gdn2_in_2700s=round(100.0 * t_g / step_s, 1),
                added_by_slots_s=round(add, 3),
                optimizer_term_s=round(opt_term, 3),
                optimizer=opt_label)
        return out

    table = {str(s): row(s) for s in RATIOS}
    calibration = dict(
        note="the projection model's own calibration: the 24-of-24 row "
             "reconstructs the pure-ours step from the measured pure-GDN2 "
             "step plus 24x8 per-(layer, window-group) swap deltas and the "
             "measured optimizer term -- compared against the measured "
             "pure-ours step.  The residual is the backbone/embedding/head "
             "difference and the measurement spread, stated not hidden "
             "(the two machines' backbones differ by the capacity-matching "
             "deconfound: ours carries the wider MLP on every layer, which "
             "the layer_swap delta includes and the mixer_swap delta does "
             "not).",
        measured_ours_step_s=round(t_e, 3),
        predicted_ours_step_s_layer_swap=table["24"]["layer_swap_banked_muon"][
            "step_s"],
        predicted_ours_step_s_mixer_swap=table["24"]["mixer_swap_banked_muon"][
            "step_s"],
        recorded_banked_ours_step_s=round(
            RECORDED["ours_probe_step180_active_seconds"]
            / RECORDED["probe_steps"], 3),
        recorded_banked_gdn2_step_s=round(
            RECORDED["gdn2_probe_step180_active_seconds"]
            / RECORDED["probe_steps"], 3))
    return dict(
        unit=UNIT, groups_per_step=GROUPS_PER_STEP, depth=DEPTH,
        measured_step_costs=dict(
            pure_gdn2_s=round(t_g, 3), pure_ours_s=round(t_e, 3),
            gdn2_optimizer_s=round(o_g, 3), ours_optimizer_s=round(o_e, 3)),
        per_layer_medians=dict(ours=ours, gdn2=gdn2),
        per_layer_delta_ms=dict(
            mixer_fwd_bwd=round(delta_mixer_ms, 3),
            mixer_plus_mlp_fwd_bwd=round(delta_mixer_ms + delta_mlp_ms, 3)),
        projection_model="T_hybrid(s) = T_pure_gdn2 + s x 8 x delta_per_layer"
                         " [+ optimizer term when the hybrid runs the banked "
                         "protocol's Muon instead of GDN2's registered "
                         "AdamW] -- computed from measured per-(layer, "
                         "window-group) costs; no hybrid was built or run",
        ratios=table, calibration=calibration,
        time_exposure_asymmetry=dict(
            probe_protocol="matched-TARGETS (20,022,866 bytes): both banked "
                           "rows trained the SAME bytes in DIFFERENT "
                           "wall-clocks",
            ours_recorded_s=RECORDED["ours_probe_step180_active_seconds"],
            gdn2_recorded_s=RECORDED["gdn2_probe_step180_active_seconds"],
            board_protocol="matched-TIME (2700 s): the faster machine sees "
                           "more bytes; the hybrid's exposure fraction is "
                           "the step-cost ratio T_gdn2/T_hybrid"))


def verdict(proj: dict) -> dict:
    """The feasibility statement per ratio, in the board registration's own
    units.  Scale (the task's own pre-state): ~1.25x wall-clock viable,
    ~2.1x board-fatal at parity quality."""
    t_g = proj["measured_step_costs"]["pure_gdn2_s"]
    rows = []
    for s in (3, 4, 6, 8):
        entry = proj["ratios"][str(s)]
        rows.append(dict(
            ratio=f"{s}-of-24 slot layers",
            mixer_swap=dict(
                step_s=entry["mixer_swap_banked_muon"]["step_s"],
                wall_ratio=entry["mixer_swap_banked_muon"][
                    "wall_ratio_vs_pure_gdn2"],
                exposure_pct=entry["mixer_swap_banked_muon"][
                    "exposure_pct_of_gdn2_in_2700s"]),
            layer_swap=dict(
                step_s=entry["layer_swap_banked_muon"]["step_s"],
                wall_ratio=entry["layer_swap_banked_muon"][
                    "wall_ratio_vs_pure_gdn2"],
                exposure_pct=entry["layer_swap_banked_muon"][
                    "exposure_pct_of_gdn2_in_2700s"]),
            statement=(f"at {s}-of-24 the projected hybrid step is "
                       f"{entry['mixer_swap_banked_muon']['step_s']:.2f} s "
                       f"(mixer swap) / "
                       f"{entry['layer_swap_banked_muon']['step_s']:.2f} s "
                       f"(layer swap) against pure-GDN2's {t_g:.2f} s -- "
                       f"the hybrid sees "
                       f"{entry['mixer_swap_banked_muon']['exposure_pct_of_gdn2_in_2700s']:.0f}% / "
                       f"{entry['layer_swap_banked_muon']['exposure_pct_of_gdn2_in_2700s']:.0f}% "
                       f"of GDN2's exposure in 2700 s")))
    return dict(rows=rows,
                scale="the task's own pre-state: ~1.25x wall-clock viable, "
                      "~2.1x board-fatal at parity quality; no new "
                      "threshold is invented here",
                no_viable_ratio=bool(all(
                    r["mixer_swap"]["wall_ratio"] > 2.0 for r in rows)))


# ------------------------------------------------------------------- driver


def run_stage(out: str, key: str, fn):
    try:
        payload = fn()
    except Exception:
        failure = dict(error=traceback.format_exc()[-4000:])
        print(f"[{key}] FAILED:\n{failure['error']}", flush=True)
        accumulate(out, key + "_failed", failure)
        return None
    accumulate(out, key, payload)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",
                        default=str(RECEIPTS / "k20_hybrid_velocity.json"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--warmup-iters", type=int, default=25)
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--step-warmup", type=int, default=3)
    parser.add_argument("--step-measure", type=int, default=5)
    parser.add_argument("--layers", default="0,6,12,18,23",
                        help="which layers' own captured activations to time")
    args = parser.parse_args()
    assert args.seed == SEED, "the standing seed is 3401"
    card = _require_lease()
    configure_production()
    torch.set_num_threads(4)
    assert torch.cuda.is_available(), "K20 needs CUDA"
    layers = [int(x) for x in args.layers.split(",")]

    receipt = dict(
        probe="k20_hybrid_velocity",
        question="which slot/dense interleave ratios are viable under the "
                 "matched-TIME board protocol -- measured, not estimated",
        pre_state="no directional prediction; the measurement IS the result",
        unit=UNIT,
        seed=args.seed,
        session=session_meta(card), git=git_state(),
        bindings=engine_bindings(), recorded=RECORDED)
    dump(args.out, receipt)
    stream = make_stream()

    ours_row = run_stage(args.out, "per_layer_ours", lambda: per_layer_ours(
        stream, args.warmup_iters, args.iters, layers))
    gdn2_row = run_stage(args.out, "per_layer_gdn2", lambda: per_layer_gdn2(
        stream, args.warmup_iters, args.iters, layers))
    ours_steps = run_stage(args.out, "steps_ours", lambda: steps_ours(
        stream, args.step_warmup, args.step_measure))
    gdn2_steps = run_stage(args.out, "steps_gdn2", lambda: steps_gdn2(
        stream, args.step_warmup, args.step_measure))

    if None not in (ours_row, gdn2_row, ours_steps, gdn2_steps):
        proj = projection(dict(ours=ours_steps, gdn2=gdn2_steps),
                          ours_row, gdn2_row)
        accumulate(args.out, "projection", proj)
        accumulate(args.out, "verdict", verdict(proj))
        print(json.dumps(proj["per_layer_medians"], indent=1), flush=True)
        print(json.dumps(proj["measured_step_costs"], indent=1), flush=True)
        print(json.dumps(verdict(proj)["rows"], indent=1), flush=True)
    else:
        print("a stage failed -- the projection is NOT computed; the "
              "receipt carries the failure", flush=True)
    print(f"receipt written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
