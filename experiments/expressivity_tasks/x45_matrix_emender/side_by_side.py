"""SIDE-BY-SIDE GAP DISSECTION: why gdn2 (1.5837) beats the tuned winner
(1.8227) by 0.2390 nats at the matched-exposure probe protocol.

DIAGNOSIS ONLY.  No model changes, no fixes; ranked causes with receipts
(`RESULT_SIDE_BY_SIDE_0239.md`).  ``k10_*`` / ``k6_*`` / gdn2 arm code, the
PREREGISTRATIONs and ``lattice.py`` are NOT edited: the banked builders are
imported verbatim (``matched_arms.build_gdn2``, ``velocity_ladder._build_k10``),
as are the mechanism instruments (``k10_probe_separated_keys.probe_diagnostics``).

Both arms run at the EXACT probe protocol (20,000,000-target matched-exposure
rule -> the 20,022,866 landing, seed 3401, the registered flagship stream, the
document-disjoint calibration eval at 7 windows / 14,336 targets, the honest
clock), with dense instrumentation:

* **checkpoint evals at steps 1/5/10/20/40/.../180 (both arms, the same
  checkpoints) -> the gap-vs-training-phase curve**, the headline artifact.
  The model state is snapshotted (CPU ``state_dict`` clones) at each
  checkpoint step INSIDE the training loop; every evaluation and telemetry
  pass runs on restored snapshots AFTER training, so the training path is
  unperturbed by construction.  The protocol-equivalence proof is the ladder
  row's own template, asserted in-code: every training-log row must be
  bit-identical to the banked run's row, and the step-180 checkpoint's eval
  must reproduce the banked ladder number bit-for-bit.
* **per-layer store telemetry for our arm at every checkpoint** (the banked
  channels: op rates, occupancy, read entropy, identity-rewrite share, the
  key-collapse watch) plus the DECAY gate's per-layer mean (a non-invasive
  forward hook re-computing the trained walk's own gate from the mixer's
  projections).
* **gdn2 telemetry where its arm code exposes it**: the per-checkpoint eval,
  the train-loss curve, and its forward's own gate diagnostics (the
  channel-wise log-decay ``g``, erase gate ``b``, write gate ``w``, read from
  its own projection modules in a forward hook; its source is untouched).
  Its recurrent state is internal to the FLA chunk kernel and not exposed by
  the wrapper -- stated honestly; the gates are the state's own write/forget
  coefficients, so their telemetry IS the state-utilization receipt.
* **the C4 read-mask probes** (a diagnostic receipt only, no model change):
  at the winner's final checkpoint, one layer at a time, the layer's mixer
  output -- exactly its store read, projected -- is zeroed for the eval pass
  and the calibration NLL re-measured; plus one all-layers-masked cell (the
  storeless floor).
* **the C1 bounded fair-tuning extension** is driven by the banked machinery
  (``run_velocity_ladder_leased.sh search`` with the pre-stated knobs,
  RESULT_SIDE_BY_SIDE_0239.md 0.1); it is NOT re-implemented here.

Subcommands:

``k10-armed``    the K10 arm (tuned winner or transferred config) at the probe
                protocol with the checkpoint/telemetry/read-mask receipts.
``gdn2-armed``  the gdn2 arm at the probe protocol with the checkpoint/eval
                receipts and the gate telemetry.
``assemble``    reads the armed receipts + the C1 extension receipt and emits
                the analysis summary (``receipts/side_by_side_summary.json``).

Usage (see ``run_side_by_side_leased.sh`` for the detached leased driver):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.side_by_side k10-armed \\
        --config CONFIG_JSON --label k10_tuned_armed \\
        --output side_by_side_runs/k10_tuned_armed_seed3401 \\
        --banked-log velocity_runs/k10_tuned_confirm_seed3401/training.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# The probe subprocess sets these BEFORE importing torch (the matched-arms
# discipline; configure_production() fail-closes on a missing allocator knob).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
X45_RECEIPTS = ROOT / "receipts"
RUNS = ROOT / "side_by_side_runs"

SEED = 3401
EXPOSURE_TARGETS = 20_000_000
PARAMETERS_AUDITED = 149_981_808
# The checkpoint plan: dense early (the C2 test lives at steps 1-20), then
# every 20 steps to the landing.  Both arms see byte-identical windows at the
# same step (same stream, same batch composition), so the checkpoints match.
CHECKPOINT_STEPS = (1, 5, 10, 20, 40, 60, 80, 100, 120, 140, 160, 180)

# The pre-stated C1 extension's incumbent (velocity_k10_cmaes_result.json's
# best, eval 4) and the transferred config (cmaes_search.TRANSFERRED) -- the
# two K10 configurations this lane arms.
TUNED_INCUMBENT = {
    "lr": 0.00033427141575371284, "weight_decay": 0.17422187309224696,
    "warmup_steps": 45, "read_scale": 2.7870458737509636,
    "tau_match": 0.45, "tau_pop": 0.1, "tau_merge": 0.85,
}
TRANSFERRED_CONFIG = {
    "lr": 5.420185524740253e-4, "weight_decay": 0.20059436743465126,
    "warmup_steps": 69, "read_scale": 2.0,
    "tau_match": 0.45, "tau_pop": 0.10, "tau_merge": 0.85,
}

# The banked ladder rows this lane must reproduce bit-for-bit (the
# protocol-equivalence proof, the GDN2 row's own template).
BANKED = {
    "gdn2_probe": dict(
        nll=1.5836738858904158,
        training_jsonl=str(ROOT / "velocity_runs/gdn2_probe_seed3401/training.jsonl"),
        run=str(ROOT / "velocity_runs/gdn2_probe_seed3401")),
    "k10_tuned_confirm": dict(
        nll=1.822737353188651,
        training_jsonl=str(ROOT / "velocity_runs/k10_tuned_confirm_seed3401/training.jsonl"),
        run=str(ROOT / "velocity_runs/k10_tuned_confirm_seed3401")),
    "k10_untuned_probe": dict(
        nll=1.841756020273481,
        training_jsonl=str(ROOT / "k10_runs/separated_keys_seed3401/training.jsonl"),
        run=str(ROOT / "k10_runs/separated_keys_seed3401")),
}


def _require_lease() -> str:
    card = os.environ.get("GPU_LEASE_HELD")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if card is None or visible is None:
        raise ValueError("GPU lease: GPU_LEASE_HELD / CUDA_VISIBLE_DEVICES required")
    if str(card) != str(visible):
        raise ValueError(f"lease mismatch: {card} vs CUDA_VISIBLE_DEVICES {visible}")
    return card


def _calibration_windows(stream, windows: int):
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    rows = [row for row in stream.windows("calibration") if len(row[2]) == L.CONTEXT]
    if len(rows) < windows:
        raise ValueError(f"calibration split has {len(rows)} full windows; "
                         f"the probe needs {windows}")
    return rows[:windows]


def _snapshot(model) -> dict:
    """A CPU clone of the whole state at a checkpoint step (the training loop's
    ONLY addition over the banked loops; every eval/telemetry pass later runs
    on restored snapshots, so the training path is unperturbed by construction)."""
    import torch
    _ = torch  # (imported for clarity: state_dict tensors are detached views)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _assert_banked_equivalence(rows: list, banked_log: str) -> dict:
    """The ladder row's protocol-equivalence proof: every training-log row
    bit-identical (bytes AND loss) to the banked run's own row.  Fails loudly."""
    banked = [json.loads(line) for line in Path(banked_log).read_text()
              .splitlines() if line.strip()]
    if len(banked) != len(rows):
        raise ValueError(f"row count mismatch: {len(rows)} vs banked {len(banked)}")
    for mine, theirs in zip(rows, banked):
        if (mine["step"] != theirs["step"] or mine["bytes"] != theirs["bytes"]
                or mine["loss"] != theirs["loss"]):
            raise ValueError(
                f"protocol equivalence broken at step {mine['step']}: "
                f"mine(bytes={mine['bytes']}, loss={mine['loss']!r}) vs "
                f"banked(bytes={theirs['bytes']}, loss={theirs['loss']!r})")
    return dict(banked_log=str(banked_log), rows=len(rows), bit_identical=True)


def _write_checkpoints(out: Path, checkpoint_rows: list):
    with (out / "checkpoints.jsonl").open("w", buffering=1) as f:
        for row in checkpoint_rows:
            f.write(json.dumps(row) + "\n")


# ------------------------------------------------------- our arm's instruments


def k10_decay_telemetry(model, stream, windows: int = 4) -> dict:
    """The DECAY gate's per-layer mean over the probe windows, from the trained
    walk's own projections (the same recomputation pattern the banked probe
    instruments use; a non-invasive forward hook, no code changes)."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    from experiments.expressivity_tasks.x45_matrix_emender.model import DECAY_FLOOR
    rows = _calibration_windows(stream, windows)
    acc: dict[int, list] = {}

    def make_hook(i):
        def hook(module, args, _output):
            x = args[0]
            with torch.no_grad():
                decay = torch.sigmoid(module.decay(x)).clamp(
                    min=DECAY_FLOOR, max=1.0 - 1e-6)
                acc.setdefault(i, []).append(float(decay.float().mean()))
        return hook

    handles = [block.mixer.register_forward_hook(make_hook(i))
               for i, block in enumerate(model.blocks)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                x, _y, _v = next(grouped([row], 1))
                model.features(x)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    per_layer = {str(i): sum(v) / len(v) for i, v in sorted(acc.items())}
    return dict(
        note=("the DECAY gate's per-token mean per layer (the store rows' "
              "survival factor per token), from the trained walk's own "
              "projections on the calibration probe windows"),
        mean_decay_per_layer=per_layer)


def k10_read_mask_sweep(model, stream, layers=None) -> dict:
    """The C4 diagnostic receipt: each layer's store read, masked at eval time.

    The K10 mixer's output IS its store read (``out(reads)`` -- there is no
    other path), so zeroing one layer's mixer output removes exactly that
    layer's store read from the eval pass.  One layer at a time, then the
    all-layers cell (the storeless floor).  Diagnostic only: the intervention
    is a forward hook on the trained, restored checkpoint; no weights, no
    code, no training change.  Stated caveat: zeroing layer L shifts the
    downstream input distribution, so the per-layer deltas are ablation
    effects, not additive marginal contributions.
    """
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    os.environ["X40_BATCHED_READS"] = "0"   # evaluation path is recorded-read only
    base = L.evaluate(model, stream, "matrix_ops", "calibration")["score"]
    n_layers = len(model.blocks)
    layers = list(range(n_layers)) if layers is None else list(layers)
    per_layer = {}
    for i in layers:
        handle = model.blocks[i].mixer.register_forward_hook(
            lambda _m, _a, out: torch.zeros_like(out))
        try:
            nll = L.evaluate(model, stream, "matrix_ops", "calibration")["score"]
        finally:
            handle.remove()
        per_layer[str(i)] = nll
        print(f"  read-mask layer {i}: nll {nll:.4f} (base {base:.4f})",
              flush=True)
    all_handles = [block.mixer.register_forward_hook(
        lambda _m, _a, out: torch.zeros_like(out)) for block in model.blocks]
    try:
        storeless = L.evaluate(model, stream, "matrix_ops", "calibration")["score"]
    finally:
        for handle in all_handles:
            handle.remove()
    return dict(
        note=("C4 diagnostic: one layer's mixer output (== its store read, "
              "projected) zeroed for the eval pass, one layer at a time, at "
              "the step-180 restored checkpoint; plus the all-layers cell "
              "(the storeless floor). Ablation effects, not marginals."),
        base_nll=base, masked_nll_per_layer=per_layer,
        all_layers_masked_nll=storeless,
        delta_per_layer={k: v - base for k, v in per_layer.items()},
        all_layers_delta=storeless - base)


# --------------------------------------------------- gdn2's gate instrument


def gdn2_gate_telemetry(model, stream, windows: int = 4) -> dict:
    """gdn2's own gate diagnostics, per layer, from its forward's own
    arithmetic: the channel-wise log-decay g (the state's per-token retention
    exp(g)), the channel-wise erase gate b and the write gate w -- recomputed
    in a forward hook from the GDN-2 core module's own projections (its source
    is NOT modified; the expressions are its forward's own lines).  The
    recurrent state itself is internal to the FLA chunk kernel and is not
    exposed by the arm wrapper; these gates are the state's write/forget
    coefficients, so their telemetry is the state-utilization receipt."""
    import torch
    import torch.nn.functional as F

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped
    rows = _calibration_windows(stream, windows)
    cores = [block.layer.gdn2.gdn2 for block in model.model.blocks]
    acc: dict[int, list] = {}

    def make_hook(i):
        def hook(module, args, _output):
            x = args[0]
            with torch.no_grad():
                # the layer's own gate arithmetic (gdn2.py forward, verbatim
                # expressions over the module's own projections/parameters)
                g = (-module.A_log.float().exp().repeat_interleave(module.head_k_dim)
                     * F.softplus(module.f_proj(x).float() + module.dt_bias))
                b = module.b_proj(x).sigmoid()
                w = module.w_proj(x).sigmoid()
                retention = g.exp()
                acc.setdefault(i, []).append(dict(
                    mean_retention=float(retention.float().mean()),
                    mean_erase_b=float(b.float().mean()),
                    mean_write_w=float(w.float().mean())))
        return hook

    handles = [core.register_forward_hook(make_hook(i))
               for i, core in enumerate(cores)]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for row in rows:
                x, y, _v = next(grouped([row], 1))
                model.loss(x, y)          # drives the same forward; loss unused
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    per_layer = {}
    for i in sorted(acc):
        entries = acc[i]
        mean_retention = sum(e["mean_retention"] for e in entries) / len(entries)
        per_layer[str(i)] = dict(
            mean_state_retention_exp_g=mean_retention,
            implied_memory_horizon_tokens=1.0 / max(1e-9, 1.0 - mean_retention),
            mean_erase_gate_b=sum(e["mean_erase_b"] for e in entries) / len(entries),
            mean_write_gate_w=sum(e["mean_write_w"] for e in entries) / len(entries))
    head_decay_base = {}
    for i, core in enumerate(cores):
        head_decay_base[str(i)] = [float(v) for v in
                                   core.A_log.detach().float().exp().cpu()]
    return dict(
        note=("gdn2's own gate telemetry per layer, averaged over the "
              "calibration probe windows: the channel-wise log-decay's "
              "retention exp(g) (the state's per-token survival factor; its "
              "reciprocal-of-complement is the implied memory horizon), the "
              "erase gate b and the write gate w -- the state's own "
              "write/forget coefficients, recomputed from the module's own "
              "projections in a non-invasive forward hook"),
        per_layer=per_layer, head_decay_base_exp_A=head_decay_base)


# ------------------------------------------------------------- the armed runs


def _train_with_snapshots_k10(model, optimizer_config, stream, targets, label,
                              log_path, checkpoint_steps):
    """velocity_ladder._train_to_targets' loop VERBATIM, with exactly one
    addition: the CPU state snapshot at each checkpoint step (after the row is
    logged; no eval, no telemetry, no RNG draw inside the loop)."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = seen = 0
    rows = []
    snapshots = {}
    diverged = False
    log = Path(log_path).open("w", buffering=1) if log_path else None
    try:
        while seen < targets:
            torch.cuda.synchronize()
            start = time.monotonic()
            try:
                batch_rows = next(batches)
            except StopIteration:
                raise RuntimeError("train split exhausted before the probe budget")
            n = sum(len(r[3]) for r in batch_rows)
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = optimizer_config["lr"] * min(
                    1., (step + 1) / max(1, optimizer_config["warmup_steps"]))
            loss_sum = 0.0
            for x, y, valid in grouped(batch_rows, optimizer_config["microbatch"]):
                loss = model.loss(x, y)
                if not torch.isfinite(loss):
                    diverged = True
                    break
                (loss * (valid / n)).backward()
                loss_sum += float(loss.detach()) * valid / n
            if diverged:
                elapsed += time.monotonic() - start
                return dict(status="diverged", step=step, seen=seen,
                            elapsed=elapsed, rows=rows, snapshots=snapshots)
            torch.cuda.synchronize()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.,
                                           error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            elapsed += time.monotonic() - start
            step += 1
            seen += n
            rows.append(dict(step=step, active_seconds=elapsed, bytes=seen,
                             loss=loss_sum))
            if log is not None:
                log.write(json.dumps(rows[-1]) + "\n")
            if step in checkpoint_steps:
                snapshots[step] = _snapshot(model)
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s {seen/1e6:.1f}MB "
                      f"loss {loss_sum:.4f}", flush=True)
    finally:
        if log is not None:
            log.close()
    return dict(status="completed", step=step, seen=seen, elapsed=elapsed,
                rows=rows, snapshots=snapshots)


def _train_with_snapshots_gdn2(model, optimizer_config, stream, targets, label,
                               log_path, checkpoint_steps):
    """matched_arms.train's gdn2 loop VERBATIM (the honest clock, the per-step
    log with time_ns), with exactly one addition: the CPU state snapshot at
    each checkpoint step."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale.byte_velocity import grouped

    microbatch = optimizer_config["microbatch"]
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=optimizer_config["lr"], betas=tuple(optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"], fused=True)
    batches = iter(stream.batches("train"))
    elapsed = 0.0
    step = targets_seen = 0
    log_rows = []
    snapshots = {}
    with Path(log_path).open("w", buffering=1) as log:
        while targets_seen < targets:
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
            targets_seen += n
            row = dict(step=step, active_seconds=elapsed, bytes=targets_seen,
                       loss=loss_sum, time_ns=time.time_ns())
            log.write(json.dumps(row) + "\n")
            log_rows.append(row)
            if step in checkpoint_steps:
                snapshots[step] = _snapshot(model)
            if step % 25 == 0:
                print(f"  {label} step {step} {elapsed:.0f}s "
                      f"{targets_seen/1e6:.1f}MB loss {loss_sum:.4f}", flush=True)
    return dict(status="completed", step=step, seen=targets_seen, elapsed=elapsed,
                rows=log_rows, snapshots=snapshots)


def _armed_common(stream, out: Path, config: dict, args, run_config: dict):
    """The shared receipt spine: start.json (before training) and the lease /
    stream / identity bindings, the banked arms-lane convention."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    card = _require_lease()
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns(),
        checkpoint_steps=list(args.checkpoint_steps),
        checkpoint_note=("state snapshots taken inside the training loop at "
                         "each checkpoint step; every eval/telemetry pass runs "
                         "on restored snapshots after training, so the "
                         "training path is unperturbed by construction")))
    return card


def k10_armed(args) -> dict:
    """The K10 arm (the tuned winner, or the transferred config) at the exact
    probe protocol with the checkpoint curve, the per-layer telemetry and the
    C4 read-mask sweep."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.k10_probe_separated_keys \
        import probe_diagnostics
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder import (
        _build_k10)

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    config = json.loads(args.config)
    label = args.label
    out = Path(args.output)
    model, optimizer_config, summary = _build_k10(config, args.seed)
    run_config = dict(
        probe="side_by_side k10-armed", arm="k10_separated_keys", label=label,
        seed=args.seed, exposure_targets=int(args.bytes),
        optimizer=optimizer_config, summary=summary, config=config,
        knobs_that_differ_from_transfer={
            k: dict(transferred=TRANSFERRED_CONFIG[k], searched=config[k])
            for k in config
            if k in TRANSFERRED_CONFIG and config[k] != TRANSFERRED_CONFIG[k]},
        identity=identity,
        bindings=dict(
            matched_arms.registration_bindings(),
            k10_preregistration=str(ROOT / "PREREGISTRATION_SEPARATED_KEYS.md"),
            k10_gates_receipt=str(X45_RECEIPTS / "k10_gates.json"),
            k10_banked_probe=str(X45_RECEIPTS / "k10_probe.json")),
        git=matched_arms.git_state(),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} targets "
                       "seen (~20M, the probe exposure); the honest elapsed "
                       "wall is reported first-class"))
    card = _armed_common(stream, out, config, args, run_config)
    print(json.dumps(dict(arm="k10_separated_keys", label=label, card=card,
                          seed=args.seed, parameters=PARAMETERS_AUDITED)),
          flush=True)

    run = _train_with_snapshots_k10(
        model, optimizer_config, stream, int(args.bytes), label,
        out / "training.jsonl", tuple(args.checkpoint_steps))
    if run["status"] != "completed":
        raise FloatingPointError(f"the armed run diverged at step "
                                  f"{run['step'] + 1}; no receipt is written")
    equivalence = (None if args.smoke
                  else _assert_banked_equivalence(run["rows"], args.banked_log))
    if equivalence is not None:
        print(f"  protocol equivalence: {equivalence['rows']} rows bit-identical "
              f"to {equivalence['banked_log']}", flush=True)

    # ---- the checkpoint pass (restored snapshots; training is over)
    rows_by_step = {r["step"]: r for r in run["rows"]}
    checkpoint_rows = []
    for step in args.checkpoint_steps:
        model.load_state_dict(run["snapshots"][step])
        diagnostics = probe_diagnostics(model, stream, windows=args.probe_windows)
        decay = k10_decay_telemetry(model, stream, windows=args.probe_windows)
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, "matrix_ops", "calibration")
        row = dict(step=step, bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"], eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"],
                   diagnostics=diagnostics, decay=decay)
        checkpoint_rows.append(row)
        print(f"  checkpoint step {step}: bytes {row['bytes']} "
              f"train {row['train_loss']:.4f} eval {row['eval_nll']:.4f}",
              flush=True)
    _write_checkpoints(out, checkpoint_rows)

    # ---- the final checkpoint's eval must reproduce the banked ladder number
    final_eval = checkpoint_rows[-1]["eval_nll"]
    banked_final = BANKED["k10_tuned_confirm" if label == "k10_tuned_armed"
                          else "k10_untuned_probe"]["nll"]
    if not args.smoke and final_eval != banked_final:
        raise ValueError(f"final-checkpoint eval {final_eval!r} != the banked "
                         f"ladder number {banked_final!r}")

    # ---- the C4 read-mask sweep at the final checkpoint
    model.load_state_dict(run["snapshots"][run["step"]])
    sweep = k10_read_mask_sweep(model, stream,
                                layers=(None if not args.smoke else [0, 1]))

    receipt = dict(
        probe="side_by_side k10-armed", arm="k10_separated_keys", label=label,
        seed=args.seed, exposure_bytes=run["seen"], exposure_steps=run["step"],
        train_seconds=round(run["elapsed"], 1),
        bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
        parameters=PARAMETERS_AUDITED, config=config,
        protocol_equivalence=(equivalence if equivalence is not None
                              else "smoke: skipped"),
        final_eval_nll=final_eval,
        banked_final_nll=banked_final,
        final_eval_bit_identical=bool(final_eval == banked_final),
        checkpoints=checkpoint_rows, read_mask_sweep=sweep,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        session=matched_arms.session_meta(card))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "exposure_bytes", "exposure_steps",
                       "final_eval_nll", "bytes_per_s")}), flush=True)
    return receipt


def gdn2_armed(args) -> dict:
    """The gdn2 arm at the exact probe protocol with the checkpoint curve and
    the gate/state telemetry."""
    import torch

    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    label = args.label
    out = Path(args.output)
    model, optimizer_config, summary = matched_arms.build_gdn2(args.seed)
    run_config = dict(
        probe="side_by_side gdn2-armed", arm="gdn2", label=label,
        seed=args.seed, exposure_targets=int(args.bytes),
        optimizer=optimizer_config, summary=summary, identity=identity,
        bindings=dict(
            matched_arms.registration_bindings()),
        git=matched_arms.git_state(),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} bytes "
                       "seen (the registration's twin rule); the honest "
                       "elapsed wall is reported first-class"))
    card = _armed_common(stream, out, {}, args, run_config)
    print(json.dumps(dict(arm="gdn2", label=label, card=card, seed=args.seed)),
          flush=True)

    run = _train_with_snapshots_gdn2(
        model, optimizer_config, stream, int(args.bytes), label,
        out / "training.jsonl", tuple(args.checkpoint_steps))
    equivalence = (None if args.smoke
                  else _assert_banked_equivalence(run["rows"], args.banked_log))
    if equivalence is not None:
        print(f"  protocol equivalence: {equivalence['rows']} rows bit-identical "
              f"to {equivalence['banked_log']}", flush=True)

    # ---- the checkpoint pass (restored snapshots; training is over)
    rows_by_step = {r["step"]: r for r in run["rows"]}
    checkpoint_rows = []
    for step in args.checkpoint_steps:
        model.load_state_dict(run["snapshots"][step])
        gates = gdn2_gate_telemetry(model, stream, windows=args.probe_windows)
        os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
        result = L.evaluate(model, stream, "gdn2", "calibration")
        row = dict(step=step, bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"], eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"], gates=gates)
        checkpoint_rows.append(row)
        print(f"  checkpoint step {step}: bytes {row['bytes']} "
              f"train {row['train_loss']:.4f} eval {row['eval_nll']:.4f}",
              flush=True)
    _write_checkpoints(out, checkpoint_rows)

    final_eval = checkpoint_rows[-1]["eval_nll"]
    banked_final = BANKED["gdn2_probe"]["nll"]
    if not args.smoke and final_eval != banked_final:
        raise ValueError(f"final-checkpoint eval {final_eval!r} != the banked "
                         f"ladder number {banked_final!r}")

    receipt = dict(
        probe="side_by_side gdn2-armed", arm="gdn2", label=label,
        seed=args.seed, exposure_bytes=run["seen"], exposure_steps=run["step"],
        train_seconds=round(run["elapsed"], 1),
        bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
        protocol_equivalence=(equivalence if equivalence is not None
                              else "smoke: skipped"),
        final_eval_nll=final_eval, banked_final_nll=banked_final,
        final_eval_bit_identical=bool(final_eval == banked_final),
        checkpoints=checkpoint_rows,
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        session=matched_arms.session_meta(card))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "exposure_bytes", "exposure_steps",
                       "final_eval_nll", "bytes_per_s")}), flush=True)
    return receipt


# --------------------------------------------------------------------- assemble


def assemble(args) -> dict:
    """Read the armed receipts and the C1 extension receipt; emit the analysis
    summary (the curve, the per-layer tables, the C1..C5 evidence pack) that
    RESULT_SIDE_BY_SIDE_0239.md cites."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    def load_checkpoints(name):
        path = RUNS / name / "checkpoints.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    tuned = load_checkpoints("k10_tuned_armed_seed3401")
    transferred = (load_checkpoints("k10_transferred_armed_seed3401")
                   if (RUNS / "k10_transferred_armed_seed3401").exists() else [])
    gdn2_rows = load_checkpoints("gdn2_armed_seed3401")
    curve = []
    for k_row, g_row in zip(tuned, gdn2_rows):
        assert k_row["step"] == g_row["step"]
        entry = dict(step=k_row["step"], bytes=k_row["bytes"],
                     gdn2_eval_nll=g_row["eval_nll"],
                     k10_tuned_eval_nll=k_row["eval_nll"],
                     gap=g_row["eval_nll"] - k_row["eval_nll"],
                     gdn2_train_loss=g_row["train_loss"],
                     k10_tuned_train_loss=k_row["train_loss"])
        if transferred:
            t_row = next(r for r in transferred if r["step"] == k_row["step"])
            entry["k10_transferred_eval_nll"] = t_row["eval_nll"]
            entry["tuned_minus_transferred"] = (k_row["eval_nll"]
                                                - t_row["eval_nll"])
        curve.append(entry)

    final = tuned[-1]
    summary = dict(
        probe="side_by_side assemble", curve=curve,
        final=dict(step=final["step"], gap=curve[-1]["gap"]),
        per_layer_final=dict(
            occupancy={k: v["occupancy_mean_live_slots"]
                       for k, v in final["diagnostics"]["per_layer"].items()},
            read_entropy={k: v["read_weight_entropy_nats"]
                          for k, v in final["diagnostics"]["per_layer"].items()},
            identity_share={k: v["identity_rewrite_share"]
                            for k, v in final["diagnostics"]["per_layer"].items()},
            mean_decay=final["decay"]["mean_decay_per_layer"]),
        read_mask=final and json.loads(
            (RUNS / "k10_tuned_armed_seed3401/result.json").read_text()
        )["read_mask_sweep"],
        gdn2_gates_final=gdn2_rows[-1]["gates"]["per_layer"],
    )
    gen2_path = X45_RECEIPTS / "velocity_k10_cmaes_result_gen2.json"
    if gen2_path.exists():
        gen2 = json.loads(gen2_path.read_text())
        banked_gen1 = json.loads(
            (X45_RECEIPTS / "velocity_k10_cmaes_result.json").read_text())
        summary["c1_extension"] = dict(
            gen2_result=dict(
                n_evaluations=gen2["n_evaluations"],
                best=gen2["best"],
                incumbent_trajectory=gen2["incumbent_trajectory"],
                settings=gen2["settings"]),
            gen1_incumbent=banked_gen1["best"])
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(final_gap=summary["final"],
                          curve_points=len(curve))), flush=True)
    return summary


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    k = sub.add_parser("k10-armed")
    k.add_argument("--config", required=True)
    k.add_argument("--label", default="k10_tuned_armed")
    k.add_argument("--seed", type=int, default=SEED)
    k.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    k.add_argument("--output", required=True)
    k.add_argument("--banked-log", required=True,
                   help="the banked run's training.jsonl this armed run must "
                        "reproduce bit-for-bit (the protocol-equivalence proof)")
    k.add_argument("--probe-windows", type=int, default=4)
    k.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    k.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, 2 probe windows, a "
                        "2-layer read-mask sweep, no banked assertions")
    g = sub.add_parser("gdn2-armed")
    g.add_argument("--label", default="gdn2_armed")
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    g.add_argument("--output", required=True)
    g.add_argument("--banked-log", required=True)
    g.add_argument("--probe-windows", type=int, default=4)
    g.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    g.add_argument("--smoke", action="store_true")
    a = sub.add_parser("assemble")
    a.add_argument("--out", default=str(X45_RECEIPTS / "side_by_side_summary.json"))
    args = ap.parse_args()
    if args.cmd == "k10-armed":
        if args.smoke:
            args.bytes = 300_000
            args.checkpoint_steps = [1, 2, 3]
            args.probe_windows = 2
        k10_armed(args)
        return
    if args.cmd == "gdn2-armed":
        if args.smoke:
            args.bytes = 300_000
            args.checkpoint_steps = [1, 2, 3]
            args.probe_windows = 2
        gdn2_armed(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
