"""K19 -- THE THRESHOLD RE-TUNE ON THE COMPOSED MACHINE: a bounded CMA-ES
staleness re-check of the op-routing thresholds, under the operator's
overnight authorization ("run experiments to explore potential additional
wins").

THE STALE AXIS THIS LANE RECEIPTS
(``PREREGISTRATION_K19_THRESHOLD_RETUNE.md``, committed before any probe
measurement): the composed machine (K15: the K10 winner + the K13 ADDITIVE
READ + the K11B p3 per-channel gate stack -- the program's best absolute
rows, post-cooldown 1.3811104467936925 / 1.383232252938407 at seeds
3401/3402) carries its op-routing thresholds from a machine that no longer
exists.  The thresholds were tuned on the PRE-read-law machine (the
committed CMA-ES search, ``CMAES_TUNING.md`` / ``cmaes_search.py``: the
K5B chunked arm under the SOFTMAX read, where read_scale was a logit
temperature); K15 changed the read/write interface fundamentally and the
thresholds were never re-searched.  The K16 config bracket discharged the
LR/optimizer/family axis on the composed machine but NOT the thresholds.

THE PARAMETERIZATION (the original search's own, nothing invented;
supervisor-approved 2026-11-25): EXACTLY the original search's dims and
bounds for these knobs -- tau_match (linear 0.2..0.8, phase 2's own),
tau_pop (linear -0.2..0.4, phase 2's own), read_scale (linear 0.5..4.0,
phase 1's own; the registered read-channel scale, the "read temperature"
knob, whose mechanical role the additive read changed from softmax
temperature to raw magnitude).  tau_merge is HELD at the banked 0.85 per
the original's receipted reason (merges fire plenty on near-duplicates);
including it would require inventing bounds the original never set.  x0
is the composed machine's incumbent point (0.45 / 0.10 / 2.787, all
inside the bounds), sigma0 0.15 (the original phase-2's own), popsize 8,
ONE generation, CMA-ES seed 20261125.

THE EVAL CONVENTION (the original's own, horizon stated honestly): one
eval = ONE matched-exposure probe of the COMPOSED MACHINE at seed 3401
under the CORRECTED COOLDOWN-PRIMARY SCORING -- the 20,022,866-target
landing (180 constant-LR steps), then the 24-step linear cooldown, the
objective the post-cooldown calibration NLL -- the same protocol the
composed anchors and GDN2's 1.3522 live at.  The search tunes to its own
budget: its eval IS the registered probe protocol, so the winner's
protocol-confirm is a determinism + full-receipt pass at the SAME
protocol, not a horizon transfer; no board-scale claim is made.  The
search-internal scoreboard is NEVER a ladder row (the standing rule):
only the protocol-confirmed winner at seed 3401 banks anything, and the
second seed (3402) is owed only if the win bar (>= 0.01 vs the composed
machine's own same-seed row) is met -- the pre-stated adjudication of the
registration's 3, with every margin read against BOTH bands (0.075
endpoint variance, 0.0119 winner seed delta) and all anchors quoted
(composed 1.3811/1.3832, GDN2 1.3522).

THE HARNESS RECEIPT (the original phase-2's own convention): eval 0 is a
RE-RUN of the incumbent through this probe -- it must reproduce the
composed machine's banked seed-3401 post-cooldown row 1.3811104467936925
and the K15 arm's own step-1 row (the nearest-kin parity).  If it does
NOT reproduce, the search STOPS and escalates: no search result is
interpretable until the composed machine's banked row is re-established.

MECHANISM DUTIES: the op-mix instrument per eval (the search's own
telemetry -- the K15 mechanism instrument imported verbatim, run at every
eval's post-cooldown state), and the winner's full mechanism receipts at
the confirm (identity share, effective rows, occupancy, the op mix, the
decorative sweep, at both states).  The pre-stated mechanism question: did
the re-tuned thresholds CHANGE THE OP MIX?  (The K14 receipts said the
model REJECTS learned routing; this tests whether SEARCHED routing on the
new interface does better.)

PROVENANCE (the lane's own rules): the K15 ENGINE IS COMMITTED AND
READ-ONLY -- ``k15_composition.py`` imported VERBATIM, never modified; the
committed CMA search machinery's conventions are IMPORTED
(``cmaes_search.encode`` verbatim; the decode's unit-cube scale handling
followed with THIS lane's base, because the committed decode's base is the
chunked arm's TRANSFERRED config -- a different machine's knobs); the
corrected-scoring machinery, the K15/K16 anchors, the mechanism instrument
and the decorative sweep are imported verbatim.  No ``k10_*``/``k11*``/
``k12_*``/``k13_*``/``k15_*``/``x46_*`` source, no committed
PREREGISTRATION, no ``lattice.py`` is edited.  This lane owns cards 5/6
only (the K18 lane owns 0/1, the X46 re-run owns 2/3 -- untouched).

Deliverables: per-eval receipts ``receipts/k19_probes/eval_*.json`` + the
fsynced log ``receipts/k19_threshold_evals.jsonl``, the search result
``receipts/k19_threshold_search_result.json`` (written incrementally,
committed per eval -- the ceiling is real), the confirm runs
``k19_runs/winner_seed{3401,3402}/`` and the assemble
``receipts/k19_threshold_search_summary.json``.

Usage (see ``run_k19_threshold_search_leased.sh`` for the detached leased
driver, cards 5/6):

    /home/erikg/emender/.venv/bin/python -m \\
        experiments.expressivity_tasks.x45_matrix_emender.k19_threshold_search \\
        search
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
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
                                  LR_MULTIPLIER, PRIMARY_SEED, PROBE_WINDOWS,
                                  SECOND_SEED, k15_probe_diagnostics,
                                  _k13_additive_step1)
from experiments.expressivity_tasks.x45_matrix_emender.k16_config_bracket \
    import (BANDS, GDN2_POST_COOLDOWN, composed_4x_banked)

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
RECEIPTS = ROOT / "receipts"
PROBE_RECEIPTS = RECEIPTS / "k19_probes"
RUNS = ROOT / "k19_runs"

# The original search's own dims and bounds for the threshold axis
# (CMAES_TUNING.md: phase 2's tau_match/tau_pop, phase 1's read_scale --
# the registered read-channel scale, the "read temperature" knob), verbatim.
# tau_merge is NOT here: the original never searched it (its receipted
# reason stands; see the registration's 1).
K19_DIMS = [
    ("tau_match", "linear", 0.2, 0.8),
    ("tau_pop", "linear", -0.2, 0.4),
    ("read_scale", "linear", 0.5, 4.0),
]

# The composed machine's incumbent point: the K10 tuned winner's own knobs
# (optimizer_bakeoff.TUNED_WINNER) -- the config the K15/K16 arms ran and
# the x0 the CMA-ES starts from.  Every non-searched knob is held here.
INCUMBENT = dict(TUNED_WINNER)

# The pre-stated adjudication (the registration's 3).
WIN_BAR = 0.01
ANCHOR_LABEL = "incumbent_anchor"
CMA_SEED = 20261125
SIGMA0 = 0.15                       # the original phase-2's own
POPSIZE = 8                         # the original phase-2's own
GENERATIONS = 1                     # a bounded staleness re-check
PROBE_TIMEOUT_S = 4200              # one eval, honest worst case
EVAL_EST_S = 3300                   # ~40-55 min/eval, the wall guard's own

# The composed machine's banked rows (the K15 composition arm's own landed
# rows, cited -- loaded from the committed assemble receipt; the fallback
# constants are the pinned ones the tests check).
COMPOSED_FALLBACK = dict(
    post_cooldown={3401: 1.3811104467936925, 3402: 1.383232252938407},
    constant_lr={3401: 1.433674931526184, 3402: 1.4341776371002197},
    ema_180={3401: 1.424389226096017, 3402: 1.4385861498968942},
    step1_train_loss={3401: 5.782775391046648, 3402: 5.720172563202926},
    note="the K15 composition arm's own landed rows at both seeds "
         "(receipts/k15_composition_summary.json) -- the incumbent config's "
         "own rows at the identical protocol; the anchor eval must "
         "reproduce the seed-3401 post-cooldown row bit for bit")

# The K15 composition arm's own mechanism rows at seed 3401 (the incumbent's
# op mix under the SAME instrument -- the per-eval telemetry's comparison
# anchor); loaded from the committed run receipt when present.
COMPOSED_MECH_FALLBACK = dict(
    identity_rewrite_share=0.1266, effective_read_rows=16.83,
    top_row_w_mass_share=0.155, occupancy=27.20,
    note="the K15 composition arm's own post-cooldown mechanism rows at "
         "seed 3401 (k15_runs/composition_seed3401/result.json), the "
         "incumbent's op mix under the same instrument")


def composed_banked() -> dict:
    """The composed machine's banked rows (the K16 lane's own loader, whose
    constants are pinned by its tests)."""
    return composed_4x_banked()


def composed_mechanism_anchor() -> dict:
    """The K15 composition arm's own post-cooldown mechanism rows at seed
    3401 (the incumbent's op-mix comparison anchor)."""
    out = dict(COMPOSED_MECH_FALLBACK)
    path = ROOT / "k15_runs" / "composition_seed3401" / "result.json"
    if path.exists():
        diag = json.loads(path.read_text()).get("diagnostics_post_cooldown") \
            or {}
        if diag:
            out.update(
                identity_rewrite_share=diag.get("identity_rewrite_share"),
                effective_read_rows=diag.get("effective_read_rows"),
                top_row_w_mass_share=(diag.get("top_row_w_mass_share")
                                      or {}).get("mean"),
                occupancy=(diag.get("occupancy") or {}).get("mean_live_slots"),
                source=str(path))
    return out


def win_clause(win: float) -> dict:
    """The pre-stated adjudication of a single seed's win (the registration's
    3, applied exactly as stated): the win is the composed machine's OWN
    same-seed banked row minus the confirmed winner's post-cooldown row,
    read against the 0.01 win bar with BOTH bands quoted on every margin."""
    if win >= WIN_BAR:
        return dict(clause="STALE", win=win,
                   text=f"the re-tune closes {win:.6f} >= {WIN_BAR} at the "
                        f"protocol -- the thresholds were stale across the "
                        f"interface change and the win banks (the second "
                        f"seed is owed)")
    if win <= -WIN_BAR:
        return dict(clause="INCUMBENT_BETTER", win=win,
                   text=f"the searched point loses by {-win:.6f} >= "
                        f"{WIN_BAR} -- the incumbent's thresholds beat the "
                        f"searched point at this budget; receipted and "
                        f"escalated")
    return dict(clause="PARITY", win=win,
               text=f"|win| = {abs(win):.6f} < {WIN_BAR} -- the thresholds "
                    f"were ROBUST across the interface change; the "
                    f"search-symmetry duty is discharged by receipt")


# ------------------------------------------------- the unit-cube conventions


def encode(dims, params: dict) -> list:
    """Real knobs -> unit cube: the committed search machinery's own encoder,
    imported VERBATIM (cmaes_search.encode)."""
    from experiments.expressivity_tasks.x45_matrix_emender.cmaes_search \
        import encode as _encode
    return _encode(dims, params)


def decode(dims, u) -> dict:
    """Unit cube -> real knobs, following the committed machinery's own
    scale handling (``cmaes_search.decode``'s convention: clip, log/logint/
    linear scales, u decoded left-to-right) with THIS lane's base -- the
    composed machine's incumbent config.  The committed decode's own base
    is the chunked arm's TRANSFERRED config (a different machine's knobs:
    lr 5.42e-4 etc.), so it cannot be imported for the base; the convention
    is followed, not restated from invention."""
    out = dict(INCUMBENT)
    for (name, scale, lo, hi), v in zip(dims, u):
        v = min(1.0, max(0.0, float(v)))
        if scale == "log":
            value = lo * (hi / lo) ** v
        elif scale == "logint":
            value = round(lo * (hi / lo) ** v)
        else:
            value = lo + v * (hi - lo)
        out[name] = int(value) if scale == "logint" else float(value)
    return out


def guard_config(config: dict) -> None:
    """The drift guards (the K16 bracket's own discipline, applied to the
    threshold axis): every NON-searched knob must be the composed machine's
    banked value, tau_merge held at the original's 0.85, and the searched
    knobs inside the original's bounds."""
    fixed = dict(lr=TUNED_WINNER["lr"],
                 weight_decay=TUNED_WINNER["weight_decay"],
                 warmup_steps=TUNED_WINNER["warmup_steps"])
    for key, want in fixed.items():
        if config.get(key) != want:
            raise ValueError(f"the non-searched knob {key} drifted: "
                             f"{config.get(key)!r} != the banked {want!r}")
    if config.get("tau_merge", 0.85) != 0.85:
        raise ValueError(f"tau_merge is HELD at the banked 0.85 (the "
                         f"original search's own choice, its receipted "
                         f"reason standing); got {config.get('tau_merge')!r}")
    for name, _scale, lo, hi in K19_DIMS:
        v = float(config[name])
        if not lo <= v <= hi:
            raise ValueError(f"{name}={v!r} is outside the original search's "
                             f"own bounds [{lo}, {hi}]")


# ------------------------------------------------------------------ helpers


def _require_lease() -> str:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _require_lease
    return _require_lease()


def _versions() -> dict:
    from experiments.expressivity_tasks.x45_matrix_emender.optimizer_bakeoff \
        import _versions
    return _versions()


def _mechanism_excerpt(diag: dict | None) -> dict | None:
    """The per-eval telemetry excerpt (the op-mix instrument's own rows)."""
    if not diag:
        return None
    return dict(
        op_rate_per_token_lane0=diag.get("op_rate_per_token_lane0"),
        op_histogram_all_lanes=diag.get("op_histogram_all_lanes"),
        occupancy=diag.get("occupancy"),
        effective_read_rows=diag.get("effective_read_rows"),
        read_weight_entropy=diag.get("read_weight_entropy"),
        top_row_w_mass_share=diag.get("top_row_w_mass_share"),
        identity_rewrite_share=diag.get("identity_rewrite_share"),
        gate_telemetry={k: (diag.get("gate_telemetry") or {}).get(k)
                        for k in ("decay_gate_mean_distance_from_registered_"
                                   "scalar",
                                   "write_gate_mean_distance_from_1",
                                   "read_gate_mean_distance_from_1")},
        probe=diag.get("probe"))


def _build_composed(config: dict, seed: int):
    """The K15 composed machine byte for byte with THIS eval's thresholds:
    the tuned winner's build (the searched-knobs path) + the K15 conversion,
    the parameter accounting asserted, the p3 gate routing split + the
    hybrid-Muon 4x optimizer -- the K15 arm's own plumbing, imported."""
    import torch
    from experiments.expressivity_tasks.x45_matrix_emender.velocity_ladder \
        import _build_k10
    from experiments.expressivity_tasks.x45_matrix_emender.\
        k11b_probe_channel_gates import (_routing_cross_check,
                                         split_parameters_gated)
    from experiments.expressivity_tasks.x45_matrix_emender.k15_composition \
        import (composed_parameter_delta, convert_model_to_composition)
    from experiments.expressivity_tasks.x45_matrix_emender.bakeoff_optimizers \
        import HybridMuonX45

    guard_config(config)
    model, optimizer_config, summary = _build_k10(dict(config), seed)
    parameters_before = sum(p.numel() for p in model.parameters())
    model = convert_model_to_composition(model)
    model = model.cuda().train()
    depth = len(model.blocks)
    mixer0 = model.blocks[0].mixer
    delta = composed_parameter_delta(
        depth, mixer0.width, mixer0.heads, mixer0.value_dim)
    parameters_after = sum(p.numel() for p in model.parameters())
    if parameters_after != parameters_before + delta \
            or delta != 8_657_040 or parameters_after != 158_638_848:
        raise ValueError(f"the K15 composition's parameter accounting must "
                         f"hold: {parameters_before} + {delta} != "
                         f"{parameters_after} (the banked 149,981,808 + "
                         f"8,657,040)")
    muon_params, adamw_params, muon_names, adamw_names = \
        split_parameters_gated(model, "p3")
    routing_receipt = _routing_cross_check(model, "p3", seed, False)
    peak_lr = LR_MULTIPLIER * optimizer_config["lr"]
    optimizer = HybridMuonX45(
        muon_params, adamw_params, lr=peak_lr, betas=tuple(
            optimizer_config["betas"]),
        weight_decay=optimizer_config["weight_decay"])
    return model, optimizer, optimizer_config, summary, dict(
        parameters_before=parameters_before, parameters_added=delta,
        parameters_after=parameters_after, peak_lr=peak_lr,
        muon=len(muon_names), adamw=len(adamw_names),
        routing_receipt=routing_receipt)


# ------------------------------------------------------------------- the probe


def probe(args) -> dict:
    """One evaluation at the fixed protocol: train the composed machine
    (THIS eval's thresholds) to the matched-exposure landing, run the
    24-step cooldown, take the post-cooldown calibration NLL as the
    objective, and run the op-mix instrument at the final state (the
    search's own per-eval telemetry).  Writes the receipt, prints JSON."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms

    config = json.loads(args.config)
    guard_config(config)
    started = time.time()

    def finish(status, nll, run, extra=None):
        receipt = dict(
            probe="k19_threshold_search", label=args.label, seed=args.seed,
            status=status, nll=nll, config=config,
            searched_knobs={k: config[k] for k, _, _, _ in K19_DIMS},
            tau_merge_held=config["tau_merge"],
            steps=(run or {}).get("step"), targets=(run or {}).get("seen"),
            exposure_targets=int(args.bytes),
            train_seconds_constant=(run or {}).get("constant_elapsed"),
            train_seconds_total=(run or {}).get("elapsed"),
            wall_seconds=round(time.time() - started, 1),
            step_log=(run or {}).get("rows") or [],
            **(extra or {}))
        receipt["versions"] = _versions()
        receipt["git"] = matched_arms.git_state()
        if args.out:
            atomic_json(Path(args.out), receipt)
        line = dict(label=args.label, status=status, nll=nll,
                    steps=receipt["steps"], targets=receipt["targets"])
        line.update({k: receipt.get(k) for k in (
            "anchor_reproduced_banked_row", "step1_k13_parity")})
        print(json.dumps(line), flush=True)
        return receipt

    L.check_python()
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    seed = args.seed

    model, optimizer, optimizer_config, summary, build = _build_composed(
        config, seed)
    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    run = _train_corrected(
        model, optimizer, build["peak_lr"], optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"k19_{args.label}", None, (), cooldown_steps, EMA_DECAY)
    if run["status"] != "completed":
        return finish("diverged", float("inf"), run)

    # ---- the objective: the post-cooldown calibration NLL (PRIMARY)
    model.load_state_dict(run["snapshots"]["final"])
    os.environ["X40_BATCHED_READS"] = "0"   # recorded-read eval path only
    result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
    nll = result["score"]

    # ---- the op-mix instrument (the search's own per-eval telemetry)
    diag = k15_probe_diagnostics(model, stream, windows=args.probe_windows)

    step1 = run["rows"][0]["loss"]
    extra = dict(
        eval_windows=result["eval_windows"],
        eval_targets=result["eval_targets"],
        bytes_per_s=run["seen"] / max(1e-9, run["elapsed"]),
        step1_train_loss=step1,
        parameters=build["parameters_after"],
        peak_lr=build["peak_lr"],
        routing_receipt=build["routing_receipt"],
        mechanism_post_cooldown=diag,
        op_mix=_mechanism_excerpt(diag),
        identity=identity, session=matched_arms.session_meta(card),
        summary=summary,
        anchors=dict(
            composed_banked=composed_banked(),
            gdn2_post_cooldown=GDN2_POST_COOLDOWN,
            win_bar=WIN_BAR,
            note="the composed machine's own banked rows (cited) and GDN2's "
                 "row -- quoted on every receipt"),
        corrected_scoring=CORRECTED_SCORING,
        both_bands=BANDS)
    if args.label == ANCHOR_LABEL:
        banked = composed_banked()
        extra["anchor_receipt"] = dict(
            banked_post_cooldown=banked["post_cooldown"][seed],
            reproduced_banked_row=bool(nll == banked["post_cooldown"][seed]),
            banked_step1=banked["step1_train_loss"][seed],
            k13_additive_step1=_k13_additive_step1(seed),
            step1_k13_parity=bool(step1 == _k13_additive_step1(seed)),
            note="the incumbent config through THIS probe must reproduce "
                 "the composed machine's banked same-seed post-cooldown row "
                 "bit for bit (the harness receipt, the original phase-2's "
                 "own convention) and the K15 arm's own step-1 row (the "
                 "nearest-kin parity); a mismatch STOPS the search and "
                 "escalates")
        extra["anchor_reproduced_banked_row"] = \
            extra["anchor_receipt"]["reproduced_banked_row"]
        extra["step1_k13_parity"] = extra["anchor_receipt"]["step1_k13_parity"]
    return finish("completed", nll, run, extra)


# ------------------------------------------------- the driver's own receipts


def append_eval(path: Path, record: dict):
    from experiments.expressivity_tasks.x45_matrix_emender.cmaes_search \
        import append_eval as _append
    _append(path, record)


def load_evals(path: Path) -> list:
    from experiments.expressivity_tasks.x45_matrix_emender.cmaes_search \
        import load_evals as _load
    return _load(path)


def commit(paths, message):
    """The per-eval commit (the lane's rule: a committed partial beats an
    uncommitted complete), with the index.lock-contention retry the
    concurrent lanes' drivers use."""
    paths = [str(p) for p in paths if Path(p).exists()]
    if not paths:
        return
    for attempt in range(12):
        done = subprocess.run(["git", "add", "--", *paths], cwd=REPO,
                              capture_output=True)
        if done.returncode == 0:
            done = subprocess.run(["git", "commit", "-m", message, "--",
                                   *paths], cwd=REPO, capture_output=True)
            if done.returncode == 0:
                return
        time.sleep(10)
    print(f"[k19] commit failed after retries; artifacts remain on disk "
          f"(the next commit picks them up)", flush=True)


def run_probe_subprocess(cfg: dict, label: str, out: Path, seed: int,
                         targets: float, probe_windows: int, timeout: int,
                         smoke: bool = False) -> dict:
    cmd = [sys.executable, "-m",
           "experiments.expressivity_tasks.x45_matrix_emender."
           "k19_threshold_search",
           "probe", "--config", json.dumps(cfg), "--label", label,
           "--seed", str(seed), "--bytes", str(int(targets)),
           "--probe-windows", str(probe_windows), "--out", str(out)]
    if smoke:
        cmd.append("--smoke")
    try:
        done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                              timeout=timeout)
        if done.returncode != 0:
            return dict(label=label, status="crashed", nll=float("inf"),
                        returncode=done.returncode,
                        stderr_tail=done.stderr[-2000:])
        line = done.stdout.strip().splitlines()[-1]
        return json.loads(line)
    except subprocess.TimeoutExpired:
        return dict(label=label, status="timeout", nll=float("inf"))
    except Exception as exc:                       # defensive: the search survives
        return dict(label=label, status="error", nll=float("inf"),
                    error=repr(exc))


# ------------------------------------------------------------------- the search


def search(args) -> dict:
    import cma

    evals_path = Path(args.evals)
    result_path = Path(args.result)
    probes_dir = Path(args.probes)
    probes_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    log = load_evals(evals_path)
    dims = list(K19_DIMS)

    def remaining_wall():
        return args.max_wall - (time.monotonic() - t0)

    def write_result(best, trajectory, extra=None):
        """The result receipt, written INCREMENTALLY (a lane death mid-search
        leaves a committed partial the next session harvests)."""
        from experiments.expressivity_tasks.x42_pile_scale.velocity_state \
            import atomic_json
        result = dict(
            probe="k19_threshold_search", space=[list(d) for d in dims],
            settings=dict(
                popsize=args.popsize, generations_requested=args.generations,
                sigma0=args.sigma0, cma_seed=args.cma_seed,
                probe_seed=args.seed, probe_targets=int(args.bytes),
                exposure_landing=20_022_866 if not args.smoke else None,
                cma_version=cma.__version__,
                tau_merge_held=0.85,
                tau_merge_note="the original search's own choice (its "
                               "receipted reason standing); including it "
                               "would require inventing bounds the "
                               "original never set",
                eval_convention="one eval = one matched-exposure probe of "
                                "the composed machine at seed 3401 under the "
                                "corrected cooldown-primary scoring (the "
                                "objective is the post-cooldown calibration "
                                "NLL); the search tunes to its own budget -- "
                                "its eval IS the registered probe protocol, "
                                "so the winner's protocol-confirm is a "
                                "determinism + full-receipt pass at the SAME "
                                "protocol, not a horizon transfer; no "
                                "board-scale claim is made",
                infeasible_corner_note="the original's bounds admit "
                                       "tau_pop > tau_match corners, which "
                                       "crash the policy constructor and "
                                       "receipt as nll=inf (CMA-ES is "
                                       "rank-based; the crash is receipted, "
                                       "not hidden)",
                fixed_seed_note="every evaluation uses seed 3401 and the "
                                "same deterministic stream prefix: the "
                                "search tunes the seed and the data draw "
                                "as well as the thresholds"),
            incumbent_trajectory=trajectory, best=best,
            anchors=dict(composed_banked=composed_banked(),
                          gdn2_post_cooldown=GDN2_POST_COOLDOWN,
                          win_bar=WIN_BAR, bands=BANDS),
            n_evaluations=len(log), wall_s=round(time.monotonic() - t0, 1),
            **(extra or {}))
        atomic_json(result_path, result)
        return result

    def evaluate(cfg, label):
        """Run one probe subprocess, append the receipt line, commit per
        eval (the lane's own rule), refresh the result receipt."""
        nonlocal log
        eval_id = len(log)
        out = probes_dir / f"eval_{eval_id:03d}.json"
        record = run_probe_subprocess(cfg, label, out, args.seed, args.bytes,
                                      args.probe_windows, args.timeout,
                                      args.smoke)
        record.update(eval_id=eval_id,
                      params={k: cfg[k] for k in cfg},
                      wall_s=round(time.monotonic() - t0, 1))
        log.append(record)
        append_eval(evals_path, record)
        print(f"[k19] eval {eval_id} ({label}): nll={record.get('nll')} "
              f"status={record.get('status')} "
              f"knobs={{{', '.join(f'{k}={cfg[k]:.4g}' for k, _, _, _ in dims)}}} "
              f"wall={record['wall_s']}s", flush=True)
        completed = [r for r in log if r.get("status") == "completed"
                     and r.get("nll") is not None]
        best = None
        trajectory = []
        for r in completed:
            if best is None or r["nll"] < best["nll"]:
                best = dict(nll=r["nll"], config=r["params"],
                            eval_id=r["eval_id"], label=r.get("label"))
            trajectory.append(dict(eval_id=r["eval_id"], label=r.get("label"),
                                    nll=r["nll"], best_so_far=best["nll"],
                                    read_scale=r["params"]["read_scale"],
                                    tau_match=r["params"]["tau_match"],
                                    tau_pop=r["params"]["tau_pop"]))
        anchor_rows = [r for r in log if r.get("label") == ANCHOR_LABEL]
        write_result(best, trajectory, dict(
            incumbent_anchor=(anchor_rows[-1] if anchor_rows else None),
            read_scale_trajectory=[dict(
                eval_id=t["eval_id"], read_scale=t["read_scale"],
                nll=t["nll"]) for t in trajectory]))
        commit([evals_path, result_path, probes_dir,
                Path(__file__), ROOT / "PREREGISTRATION_K19_THRESHOLD_RETUNE"
                ".md"],
               f"X45 K19 threshold re-tune: eval {len(log)} ({label}) "
               f"landed -- the composed machine's op-routing thresholds "
               f"re-searched on the interface that actually runs them (the "
               "original CMA-ES parameterization, cards 5/6, commit per eval)")
        return record

    # ---- eval 0: the incumbent anchor RE-RUN (the harness receipt).  It
    # must reproduce the composed machine's banked seed-3401 post-cooldown
    # row bit for bit and the K15 arm's own step-1 row; if it does not, the
    # search STOPS and escalates (no search result is interpretable).
    if not any(r.get("label") == ANCHOR_LABEL for r in log):
        evaluate(dict(INCUMBENT), ANCHOR_LABEL)
    anchor = [r for r in log if r.get("label") == ANCHOR_LABEL][-1]
    banked = composed_banked()
    if args.smoke:
        print("[k19] SMOKE: the anchor gate is SKIPPED (the tiny exposure "
              "cannot reproduce the banked row; the full run gates on it)",
              flush=True)
    else:
        anchor_ok = (anchor.get("status") == "completed"
                     and anchor.get("nll") == banked["post_cooldown"][args.seed]
                     and anchor.get("step1_k13_parity"))
        if not anchor_ok:
            write_result(
                None, [], dict(
                    incumbent_anchor=anchor,
                    stopped="ANCHOR MISMATCH -- the incumbent through the K19 "
                           "probe did NOT reproduce the composed machine's "
                           "banked seed-3401 post-cooldown row bit for bit "
                           "(and/or the nearest-kin step-1 parity): the search "
                           "is UNINTERPRETABLE until the composed machine's "
                           "banked row is re-established -- ESCALATED, per the "
                           "registration's 2"))
            commit([evals_path, result_path, probes_dir],
                   "X45 K19 threshold re-tune: ANCHOR MISMATCH -- the search "
                   "stopped and escalated (the incumbent did not reproduce "
                   "the banked 1.3811104467936925 row)")
            print(json.dumps(dict(ESCALATION="anchor mismatch",
                                  anchor=anchor,
                                  banked_post_cooldown=banked[
                                      "post_cooldown"][args.seed])), flush=True)
            sys.exit(2)
        print(f"[k19] anchor reproduced the banked row "
              f"{banked['post_cooldown'][args.seed]!r} bit for bit -- the "
              f"harness receipt holds; the search proceeds", flush=True)

    # ---- the bounded CMA-ES pass: x0 the incumbent, the original phase-2's
    # own sigma0, popsize 8, ONE generation (a bounded staleness re-check,
    # not an exhaustive search).
    x0 = encode(dims, INCUMBENT)
    es = cma.CMAEvolutionStrategy(
        x0, args.sigma0,
        dict(bounds=[0, 1], popsize=args.popsize, seed=args.cma_seed,
             verbose=-9, CMA_active=True))
    generation = 0
    while generation < args.generations:
        if remaining_wall() < EVAL_EST_S * 1.05:
            print(f"[k19] wall budget short ({remaining_wall():.0f}s < "
                  f"{EVAL_EST_S * 1.05:.0f}s): stopping before generation "
                  f"{generation + 1}", flush=True)
            break
        xs = es.ask()
        print(f"[k19] generation {generation + 1}/{args.generations} "
              f"popsize {len(xs)}", flush=True)
        recs = []
        for i, x in enumerate(xs):
            cfg = decode(dims, x)
            recs.append(evaluate(cfg, f"g{generation + 1}_{i}"))
        es.tell(list(xs), [r.get("nll") if r.get("nll") is not None
                           else float("inf") for r in recs])
        generation += 1

    completed = [r for r in log if r.get("status") == "completed"
                 and r.get("nll") is not None]
    best = None
    trajectory = []
    for r in completed:
        if best is None or r["nll"] < best["nll"]:
            best = dict(nll=r["nll"], config=r["params"],
                        eval_id=r["eval_id"], label=r.get("label"))
        trajectory.append(dict(eval_id=r["eval_id"], label=r.get("label"),
                               nll=r["nll"], best_so_far=best["nll"],
                               read_scale=r["params"]["read_scale"],
                               tau_match=r["params"]["tau_match"],
                               tau_pop=r["params"]["tau_pop"]))
    result = write_result(best, trajectory, dict(
        incumbent_anchor=anchor,
        generations_run=generation,
        read_scale_trajectory=[dict(eval_id=t["eval_id"],
                                   read_scale=t["read_scale"], nll=t["nll"])
                               for t in trajectory],
        search_internal_scoreboard_note="the search-internal scoreboard is "
                                        "NEVER a ladder row (the standing "
                                        "rule): only the protocol-confirmed "
                                        "winner at seed 3401 banks anything"))
    commit([evals_path, result_path, probes_dir],
           f"X45 K19 threshold re-tune: the bounded CMA-ES pass complete "
           f"({len(log)} evaluations, anchor + {generation} generation(s) "
           f"at popsize {args.popsize}); best search-internal nll "
           f"{None if best is None else round(best['nll'], 6)} -- the "
           f"protocol-confirm at seed 3401 is next")
    print(json.dumps(dict(best=None if best is None else best["nll"],
                          best_config=None if best is None else best["config"],
                          evaluations=len(log))), flush=True)
    return result


# ------------------------------------------------------------------- the confirm


def confirm(args) -> dict:
    """The winner's PROTOCOL-CONFIRM at the full probe protocol with the
    FULL receipts: the constant-180 curve, EMA-180 (ranking-only), the
    post-cooldown PRIMARY row, the mechanism instrument and the
    decorative-store sweep at BOTH states -- the K15/K16 arms' own receipt
    convention.  At seed 3401 the run must reproduce the search eval's own
    NLL bit for bit (the determinism receipt; --expect-nll); a mismatch is
    written, flagged and escalated (exit 3), not excused."""
    import torch
    from experiments.expressivity_tasks.x42_pile_scale import lattice as L
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)
    from experiments.expressivity_tasks.x45_matrix_emender import matched_arms
    from experiments.expressivity_tasks.x45_matrix_emender.side_by_side \
        import k10_read_mask_sweep

    config = json.loads(args.config)
    guard_config(config)
    L.check_python()
    if args.seed not in (PRIMARY_SEED, SECOND_SEED):
        raise ValueError(f"unregistered seed: {args.seed} (the confirm runs "
                         f"seed {PRIMARY_SEED}, and seed {SECOND_SEED} only "
                         f"if the win bar is met)")
    identity = L.flagship_identity()
    stream = L.Stream(identity["source"], identity["manifest"])
    L.assert_flagship_stream(stream)
    card = _require_lease()
    seed = args.seed

    model, optimizer, optimizer_config, summary, build = _build_composed(
        config, seed)
    banked = composed_banked()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    run_config = dict(
        probe="k19_threshold_search confirm", label=args.label, seed=seed,
        corrected_scoring=CORRECTED_SCORING, exposure_targets=int(args.bytes),
        config=config,
        searched_knobs={k: config[k] for k, _, _, _ in K19_DIMS},
        tau_merge_held=config["tau_merge"],
        model_build="the K15 composition VERBATIM (velocity_ladder._build_k10 "
                    "with THIS run's thresholds + "
                    "convert_model_to_composition), the K15 engine imported "
                    "never modified; the ONLY difference vs the K15 "
                    "composition arm is the searched policy thresholds",
        optimizer=dict(optimizer="bakeoff_optimizers.HybridMuonX45 (imported "
                                "verbatim)", peak_lr=build["peak_lr"],
                      lr_multiplier=LR_MULTIPLIER,
                      lr_convention=f"{LR_MULTIPLIER}x the tuned AdamW base "
                                   "(the banked winner lineage)",
                      betas=(0.9, BETA2),
                      weight_decay=TUNED_WINNER["weight_decay"],
                      warmup_steps=TUNED_WINNER["warmup_steps"],
                      microbatch=MICROBATCH, grad_clip=GRAD_CLIP),
        parameters=dict(before=build["parameters_before"],
                        added=build["parameters_added"],
                        total=build["parameters_after"],
                        note="p3's own gate stack (the read channel "
                             "instantiated once); the threshold axis is "
                             "parameter-free"),
        stopping_rule=(f"matched-EXPOSURE: train until {int(args.bytes)} "
                       "targets seen (the banked probe exposure, constant "
                       f"LR), THEN the {COOLDOWN_STEPS}-step linear cooldown "
                       "(the corrected-scoring extension, same stream); the "
                       "step-180 constant-LR eval is retained for banked "
                       "comparability"),
        determinism_rule="at seed 3401 the confirm must reproduce the search "
                         "eval's own post-cooldown NLL bit for bit (the "
                         "same deterministic computation); a mismatch is "
                         "escalated, not excused",
        expected_nll=args.expect_nll,
        win_bar=WIN_BAR,
        win_bar_rule="the win bar reads the confirm's post-cooldown row vs "
                     "the composed machine's OWN same-seed banked row; met "
                     "at seed 3401 => the thresholds were STALE and the "
                     "second seed (3402) is owed; not met => PARITY, the "
                     "thresholds were robust across the interface change",
        anchors=dict(composed_banked=banked, gdn2_post_cooldown=(
            GDN2_POST_COOLDOWN), bands=BANDS),
        bindings=dict(matched_arms.registration_bindings(),
                      k19_preregistration=str(
                          ROOT / "PREREGISTRATION_K19_THRESHOLD_RETUNE.md"),
                      k19_search_result=str(
                          RECEIPTS / "k19_threshold_search_result.json"),
                      k15_summary=str(RECEIPTS / "k15_composition_summary.json")),
        summary=summary, identity=identity, versions=_versions(),
        git=matched_arms.git_state())
    atomic_json(out / "start.json", dict(
        run_config, config_sha256=matched_arms.config_sha256(run_config),
        pid=os.getpid(), session=matched_arms.session_meta(card),
        started_ns=time.time_ns()))
    print(json.dumps(dict(label=args.label, card=card, seed=seed,
                          parameters=build["parameters_after"],
                          peak_lr=build["peak_lr"],
                          searched_knobs=run_config["searched_knobs"])),
          flush=True)

    cooldown_steps = 3 if args.smoke else COOLDOWN_STEPS
    checkpoint_steps = (1, 2, 3) if args.smoke else tuple(args.checkpoint_steps)
    run = _train_corrected(
        model, optimizer, build["peak_lr"], optimizer_config["warmup_steps"],
        optimizer_config["microbatch"], stream, int(args.bytes),
        f"k19_{args.label}_seed{seed}", out / "training.jsonl",
        checkpoint_steps, cooldown_steps, EMA_DECAY)
    if run["status"] != "completed":
        atomic_json(out / "result.json", dict(
            probe="k19_threshold_search confirm", label=args.label, seed=seed,
            status="diverged", config=config, step=run["step"],
            seen=run["seen"]))
        print(json.dumps(dict(label=args.label, status="diverged")),
              flush=True)
        sys.exit(4)
    rows_by_step = {r["step"]: r for r in run["rows"]}

    # ---- the scoring pass (restored snapshots; training is over)
    scores = []
    for step in checkpoint_steps:                 # the constant-LR curve
        if step not in run["snapshots"]:
            continue
        model.load_state_dict(run["snapshots"][step])
        os.environ["X40_BATCHED_READS"] = "0"     # recorded-read eval only
        result = L.evaluate(model, stream, "matrix_ops", args.eval_split)
        row = dict(kind="constant", step=step,
                   bytes=rows_by_step[step]["bytes"],
                   train_loss=rows_by_step[step]["loss"],
                   eval_nll=result["score"],
                   eval_windows=result["eval_windows"],
                   eval_targets=result["eval_targets"])
        scores.append(row)
        print(f"  constant step {step}: eval {row['eval_nll']:.4f}",
              flush=True)
    if run["ema"] is not None and run["constant_steps"] in run["snapshots"]:
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

    # ---- the FULL mechanism receipts at BOTH states + the decorative sweep
    diagnostics_180 = diagnostics_final = None
    sweep_180 = sweep_final = None
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
    steady = run["opt_seconds"][9:] if len(run["opt_seconds"]) > 9 \
        else run["opt_seconds"]

    # ---- the win-bar + determinism receipts (the registration's 3)
    determinism = None
    if args.expect_nll is not None:
        determinism = dict(expected_search_eval_nll=args.expect_nll,
                           measured_post_cooldown=post_cooldown,
                           reproduced=bool(post_cooldown == args.expect_nll),
                           note="the confirm at seed 3401 must reproduce the "
                                "search eval's own NLL bit for bit (the "
                                "same deterministic computation); a mismatch "
                                "is escalated, not excused")
    win = None
    if post_cooldown is not None:
        win = dict(measured_post_cooldown=post_cooldown,
                   composed_banked_same_seed=banked["post_cooldown"][seed],
                   win=banked["post_cooldown"][seed] - post_cooldown,
                   win_bar=WIN_BAR,
                   win_bar_met=bool(banked["post_cooldown"][seed]
                                    - post_cooldown >= WIN_BAR),
                   vs_gdn2=post_cooldown - GDN2_POST_COOLDOWN,
                   gdn2_contact_flag=bool(post_cooldown
                                          <= GDN2_POST_COOLDOWN),
                   bands=BANDS)
    receipt = dict(
        probe="k19_threshold_search confirm", label=args.label, seed=seed,
        status="completed", corrected_scoring=CORRECTED_SCORING,
        config=config, searched_knobs=run_config["searched_knobs"],
        tau_merge_held=config["tau_merge"],
        exposure_bytes=rows_by_step[run["constant_steps"]]["bytes"],
        exposure_steps=run["constant_steps"],
        total_bytes=run["seen"], total_steps=run["step"],
        cooldown_steps=cooldown_steps,
        train_seconds_constant=round(run.get("constant_elapsed", 0.0), 1),
        train_seconds_total=round(run["elapsed"], 1),
        optimizer_step_seconds=dict(
            mean_steady=(sum(steady) / len(steady)) if steady else None,
            note="steps 10+; the honest clock's own measure, per step"),
        parameters=build["parameters_after"], peak_lr=build["peak_lr"],
        final_eval_nll_constant_lr=constant_final,
        final_eval_nll_post_cooldown=post_cooldown,
        ema_eval_nll=ema_final,
        cooldown_drop=(constant_final - post_cooldown
                       if constant_final is not None
                       and post_cooldown is not None else None),
        step1_train_loss=run["rows"][0]["loss"],
        step1_note="searched thresholds are a DIFFERENT function: NO "
                   "nearest-kin step-1 parity is asserted on a searched "
                   "config (the anchor eval's own parity is the harness "
                   "receipt)",
        scores=scores, win_bar_receipt=win, determinism_receipt=determinism,
        diagnostics_step180=diagnostics_180,
        diagnostics_post_cooldown=diagnostics_final,
        read_mask_sweep_step180=sweep_180,
        read_mask_sweep_post_cooldown=sweep_final,
        op_mix=_mechanism_excerpt(diagnostics_final),
        routing_receipt=build["routing_receipt"],
        peak_allocated_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2),
        config_sha256=matched_arms.config_sha256(run_config),
        session=matched_arms.session_meta(card),
        trajectory=matched_arms.trajectory_sample(
            run["rows"], keep_first=min(5, max(1, len(run["rows"])))))
    atomic_json(out / "result.json", receipt)
    print(json.dumps({k: receipt[k] for k in
                      ("label", "seed", "status", "exposure_steps",
                       "final_eval_nll_constant_lr",
                       "final_eval_nll_post_cooldown", "ema_eval_nll",
                       "win_bar_receipt")}), flush=True)
    if determinism is not None and not determinism["reproduced"]:
        print(json.dumps(dict(
            ESCALATION="the confirm did NOT reproduce the search eval's NLL "
                       "bit for bit -- the determinism receipt failed; no "
                       "claim is bankable until resolved",
            determinism_receipt=determinism)), flush=True)
        sys.exit(3)
    return receipt


# ------------------------------------------------------------------- assemble


def _confirm_row(path: Path) -> dict | None:
    if not (path / "result.json").exists():
        return None
    r = json.loads((path / "result.json").read_text())
    diag = r.get("diagnostics_post_cooldown") or {}
    return dict(
        label=r["label"], seed=r["seed"], status=r["status"],
        post_cooldown=r["final_eval_nll_post_cooldown"],
        constant_lr=r["final_eval_nll_constant_lr"],
        ema_180=r["ema_eval_nll"], config=r["config"],
        searched_knobs=r["searched_knobs"],
        determinism_reproduced=(r.get("determinism_receipt") or {}).get(
            "reproduced"),
        win_receipt=r.get("win_bar_receipt"),
        effective_rows=diag.get("effective_read_rows"),
        top_row_share=(diag.get("top_row_w_mass_share") or {}).get("mean"),
        occupancy=(diag.get("occupancy") or {}).get("mean_live_slots"),
        identity_share=diag.get("identity_rewrite_share"),
        op_rate_per_token_lane0=diag.get("op_rate_per_token_lane0"),
        read_mask_sweep_layers_ge_threshold=sum(
            1 for v in ((r.get("read_mask_sweep_post_cooldown") or {})
                        .get("delta_per_layer") or {}).values()
            if v >= 0.02))


def assemble(args) -> dict:
    """The pre-stated adjudication (the registration's 3), applied exactly
    as stated: the search-internal scoreboard is NEVER a ladder row; the
    protocol row at seed 3401 reads against the win bar vs the composed
    machine's OWN same-seed banked row; the second seed's clause and the
    mixed-seed conservative reading govern any claim (the supervisor
    adjudicates the banking -- the K13/K15 clause-conflict precedent);
    every margin is quoted with BOTH bands and all anchors."""
    from experiments.expressivity_tasks.x42_pile_scale.velocity_state import (
        atomic_json)

    banked = composed_banked()
    search_result = json.loads(Path(args.search_result).read_text())
    confirm_3401 = _confirm_row(RUNS / f"{args.label}_seed{PRIMARY_SEED}")
    confirm_3402 = _confirm_row(RUNS / f"{args.label}_seed{SECOND_SEED}")
    if confirm_3401 is None or confirm_3401["status"] != "completed":
        raise ValueError("no completed seed-3401 confirm receipt to assemble")

    escalation_flags = []
    post1 = confirm_3401["post_cooldown"]
    win1 = banked["post_cooldown"][PRIMARY_SEED] - post1
    clause1 = win_clause(win1)
    if clause1["clause"] == "STALE":
        clause1_text = (f"STALE at seed {PRIMARY_SEED}: {clause1['text']}")
    elif clause1["clause"] == "INCUMBENT_BETTER":
        clause1_text = (f"INCUMBENT BETTER at seed {PRIMARY_SEED}: "
                        f"{clause1['text']}")
        escalation_flags.append(
            f"the searched point is WORSE than the incumbent by "
            f"{-win1:.4f} at seed {PRIMARY_SEED} -- the stale-axis "
            f"hypothesis is refuted at this search budget")
    else:
        clause1_text = (f"PARITY at seed {PRIMARY_SEED}: {clause1['text']}")
    clause1 = clause1_text

    second_seed = None
    if win1 >= WIN_BAR:
        if confirm_3402 is not None:
            post2 = confirm_3402["post_cooldown"]
            win2 = banked["post_cooldown"][SECOND_SEED] - post2
            met2 = win2 >= WIN_BAR
            second_seed = dict(
                seed=SECOND_SEED, post_cooldown=post2,
                composed_banked_same_seed=banked["post_cooldown"][SECOND_SEED],
                win=win2, win_bar=WIN_BAR, win_bar_met=met2,
                clause=(f"STALE at seed {SECOND_SEED} too (win {win2:.6f} "
                        f">= {WIN_BAR})" if met2 else
                        f"inside the band at seed {SECOND_SEED} (win "
                        f"{win2:.6f} < {WIN_BAR}) -- the mixed-seed "
                        f"conservative reading governs any 'new best "
                        f"machine' claim; both rows quoted; the supervisor "
                        f"adjudicates the banking"),
                bands=BANDS)
        else:
            second_seed = dict(
                seed=SECOND_SEED, status="owed_not_run",
                note=f"the win bar was met at seed {PRIMARY_SEED}; the "
                     f"second seed ({SECOND_SEED}) is OWED and must run "
                     f"before any banking claim")

    # the GDN2 contact rule + the standing-conclusion inversions
    for row, seed in ((confirm_3401, PRIMARY_SEED),
                      (confirm_3402, SECOND_SEED)):
        if row is not None and row.get("post_cooldown") is not None \
                and row["post_cooldown"] <= GDN2_POST_COOLDOWN:
            escalation_flags.append(
                f"the confirmed winner at seed {seed} landed at/past GDN2's "
                f"{GDN2_POST_COOLDOWN} (post-cooldown "
                f"{row['post_cooldown']}) -- the band-discipline clause: "
                f"BOTH scorings and BOTH seeds are reported before any "
                f"claim, and the result is escalated")
    if win1 >= 0.02:
        escalation_flags.append(
            f"the threshold re-tune closes {win1:.4f} at seed "
            f"{PRIMARY_SEED} -- a config-axis win large enough to challenge "
            f"the 'the plateau is architectural' reading of the remaining "
            f"~0.03; escalated per the registration's 3 (and the searched-"
            f"routing-vs-K14-learned-routing reading is the supervisor's)")

    # the mechanism comparison: did the re-tuned thresholds change the op
    # mix?  (the winner's rows vs the incumbent's OWN rows under the SAME
    # instrument -- the K15 arm's banked mechanism receipt at seed 3401)
    incumbent_mech = composed_mechanism_anchor()
    anchor_eval = None
    if search_result.get("incumbent_anchor") is not None:
        anchor_eval = dict(
            nll=search_result["incumbent_anchor"].get("nll"),
            note="the incumbent's own row through the K19 probe (the "
                 "harness receipt; reproduced the banked row bit for bit)")
    mechanism = None
    if confirm_3401.get("op_rate_per_token_lane0") is not None:
        mechanism = dict(
            winner_op_rate_per_token_lane0=confirm_3401[
                "op_rate_per_token_lane0"],
            winner_identity_share=confirm_3401["identity_share"],
            winner_effective_rows=confirm_3401["effective_rows"],
            winner_occupancy=confirm_3401["occupancy"],
            winner_top_row_share=confirm_3401["top_row_share"],
            winner_read_mask_sweep_layers_ge_threshold=confirm_3401[
                "read_mask_sweep_layers_ge_threshold"],
            incumbent_same_instrument=incumbent_mech,
            op_mix_changed=None,     # filled below from the per-eval receipts
            note="the pre-stated mechanism question: did the re-tuned "
                 "thresholds CHANGE THE OP MIX? (the K14 receipts said the "
                 "model REJECTS learned routing; this tests whether "
                 "SEARCHED routing on the new interface does better)")
    anchor_probe = PROBE_RECEIPTS / "eval_000.json"   # the anchor is eval 0 by construction
    if mechanism is not None and anchor_probe.exists():
        anchor_diag = json.loads(anchor_probe.read_text()).get("op_mix") or {}
        winner_mix = confirm_3401.get("op_rate_per_token_lane0") or {}
        anchor_mix = anchor_diag.get("op_rate_per_token_lane0") or {}
        if anchor_mix and winner_mix:
            deltas = {k: (winner_mix.get(k) or 0.0) - (anchor_mix.get(k) or 0.0)
                      for k in sorted(set(anchor_mix) | set(winner_mix))}
            mechanism["incumbent_op_rate_per_token_lane0"] = anchor_mix
            mechanism["op_mix_delta_winner_minus_incumbent"] = deltas
            mechanism["op_mix_changed"] = bool(
                max((abs(v) for v in deltas.values()), default=0.0) > 0.005)

    summary = dict(
        probe="k19_threshold_search assemble",
        corrected_scoring=CORRECTED_SCORING,
        parameterization=dict(
            dims=[list(d) for d in K19_DIMS],
            tau_merge_held=0.85,
            note="the original search's own dims and bounds, verbatim; "
                 "x0 the composed machine's incumbent; sigma0 0.15; "
                 "popsize 8; ONE generation; CMA-ES seed "
                 f"{CMA_SEED}"),
        search_internal=dict(
            best=search_result.get("best"),
            incumbent_trajectory=search_result.get("incumbent_trajectory"),
            read_scale_trajectory=search_result.get("read_scale_trajectory"),
            n_evaluations=search_result.get("n_evaluations"),
            scoreboard_rule="the search-internal scoreboard is NEVER a "
                            "ladder row (the standing rule): only the "
                            "protocol-confirmed winner at seed 3401 banks "
                            "anything"),
        incumbent_anchor=anchor_eval or search_result.get("incumbent_anchor"),
        confirm_seed_3401=confirm_3401,
        confirm_seed_3402=confirm_3402,
        win_bar=WIN_BAR,
        clause_seed_3401=clause1,
        win_seed_3401=win1,
        second_seed=second_seed,
        mechanism=mechanism,
        anchors=dict(composed_banked=banked,
                     gdn2_post_cooldown=GDN2_POST_COOLDOWN,
                     banked_corrected_anchor=BANKED["post_cooldown"][
                         "seed3402"],
                     bands=BANDS),
        escalation_flags=escalation_flags,
        notes=dict(
            scoring="PRIMARY: post-cooldown (24-step linear); SECONDARY: "
                    "EMA-0.99 at step 180 (ranking-only); constant-180 "
                    "retained for banked comparability",
            bands=f"every margin is read against BOTH the "
                  f"{ENDPOINT_VARIANCE_BAND}-nat endpoint-variance band and "
                  f"the winner's {WINNER_SEED_DELTA:.4f} seed delta",
            horizon="the search tuned to its own budget -- its eval IS the "
                    "registered probe protocol (180+24 steps at the "
                    "matched exposure), the same horizon the composed "
                    "anchors and the GDN2 row live at; no board-scale "
                    "(2700 s) claim is made or implied",
            pre_state="NO directional prediction was pre-stated; stale "
                      "thresholds, robust thresholds and an incumbent "
                      "win are ALL receipts (the search-symmetry duty is "
                      "discharged either way)",
            banking="the banking of any win and the final gap-quote "
                    "phrasing are the supervisor's to adjudicate (the "
                    "K13/K15 clause-conflict precedent); mixed seeds "
                    "adjudicate to the CONSERVATIVE reading"))
    atomic_json(Path(args.out), summary)
    print(json.dumps(dict(
        clause_seed_3401=clause1, win_seed_3401=win1,
        second_seed=(second_seed or {}).get("clause"),
        search_internal_best=(search_result.get("best") or {}).get("nll"),
        escalation_flags=escalation_flags), indent=1), flush=True)
    return summary


# ------------------------------------------------------------------------ main


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--config", required=True,
                   help="JSON config (the banked winner's knobs with the "
                        "searched thresholds)")
    p.add_argument("--label", default="probe")
    p.add_argument("--seed", type=int, default=PRIMARY_SEED)
    p.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    p.add_argument("--eval-split", default="calibration")
    p.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true",
                   help="harness shakedown: tiny exposure, 3 cooldown steps")
    c = sub.add_parser("confirm")
    c.add_argument("--config", required=True)
    c.add_argument("--label", default="winner")
    c.add_argument("--seed", type=int, default=PRIMARY_SEED)
    c.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    c.add_argument("--eval-split", default="calibration")
    c.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    c.add_argument("--checkpoint-steps", type=int, nargs="+",
                   default=list(CHECKPOINT_STEPS))
    c.add_argument("--expect-nll", type=float, default=None,
                   help="the search eval's own NLL: the seed-3401 confirm "
                        "must reproduce it bit for bit (the determinism "
                        "receipt)")
    c.add_argument("--output", required=True)
    c.add_argument("--smoke", action="store_true")
    s = sub.add_parser("search")
    s.add_argument("--evals", default=str(RECEIPTS
                                         / "k19_threshold_evals.jsonl"))
    s.add_argument("--result", default=str(RECEIPTS /
                                           "k19_threshold_search_result.json"))
    s.add_argument("--probes", default=str(PROBE_RECEIPTS))
    s.add_argument("--popsize", type=int, default=POPSIZE)
    s.add_argument("--generations", type=int, default=GENERATIONS)
    s.add_argument("--sigma0", type=float, default=SIGMA0)
    s.add_argument("--cma-seed", type=int, default=CMA_SEED)
    s.add_argument("--seed", type=int, default=PRIMARY_SEED)
    s.add_argument("--bytes", type=float, default=float(EXPOSURE_TARGETS))
    s.add_argument("--probe-windows", type=int, default=PROBE_WINDOWS)
    s.add_argument("--timeout", type=float, default=PROBE_TIMEOUT_S)
    s.add_argument("--max-wall", type=float, default=10 * 3600)
    s.add_argument("--smoke", action="store_true")
    a = sub.add_parser("assemble")
    a.add_argument("--search-result", default=str(
        RECEIPTS / "k19_threshold_search_result.json"))
    a.add_argument("--label", default="winner")
    a.add_argument("--out", default=str(
        RECEIPTS / "k19_threshold_search_summary.json"))
    args = ap.parse_args()
    if args.cmd == "probe":
        if args.smoke:
            args.bytes = 300_000
            args.probe_windows = 2
        probe(args)
        return
    if args.cmd == "confirm":
        if args.smoke:
            args.bytes = 300_000
            args.probe_windows = 2
        confirm(args)
        return
    if args.cmd == "search":
        if args.smoke:
            args.bytes = 300_000
            args.probe_windows = 2
        search(args)
        return
    assemble(args)


if __name__ == "__main__":
    main()
