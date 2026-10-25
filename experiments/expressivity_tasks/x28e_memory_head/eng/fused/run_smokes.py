"""Per-arm numerics smokes — registration gate 4 discharged PER ARM
(M2-review correction F5), pre-launch, ZERO optimizer updates.

Every smoke is a fresh zero-update forward + backward + fused evaluation
totalling the house max-length standard of 90,904 TOKENS per arm, processed
at the arm's freeze-bound microbatch budgets (configs/<arm>.json:
``microbatch.token_budget`` / ``max_sequences``; the effective-batch
accumulation mechanism of trainer.py), on the arm's ACTUAL training-path
model (trainer.build_tape_model / trainer.build_native_model — the
``tape_appendonly`` arm runs its pop-disabled forward wiring). No optimizer
exists anywhere in this leg; zero updates by construction.

  * PER-ARM MAX-LENGTH SMOKE (7 arms): summed-loss two-logit-margin BCE at
    the word's final token (G8), backward per microbatch with gradient
    accumulation (the F5 mechanism), fused evaluation forward per
    microbatch. Receipts per arm: the microbatch plan, predicted
    (allocation model) vs measured peak CUDA allocation per microbatch,
    finiteness of loss/outputs/every parameter gradient, wall times, trace
    case counts, final occupancy, the T2 fade-horizon calculation, and the
    eval-vs-train forward determinism.
    Receipt -> receipts/per_arm_smokes/<arm>.json (tape_B0 additionally
    mirrors the house-standard single-sequence receipt at
    receipts/numerics_smoke_max_length.json).

  * CAP-STRESS SMOKE (the two cap arms: tape_unbounded, tape_appendonly):
    the registered machine cap (T5: 2^17 = 131,072 slots per (batch
    element, head)) with forward + backward at each arm's actual wiring,
    receipting peak allocation, finite gradients, and that no allocation
    ever targets a slot >= the cap.
    Receipt -> receipts/cap_stress_smoke.json (both arms).

Registered zero-update smoke splits (n_seq x length, total 90,904 tokens
per arm; divisors of the house standard, receipted):

    tape_B0        1 x 90,904   (the house-standard single sequence)
    tape_B0x4      4 x 22,726
    tape_B0x16    11 x  8,264
    tape_unbounded 88 x  1,033
    tape_appendonly 88 x 1,033
    gdn2_native    1 x 90,904
    e97_native     1 x 90,904

Usage: .venv/bin/python run_smokes.py --arm <arm> --gpu <0..6>
       .venv/bin/python run_smokes.py --cap-stress --gpu <0..6>
Card 7 is NEVER used (live X25 run).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE_ROOT))

import trainer  # noqa: E402  (the training-path wiring: build_*_model, F5/F10/F11)
from eng.routing_trace import CASE_NAMES  # noqa: E402
from eng.tape_head import fade_horizon  # noqa: E402
from generators.stage3_ladder import (  # noqa: E402
    A, B, EOS, LADDER_VOCAB, SOS, binary_report,
)

RECEIPTS = Path(__file__).resolve().parent / "receipts"
SEED = 20260128
HOUSE_MAX_LENGTH = 90_904
ZERO_UPDATES = True
RUNG_BYTES = trainer.MICROBATCH_RUNG_BYTES

SMOKE_SPLITS = {          # n_seq x length, n_seq * length == 90,904 exactly
    "tape_B0": (1, 90904),
    "tape_B0x4": (4, 22726),
    "tape_B0x16": (11, 8264),
    "tape_unbounded": (88, 1033),
    "tape_appendonly": (88, 1033),
    "gdn2_native": (1, 90904),
    "e97_native": (1, 90904),
}
ARM_ORDER = ("tape_B0", "tape_B0x4", "tape_B0x16", "tape_unbounded",
             "tape_appendonly", "gdn2_native", "e97_native")


def load_arm_model(arm: str, cfg: dict, dev: str):
    """The arm's ACTUAL training-path model (zero updates: no optimizer)."""
    if cfg["architecture"] == "slot_store_tape_head":
        model = trainer.build_tape_model(cfg)
        model.attach_binary_stage(LADDER_VOCAB)   # attach BEFORE .to(dev)
        model = model.to(dev).float()
        model.eval()          # zero-update smoke; dropout-free modules
        slots = model.cfg.slots
    else:
        model = trainer.build_native_model(cfg, LADDER_VOCAB).to(dev)
        model.eval()
        slots = 0
    return model, slots


def smoke_data(arm: str, dev: str):
    """Seeded ladder-convention words (SOS ... EOS), n_seq x L tokens,
    random accept/reject labels — the G8 end-of-word supervision surface."""
    n_seq, length = SMOKE_SPLITS[arm]
    g = torch.Generator().manual_seed(SEED + ARM_ORDER.index(arm))
    body = torch.randint(1, 3, (n_seq, length - 2), generator=g)   # A or B
    x = torch.full((n_seq, length), 0, dtype=torch.long)
    x[:, 0] = SOS
    x[:, 1:length - 1] = body
    x[:, length - 1] = EOS
    labels = torch.randint(0, 2, (n_seq,), generator=g)
    return (x.to(dev), labels.to(dev),
            torch.full((n_seq,), length, dtype=torch.long, device=dev))


def _forward_logits(model, cfg, x, lengths, collect_trace):
    """The trainer's end-of-word two-logit margin (G8), on the arm's model.
    Returns (logits, final_token_layer_output | None, trace | None) — the
    final-token output feeds the eval-vs-train determinism receipt."""
    if cfg["architecture"] == "slot_store_tape_head":
        y, _, _, trace = model.forward(x, collect_trace=collect_trace)
        idx = torch.arange(x.shape[0], device=x.device)
        final_y = y[idx, lengths - 1]
        return model.head_binary(final_y), final_y, trace
    idx = torch.arange(x.shape[0], device=x.device)
    return model(x)[idx, lengths - 1, :2], None, None


def _finite_receipts(model, loss, logits, extra_outputs,
                    expect_no_grad=()):
    """Finiteness of the loss/outputs and every parameter gradient.

    A parameter whose gradient is None is finite-BY-CONSTRUCTION only when
    its name is in ``expect_no_grad`` — the F10 pop-disabled wiring never
    uses the pop projection in the forward pass, so autograd produces no
    gradient for it (there is no gradient VALUE to be nonfinite). An
    UNEXPECTED None gradient is a failure (False), never silently passed.
    """
    finite = {"loss": bool(torch.isfinite(loss.detach()).item()),
              "eval_logits": bool(torch.isfinite(logits).all().item())}
    for name, t in extra_outputs.items():
        finite[name] = bool(torch.isfinite(t).all().item())
    for name, p in model.named_parameters():
        if p.grad is None:
            finite[f"grad::{name}"] = bool(name in expect_no_grad)
        else:
            finite[f"grad::{name}"] = bool(
                torch.isfinite(p.grad).all().item())
    return finite


def expected_no_grad_names(cfg: dict):
    """Parameters with NO gradient BY CONSTRUCTION: the pop projections of
    the pop-disabled arm (F10 — pop strengths are identically zero constants
    in the forward pass, so ``W_pop`` is architecturally unused there)."""
    if (cfg["architecture"] == "slot_store_tape_head"
            and not cfg["pop_enabled"]):
        return {name for name in (
            "layer.head.head.W_pop.weight", "layer.head.head.W_pop.bias")}
    return set()


def smoke_arm(arm: str, gpu: int) -> dict:
    assert 0 <= gpu <= 6, "card 7 is EXCLUDED (live X25 run)"
    dev = f"cuda:{gpu}"
    cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
    micro = cfg["microbatch"]
    n_seq, length = SMOKE_SPLITS[arm]
    assert n_seq * length == HOUSE_MAX_LENGTH
    model, slots = load_arm_model(arm, cfg, dev)
    x, labels, lengths = smoke_data(arm, dev)
    is_tape = cfg["architecture"] == "slot_store_tape_head"

    rec: dict = {
        "smoke": "per-arm max-length 90,904-token zero-update fwd/bwd/eval "
                 "(registration gate 4, discharged per arm; M2-review "
                 "correction F5)",
        "arm": arm, "seed": SEED, "zero_updates": ZERO_UPDATES,
        "physical_card": int(os.environ["X28E_PHYSICAL_CARD"])
        if os.environ.get("X28E_PHYSICAL_CARD") is not None else None,
        "supervision": "end-of-word BCE on a two-logit margin reading the "
                      "final READ output (G8); summed loss over all "
                      "microbatches (the F5 effective-batch mechanism, zero "
                      "optimizer steps)",
        "total_tokens": HOUSE_MAX_LENGTH,
        "split": {"n_sequences": n_seq, "length": length},
        "microbatch_binding": {
            "token_budget": micro["token_budget"],
            "max_sequences": micro["max_sequences"],
            "rung_gib": RUNG_BYTES / 2 ** 30},
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }
    if is_tape:
        rec["dims"] = {"d_model": cfg["d_model"], "NH": cfg["head"]["n_heads"],
                       "K": cfg["head"]["key_dim"], "V": cfg["head"]["value_dim"],
                       "budget_slots": cfg.get("budget_slots"),
                       "slots": slots, "CKPT_INTERVAL":
                       cfg["constants"]["ckpt_interval"],
                       "checkpoint_dtype": cfg["constants"]["checkpoint_dtype"],
                       "pop_enabled": cfg["pop_enabled"]}

    lengths_np = lengths.cpu().numpy()
    mbs = trainer.microbatch_plan(lengths_np, micro["token_budget"],
                                   micro["max_sequences"])
    rec["microbatch_plan"] = [
        {"microbatch": i, "n_sequences": len(mb),
         "t_pad": int(lengths_np[mb].max()),
         "tokens": int(lengths_np[mb].sum())}
        for i, mb in enumerate(mbs)]
    if is_tape:
        rec["microbatch_plan_predicted_peak_gib"] = [
            round(trainer.allocation_model_bytes(
                len(mb), int(lengths_np[mb].max()), slots) / 2 ** 30, 3)
            for mb in mbs]

    # ---- zero-update forward + backward with F5 gradient accumulation ----
    for p in model.parameters():
        p.grad = None
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    walls = {"forward": 0.0, "backward": 0.0, "eval_forward": 0.0}
    per_mb_peaks = []
    train_final_outputs = {}
    case_counts: dict = {}
    pop_fired = free_pop = free_decay = 0
    final_live = 0
    loss_total = 0.0
    for i, mb in enumerate(mbs):
        mb_t = torch.as_tensor(mb, device=dev)
        x_mb = x[mb_t]
        ln_mb = lengths[mb_t]
        y_mb = labels[mb_t]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        # M2-review-2 finding 6: the training loop runs in TRAIN mode (true
        # emulation) and captures final outputs for ALL arms; the eval loop
        # switches to eval mode and MEASURES the diff (the native receipts'
        # former 0.0 was vacuous). The native configs are dropout-free, so
        # the measured diff is real, not defaulted.
        model.train()
        logits, final_y, trace = _forward_logits(model, cfg, x_mb, ln_mb,
                                                 collect_trace=is_tape)
        # natives return final_y=None; their comparable deterministic output
        # is the final two-logit readout (M2-review-2 finding 6)
        cap = final_y if final_y is not None else logits
        train_final_outputs[i] = cap.detach().clone()
        if is_tape:
            for c in trace.case.unique().tolist():
                name = CASE_NAMES[int(c)]
                case_counts[name] = case_counts.get(name, 0) + int(
                    (trace.case == int(c)).sum())
            pop_fired += int(trace.pop_fired.sum())
            free_pop += int((trace.freed == 1).sum())
            free_decay += int((trace.freed == 2).sum())
        margin = logits[:, 1] - logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(
            margin, y_mb.float(), reduction="sum")   # summed loss (F5)
        loss_total += float(loss.detach().item())
        torch.cuda.synchronize()
        walls["forward"] += time.time() - t0
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        walls["backward"] += time.time() - t0
        per_mb_peaks.append(torch.cuda.max_memory_allocated() / 2 ** 30)

    # ---- fused evaluation forward per microbatch (eval-mode) ----
    eval_diff = 0.0
    with torch.no_grad():
        model.eval()      # M2-review-2 finding 6: measured for ALL arms
        for i, mb in enumerate(mbs):
            mb_t = torch.as_tensor(mb, device=dev)
            x_mb = x[mb_t]
            ln_mb = lengths[mb_t]
            torch.cuda.synchronize()
            t0 = time.time()
            logits_eval, final_y_eval, trace = _forward_logits(
                model, cfg, x_mb, ln_mb, collect_trace=is_tape)
            torch.cuda.synchronize()
            walls["eval_forward"] += time.time() - t0
            if not bool(torch.isfinite(logits_eval).all()):
                raise RuntimeError(f"{arm}: nonfinite evaluation logits")
            cap_eval = final_y_eval if final_y_eval is not None else logits_eval
            eval_diff = max(
                eval_diff,
                float((cap_eval - train_final_outputs[i])
                      .abs().max().item()))
    if is_tape:
        # final occupancy from one clean eval forward of the LAST microbatch
        with torch.no_grad():
            mb = mbs[-1]
            mb_t = torch.as_tensor(mb, device=dev)
            _, _, st_eval, _ = model.forward(x[mb_t], collect_trace=False)
            final_live = int(st_eval.live.sum().item())

    peak_gib = max(per_mb_peaks)
    no_grad_expected = sorted(expected_no_grad_names(cfg))
    finite = _finite_receipts(model, torch.tensor(loss_total), logits,
                              {}, expect_no_grad=no_grad_expected)
    rec["finite"] = finite
    rec["all_finite"] = all(finite.values())
    rec["grad_none_by_construction"] = no_grad_expected
    rec["summed_loss"] = loss_total
    rec["peak_alloc_gib_per_microbatch"] = [round(p, 3) for p in per_mb_peaks]
    rec["peak_alloc_gib"] = peak_gib
    rec["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2 ** 30
    rec["wall_seconds"] = {k: round(v, 3) for k, v in walls.items()}
    rec["n_microbatches"] = len(mbs)
    rec["all_microbatches_within_rung"] = bool(
        all(p <= RUNG_BYTES / 2 ** 30 for p in per_mb_peaks))
    if is_tape:
        rec["trace_case_counts"] = case_counts
        rec["pop_fired_count"] = pop_fired
        rec["free_by_pop_count"] = free_pop
        rec["free_by_decay_count"] = free_decay
        rec["final_live_slots"] = final_live
        if arm == "tape_appendonly":
            rec["pop_disabled_receipt"] = {
                "pop_fired_count": pop_fired,
                "pop_architecturally_off": bool(pop_fired == 0),
                "note": "F10 wiring: single construction, pop strengths "
                        "identically zero in the forward pass"}
        horizon = fade_horizon(model.cfg)
        rec["t2_fade_horizon"] = {"decay_rate": model.cfg.decay_rate,
                                  "free_threshold": model.cfg.free_threshold,
                                  "horizon_tokens": horizon,
                                  "required_min": 10 * HOUSE_MAX_LENGTH,
                                  "satisfies_10x": bool(
                                      horizon >= 10 * HOUSE_MAX_LENGTH)}
        rec["eval_vs_train_output_max_abs"] = eval_diff
        predicted_ok = all(
            trainer.allocation_model_bytes(len(mb), int(lengths_np[mb].max()),
                                            slots) <= RUNG_BYTES
            for mb in mbs)
        rec["allocation_model_within_rung"] = bool(predicted_ok)
        rec["eval_vs_train_exact"] = bool(eval_diff == 0.0)
        rec["t2_satisfies_10x"] = bool(
            rec["t2_fade_horizon"]["satisfies_10x"])
        rec["pop_architecturally_off"] = bool(
            arm != "tape_appendonly" or pop_fired == 0)
        rec["pass"] = bool(rec["all_finite"]
                           and rec["all_microbatches_within_rung"]
                           and predicted_ok
                           and rec["t2_satisfies_10x"]
                           and rec["eval_vs_train_exact"]
                           and rec["pop_architecturally_off"])
    else:
        rec["eval_vs_train_output_max_abs"] = eval_diff
        rec["eval_vs_train_exact"] = bool(eval_diff == 0.0)
        rec["pass"] = bool(rec["all_finite"]
                           and rec["all_microbatches_within_rung"]
                           and rec["eval_vs_train_exact"])
    return rec


PASS_CRITERIA = ("all_finite", "all_microbatches_within_rung",
                 "allocation_model_within_rung", "eval_vs_train_exact",
                 "t2_satisfies_10x", "pop_architecturally_off")

CENSOR_NOTE = ("CENSORED infrastructure; no failure-to-learn adjudicated; "
               "any replacement arm requires a NEW registration")


def write_censor_receipt(arm: str, receipt: dict, why: str,
                         diagnostics: dict | None = None):
    """Fail-closed censor receipt for a FAILED per-arm smoke (the F6
    censor-path discipline extended to the gate-4 smokes): the full failed
    smoke receipt is preserved inside it as evidence; the arm is NEVER
    adjudicated as science here."""
    out = RECEIPTS / "per_arm_smokes" / f"{arm}_CENSORED.json"
    payload = {
        "censor": why,
        "arm": arm,
        "zero_updates": True,
        "failed_criteria": sorted(
            c for c in PASS_CRITERIA if receipt.get(c, True) is False),
        "failed_smoke_receipt": receipt,
        "diagnostics": diagnostics or {},
        "note": CENSOR_NOTE,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    return out


def smoke_cap_stress(gpu: int) -> dict:
    """The cap-stress smoke for BOTH cap arms (tape_unbounded,
    tape_appendonly) at their ACTUAL arm wiring (F5: the cap arms keep the
    cap-stress smoke; T5: 2^17 slots per (batch element, head))."""
    assert 0 <= gpu <= 6, "card 7 is EXCLUDED (live X25 run)"
    dev = f"cuda:{gpu}"
    receipt = {"smoke": "cap_stress (T5: unbounded-arm machine cap, 2^17 "
                        "slots per (batch element, head)); per M2-review "
                        "correction F5 run for BOTH cap arms at their actual "
                        "arm wiring",
               "seed": SEED, "zero_updates": ZERO_UPDATES,
               "physical_card": int(os.environ["X28E_PHYSICAL_CARD"])
               if os.environ.get("X28E_PHYSICAL_CARD") is not None else None,
               "cap_slots": 2 ** 17, "dims": {"T": 64, "CKPT_INTERVAL": 16,
                                              "checkpoint_dtype": "bfloat16"},
               "arms": {}}
    for arm in ("tape_unbounded", "tape_appendonly"):
        cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
        model, slots = load_arm_model(arm, cfg, dev)
        assert slots == 2 ** 17
        g = torch.Generator().manual_seed(SEED + 1)
        n, T = 4, 64
        body = torch.randint(1, 3, (n, T - 2), generator=g)
        x = torch.full((n, T), 0, dtype=torch.long)
        x[:, 0] = SOS
        x[:, 1:T - 1] = body
        x[:, T - 1] = EOS
        x = x.to(dev)
        labels = torch.randint(0, 2, (n,), generator=g).to(dev)
        lengths = torch.full((n,), T, dtype=torch.long, device=dev)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        logits, _, trace = _forward_logits(model, cfg, x, lengths,
                                           collect_trace=True)
        margin = logits[:, 1] - logits[:, 0]
        loss = F.binary_cross_entropy_with_logits(
            margin, labels.float(), reduction="sum")
        torch.cuda.synchronize()
        t_fwd = time.time() - t0
        t0 = time.time()
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.time() - t0
        finite = _finite_receipts(
            model, loss, logits, {},
            expect_no_grad=sorted(expected_no_grad_names(cfg)))
        counts = {CASE_NAMES[int(c)]: int((trace.case == int(c)).sum())
                  for c in trace.case.unique().tolist()}
        arm_rec = {
            "arm": arm, "pop_enabled": cfg["pop_enabled"],
            "finite": finite, "all_finite": all(finite.values()),
            "trace_case_counts": counts,
            "max_alloc_target": int(trace.alloc_target.max().item()),
            "alloc_below_cap": bool(
                int(trace.alloc_target.max().item()) < 2 ** 17),
            "pop_fired_count": int(trace.pop_fired.sum()),
            "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2 ** 30,
            "wall_seconds": {"forward": round(t_fwd, 3),
                             "backward": round(t_bwd, 3)},
        }
        with torch.no_grad():
            _, _, st, _ = model.forward(x, collect_trace=False)
        arm_rec["final_live_slots"] = int(st.live.sum().item())
        arm_rec["pass"] = bool(arm_rec["all_finite"]
                               and arm_rec["alloc_below_cap"]
                               and (cfg["pop_enabled"]
                                    or arm_rec["pop_fired_count"] == 0))
        receipt["arms"][arm] = arm_rec
    receipt["pass"] = all(a["pass"] for a in receipt["arms"].values())
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_ORDER)
    parser.add_argument("--cap-stress", action="store_true")
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    if not 0 <= args.gpu <= 6:
        parser.error("cards 0-6 only; card 7 is EXCLUDED (live X25 run)")
    torch.cuda.set_device(args.gpu)
    if args.cap_stress:
        receipt = smoke_cap_stress(args.gpu)
        out = RECEIPTS / "cap_stress_smoke.json"
        out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"=== CAP-STRESS (both cap arms) pass={receipt['pass']} ===")
        for arm, a in receipt["arms"].items():
            print(f"  {arm}: all_finite={a['all_finite']} "
                  f"max_alloc={a['max_alloc_target']} < cap: "
                  f"{a['alloc_below_cap']} pops={a['pop_fired_count']} "
                  f"peak {a['peak_alloc_gib']:.2f} GiB")
        print(f"receipt -> {out}")
        return 0 if receipt["pass"] else 1
    if not args.arm:
        parser.error("--arm or --cap-stress required")
    receipt = smoke_arm(args.arm, args.gpu)
    out_dir = RECEIPTS / "per_arm_smokes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.arm}.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    if not receipt["pass"]:
        # fail-closed (F6 discipline extended to the gate-4 smokes): a FAILED
        # smoke is CENSORED infrastructure, never a scientific verdict; the
        # censor receipt preserves the full failure evidence.
        failed = sorted(c for c in PASS_CRITERIA
                        if receipt.get(c, True) is False)
        cout = write_censor_receipt(
            args.arm, receipt,
            f"per-arm max-length smoke FAILED the registered gate-4 "
            f"criteria {failed} (fail-closed; CENSORED infrastructure, "
            f"never a scientific verdict)")
        print(f"CENSOR receipt (fail-closed) -> {cout}")
    if args.arm == "tape_B0":
        # the house-standard single-sequence receipt keeps its registered
        # filename (configs reference it); same bytes, same evidence.
        out2 = RECEIPTS / "numerics_smoke_max_length.json"
        out2.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"receipt (house-standard copy) -> {out2}")
    print(f"=== PER-ARM SMOKE {args.arm}: pass={receipt['pass']} ===")
    print(f"  split {receipt['split']} -> {receipt['n_microbatches']} "
          f"microbatches (plan "
          f"{[(m['n_sequences'], m['t_pad']) for m in receipt['microbatch_plan']]})")
    print(f"  all_finite={receipt['all_finite']} "
          f"summed_loss={receipt['summed_loss']:.4f} "
          f"peak {receipt['peak_alloc_gib']:.2f} GiB "
          f"(reserved {receipt['peak_reserved_gib']:.2f}) "
          f"within 40-GiB rung: {receipt['all_microbatches_within_rung']}")
    print(f"  wall: fwd {receipt['wall_seconds']['forward']:.1f}s "
          f"bwd {receipt['wall_seconds']['backward']:.1f}s "
          f"eval {receipt['wall_seconds']['eval_forward']:.1f}s")
    if "trace_case_counts" in receipt:
        print(f"  case counts: {receipt['trace_case_counts']}")
        print(f"  pops={receipt['pop_fired_count']} "
              f"free_by_pop={receipt['free_by_pop_count']} "
              f"free_by_decay={receipt['free_by_decay_count']} "
              f"final_live={receipt['final_live_slots']}")
        print(f"  T2 horizon {receipt['t2_fade_horizon']['horizon_tokens']:.0f} "
              f">= 10x length: "
              f"{receipt['t2_fade_horizon']['satisfies_10x']}")
        print(f"  eval-vs-train output max abs: "
              f"{receipt['eval_vs_train_output_max_abs']}")
    print(f"receipt -> {out}")
    return 0 if receipt["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
