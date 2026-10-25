"""Phase-4 freeze receipt generator (M2-review corrections F5 + F11).

Produces two manifest-covered freeze receipts (pure CPU, zero optimizer
updates, byte-stable):

  * receipts/microbatch_allocation_model.json (F5): the per-arm freeze-bound
    microbatch budgets, the allocation-model prediction for every in-plan
    worst case, the over-plan single worst cases among the ladder's
    registered inputs, the 40-GiB rung, and the margins the budgets were
    chosen against. The registered amendment: batch 64 = effective batch 64
    via length-aware microbatch accumulation; per-arm token budgets are
    freeze-bound receipts; updates are identical to a single batch-64 step.

  * receipts/native_parameter_totals.json (F11): the ACTUAL constructor
    totals of the two native controls at the frozen vocab-5 encoding
    (~562,834 GDN2 / ~569,888 E97, verified BY CONSTRUCTION), explicitly
    distinguishing the inherited X27 vocab-8 whole-model labels
    (563,218 / 570,464 — same backbone dimensions) from the vocab-5 counts:
    the shared TIED embedding/output weight changes by (8-5)*dim.

Usage: python make_freeze_receipts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE_ROOT))

RECEIPTS = PACKAGE_ROOT / "receipts"
ARMS = ("tape_B0", "tape_B0x4", "tape_B0x16", "tape_unbounded",
        "tape_appendonly", "gdn2_native", "e97_native")
TAPE_ARMS = ("tape_B0", "tape_B0x4", "tape_B0x16", "tape_unbounded",
             "tape_appendonly")
RUNG_GIB = 40.0
# the ladder's registered input extremes (tokens incl. SOS/EOS):
#   balanced / leading_b at the 9x bin top k=1152: 2k+2 = 2,306
#   count-mismatch at the 9x bin top: j+k+2 with j,k <= 1152, j!=k: 2,307 max
#   trailing_a at the 9x top: 3k+2 = 3,458 (the longest registered word)
LADDER_WORST_TOKENS = {"balanced_9x": 2306, "count_mismatch_9x": 2307,
                       "trailing_a_9x": 3458}


def slots_of(cfg: dict) -> int:
    if cfg["architecture"] != "slot_store_tape_head":
        return 0
    return (cfg.get("budget_slots")
            or cfg["constants"]["unbounded_cap_slots"])


def in_plan_worst_gib(cfg: dict) -> dict:
    """Worst predicted peak over every microbatch the bound budgets can
    produce: n sequences at the largest T with n*T <= token_budget,
    for every n in [1, max_sequences]."""
    micro = cfg["microbatch"]
    W, M = micro["token_budget"], micro["max_sequences"]
    worst = None
    arg = None
    for n in range(1, M + 1):
        T = max(1, W // n)
        gib = __import__("trainer").allocation_model_bytes(
            n, T, slots_of(cfg)) / 2 ** 30
        if worst is None or gib > worst:
            worst, arg = gib, (n, T)
    return {"n_sequences": arg[0], "t_pad": arg[1],
            "predicted_peak_gib": round(worst, 2)}


def main() -> int:
    import trainer

    # ---- F5: the per-arm microbatch allocation-model receipt --------------
    arms = {}
    for arm in ARMS:
        cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
        micro = cfg["microbatch"]
        entry = {
            "token_budget": micro["token_budget"],
            "max_sequences": micro["max_sequences"],
        }
        if cfg["architecture"] == "slot_store_tape_head":
            slots = slots_of(cfg)
            worst = in_plan_worst_gib(cfg)
            entry.update({
                "slots": slots,
                "rung_gib": RUNG_GIB,
                "in_plan_worst": worst,
                "in_plan_margin_gib": round(RUNG_GIB - worst["predicted_peak_gib"], 2),
                # over-plan singles among the ladder's registered inputs
                "over_plan_singles": {},
            })
            for name, T in LADDER_WORST_TOKENS.items():
                gib_bwd = trainer.allocation_model_bytes(
                    1, T, slots, backward=True) / 2 ** 30
                gib_fwd = trainer.allocation_model_bytes(
                    1, T, slots, backward=False) / 2 ** 30
                within = gib_bwd <= RUNG_GIB
                entry["over_plan_singles"][name] = {
                    "tokens": T,
                    "predicted_peak_gib_fwd_bwd": round(gib_bwd, 2),
                    "predicted_peak_gib_fwd_only": round(gib_fwd, 2),
                    "within_rung_fwd_bwd": bool(within),
                    "within_rung_fwd_only": bool(gib_fwd <= RUNG_GIB),
                    "policy": ("in-plan / within rung: normal zero-update "
                               "measurement" if within else
                               "ABOVE the 40-GiB rung: the trainer's "
                               "fail-closed rung check writes a CENSORED "
                               "receipt (right-censored infrastructure, "
                               "never a scientific verdict) BEFORE any "
                               "OOM; flagged in ENGINEERING_FREEZE.md"),
                }
            entry["pass"] = bool(entry["in_plan_margin_gib"] >= 8.0)
        else:
            entry.update({
                "note": "native arms have no tape store; chunked native "
                        "kernels; peak allocation is receipted MEASURED per "
                        "arm in eng/fused/receipts/per_arm_smokes/",
                "rung_gib": RUNG_GIB,
                "pass": True,
            })
        arms[arm] = entry
    f5 = {
        "receipt": "registered effective-batch accumulation: per-arm "
                   "microbatch budgets vs the 40-GiB rung (M2-review "
                   "correction F5)",
        "amendment": "batch 64 = effective batch 64 via length-aware "
                     "microbatch accumulation; per-arm token budgets are "
                     "freeze-bound receipts; updates are identical to a "
                     "single batch-64 step",
        "allocation_model": "trainer.allocation_model_bytes: bf16 sparse "
                            "checkpoints every 16 tokens + int8 free-trace + "
                            "spilled fwd scratch + fp32 per-step backward "
                            "replay cache + adjoints (the G7/G14 frozen "
                            "constants); prediction is validated MEASURED "
                            "against the per-arm smokes",
        "rung_gib": RUNG_GIB,
        "margin_rule": "every in-plan worst case keeps >= 8 GiB of rung "
                       "margin",
        "arms": arms,
        "zero_updates": True,
        "pass": all(a["pass"] for a in arms.values()),
    }
    RECEIPTS.mkdir(parents=True, exist_ok=True)
    (RECEIPTS / "microbatch_allocation_model.json").write_text(
        json.dumps(f5, indent=2) + "\n")

    # ---- F11: the ACTUAL native constructor totals at vocab 5 --------------
    from generators.stage3_ladder import LADDER_VOCAB
    native = {}
    for arm in ("gdn2_native", "e97_native"):
        cfg = json.loads((PACKAGE_ROOT / "configs" / f"{arm}.json").read_text())
        model = trainer.build_native_model(cfg, LADDER_VOCAB)
        actual = sum(p.numel() for p in model.parameters())
        dim = cfg["constructor_size"]["dim"]
        inherited = actual + (8 - LADDER_VOCAB) * dim
        native[arm] = {
            "actual_constructor_total_vocab5": actual,
            "inherited_x27_label_vocab8": (563218 if arm == "gdn2_native"
                                            else 570464),
            "independent_reconstruction_vocab8": inherited,
            "shared_tied_embedding_output_delta": (8 - LADDER_VOCAB) * dim,
            "backbone_dimensions": cfg["constructor_size"],
            "note": "whole-model count at the frozen vocab-5 encoding; the "
                    "inherited X27 backbone dimensions are UNCHANGED — only "
                    "the shared tied embedding/output weight "
                    "(vocab_size * dim) differs from the vocab-8 label",
            "verified_by_construction": bool(
                actual == cfg["parameters"]
                and inherited == (563218 if arm == "gdn2_native"
                                  else 570464)),
        }
    f11 = {
        "receipt": "native whole-model constructor totals at the frozen "
                   "vocab-5 encoding (M2-review correction F11)",
        "arms": native,
        "zero_updates": True,
        "pass": all(v["verified_by_construction"] for v in native.values()),
    }
    (RECEIPTS / "native_parameter_totals.json").write_text(
        json.dumps(f11, indent=2) + "\n")

    print(f"microbatch_allocation_model: pass={f5['pass']}")
    for arm, a in arms.items():
        if "in_plan_worst" in a:
            print(f"  {arm}: W={a['token_budget']} M={a['max_sequences']} "
                  f"slots={a['slots']} in-plan worst "
                  f"{a['in_plan_worst']['predicted_peak_gib']} GiB "
                  f"(margin {a['in_plan_margin_gib']} GiB)")
            for name, s in a["over_plan_singles"].items():
                print(f"    over-plan single {name}: {s['tokens']} tokens -> "
                      f"{s['predicted_peak_gib_fwd_bwd']} GiB "
                      f"(within rung: {s['within_rung_fwd_bwd']})")
    print(f"native_parameter_totals: pass={f11['pass']}")
    for arm, v in native.items():
        print(f"  {arm}: vocab5 total {v['actual_constructor_total_vocab5']} "
              f"(inherited vocab8 label {v['inherited_x27_label_vocab8']}; "
              f"reconstruction {v['independent_reconstruction_vocab8']})")
    return 0 if f5["pass"] and f11["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
