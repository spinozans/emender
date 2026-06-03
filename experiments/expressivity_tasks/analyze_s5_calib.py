#!/usr/bin/env python
"""Analyze S5 candidate-budget calibration runs (task s5sym-calibrate).

Reads all run JSONs under results/s5_calib_20260603/ and computes, per arm:
  - acc-vs-step curves (text table),
  - earliest step each config rises above chance (1/120 = 0.0083),
  - earliest step at which within-arm configs become STABLY rank-ordered:
      the order at step N equals the order at the final eval AND never flips
      again through 4000 (a stable span of >= 300 steps, i.e. >=4 dense evals).

Emits a single parseable line: recommended_candidate_steps: <N>.
"""
import os, json, glob

THIS = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(THIS, "results", "s5_calib_20260603")
BASELINE = 1.0 / 120.0           # 0.008333
ABOVE_CHANCE = 0.02              # ~2.4x chance: clearly above noise floor
STABLE_SPAN = 300               # consecutive steps the order must hold

ARMS = {
    "E88":   ["E88_lr9e-5_tanh", "E88_lr3e-4_tanh", "E88_lr9e-4_tanh", "E88_lr3e-4_linear"],
    "M2RNN": ["M2RNN_lr9e-5", "M2RNN_lr3e-4", "M2RNN_lr9e-4"],
    "GDN":   ["GDN_lr9e-5", "GDN_lr3e-4", "GDN_lr9e-4"],
}


def load(label):
    p = os.path.join(DIR, f"{label}.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    steps = [s["step"] for s in d["steps"]]
    accs = [s["eval_acc"] for s in d["steps"]]
    return {"label": label, "steps": steps, "accs": accs,
            "final_acc": d.get("final_acc"), "lr": d.get("lr"),
            "linear_state": d.get("linear_state")}


def common_grid(configs):
    """Eval steps present in every config, sorted."""
    sets = [set(c["steps"]) for c in configs]
    grid = sorted(set.intersection(*sets)) if sets else []
    return grid


def acc_at(c, step):
    return c["accs"][c["steps"].index(step)]


def order_at(configs, step):
    """Return tuple of config labels sorted by acc (desc) at given step."""
    return tuple(sorted((c["label"] for c in configs),
                        key=lambda lab: -acc_at(next(c for c in configs if c["label"] == lab), step)))


def first_above_chance(c):
    for s, a in zip(c["steps"], c["accs"]):
        if a > ABOVE_CHANCE:
            return s
    return None


def stable_rank_step(configs, grid):
    """Earliest grid step N s.t. order_at(N) == final order and never flips
    again through the end, with the stable span >= STABLE_SPAN steps."""
    if len(grid) < 2:
        return None, None
    final_order = order_at(configs, grid[-1])
    # Walk backward: find the last step where the order != final_order.
    last_flip_idx = -1
    for i, s in enumerate(grid):
        if order_at(configs, s) != final_order:
            last_flip_idx = i
    if last_flip_idx == len(grid) - 1:
        return None, final_order  # order still changing at the very end
    n_idx = last_flip_idx + 1
    n_step = grid[n_idx]
    span = grid[-1] - n_step
    if span < STABLE_SPAN:
        return None, final_order  # stabilized too late to confirm >=300-step hold
    return n_step, final_order


def fmt_curve(configs, grid, every=500):
    show = [s for s in grid if s % every == 0]
    if grid and grid[-1] not in show:
        show.append(grid[-1])
    header = "step   " + "  ".join(f"{c['label']:>20s}" for c in configs)
    lines = [header]
    for s in show:
        row = f"{s:>5d}  " + "  ".join(f"{acc_at(c, s):>20.4f}" for c in configs)
        lines.append(row)
    return "\n".join(lines)


def main():
    report = []
    report.append("# S5 CANDIDATE-BUDGET CALIBRATION\n")
    report.append(f"Task: s5_permutation (S5, |classes|=120, chance={BASELINE:.4f}), "
                  f"seq_len=128, batch=32, seed=42, schedule-free AdamW, 4000 steps, "
                  f"dense S5 eval every 100 steps.\n")
    report.append(f"above-chance threshold = {ABOVE_CHANCE} (~{ABOVE_CHANCE/BASELINE:.1f}x chance); "
                  f"stable-rank requires order == final order held for >= {STABLE_SPAN} consecutive steps.\n")

    arm_stable = {}
    for arm, labels in ARMS.items():
        configs = [load(l) for l in labels]
        configs = [c for c in configs if c is not None]
        report.append(f"\n## Arm: {arm}\n")
        if len(configs) < 2:
            report.append(f"  INCOMPLETE: only {len(configs)} configs present.\n")
            continue
        grid = common_grid(configs)
        report.append("### acc-vs-step (every 500 steps + final)\n```")
        report.append(fmt_curve(configs, grid, every=500))
        report.append("```\n")

        report.append("### above-chance step per config")
        for c in configs:
            fac = first_above_chance(c)
            report.append(f"  {c['label']:>20s}: final_acc={c['final_acc']:.4f}  "
                          f"first_above_{ABOVE_CHANCE}={fac}")
        arm_above = [first_above_chance(c) for c in configs if first_above_chance(c) is not None]
        earliest_above = min(arm_above) if arm_above else None

        n_step, final_order = stable_rank_step(configs, grid)
        report.append(f"\n### rank-separation")
        report.append(f"  final order (best->worst): {' > '.join(final_order)}")
        report.append(f"  earliest above-chance (any config): {earliest_above}")
        if n_step is None:
            report.append(f"  STABLE-RANK STEP: NOT STABLE by 4000 "
                          f"(order still flipping near the end OR stabilized <{STABLE_SPAN} steps from end)")
            arm_stable[arm] = None
        else:
            report.append(f"  STABLE-RANK STEP: {n_step}  "
                          f"(order held from step {n_step} through 4000)")
            arm_stable[arm] = n_step
        report.append("")

    # Recommendation
    valid = {a: s for a, s in arm_stable.items() if s is not None}
    unstable = [a for a, s in arm_stable.items() if s is None]
    report.append("\n## RECOMMENDATION\n")
    for a, s in arm_stable.items():
        report.append(f"  arm {a}: stable-rank step = {s}")
    if valid:
        worst = max(valid.values())
        # margin: +1 dense-eval window beyond worst, then round up to a CMA-friendly budget
        ladder = [100, 250, 500, 1000, 2000]
        rec = next((x for x in ladder if x >= worst + 100), worst + 100)
        report.append(f"\n  max stable-rank step across arms = {worst}")
        report.append(f"\nrecommended_candidate_steps: {rec}")
        report.append("")
        for x in ladder:
            ok = "SUFFICIENT" if x >= worst else "INSUFFICIENT"
            report.append(f"  {x:>5d} steps: {ok} "
                          f"(>= max stable-rank step {worst}? {x >= worst})")
    else:
        report.append("\nrecommended_candidate_steps: UNRESOLVED (no arm stably ranked by 4000)")
    if unstable:
        report.append(f"\n  WARNING: arm(s) NOT stably ranked by 4000 steps: {unstable} "
                      f"-> would force a larger per-candidate budget.")

    text = "\n".join(report)
    print(text)
    return text


if __name__ == "__main__":
    main()
