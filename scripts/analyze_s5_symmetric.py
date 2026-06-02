#!/usr/bin/env python3
"""Analyze the s5-symmetric-budget runs: print the per-model S5 acc-vs-T table at
the new symmetric budget side by side with the prior 2500-step matched and the
prior to-competence numbers, classify E88's trajectory (climbing / plateau /
ceiling) from the recorded traj_curve, and emit an acc-vs-T plot.

Reads ONLY real run JSONs. No fabrication.
"""
import json, sys
from pathlib import Path

NEW = Path("paper/review/s5_symmetric_data")
MATCHED = Path("paper/review/s3_s5_finetune_v03_data_matched")
TOCOMP = Path("paper/review/s3_s5_finetune_v03_data_tocomp")
MODELS = ["e88", "gdn", "m2rnn"]


def s5_grid(d):
    return d["tasks"]["s5_permutation"]["acc_vs_T"]


def load_grid(base, model):
    f = base / f"{model}.json"
    if not f.exists():
        return None, None
    d = json.load(open(f))
    return s5_grid(d), d.get("recipe", {})


def fmt(g, T):
    if g is None:
        return "  —  "
    v = g.get(str(T)) if str(T) in g else g.get(T)
    return f"{v:.3f}" if v is not None else "  —  "


def main():
    new = {m: load_grid(NEW, m) for m in MODELS}
    matched = {m: load_grid(MATCHED, m)[0] for m in MODELS}
    tocomp = {m: load_grid(TOCOMP, m)[0] for m in MODELS}

    # union of eval lengths from the new runs
    Ts = set()
    for m in MODELS:
        g, _ = new[m]
        if g:
            Ts.update(int(k) for k in g)
    Ts = sorted(Ts)

    print("\n=== Recipe (symmetric, a-priori) ===")
    for m in MODELS:
        _, r = new[m]
        if r:
            print(f"  {m}: steps={r['steps']} lr={r['lr']} gc={r['grad_clip']} "
                  f"warmup={r['warmup']} sched={r['lr_schedule']} "
                  f"train_lens={r['train_lens']}")

    print("\n=== S5 acc-vs-T @ SYMMETRIC budget (chance 0.0083) ===")
    hdr = "model    " + "".join(f"T{T:>5d}" for T in Ts)
    print(hdr)
    for m in MODELS:
        g, _ = new[m]
        print(f"{m:<8} " + "".join(f"{fmt(g,T):>6}" for T in Ts))

    print("\n=== Side-by-side @ key lengths: prior-matched(2500) | prior-tocomp | NEW-symmetric ===")
    for T in [64, 128, 256, 512, 768, 1024]:
        print(f"  --- T={T} ---")
        for m in MODELS:
            g, _ = new[m]
            print(f"    {m:<7} matched={fmt(matched[m],T)}  tocomp={fmt(tocomp[m],T)}  "
                  f"NEW={fmt(g,T)}")

    print("\n=== TRAJECTORY (acc vs step at length) — climbing / plateau / ceiling ===")
    for m in MODELS:
        f = NEW / f"{m}.json"
        if not f.exists():
            continue
        d = json.load(open(f))
        tc = d["tasks"]["s5_permutation"].get("traj_curve", [])
        if not tc:
            continue
        tl = d["tasks"]["s5_permutation"].get("traj_lens", [])
        print(f"\n  {m}: traj at lengths {tl}")
        print("    step   " + "".join(f"T{T:>5d}" for T in tl))
        for pt in tc:
            a = pt["acc_vs_T"]
            print(f"    {pt['step']:>5d}  " +
                  "".join(f"{a.get(str(T), a.get(T,0)):>6.3f}" for T in tl))
        # climbing analysis on the longest traj length using last vs prior window
        Tlong = max(tl)
        series = [(pt["step"], pt["acc_vs_T"].get(str(Tlong), pt["acc_vs_T"].get(Tlong)))
                  for pt in tc]
        if len(series) >= 4:
            last = series[-1][1]
            prev = series[-4][1]
            d_per_kstep = (last - prev) / max(1, (series[-1][0] - series[-4][0])) * 1000
            verdict = ("CLIMBING" if d_per_kstep > 0.01 else
                       "PLATEAU/CEILING" if abs(d_per_kstep) <= 0.01 else "DECLINING")
            print(f"    -> T{Tlong}: last={last:.3f} (step {series[-1][0]}), "
                  f"slope={d_per_kstep:+.4f}/1k-step over last window => {verdict}")

    # plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5.5))
        colors = {"e88": "#1f77b4", "gdn": "#2ca02c", "m2rnn": "#d62728"}
        for m in MODELS:
            g, _ = new[m]
            if not g:
                continue
            xs = sorted(int(k) for k in g)
            ys = [g[str(x)] if str(x) in g else g[x] for x in xs]
            ax.plot(xs, ys, "-o", color=colors[m], label=f"{m} (symmetric 24k)", lw=2)
            tg = tocomp[m]
            if tg:
                xs2 = sorted(int(k) for k in tg)
                ys2 = [tg[str(x)] if str(x) in tg else tg[x] for x in xs2]
                ax.plot(xs2, ys2, "--", color=colors[m], alpha=0.5,
                        label=f"{m} (prior to-comp)")
        ax.axhline(0.0083, color="gray", ls=":", label="chance 0.0083")
        ax.axvline(64, color="k", ls=":", alpha=0.3)
        ax.set_xlabel("eval length T (trained up to T=64)")
        ax.set_ylabel("S5 running-state accuracy")
        ax.set_title("S5 length-extrapolation @ symmetric 24k-step budget")
        ax.set_xscale("log"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
        out = "paper/review/s5_symmetric_acc_vs_T.png"
        fig.tight_layout(); fig.savefig(out, dpi=130)
        print(f"\nwrote {out}")
    except Exception as e:
        print(f"plot skipped: {e}")


if __name__ == "__main__":
    main()
