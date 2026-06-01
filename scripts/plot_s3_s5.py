#!/usr/bin/env python3
"""Plot accuracy-vs-T length-generalization curves (one panel per task, one line
per model) from the fine-tune result JSONs. Real numbers only."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("paper/review/s3_s5_finetune_data")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("paper/review/s3_s5_finetune_acc_vs_T.png")
MODELS = ["e88", "gdn", "m2rnn"]
LABEL = {"e88": "E88 (delta, nonlinear)", "gdn": "GDN (linear recurrent)", "m2rnn": "M2RNN (raw-write, nonlinear)"}
COLOR = {"e88": "tab:blue", "gdn": "tab:red", "m2rnn": "tab:green"}
TASKS = ["parity", "s3_permutation", "s5_permutation"]
TITLE = {"parity": "Parity (S2, solvable)", "s3_permutation": "S3 (solvable)",
         "s5_permutation": "S5 (non-solvable / NC1-complete)"}

res = {}
for m in MODELS:
    p = DATA / f"{m}.json"
    if p.exists():
        res[m] = json.loads(p.read_text())

fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
for ax, task in zip(axes, TASKS):
    base = None
    train_max = None
    for m in MODELS:
        if m not in res or task not in res[m]["tasks"]:
            continue
        rec = res[m]["tasks"][task]
        Ts = sorted(int(k) for k in rec["acc_vs_T"])
        ys = [rec["acc_vs_T"][str(t)] for t in Ts]
        base = rec["random_baseline_acc"]
        train_max = rec["train_max_T"]
        ax.plot(Ts, ys, "o-", color=COLOR[m], label=LABEL[m])
    if base is not None:
        ax.axhline(base, ls=":", color="gray", lw=1, label=f"chance ({base:.3f})")
    if train_max is not None:
        ax.axvline(train_max, ls="--", color="k", lw=1, alpha=0.5, label=f"max trained T={train_max}")
    ax.set_xscale("log", base=2)
    ax.set_title(TITLE[task])
    ax.set_xlabel("sequence length T (log2)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
axes[0].set_ylabel("running-state accuracy")
fig.suptitle("1.3B production checkpoints fine-tuned on state-tracking — length generalization", fontsize=13)
fig.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=130)
print(f"wrote {OUT}")
