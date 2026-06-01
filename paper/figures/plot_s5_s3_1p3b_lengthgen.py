#!/usr/bin/env python3
"""
Generate the §6 1.3 B length-generalisation curves (S5, S3, parity) for the
three deployed architectures, matching the styling of the other §6 figures
(plot_expressivity_seeds.py / plot_hybrid_degradation.py).

Single-length bars are the right form for the 8 M S5 probe; the 1.3 B story is
*length generalisation*, so this figure plots accuracy vs sequence length T as
curves, with the longest trained length marked and the chance line shown.

Data source (REAL, confound-removed "to-competence" fine-tune run, the same run
the numbers in tab_s5_1p3b are read from):
  paper/review/s3_s5_finetune_v03_data_tocomp/{e88,gdn,m2rnn}.json
Each model JSON carries, per task, eval_lens + acc_vs_T measured on held-out,
test-disjoint sequences after a strict load of the public v0.3 checkpoint.
No mock / synthetic data.

Paper colour convention: Emender = blue, GDN = orange, M²RNN-CMA = red.
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(
    os.path.join(HERE, "..", "review", "s3_s5_finetune_v03_data_tocomp")
)

# Model file -> (paper label, colour, marker).  Order = legend/plot order.
MODELS = [
    ("e88",   "E88 (delta)",        "#4477AA", "o"),
    ("m2rnn", "M²RNN (raw-write)", "#CC3311", "s"),
    ("gdn",   "GDN (linear)",       "#EE7733", "^"),
]

# Tasks to plot, left-to-right, with display title and the group order g
# (chance = 1/g!).  S5 first (the headline non-solvable probe), then the
# solvable S3 control, then parity.
TASKS = [
    ("s5_permutation", "$S_5$ (non-solvable)", 0.008333333333333333),
    ("s3_permutation", "$S_3$ (solvable control)", 0.16666666666666666),
    ("parity",         "Parity (solvable control)", 0.5),
]


def load(model_file):
    with open(os.path.join(DATA_DIR, f"{model_file}.json")) as f:
        return json.load(f)


DATA = {mf: load(mf) for mf, *_ in MODELS}

# Trained length is shared across the cohort (train_lens 16,32,48,64).
TRAIN_MAX_T = int(DATA["e88"]["tasks"]["s5_permutation"]["train_max_T"])

# Style matched to plot_expressivity_seeds.py / plot_hybrid_degradation.py.
plt.rcParams.update({
    "font.family":     "DejaVu Sans",
    "font.size":        9,
    "axes.titlesize":  10,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
    "legend.fontsize":  8,
    "figure.dpi":     150,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.6), sharey=True)

for ax, (task, title, chance) in zip(axes, TASKS):
    for model_file, label, color, marker in MODELS:
        td = DATA[model_file]["tasks"][task]
        lens = [int(L) for L in td["eval_lens"]]
        accs = [float(td["acc_vs_T"][str(L)]) for L in lens]
        ax.plot(
            lens, accs,
            color=color, marker=marker, markersize=5,
            markeredgecolor="white", markeredgewidth=0.6,
            linewidth=1.5, zorder=4, label=label,
        )

    # Trained-length marker (everything to its right is extrapolation).
    ax.axvline(
        TRAIN_MAX_T, color="black", linestyle=":", linewidth=0.9,
        alpha=0.5, zorder=1,
    )
    ax.text(
        TRAIN_MAX_T, 0.012, f" trained $T{'='}{TRAIN_MAX_T}$",
        fontsize=7, color="black", alpha=0.65,
        ha="left", va="bottom",
    )

    # Chance line.
    ax.axhline(
        chance, color="black", linestyle="--", linewidth=0.8,
        alpha=0.55, zorder=1, label=f"chance = {chance:.4f}",
    )

    ax.set_xscale("log", base=2)
    all_lens = [int(L) for L in DATA["e88"]["tasks"][task]["eval_lens"]]
    ax.set_xticks(all_lens)
    ax.set_xticklabels([str(L) for L in all_lens], rotation=0)
    ax.minorticks_off()
    ax.set_xlabel("Sequence length $T$")
    ax.set_title(title, pad=6)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(loc="upper right", frameon=False, fontsize=7)

axes[0].set_ylabel("Accuracy")

fig.suptitle(
    "Length generalisation at the deployed 1.3 B scale: train $T \\leq 64$, "
    "evaluate to $T = 512$ (to-competence fine-tune)",
    fontsize=9, y=1.02,
)
fig.tight_layout()

out = os.path.join(HERE, "s5_s3_1p3b_lengthgen.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
