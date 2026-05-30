#!/usr/bin/env python3
"""Render the v2 CMA-ES candidate-burst diagnostic figure.

Run from the repository root:

    python3 paper/results/cmaes_burst_v2/plot_cmaes_burst.py

The script reads only the normalized files in this directory and writes
`cmaes_burst_v2.png` plus `cmaes_burst_v2.pdf`. The x-axis is candidate-local
elapsed time from 0 to 15 minutes, not full CMA-ES sweep elapsed time.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


OUT = Path(__file__).resolve().parent
POINTS_CSV = OUT / "cmaes_trajectory_points.csv.gz"
EVAL_SUMMARY_CSV = OUT / "cmaes_eval_summary.csv"
OUT_PREFIX = OUT / "cmaes_burst_v2"

WINDOW_MINUTES = (0.0, 15.0)
SUMMARY_STEP_MINUTES = 0.25

TRAJECTORY_KEYS = [
    "architecture_family",
    "sweep_id",
    "run_id",
    "config_id",
    "trial_id",
]

ARCHITECTURE_ORDER = [
    "E88 delta",
    "E88 normal rerun",
    "E88 raw-write",
    "E97",
    "E97 normal",
    "E97 raw-update",
    "E97 linear-state",
    "GDN2",
    "FLA-GDN",
    "M2RNN",
    "Mamba2",
    "Transformer",
    "Transformer rerun",
]

ARCHITECTURE_COLORS = {
    "E88 delta": "#1f77b4",
    "E88 normal rerun": "#4e79a7",
    "E88 raw-write": "#76b7b2",
    "E97": "#59a14f",
    "E97 normal": "#2ca02c",
    "E97 raw-update": "#8cd17d",
    "E97 linear-state": "#b6992d",
    "GDN2": "#f28e2b",
    "FLA-GDN": "#ff7f0e",
    "M2RNN": "#d62728",
    "Mamba2": "#9467bd",
    "Transformer": "#8c564b",
    "Transformer rerun": "#e377c2",
}

STATUS_LABELS = {
    "complete_usable": "complete",
    "partial_in_flight_no_results_json": "partial",
    "partial_no_results_json": "partial",
    "failed_cma_update_after_one_eval": "failed",
}

BEST_STYLES = {
    "avg": {
        "label": "CMA-ES avg-loss best flag",
        "color": "#b2182b",
        "linestyle": "-",
        "marker": "o",
    },
    "final": {
        "label": "CMA-ES final-loss best flag",
        "color": "#2166ac",
        "linestyle": (0, (3.2, 2.2)),
        "marker": "s",
    },
    "avg_final": {
        "label": "CMA-ES avg+final best flags",
        "color": "#5e3c99",
        "linestyle": "-",
        "marker": "D",
    },
}

POINT_COLUMNS = [
    "architecture_family",
    "sweep_id",
    "run_id",
    "run_status",
    "config_id",
    "trial_id",
    "candidate_elapsed_minutes",
    "loss",
    "is_results_best_loss",
    "is_results_best_final_loss",
    "is_observed_best_loss",
    "is_observed_best_final_loss",
]

EVAL_COLUMNS = [
    "architecture_family",
    "run_status",
    "sweep_id",
    "run_id",
    "config_id",
    "trial_id",
    "trajectory_rows",
    "wallclock_span_minutes",
    "train_minutes",
    "is_results_best_loss",
    "is_results_best_final_loss",
    "is_observed_best_loss",
    "is_observed_best_final_loss",
    "selected_best_flag_source",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot v2 CMA-ES per-candidate training bursts from normalized data."
        )
    )
    parser.add_argument(
        "--points",
        type=Path,
        default=POINTS_CSV,
        help="Normalized trajectory-point CSV or CSV.GZ.",
    )
    parser.add_argument(
        "--eval-summary",
        type=Path,
        default=EVAL_SUMMARY_CSV,
        help="Normalized eval-summary CSV.",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=OUT_PREFIX,
        help="Output path prefix; .png and .pdf are written.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG rasterization DPI.",
    )
    return parser.parse_args()


def load_points(path: Path) -> pd.DataFrame:
    points = pd.read_csv(path, usecols=POINT_COLUMNS)
    points["candidate_elapsed_minutes"] = pd.to_numeric(
        points["candidate_elapsed_minutes"], errors="coerce"
    )
    points["loss"] = pd.to_numeric(points["loss"], errors="coerce")
    points = points.dropna(subset=["candidate_elapsed_minutes", "loss"])
    lo, hi = WINDOW_MINUTES
    points = points[
        (points["candidate_elapsed_minutes"] >= lo)
        & (points["candidate_elapsed_minutes"] <= hi)
    ].copy()
    if points.empty:
        raise RuntimeError(
            f"No trajectory points remain after filtering to {lo:g}-{hi:g} minutes."
        )
    return points


def load_eval_summary(path: Path) -> pd.DataFrame:
    evals = pd.read_csv(path, usecols=EVAL_COLUMNS)
    if evals.empty:
        raise RuntimeError(f"No eval-summary rows found in {path}.")
    return evals


def architecture_order(points: pd.DataFrame) -> list[str]:
    present = set(points["architecture_family"].unique())
    ordered = [arch for arch in ARCHITECTURE_ORDER if arch in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def normalize_status(status: str) -> str:
    return STATUS_LABELS.get(str(status), str(status))


def status_text(evals: pd.DataFrame) -> str:
    counts = (
        evals["run_status"]
        .map(normalize_status)
        .value_counts()
        .sort_index()
    )
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def raw_line_alpha(n_trajectories: int) -> float:
    if n_trajectories <= 2:
        return 0.45
    if n_trajectories <= 40:
        return 0.18
    if n_trajectories <= 100:
        return 0.085
    return 0.055


def group_trajectories(points: pd.DataFrame) -> Iterable[tuple[tuple[object, ...], pd.DataFrame]]:
    return points.groupby(TRAJECTORY_KEYS, sort=False, observed=True)


def unique_xy(group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ordered = group.sort_values("candidate_elapsed_minutes")
    xy = (
        ordered.groupby("candidate_elapsed_minutes", sort=True, as_index=False)["loss"]
        .mean()
        .dropna()
    )
    return (
        xy["candidate_elapsed_minutes"].to_numpy(dtype=float),
        xy["loss"].to_numpy(dtype=float),
    )


def interpolate_summary(points: pd.DataFrame, grid: np.ndarray) -> dict[str, np.ndarray]:
    rows: list[np.ndarray] = []
    for _, group in group_trajectories(points):
        x, y = unique_xy(group)
        if len(x) < 1:
            continue
        interpolated = np.full_like(grid, np.nan, dtype=float)
        if len(x) == 1:
            nearest = int(np.argmin(np.abs(grid - x[0])))
            if abs(grid[nearest] - x[0]) <= SUMMARY_STEP_MINUTES / 2:
                interpolated[nearest] = y[0]
        else:
            mask = (grid >= x[0]) & (grid <= x[-1])
            interpolated[mask] = np.interp(grid[mask], x, y)
        rows.append(interpolated)

    if not rows:
        nan = np.full_like(grid, np.nan, dtype=float)
        return {
            "count": np.zeros_like(grid, dtype=int),
            "q10": nan,
            "q25": nan,
            "q50": nan,
            "q75": nan,
            "q90": nan,
        }

    values = np.vstack(rows)
    counts = np.sum(~np.isnan(values), axis=0)

    def quantile(q: float) -> np.ndarray:
        out = np.full_like(grid, np.nan, dtype=float)
        for idx in np.where(counts > 0)[0]:
            out[idx] = float(np.nanquantile(values[:, idx], q))
        return out

    return {
        "count": counts,
        "q10": quantile(0.10),
        "q25": quantile(0.25),
        "q50": quantile(0.50),
        "q75": quantile(0.75),
        "q90": quantile(0.90),
    }


def best_kind(group: pd.DataFrame) -> str | None:
    avg_best = bool(
        group["is_results_best_loss"].any() or group["is_observed_best_loss"].any()
    )
    final_best = bool(
        group["is_results_best_final_loss"].any()
        or group["is_observed_best_final_loss"].any()
    )
    if avg_best and final_best:
        return "avg_final"
    if avg_best:
        return "avg"
    if final_best:
        return "final"
    return None


def draw_panel(
    ax: plt.Axes,
    arch: str,
    points: pd.DataFrame,
    evals: pd.DataFrame,
    grid: np.ndarray,
) -> None:
    color = ARCHITECTURE_COLORS.get(arch, "#4d4d4d")
    trajectory_count = evals.drop_duplicates(TRAJECTORY_KEYS).shape[0]
    alpha = raw_line_alpha(trajectory_count)

    for _, group in group_trajectories(points):
        x, y = unique_xy(group)
        ax.plot(
            x,
            y,
            color="#30343b",
            alpha=alpha,
            linewidth=0.42,
            rasterized=True,
            zorder=1,
        )

    summary = interpolate_summary(points, grid)
    support = summary["count"] > 0
    band_support = summary["count"] >= 2
    if np.any(band_support):
        ax.fill_between(
            grid,
            summary["q10"],
            summary["q90"],
            where=band_support,
            color=color,
            alpha=0.12,
            linewidth=0.0,
            zorder=2,
        )
        ax.fill_between(
            grid,
            summary["q25"],
            summary["q75"],
            where=band_support,
            color=color,
            alpha=0.22,
            linewidth=0.0,
            zorder=3,
        )
    if np.any(support):
        ax.plot(
            grid[support],
            summary["q50"][support],
            color=color,
            linewidth=1.7,
            zorder=4,
        )

    for _, group in group_trajectories(points):
        kind = best_kind(group)
        if kind is None:
            continue
        style = BEST_STYLES[kind]
        x, y = unique_xy(group)
        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.35,
            alpha=0.96,
            zorder=6,
        )
        ax.scatter(
            [x[-1]],
            [y[-1]],
            s=17,
            marker=style["marker"],
            color=style["color"],
            edgecolor="white",
            linewidth=0.35,
            zorder=7,
        )

    ax.set_title(
        f"{arch}\n{trajectory_count} configs; {status_text(evals)}",
        fontsize=8.6,
        pad=5.0,
    )
    ax.set_xlim(WINDOW_MINUTES)
    ax.set_yscale("log")
    ax.set_ylim(4.6, 30.0)
    ax.set_xticks([0, 5, 10, 15])
    ax.set_yticks([5, 6, 8, 10, 15, 20, 30])
    ax.set_yticklabels(["5", "6", "8", "10", "15", "20", "30"])
    ax.grid(True, which="both", axis="both", alpha=0.24, linewidth=0.55)
    ax.tick_params(labelsize=7.5, length=2.5, pad=2)


def legend_handles() -> list[object]:
    return [
        Line2D(
            [0],
            [0],
            color="#30343b",
            lw=0.8,
            alpha=0.22,
            label="individual candidate trajectory",
        ),
        Line2D([0], [0], color="#1f77b4", lw=1.8, label="architecture median"),
        Patch(facecolor="#1f77b4", alpha=0.22, label="25-75% band"),
        Patch(facecolor="#1f77b4", alpha=0.12, label="10-90% band"),
        Line2D(
            [0],
            [0],
            color=BEST_STYLES["avg"]["color"],
            lw=1.4,
            marker=BEST_STYLES["avg"]["marker"],
            label=BEST_STYLES["avg"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=BEST_STYLES["final"]["color"],
            lw=1.4,
            linestyle=BEST_STYLES["final"]["linestyle"],
            marker=BEST_STYLES["final"]["marker"],
            label=BEST_STYLES["final"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=BEST_STYLES["avg_final"]["color"],
            lw=1.4,
            marker=BEST_STYLES["avg_final"]["marker"],
            label=BEST_STYLES["avg_final"]["label"],
        ),
    ]


def render(points: pd.DataFrame, evals: pd.DataFrame, out_prefix: Path, dpi: int) -> None:
    order = architecture_order(points)
    grid = np.arange(
        WINDOW_MINUTES[0],
        WINDOW_MINUTES[1] + SUMMARY_STEP_MINUTES / 2,
        SUMMARY_STEP_MINUTES,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    ncols = 4
    nrows = int(np.ceil(len(order) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(14.4, 11.0),
        sharex=True,
        sharey=True,
    )
    axes_array = np.asarray(axes).reshape(-1)

    for ax, arch in zip(axes_array, order):
        draw_panel(
            ax,
            arch,
            points.loc[points["architecture_family"] == arch],
            evals.loc[evals["architecture_family"] == arch],
            grid,
        )

    for ax in axes_array[len(order) :]:
        ax.axis("off")

    for ax in axes_array:
        if ax.has_data():
            ax.label_outer()

    fig.suptitle(
        "CMA-ES candidate bursts by architecture",
        fontsize=14.5,
        y=0.988,
    )
    fig.text(
        0.5,
        0.958,
        (
            "All normalized candidate trajectories are shown for candidate_elapsed_minutes "
            "0-15. This is per-candidate training time, not full sweep elapsed time."
        ),
        ha="center",
        va="center",
        fontsize=9.2,
    )
    fig.supxlabel("Candidate-local elapsed minutes", fontsize=10.2, y=0.083)
    fig.supylabel(
        "Training loss (natural-log cross entropy; log-scaled axis)",
        fontsize=10.2,
        x=0.012,
    )
    fig.legend(
        handles=legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.012),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.8,
        columnspacing=1.3,
    )
    fig.tight_layout(rect=[0.035, 0.095, 0.998, 0.94], h_pad=1.05, w_pad=0.55)

    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_prefix.with_suffix(f".{ext}")
        kwargs = {"bbox_inches": "tight"}
        if ext == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        print(f"wrote {path}")
    plt.close(fig)


def print_coverage(points: pd.DataFrame, evals: pd.DataFrame) -> None:
    represented = evals.drop_duplicates(TRAJECTORY_KEYS).shape[0]
    status_counts = (
        evals["run_status"].map(normalize_status).value_counts().sort_index().to_dict()
    )
    best_counts = {
        "avg_loss_best_flags": int(
            (
                evals["is_results_best_loss"].astype(bool)
                | evals["is_observed_best_loss"].astype(bool)
            ).sum()
        ),
        "final_loss_best_flags": int(
            (
                evals["is_results_best_final_loss"].astype(bool)
                | evals["is_observed_best_final_loss"].astype(bool)
            ).sum()
        ),
    }
    print(
        "coverage: "
        f"{represented} candidate trajectories, {len(points)} step-loss rows, "
        f"{points['architecture_family'].nunique()} architecture panels"
    )
    print(f"status counts: {status_counts}")
    print(f"best-flag counts: {best_counts}")
    print(
        "candidate elapsed minute range: "
        f"{points['candidate_elapsed_minutes'].min():.3f}-"
        f"{points['candidate_elapsed_minutes'].max():.3f}"
    )


def main() -> None:
    args = parse_args()
    points = load_points(args.points)
    evals = load_eval_summary(args.eval_summary)
    print_coverage(points, evals)
    render(points, evals, args.out_prefix, args.dpi)


if __name__ == "__main__":
    main()
