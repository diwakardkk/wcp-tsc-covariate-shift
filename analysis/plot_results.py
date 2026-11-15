#!/usr/bin/env python3
"""Figures from results/summary.csv (written by run_experiment.py).

Usage:  python plot_results.py --summary results/summary.csv --out-dir results/figures

Saves (one panel per dataset, error bars = 95% CI over seeds):
  coverage_vs_lambda      coverage vs lambda per method, dashed line at 1 - alpha
  set_size_vs_lambda      mean prediction-set size vs lambda
  ess_ratio_vs_lambda     ESS / n_cal of the calibration weights vs lambda
  coverage_vs_ncal        coverage vs calibration size (at --lam-ncal)
  clipping                coverage (top) and set size (bottom) vs clipping threshold
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

MAIN_METHODS = ["standard", "oracle", "estimated"]
LABELS = {"standard": "Standard CP", "oracle": "Oracle WCP", "estimated": "Estimated WCP"}
# Validated categorical slots 1-3 (CVD-safe for all pairs), fixed per method.
COLORS = {"standard": "#2a78d6", "oracle": "#eb6834", "estimated": "#1baf7a"}
MARKERS = {"standard": "o", "oracle": "s", "estimated": "^"}
# Single-hue ordinal ramp for lambda (light -> dark = weak -> strong shift).
LAMBDA_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def ci(df, metric):
    """(mean, lower err, upper err) using the 95% CI columns, falling back to std."""
    m = df[f"{metric}_mean"].to_numpy()
    if f"{metric}_ci_low" in df:
        lo, hi = df[f"{metric}_ci_low"].to_numpy(), df[f"{metric}_ci_high"].to_numpy()
    else:
        sd = df[f"{metric}_std"].to_numpy()
        lo, hi = m - sd, m + sd
    err = np.vstack([m - lo, hi - m])
    return m, np.nan_to_num(np.clip(err, 0, None))


def panels(datasets, nrows=1, height=2.8):
    fig, axes = plt.subplots(nrows, len(datasets), figsize=(3.2 * len(datasets), height * nrows),
                             squeeze=False, sharey="row")
    return fig, axes


BOUNDS = {"coverage": (0.0, 1.02), "ess_ratio": (0.0, 1.05), "set_size": (0.0, None)}


def bound(ax, metric):
    """Rates live in [0, 1] and sizes are >= 0; keep wide few-seed CIs from distorting the axis."""
    lo, hi = BOUNDS.get(metric, (None, None))
    ax.set_ylim(bottom=lo, top=hi)


def legend_below(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="upper center", ncol=ncol, bbox_to_anchor=(0.5, 0.0))


def save(fig, out_dir, name, formats):
    for ext in formats:
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"saved {name}.{{{','.join(formats)}}}")


def method_lines(ax, sub, xcol, metric, methods, xticks):
    """One line per method; small horizontal dodge so error bars do not overlap."""
    span = (max(xticks) - min(xticks)) or 1.0
    dodge = 0.015 * span
    for j, meth in enumerate(methods):
        d = sub[sub["method"] == meth].sort_values(xcol)
        if d.empty:
            continue
        m, err = ci(d, metric)
        x = d[xcol].to_numpy(float) + (j - (len(methods) - 1) / 2) * dodge
        ax.errorbar(x, m, yerr=err, color=COLORS[meth], marker=MARKERS[meth], ms=5,
                    lw=2, capsize=3, elinewidth=1, label=LABELS[meth])
    ax.set_xticks(xticks)


def vs_lambda(s, metric, ylabel, name, a):
    sub = s[np.isclose(s["alpha"], a.alpha) & (s["ncal"] == a.ncal)]
    datasets = list(dict.fromkeys(sub["dataset"]))
    lams = sorted(sub["lam"].unique())
    fig, axes = panels(datasets)
    for ax, ds in zip(axes[0], datasets):
        method_lines(ax, sub[sub["dataset"] == ds], "lam", metric, MAIN_METHODS, lams)
        ax.set_title(ds, color=INK)
        ax.set_xlabel("shift strength λ")
        if metric == "coverage":
            ax.axhline(1 - a.alpha, color=MUTED, ls="--", lw=1)
        bound(ax, metric)
    axes[0, 0].set_ylabel(ylabel)
    legend_below(fig, axes[0, 0], 3)
    fig.suptitle(f"{ylabel} vs shift (α = {a.alpha:g}, n_cal = {a.ncal})", color=INK, y=1.02)
    save(fig, a.out_dir, name, a.formats)


def vs_ncal(s, a):
    lam = a.lam_ncal if a.lam_ncal is not None else s["lam"].max()
    sub = s[np.isclose(s["alpha"], a.alpha) & np.isclose(s["lam"], lam)]
    datasets = list(dict.fromkeys(sub["dataset"]))
    ncals = sorted(sub["ncal"].unique())
    fig, axes = panels(datasets)
    for ax, ds in zip(axes[0], datasets):
        method_lines(ax, sub[sub["dataset"] == ds], "ncal", "coverage", MAIN_METHODS, ncals)
        ax.axhline(1 - a.alpha, color=MUTED, ls="--", lw=1)
        ax.set_title(ds, color=INK)
        ax.set_xlabel("calibration size n_cal")
        bound(ax, "coverage")
    axes[0, 0].set_ylabel("Coverage")
    legend_below(fig, axes[0, 0], 3)
    fig.suptitle(f"Coverage vs calibration size (α = {a.alpha:g}, λ = {lam:g})", color=INK, y=1.02)
    save(fig, a.out_dir, "coverage_vs_ncal", a.formats)


def clipping(s, a):
    sub = s[np.isclose(s["alpha"], a.alpha) & (s["ncal"] == a.ncal)
            & s["method"].str.startswith("estimated")].copy()
    if sub.empty:
        print("no estimated-weight rows; skipping clipping plot")
        return
    sub["clip_val"] = sub["method"].map(
        lambda m: np.inf if m == "estimated" else float(m.replace("estimated_clip", "")))
    clips = sorted(sub["clip_val"].unique())
    pos = {c: i for i, c in enumerate(clips)}
    sub["xpos"] = sub["clip_val"].map(pos)
    ticklabels = ["none" if np.isinf(c) else f"{c:g}" for c in clips]
    datasets = list(dict.fromkeys(sub["dataset"]))
    lams = sorted(sub["lam"].unique())
    ramp = LAMBDA_RAMP[-len(lams):] if len(lams) <= len(LAMBDA_RAMP) else \
        plt.cm.Blues(np.linspace(0.35, 0.95, len(lams)))

    fig, axes = panels(datasets, nrows=2, height=2.6)
    for col, ds in enumerate(datasets):
        for row, (metric, ylabel) in enumerate([("coverage", "Coverage"), ("set_size", "Mean set size")]):
            ax = axes[row, col]
            for j, lam in enumerate(lams):
                d = sub[(sub["dataset"] == ds) & np.isclose(sub["lam"], lam)].sort_values("xpos")
                m, err = ci(d, metric)
                x = d["xpos"].to_numpy(float) + (j - (len(lams) - 1) / 2) * 0.1
                ax.errorbar(x, m, yerr=err, color=ramp[j], marker="o", ms=5, lw=2,
                            capsize=3, elinewidth=1, label=f"λ = {lam:g}")
            if metric == "coverage":
                ax.axhline(1 - a.alpha, color=MUTED, ls="--", lw=1)
                ax.set_title(ds, color=INK)
            ax.set_xticks(range(len(clips)), ticklabels)
            bound(ax, metric)
            if row == 1:
                ax.set_xlabel("weight clipping threshold")
            if col == 0:
                ax.set_ylabel(ylabel)
    legend_below(fig, axes[0, 0], len(lams))
    fig.suptitle(f"Estimated WCP: effect of weight clipping (α = {a.alpha:g}, n_cal = {a.ncal})",
                 color=INK, y=1.01)
    save(fig, a.out_dir, "clipping", a.formats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/summary.csv")
    ap.add_argument("--out-dir", default=None, help="default: <summary dir>/figures")
    ap.add_argument("--alpha", type=float, default=0.10)
    ap.add_argument("--ncal", type=int, default=90)
    ap.add_argument("--lam-ncal", type=float, default=None,
                    help="lambda for the coverage-vs-ncal plot (default: largest)")
    ap.add_argument("--formats", nargs="+", default=["png", "pdf"])
    a = ap.parse_args()
    a.out_dir = a.out_dir or os.path.join(os.path.dirname(os.path.abspath(a.summary)), "figures")
    os.makedirs(a.out_dir, exist_ok=True)

    s = pd.read_csv(a.summary)
    vs_lambda(s, "coverage", "Coverage", "coverage_vs_lambda", a)
    vs_lambda(s, "set_size", "Mean set size", "set_size_vs_lambda", a)
    vs_lambda(s, "ess_ratio", "ESS ratio", "ess_ratio_vs_lambda", a)
    vs_ncal(s, a)
    clipping(s, a)
    print(f"figures in {a.out_dir}/")


if __name__ == "__main__":
    main()
