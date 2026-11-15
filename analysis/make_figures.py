#!/usr/bin/env python3
"""Main figures (coverage vs. shift strength; NATOPS per-seed coverage; selected C), built from
results/main and results/followup.

Usage (from the repository root):  python analysis/make_figures.py [--out figures]
Writes fig_coverage.pdf, fig_natops30.pdf and fig_selected_c.pdf (and .png copies).
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--main", default="results/main")
ap.add_argument("--followup", default="results/followup")
ap.add_argument("--out", default="figures")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

DATASETS = ["Libras", "RacketSports", "NATOPS"]
A, N = 0.10, 90
# Fixed categorical order (Okabe-Ito, colour-blind safe) plus distinct markers for print.
STYLE = {
    "Standard CP": dict(color="#E69F00", marker="o"),
    "Oracle WCP": dict(color="#0072B2", marker="s"),
    "Est. WCP, $C=1$": dict(color="#D55E00", marker="v"),
    "Est. WCP, CV-$C$": dict(color="#009E73", marker="D"),
}
plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "lines.linewidth": 1.4, "lines.markersize": 4, "pdf.fonttype": 42,
})
INK, GRID = "#333333", "#DDDDDD"


def summ(path):
    return pd.read_csv(os.path.join(path, "summary.csv"))


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", os.path.join(args.out, name + ".pdf"))


# ---------------------------------------------------------------- Figure 1: coverage vs lambda
def series(path, method):
    s = summ(path)
    s = s[(s.method == method) & np.isclose(s.alpha, A) & (s.ncal == N)]
    return s


fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.6), sharex=True)
for row, (tilt, base, cv) in enumerate((
        ("Channel-mean tilt", "disjoint", "channel_Ccv_grid30"),
        ("Low-margin tilt", "margin", "margin_Ccv_grid30"))):
    sources = {
        "Standard CP": (os.path.join(args.main, base), "standard"),
        "Oracle WCP": (os.path.join(args.main, base), "oracle"),
        "Est. WCP, $C=1$": (os.path.join(args.main, base), "estimated"),
        "Est. WCP, CV-$C$": (os.path.join(args.followup, cv), "estimated"),
    }
    for col, ds in enumerate(DATASETS):
        ax = axes[row, col]
        ax.axhline(1 - A, color=INK, ls="--", lw=0.8, zorder=0)
        for k, (lab, (path, m)) in enumerate(sources.items()):
            x = series(path, m)
            x = x[x.dataset == ds].sort_values("lam")
            off = (k - 1.5) * 0.035
            lo, hi = x.coverage_ci_low.to_numpy(), x.coverage_ci_high.to_numpy()
            y = x.coverage_mean.to_numpy()
            ax.errorbar(x.lam + off, y, yerr=[y - lo, hi - y], capsize=1.5, elinewidth=0.7,
                        label=lab, **STYLE[lab])
        if ds == "NATOPS":
            ax.set_ylim(0.45, 1.02) if row == 1 else ax.set_ylim(0.70, 1.0)
        else:
            ax.set_ylim(0.75, 1.0)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_xticks([0, 0.5, 1])
        if row == 0:
            ax.set_title(ds)
        if col == 0:
            ax.set_ylabel(f"{tilt}\ncoverage")
        if row == 1:
            ax.set_xlabel(r"shift strength $\lambda$")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.04))
fig.tight_layout(rect=(0, 0, 1, 0.95))
save(fig, "fig_coverage")

# ---------------------------------------------------------------- Figure 2: NATOPS, 30 seeds
r = pd.read_csv(os.path.join(args.followup, "margin_30seeds", "raw_results.csv"))
c = r[(r.dataset == "NATOPS") & np.isclose(r.lam, 1.0) & np.isclose(r.alpha, A) & (r.ncal == N)]
cov = c.pivot_table(index="seed", columns="method", values="coverage")
full = c[c.method == "oracle"].set_index("seed").full_set_rate
order = cov.standard.sort_values().index
xs = np.arange(len(order))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.4, 3.1), sharex=True, gridspec_kw=dict(height_ratios=(2, 1)))
a1.axhline(1 - A, color=INK, ls="--", lw=0.8, zorder=0)
for i, sd in enumerate(order):
    a1.plot([i, i], [cov.standard[sd], cov.oracle[sd]], color=GRID, lw=0.8, zorder=1)
a1.plot(xs, cov.standard[order], ls="none", label="Standard CP", **STYLE["Standard CP"])
a1.plot(xs, cov.oracle[order], ls="none", label="Oracle WCP", **STYLE["Oracle WCP"])
a1.set_ylabel("target coverage")
a1.set_ylim(0, 1.03)
a1.grid(axis="y", color=GRID, lw=0.5)
a1.legend(loc="lower right", frameon=False)
a2.bar(xs, full[order], color=STYLE["Oracle WCP"]["color"], width=0.7)
a2.axhline(full.mean(), color=INK, lw=0.8, ls=":")
a2.set_ylabel("oracle full-set\nrate")
a2.set_ylim(0, 1)
a2.set_xlabel("seed (sorted by standard-CP coverage)")
a2.set_xticks([])
fig.tight_layout(h_pad=0.4)
save(fig, "fig_natops30")

# ---------------------------------------------------------------- Figure 3: selected C
r = pd.read_csv(os.path.join(args.followup, "channel_Ccv_grid30", "raw_results.csv"))
e = r[(r.method == "estimated") & np.isclose(r.alpha, A) & (r.ncal == N)]
fig, ax = plt.subplots(figsize=(3.4, 2.1))
rng = np.random.default_rng(0)
marks = {0.0: "o", 0.5: "s", 1.0: "D"}
cols = {0.0: "#999999", 0.5: "#56B4E9", 1.0: "#0072B2"}
for i, ds in enumerate(DATASETS):
    for j, lam in enumerate((0.0, 0.5, 1.0)):
        v = np.log10(e[(e.dataset == ds) & np.isclose(e.lam, lam)].ratio_C.to_numpy())
        x = i + (j - 1) * 0.25 + rng.uniform(-0.06, 0.06, len(v))
        ax.plot(x, v, ls="none", marker=marks[lam], color=cols[lam], ms=3.2, alpha=0.85,
                label=fr"$\lambda={lam:g}$" if i == 0 else None)
ax.axhline(-5, color=INK, ls="--", lw=0.8, label="floor $10^{-5}$ (v3.1)")
ax.axhline(-3, color=INK, ls=":", lw=0.8, label="floor $10^{-3}$ (v2)")
ax.set_xticks(range(3))
ax.set_xticklabels(DATASETS)
ax.set_ylabel(r"selected $\log_{10} C$")
ax.set_ylim(-5.4, 0)
ax.grid(axis="y", color=GRID, lw=0.5)
ax.legend(loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(-0.02, 1.0), columnspacing=0.8, handletextpad=0.3, fontsize=6.5)
fig.tight_layout()
save(fig, "fig_selected_c")
