#!/usr/bin/env python3
"""Analysis of the independent H2 confirmation (seeds 40-69), as fixed in docs/H2_fresh_plan.txt.
H1, H3 and H4 come from the replication (seeds 10-39); Holm correction over the four tests.

Usage (from the repository root):  python analysis/analyze_h2_fresh.py
Writes <out-dir>/h2_fresh_results.txt and <out-dir>/h2_fresh_holm.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="results/h2_fresh")
ap.add_argument("--replication-dir", default="results/replication")
ap.add_argument("--followup-dir", default="results/followup")
args = ap.parse_args()
A, N = 0.10, 90
lines = []


def p(s=""):
    print(s)
    lines.append(str(s))


def raw(folder):
    for name in ("raw_results.csv", "raw_results.csv.gz"):
        f = os.path.join(folder, name)
        if os.path.exists(f):
            return pd.read_csv(f)
    raise FileNotFoundError(f"no raw_results.csv(.gz) in {folder}")


def paired(r, ds, lam, x, y):
    r = r[(r.dataset == ds) & np.isclose(r.lam, lam) & np.isclose(r.alpha, A) & (r.ncal == N)]
    pv = r.pivot_table(index="seed", columns="method", values="coverage")
    d = (pv[x] - pv[y]).to_numpy()
    return dict(seeds=f"{pv.index.min()}-{pv.index.max()}", n=len(d), mean_x=pv[x].mean(), sd_x=pv[x].std(ddof=1),
                mean_y=pv[y].mean(), sd_y=pv[y].std(ddof=1), diff=d.mean(), pos=int((d > 0).sum()),
                neg=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue)


def holm(ps):
    order, out, run = np.argsort(ps), np.zeros(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


fresh = raw(os.path.join(args.out_dir, "margin_seeds40_69"))
assert fresh.seed.min() == 40 and fresh.seed.max() == 69, "expected seeds 40-69"

# Overlap check: no seed of this run may appear in any earlier NATOPS low-margin run.
used = set()
for folder in (os.path.join(args.followup_dir, "margin_30seeds"), os.path.join(args.replication_dir, "margin_base")):
    if os.path.exists(folder):
        used |= set(raw(folder).seed.unique())
overlap = sorted(set(fresh.seed.unique()) & used)
p("=" * 78)
p("Independent confirmation of H2 (plan: docs/H2_fresh_plan.txt)")
p("=" * 78)
p(f"Seeds in this run: 40-69. Seeds used before in NATOPS low-margin runs: {min(used)}-{max(used)}. "
  f"Overlap: {overlap if overlap else 'none'}")

v3 = raw(os.path.join(args.replication_dir, "mean0_base"))
tests = [
    ("H1", "Libras, channel, lambda=1, oracle - standard", paired(v3, "Libras", 1.0, "oracle", "standard")),
    ("H2", "NATOPS, low-margin, lambda=1, oracle - standard", paired(fresh, "NATOPS", 1.0, "oracle", "standard")),
    ("H3", "NATOPS, channel, lambda=0, estimated - oracle", paired(v3, "NATOPS", 0.0, "estimated", "oracle")),
    ("H4", "NATOPS, channel, lambda=1, estimated - oracle", paired(v3, "NATOPS", 1.0, "estimated", "oracle")),
]
h = holm(np.array([t[2]["p"] for t in tests]))
rows = []
for (key, lab, t), hp in zip(tests, h):
    rows.append(dict(test=key, comparison=lab, **t, holm_p=hp))
tab = pd.DataFrame(rows)
p()
p(tab.round(4).to_string(index=False))
tab.to_csv(os.path.join(args.out_dir, "h2_fresh_holm.csv"), index=False)

h2 = rows[1]
ok = h2["holm_p"] < 0.05 and h2["diff"] > 0
p()
p(f"H2 on fresh seeds 40-69: standard {h2['mean_y']:.3f} +/- {h2['sd_y']:.3f}, oracle {h2['mean_x']:.3f} +/- "
  f"{h2['sd_x']:.3f}, difference {h2['diff']:+.3f} ({h2['pos']}/{h2['neg']}), p = {h2['p']:.4g}, "
  f"Holm p = {h2['holm_p']:.4g}")
p(f"Decision (pre-specified): H2 {'CONFIRMED' if ok else 'NOT confirmed'}. "
  f"Holm family: {int((h < 0.05).sum())}/4 significant.")

# Failure counts of this run (descriptive), for the low-margin table with 30 independent seeds.
s = pd.read_csv(os.path.join(args.out_dir, "margin_seeds40_69", "summary.csv"))
p()
p("Failures (95% CI below 1-alpha) over 81 settings, seeds 40-69:")
for m, g in s.groupby("method"):
    f = g.coverage_ci_high < 1 - g.alpha
    per = {d: int(f[g.dataset == d].sum()) for d in ("Libras", "RacketSports", "NATOPS")}
    p(f"   {m:18s} {int(f.sum()):3d}  {per}")
o = fresh[(fresh.dataset == "NATOPS") & np.isclose(fresh.lam, 1.0) & np.isclose(fresh.alpha, A)
          & (fresh.ncal == N) & (fresh.method == "oracle")]
p(f"Oracle full-set rate (NATOPS, lambda=1): {o.full_set_rate.mean():.3f}; coverage on non-full sets "
  f"{o.coverage_nonvacuous.mean():.3f}")

open(os.path.join(args.out_dir, "h2_fresh_results.txt"), "w").write("\n".join(lines) + "\n")
