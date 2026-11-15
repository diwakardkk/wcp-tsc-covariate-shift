#!/usr/bin/env python3
"""Summary tables and key numbers of all experiments, computed from the stored results.

Usage (from the repository root):  python analysis/summary_tables.py
Writes results/summary_tables.txt (also printed).

Conventions: alpha = 0.10 and n_cal = 90 unless stated; "failure" = the 95% t-interval of
coverage over seeds lies entirely below 1 - alpha; mean_SD = mean with seed SD.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
ap.add_argument("--out", default=os.path.join("results", "summary_tables.txt"))
args = ap.parse_args()
MAIN, FU, REP, H2 = (os.path.join(args.results, d) for d in ("main", "followup", "replication", "h2_fresh"))
A, N = 0.10, 90
DS = ["Libras", "RacketSports", "NATOPS"]
LAMS = [0.0, 0.5, 1.0]
lines = []
_cache = {}


def p(s=""):
    print(s)
    lines.append(str(s))


def title(s):
    p()
    p("=" * 100)
    p(s)
    p("=" * 100)


def S(path):
    if path not in _cache:
        _cache[path] = pd.read_csv(os.path.join(path, "summary.csv"))
    return _cache[path]


def RAW(path):
    key = ("raw", path)
    if key not in _cache:
        for name in ("raw_results.csv", "raw_results.csv.gz"):
            f = os.path.join(path, name)
            if os.path.exists(f):
                _cache[key] = pd.read_csv(f)
                break
    return _cache[key]


def row(path, method, ds, lam, a=A, n=N):
    s = S(path)
    q = s[(s.method == method) & (s.dataset == ds) & np.isclose(s.lam, lam) & np.isclose(s.alpha, a)]
    if "ncal" in q and q.ncal.nunique() > 1:
        q = q[q.ncal == n]
    assert len(q) == 1, (path, method, ds, lam, a, n, len(q))
    return q.iloc[0]


def fails(path, method, by_dataset=True):
    s = S(path)
    s = s[s.method == method]
    f = s.coverage_ci_high < 1 - s.alpha
    if not by_dataset:
        return int(f.sum())
    per = "/".join(str(int(f[s.dataset == d].sum())) for d in DS)
    return f"{int(f.sum())} ({per})"


def msd(r):
    flag = "*" if r.coverage_ci_high < 1 - r.alpha else ("+" if r.coverage_std > 0.10 else "")
    return f"{r.coverage_mean:.3f}_{r.coverage_std:.3f}{flag}"


def wilcoxon(raw, ds, lam, x, y, seeds=None):
    r = raw[(raw.dataset == ds) & np.isclose(raw.lam, lam) & np.isclose(raw.alpha, A) & (raw.ncal == N)]
    if seeds is not None:
        r = r[r.seed.isin(list(seeds))]
    pv = r.pivot_table(index="seed", columns="method", values="coverage")
    d = (pv[x] - pv[y]).to_numpy()
    return dict(n=len(d), diff=d.mean(), pos=int((d > 0).sum()), neg=int((d < 0).sum()), p=stats.wilcoxon(d).pvalue)


def holm(ps):
    ps = np.asarray(ps)
    order, out, run = np.argsort(ps), np.zeros(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


# ---------------------------------------------------------------- Table: failures, main study
title("Failures out of 81 settings (total, Libras/RacketSports/NATOPS)")
p(f"{'Method':34s}{'Channel, 10 seeds':>20s}{'Low-margin, 10':>18s}{'Low-margin, 30':>18s}")
for lab, ch, mg, mg30, m in (
        ("Standard CP", "disjoint", "margin", True, "standard"),
        ("Oracle WCP", "disjoint", "margin", True, "oracle"),
        ("Est., C=1 (pre-specified)", "disjoint", "margin", True, "estimated"),
        ("Est., C=1, clip 20", "disjoint", "margin", True, "estimated_clip20"),
        ("Est., C=1, clip 10", "disjoint", "margin", True, "estimated_clip10")):
    p(f"{lab:34s}{fails(os.path.join(MAIN, ch), m):>20s}{fails(os.path.join(MAIN, mg), m):>18s}"
      f"{fails(os.path.join(FU, 'margin_30seeds'), m):>18s}")
p(f"{'Est., CV-C (grid 1e-5..1e2, 30)':34s}{fails(os.path.join(FU, 'channel_Ccv_grid30'), 'estimated'):>20s}"
  f"{fails(os.path.join(FU, 'margin_Ccv_grid30'), 'estimated'):>18s}{'--':>18s}")
p(f"{'Est., PCA 14 / PCA 42':34s}"
  f"{fails(os.path.join(MAIN, 'disjoint_pca14'), 'estimated', False)} / "
  f"{fails(os.path.join(MAIN, 'disjoint_pca42'), 'estimated', False)}")
p(f"{'Est., tilting variable only':34s}{fails(os.path.join(MAIN, 'disjoint_tilt'), 'estimated'):>20s}"
  f"{fails(os.path.join(MAIN, 'margin_tilt'), 'estimated'):>18s}")
p(f"{'Est., C=0.1 (post hoc)':34s}{fails(os.path.join(MAIN, 'disjoint_C01'), 'estimated'):>20s}")
p(f"{'In-pool protocol, standard CP':34s}{fails(os.path.join(MAIN, 'in_pool'), 'standard'):>20s}")

# ---------------------------------------------------------------- Table: coverage
title("Accuracy and coverage (mean_SD; *: failure, +: SD > 0.10), alpha=0.10, n_cal=90")
for tilt, base, cv in (("Channel-mean tilt", "disjoint", "channel_Ccv_grid30"),
                       ("Low-margin tilt", "margin", "margin_Ccv_grid30")):
    p(f"--- {tilt}")
    p(f"{'Dataset':14s}{'lam':>5s}{'Acc.':>8s}{'Standard':>15s}{'Oracle':>15s}{'Est. C=1':>15s}{'Est. CV-C':>15s}")
    for ds in DS:
        for lam in LAMS:
            b = os.path.join(MAIN, base)
            st = row(b, "standard", ds, lam)
            p(f"{ds:14s}{lam:5.1f}{st.target_acc_mean:8.3f}{msd(st):>15s}{msd(row(b, 'oracle', ds, lam)):>15s}"
              f"{msd(row(b, 'estimated', ds, lam)):>15s}{msd(row(os.path.join(FU, cv), 'estimated', ds, lam)):>15s}")
p("Calibration size, channel tilt, lambda=1 (standard / oracle / C=1 / CV-C):")
for ds in DS:
    vals = []
    for n in (30, 60, 90):
        b = os.path.join(MAIN, "disjoint")
        vals.append(f"n={n}: " + " ".join(f"{row(x, m, ds, 1.0, n=n).coverage_mean:.3f}" for x, m in
                                          ((b, "standard"), (b, "oracle"), (b, "estimated"),
                                           (os.path.join(FU, "channel_Ccv_grid30"), "estimated"))))
    p(f"   {ds:14s} " + " | ".join(vals))

# ---------------------------------------------------------------- Table: low-margin, all alpha
title("Low-margin tilt, lambda=1, n_cal=90, seeds 0-9")
raw30 = RAW(os.path.join(FU, "margin_30seeds"))
r10 = raw30[raw30.seed < 10]
p(f"{'Dataset':14s}{'alpha':>6s}{'Cov S':>8s}{'Cov O':>8s}{'Size S':>8s}{'Size O':>8s}{'Full O':>8s}"
  f"{'Inf O':>8s}{'Cov non-full O':>16s}")
for ds in DS:
    for a in (0.05, 0.10, 0.20):
        q = r10[(r10.dataset == ds) & np.isclose(r10.lam, 1.0) & np.isclose(r10.alpha, a) & (r10.ncal == N)]
        s_, o_ = q[q.method == "standard"], q[q.method == "oracle"]
        flag = "*" if row(os.path.join(MAIN, "margin"), "standard", ds, 1.0, a).coverage_ci_high < 1 - a else ""
        p(f"{ds:14s}{a:6.2f}{s_.coverage.mean():7.3f}{flag:1s}{o_.coverage.mean():8.3f}{s_.set_size.mean():8.2f}"
          f"{o_.set_size.mean():8.2f}{o_.full_set_rate.mean():8.3f}{o_.inf_threshold_rate.mean():8.3f}"
          f"{o_.coverage_nonvacuous.mean():16.3f}")
o = raw30[(raw30.dataset == "NATOPS") & np.isclose(raw30.lam, 1.0) & np.isclose(raw30.alpha, A) & (raw30.ncal == N)]
o10 = o[(o.method == "oracle") & (o.seed < 10)]
N_TEST = 500                                   # target test points per seed (run_experiment.py --n-test)
p(f"NATOPS oracle, seeds 0-9: {int(round(o10.full_set_rate.sum() * N_TEST))} of {N_TEST * len(o10)} "
  f"sets full; seeds 0-29: full-set rate {o[o.method == 'oracle'].full_set_rate.mean():.3f}, "
  f"coverage non-full {o[o.method == 'oracle'].coverage_nonvacuous.mean():.3f}")
pv = o.pivot_table(index="seed", columns="method", values="coverage")
p(f"NATOPS 30 seeds: standard {pv.standard.mean():.3f} +/- {pv.standard.std():.3f} (range {pv.standard.min():.3f}-"
  f"{pv.standard.max():.3f}; <0.80 in {(pv.standard < 0.8).sum()}, <0.50 in {(pv.standard < 0.5).sum()}), "
  f"oracle {pv.oracle.mean():.3f} +/- {pv.oracle.std():.3f}")

# ---------------------------------------------------------------- Table: NATOPS ratio models
title("NATOPS, channel-mean tilt: density-ratio models")
p(f"{'Ratio model':28s}{'Cov lam=0':>15s}{'Cov lam=1':>15s}{'ESS lam=0':>10s}{'rho lam=1':>10s}{'Fail/27':>8s}")
for lab, path, m in (("All features, C=1", "disjoint", "estimated"), ("  + clip 10", "disjoint", "estimated_clip10"),
                     ("PCA 42, C=1", "disjoint_pca42", "estimated"), ("PCA 14, C=1", "disjoint_pca14", "estimated"),
                     ("All features, CV-C", None, "estimated"), ("Tilting variable only", "disjoint_tilt", "estimated"),
                     ("All features, C=0.1", "disjoint_C01", "estimated"), ("Oracle", "disjoint", "oracle")):
    path = os.path.join(FU, "channel_Ccv_grid30") if path is None else os.path.join(MAIN, path)
    r0, r1 = row(path, m, "NATOPS", 0.0), row(path, m, "NATOPS", 1.0)
    s = S(path)
    s = s[(s.method == m) & (s.dataset == "NATOPS")]
    rho = 1.0 if m == "oracle" else r1.w_spearman_mean
    p(f"{lab:28s}{msd(r0):>15s}{msd(r1):>15s}{r0.ess_ratio_mean:10.2f}{rho:10.2f}"
      f"{int((s.coverage_ci_high < 1 - s.alpha).sum()):8d}")
for ds in DS:
    r = row(os.path.join(MAIN, "disjoint"), "estimated", ds, 0.0)
    p(f"   {ds:14s} C=1 no-shift ESS {r.ess_ratio_mean:.2f}, nMAE {r.w_nmae_mean:.2f}; "
      f"rho at lambda=1 {row(os.path.join(MAIN, 'disjoint'), 'estimated', ds, 1.0).w_spearman_mean:.2f}")
r = row(os.path.join(MAIN, "margin"), "oracle", "NATOPS", 1.0)
p(f"   NATOPS low-margin oracle: ESS {r.ess_ratio_mean:.2f}, coverage {r.coverage_mean:.3f}")

# ---------------------------------------------------------------- CV-C details
title("CV-C (follow-up grid): selected C, stability, low-margin failures")
for tilt, run in (("Channel", "channel_Ccv_grid30"), ("Low-margin", "margin_Ccv_grid30")):
    e = RAW(os.path.join(FU, run))
    e = e[(e.method == "estimated") & np.isclose(e.alpha, A) & (e.ncal == N)]
    g = e.groupby(["dataset", "lam"]).ratio_C
    t = pd.DataFrame({"median_C": g.median(), "share_floor": g.apply(lambda c: np.isclose(c, 1e-5).mean())})
    p(f"--- {tilt}: median selected C and share of seeds at the floor 1e-5")
    p(t.to_string(float_format=lambda v: f"{v:.3g}"))
    s = S(os.path.join(FU, run))
    s = s[(s.method == "estimated") & (s.coverage_ci_high < 1 - s.alpha)]
    if len(s):
        p("   failures: " + "; ".join(f"{q.dataset} lam={q.lam:g} a={q.alpha:g} n={int(q.ncal)} ({q.coverage_mean:.3f})"
                                       for _, q in s.iterrows()))
    rho = S(os.path.join(FU, run))
    rho = rho[(rho.method == "estimated") & np.isclose(rho.alpha, A) & (rho.ncal == N) & (rho.lam > 0)]
    p(f"   Spearman rho at lambda>0: {rho.w_spearman_mean.min():.2f}-{rho.w_spearman_mean.max():.2f}")
cells = []
for ds in DS:
    for a in (0.05, 0.10, 0.20):
        for path, m in ((os.path.join(MAIN, "disjoint"), "standard"), (os.path.join(MAIN, "disjoint"), "oracle"),
                        (os.path.join(MAIN, "disjoint"), "estimated"), (os.path.join(MAIN, "disjoint_Ccv"), "estimated"),
                        (os.path.join(FU, "channel_Ccv_grid30"), "estimated")):
            r = row(path, m, ds, 1.0, a)
            cells.append((ds, a, m, path, r.coverage_mean, r.coverage_std))
big = [c for c in cells if c[5] > 0.10]
p(f"Channel tilt, lambda=1: {len(big)} of {len(cells)} cells with SD > 0.10: "
  + "; ".join(f"{c[0]} a={c[1]} {c[2]} {c[4]:.3f}+/-{c[5]:.3f}" for c in big))
r = row(os.path.join(FU, "channel_Ccv_grid30"), "estimated", "NATOPS", 1.0, 0.20)
r2 = row(os.path.join(MAIN, "disjoint_Ccv"), "estimated", "NATOPS", 1.0, 0.20)
p(f"NATOPS CV-C, lambda=1, alpha=0.20: {r2.coverage_mean:.3f} +/- {r2.coverage_std:.3f} (main grid), "
  f"{r.coverage_mean:.3f} +/- {r.coverage_std:.3f} (follow-up grid)")
raw30 = RAW(os.path.join(FU, "margin_30seeds"))
for m in ("estimated", "estimated_clip10"):
    t = wilcoxon(raw30, "NATOPS", 1.0, m, "oracle")
    p(f"Exploratory, 30 seeds, NATOPS low-margin lambda=1, {m} - oracle: {t['diff']:+.3f} "
      f"({t['pos']}/{t['neg']}), p={t['p']:.3g}")

# ---------------------------------------------------------------- Holm
title("Pre-specified comparisons, Wilcoxon + Holm")
main_raw, main_mg = RAW(os.path.join(MAIN, "disjoint")), RAW(os.path.join(FU, "margin_30seeds"))
rep_raw, h2_raw = RAW(os.path.join(REP, "mean0_base")), RAW(os.path.join(H2, "margin_seeds40_69"))
blocks = {
    "Main study": [("H1", main_raw, "Libras", 1.0, "oracle", "standard", None),
                   ("H2", main_mg, "NATOPS", 1.0, "oracle", "standard", None),
                   ("H3", main_raw, "NATOPS", 0.0, "estimated", "oracle", None),
                   ("H4", main_raw, "NATOPS", 1.0, "estimated", "oracle", None)],
    "Replication (independent seeds)": [("H1", rep_raw, "Libras", 1.0, "oracle", "standard", None),
                                        ("H2", h2_raw, "NATOPS", 1.0, "oracle", "standard", None),
                                        ("H3", rep_raw, "NATOPS", 0.0, "estimated", "oracle", None),
                                        ("H4", rep_raw, "NATOPS", 1.0, "estimated", "oracle", None)]}
for name, tests in blocks.items():
    res = [wilcoxon(r, ds, lam, x, y, sd) for _, r, ds, lam, x, y, sd in tests]
    h = holm([t["p"] for t in res])
    p(f"--- {name}")
    for (key, r, *_), t, hp in zip(tests, res, h):
        seeds = f"{int(r.seed.min())}-{int(r.seed.max())}"
        p(f"{key}  seeds {seeds:6s} n={t['n']:2d}  diff {t['diff']:+.3f}  +/- {t['pos']}/{t['neg']}  "
          f"p={t['p']:.4g}  Holm p={hp:.4g}")
for lab, sd in (("seeds 0-9", range(10)), ("new seeds 10-29 only", range(10, 30))):
    t = wilcoxon(main_mg, "NATOPS", 1.0, "oracle", "standard", sd)
    p(f"H2 main study, {lab}: diff {t['diff']:+.3f} ({t['pos']}/{t['neg']}), p={t['p']:.4g}")
h2v = h2_raw[(h2_raw.dataset == "NATOPS") & np.isclose(h2_raw.lam, 1) & np.isclose(h2_raw.alpha, A) & (h2_raw.ncal == N)]
pv = h2v.pivot_table(index="seed", columns="method", values="coverage")
p(f"H2 replication: standard {pv.standard.mean():.3f} +/- {pv.standard.std():.3f} (range {pv.standard.min():.3f}-"
  f"{pv.standard.max():.3f}; <0.80 in {(pv.standard < 0.8).sum()}), oracle {pv.oracle.mean():.3f} +/- "
  f"{pv.oracle.std():.3f}")

# ---------------------------------------------------------------- Protocol
title("In-pool vs disjoint protocol, channel tilt, standard CP")
for ds in DS:
    vals = []
    for lam in LAMS:
        a_, b_ = row(os.path.join(MAIN, "in_pool"), "standard", ds, lam), row(os.path.join(MAIN, "disjoint"), "standard", ds, lam)
        vals.append(f"lam={lam:g}: acc {a_.target_acc_mean:.3f}/{b_.target_acc_mean:.3f}, cov {a_.coverage_mean:.3f}/"
                    f"{b_.coverage_mean:.3f}, empty {a_.n_empty_mean:.1f}/{b_.n_empty_mean:.1f}")
    p(f"{ds:14s} " + " | ".join(vals))

# ---------------------------------------------------------------- Replication
title("Replication (seeds 10-39, ten datasets): failures out of 270")
p(f"{'Method':24s}{'LGBM channel':>14s}{'LGBM margin':>13s}{'MR channel':>12s}{'MR margin':>11s}")


def f270(run, m):
    path = os.path.join(REP, run)
    return str(fails(path, m, False)) if os.path.exists(path) else "--"


for lab, m, runs in (("Standard CP", "standard", ("mean0_base", "margin_base", "mr_mean0_base", "mr_margin_base")),
                     ("Oracle WCP", "oracle", ("mean0_base", "margin_base", "mr_mean0_base", "mr_margin_base")),
                     ("Est., C=1", "estimated", ("mean0_base", "margin_base", "mr_mean0_base", "mr_margin_base")),
                     ("Est., C=1, clip 10", "estimated_clip10",
                      ("mean0_base", "margin_base", "mr_mean0_base", "mr_margin_base")),
                     ("Est., CV-C", "estimated", ("mean0_Ccv5", "margin_Ccv5", "mr_mean0_Ccv5", "mr_margin_Ccv5")),
                     ("Est., uLSIF", "estimated", ("mean0_ulsif", "margin_ulsif", "-", "mr_margin_ulsif")),
                     ("APS: standard CP", "standard", ("aps_mean0_base", "aps_margin_base", "-", "-")),
                     ("APS: oracle WCP", "oracle", ("aps_mean0_base", "aps_margin_base", "-", "-")),
                     ("APS: est., CV-C", "estimated", ("-", "aps_margin_Ccv5", "-", "-"))):
    p(f"{lab:24s}" + "".join(f"{f270(r, m):>13s}" for r in runs))
DS10 = sorted(S(os.path.join(REP, "margin_base")).dataset.unique())


def pooled(run, m, col):
    return np.mean([row(os.path.join(REP, run), m, d, 1.0)[col] for d in DS10])


p(f"Low-margin, lambda=1, mean over datasets: standard {pooled('margin_base', 'standard', 'coverage_mean'):.3f}, "
  f"oracle {pooled('margin_base', 'oracle', 'coverage_mean'):.3f} (full-set rate "
  f"{pooled('margin_base', 'oracle', 'full_set_rate_mean'):.3f}, size {pooled('margin_base', 'oracle', 'set_size_mean'):.2f}); "
  f"APS + CV-C size {pooled('aps_margin_Ccv5', 'estimated', 'set_size_mean'):.2f}")
low = min(((row(os.path.join(REP, "margin_base"), "standard", d, 1.0).coverage_mean, d) for d in DS10))
p(f"Lowest standard-CP coverage, low-margin lambda=1: {low[0]:.3f} ({low[1]})")
recs = []
for tilt in ("mean0", "margin"):
    for run, m in ((f"{tilt}_base", "estimated"), (f"{tilt}_base", "estimated_clip10"),
                   (f"{tilt}_Ccv5", "estimated"), (f"{tilt}_ulsif", "estimated")):
        s = S(os.path.join(REP, run))
        s = s[s.method == m].assign(tilt=tilt)
        recs.append(s)
E = pd.concat(recs)
E["fail"] = E.coverage_ci_high < 1 - E.alpha
p("ESS ratio as a failure warning (AUC): " + ", ".join(
    f"{t} {roc_auc_score(E[E.tilt == t].fail, -E[E.tilt == t].ess_ratio_mean):.2f}" for t in ("mean0", "margin")))
nb = S(os.path.join(REP, "nat_base"))
nb = nb[(nb.method == "standard") & np.isclose(nb.alpha, A)].set_index("dataset")
p("Natural shift, domain AUC: " + ", ".join(f"{d} {v:.2f}" for d, v in nb.domain_auc_mean.sort_values().items()))
for lab, run, m in (("standard", "nat_base", "standard"), ("CV-C", "nat_Ccv5", "estimated"),
                    ("APS + CV-C", "aps_nat_Ccv5", "estimated")):
    r = row(os.path.join(REP, run), m, "SelfRegulationSCP1", 0.0)
    p(f"   SCP1 {lab}: coverage {r.coverage_mean:.3f} (95% upper bound {r.coverage_ci_high:.3f})")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
open(args.out, "w").write("\n".join(lines) + "\n")
