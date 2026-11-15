#!/usr/bin/env python3
"""Analysis of the follow-up runs (results/followup; results/main for comparison).

Exp 1: CV grid for C extended to [1e-5, 1e2]; Exp 2: NATOPS low-margin test with 30 seeds;
Exp 3: exact full-label-set rate of oracle WCP; Exp 4: seed SDs.
Writes <out-dir>/followup_results.txt plus CSV tables.
Usage (from the repository root):  python analysis/analyze_followup.py
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="results/followup")
ap.add_argument("--main-dir", default="results/main")
args = ap.parse_args()
O, V2 = args.out_dir, args.main_dir
A, N = 0.10, 90
GRID = np.logspace(-5, 2, 30)
lines = []


def p(s=""):
    print(s)
    lines.append(str(s))


def raw(path):
    return pd.read_csv(os.path.join(path, "raw_results.csv"))


def summ(path):
    return pd.read_csv(os.path.join(path, "summary.csv"))


def holm(ps):
    order, out, run = np.argsort(ps), np.zeros(len(ps)), 0.0
    for i, k in enumerate(order):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ======================================================================= Exp 1
p("=" * 78)
p("EXPERIMENT 1 - CV grid for C extended to np.logspace(-5, 2, 30)")
p("=" * 78)
for tag, run, v2run in (("Channel-mean tilt", "channel_Ccv_grid30", "disjoint_Ccv"),
                        ("Low-margin tilt", "margin_Ccv_grid30", "margin_Ccv")):
    path = os.path.join(O, run)
    if not os.path.exists(path):
        p(f"[missing] {run}")
        continue
    r = raw(path)
    e = r[(r.method == "estimated") & np.isclose(r.alpha, A) & (r.ncal == N)]   # one row per seed x lambda
    g = e.groupby(["dataset", "lam"]).ratio_C
    t = pd.DataFrame({
        "median_C": g.median(),
        "share_at_floor_1e-5": g.apply(lambda c: np.isclose(c, GRID[0]).mean()),
        "share_<=1e-4": g.apply(lambda c: (c <= 1.0001e-4).mean()),
        "share_<=1e-3": g.apply(lambda c: (c <= 1.0001e-3).mean()),
        "share_at_top_1e2": g.apply(lambda c: np.isclose(c, GRID[-1]).mean())})
    v2 = raw(os.path.join(V2, v2run))
    v2 = v2[(v2.method == "estimated") & np.isclose(v2.alpha, A) & (v2.ncal == N)]
    t["main_median_C (grid 1e-3..1e2)"] = v2.groupby(["dataset", "lam"]).ratio_C.median()
    p(f"\n--- {tag}: selected C (per seed and lambda; 10 seeds) ---")
    p(t.round(6).to_string())
    t.to_csv(os.path.join(O, f"exp1_selected_C_{run}.csv"))
    floor0 = t.xs(0.0, level="lam")["share_at_floor_1e-5"]
    p(f"Finding: at lambda=0 the new floor 1e-5 is selected in {floor0.min():.0%}-{floor0.max():.0%} of seeds; "
      f"median C at lambda=0: {', '.join(f'{d} {v:.0e}' for d, v in t.xs(0.0, level='lam').median_C.items())}.")

    s, s2 = summ(path), summ(os.path.join(V2, v2run))
    keys = ["dataset", "lam", "alpha", "ncal"]
    a = s[s.method == "estimated"].set_index(keys)
    b = s2[s2.method == "estimated"].set_index(keys)
    tab = pd.DataFrame({
        "cov_follow-up": a.coverage_mean.round(3), "sd_follow-up": a.coverage_std.round(3),
        "ci_high_follow-up": a.coverage_ci_high.round(3), "ess_follow-up": a.ess_ratio_mean.round(3),
        "full_set_follow-up": a.full_set_rate_mean.round(3),
        "cov_main": b.coverage_mean.round(3), "sd_main": b.coverage_std.round(3), "ess_main": b.ess_ratio_mean.round(3)})
    tab["fail_follow-up"] = (a.coverage_ci_high < 1 - a.index.get_level_values("alpha"))
    tab["fail_main"] = (b.coverage_ci_high < 1 - b.index.get_level_values("alpha"))
    tab.to_csv(os.path.join(O, f"exp1_coverage_ess_{run}.csv"))
    main = tab.xs((A, N), level=("alpha", "ncal"))
    p(f"\n--- {tag}: Estimated WCP (CV-C) at alpha=0.10, n_cal=90 (new grid vs main grid) ---")
    p(main.to_string())
    p(f"Failures (95% CI below nominal) over all 81 settings: follow-up {int(tab['fail_follow-up'].sum())}/81, "
      f"main {int(tab['fail_main'].sum())}/81")
    for ds in ("Libras", "RacketSports", "NATOPS"):
        x = tab.xs(ds, level="dataset")
        p(f"   {ds:12s} follow-up {int(x['fail_follow-up'].sum())}/27   main {int(x['fail_main'].sum())}/27")

# ======================================================================= Exp 2
p("\n" + "=" * 78)
p("EXPERIMENT 2 - NATOPS, low-margin tilt, lambda=1, alpha=0.10, n_cal=90, 30 seeds")
p("=" * 78)
path = os.path.join(O, "margin_30seeds")
if os.path.exists(path):
    r = raw(path)
    c = r[(r.dataset == "NATOPS") & np.isclose(r.lam, 1.0) & np.isclose(r.alpha, A) & (r.ncal == N)]
    pv = c.pivot_table(index="seed", columns="method", values="coverage")
    # reproduction check of the 10 main-study seeds
    v2 = raw(os.path.join(V2, "margin"))
    v2 = v2[(v2.dataset == "NATOPS") & np.isclose(v2.lam, 1.0) & np.isclose(v2.alpha, A) & (v2.ncal == N)]
    v2p = v2.pivot_table(index="seed", columns="method", values="coverage")
    same = np.allclose(pv.loc[v2p.index, v2p.columns].to_numpy(), v2p.to_numpy())
    p(f"Seeds 0-9 reproduce the main study exactly: {same}")
    rows = []
    for lab, seeds in (("seeds 0-9 (main)", range(0, 10)), ("seeds 10-29 (new only)", range(10, 30)),
                       ("seeds 0-29 (all 30)", range(0, 30))):
        x = pv.loc[list(seeds)]
        d = (x["oracle"] - x["standard"]).to_numpy()
        rows.append(dict(seeds=lab, n=len(d),
                         std_mean=x["standard"].mean(), std_sd=x["standard"].std(ddof=1),
                         std_min=x["standard"].min(), std_max=x["standard"].max(),
                         orc_mean=x["oracle"].mean(), orc_sd=x["oracle"].std(ddof=1),
                         diff_mean=d.mean(), pos=int((d > 0).sum()), neg=int((d < 0).sum()),
                         wilcoxon_p=stats.wilcoxon(d).pvalue))
    t = pd.DataFrame(rows)
    p(t.round(4).to_string(index=False))
    t.to_csv(os.path.join(O, "exp2_natops_margin_30seeds.csv"), index=False)
    # Holm with the three other primary tests of the main study (10 seeds each)
    fam = []
    for run_, ds, lam, x_, y_ in (("disjoint", "Libras", 1.0, "oracle", "standard"),
                                  ("disjoint", "NATOPS", 0.0, "estimated", "oracle"),
                                  ("disjoint", "NATOPS", 1.0, "estimated", "oracle")):
        d0 = raw(os.path.join(V2, run_))
        d0 = d0[(d0.dataset == ds) & np.isclose(d0.lam, lam) & np.isclose(d0.alpha, A) & (d0.ncal == N)]
        q = d0.pivot_table(index="seed", columns="method", values="coverage")
        fam.append((f"{ds} lam={lam:g} {x_}-{y_} (main, 10 seeds)", stats.wilcoxon(q[x_] - q[y_]).pvalue))
    for lab, pval in (("NATOPS low-margin oracle-standard, 10 seeds (main)", rows[0]["wilcoxon_p"]),
                      ("NATOPS low-margin oracle-standard, 30 seeds", rows[2]["wilcoxon_p"])):
        ps = np.array([pval] + [f[1] for f in fam])
        h = holm(ps)
        p(f"Holm over the 4 primary tests using '{lab}': raw p = {pval:.4f}, Holm-adjusted p = {h[0]:.4f}")
    p("Caution: adding seeds after a non-significant result is optional stopping. Report the 30-seed result "
      "together with the 20 new seeds alone (row 'seeds 10-29'), which form an independent replication.")
    # Exp 3 uses the same run
    p("\n" + "=" * 78)
    p("EXPERIMENT 3 - Oracle WCP, NATOPS, low-margin tilt, lambda=1, alpha=0.10, n_cal=90")
    p("=" * 78)
    o = r[(r.dataset == "NATOPS") & np.isclose(r.lam, 1.0) & np.isclose(r.alpha, A) & (r.ncal == N)
          & (r.method == "oracle")]
    rows = []
    for lab, seeds in (("seeds 0-9 (main)", range(0, 10)), ("seeds 0-29", range(0, 30))):
        x = o[o.seed.isin(list(seeds))]
        rows.append(dict(seeds=lab, full_set_rate=x.full_set_rate.mean(), full_set_sd=x.full_set_rate.std(ddof=1),
                         inf_threshold_rate=x.inf_threshold_rate.mean(), mean_set_size=x.set_size.mean(),
                         coverage=x.coverage.mean(), coverage_non_full=x.coverage_nonvacuous.mean(),
                         n_test_points=500 * len(x)))
    t = pd.DataFrame(rows)
    p(t.round(4).to_string(index=False))
    t.to_csv(os.path.join(O, "exp3_oracle_full_sets.csv"), index=False)
    p("A full set contains all 6 NATOPS classes. Table 5's 0.598 is the infinite-threshold rate; a full set can "
      "also arise from a finite threshold, so full_set_rate >= inf_threshold_rate.")
else:
    p("[missing] margin_30seeds")

# ======================================================================= Exp 4
p("\n" + "=" * 78)
p("EXPERIMENT 4 - SD of Est. CV-C, NATOPS, channel-mean tilt, alpha=0.20, lambda=1")
p("=" * 78)
rows = []
for lab, path in (("main grid (results/main/disjoint_Ccv)", os.path.join(V2, "disjoint_Ccv")),
                  ("follow-up grid (channel_Ccv_grid30)", os.path.join(O, "channel_Ccv_grid30"))):
    if not os.path.exists(path):
        continue
    s = summ(path)
    x = s[(s.dataset == "NATOPS") & np.isclose(s.lam, 1.0) & np.isclose(s.alpha, 0.20) & (s.method == "estimated")]
    for _, q in x.sort_values("ncal").iterrows():
        rows.append(dict(grid=lab, ncal=int(q.ncal), mean=q.coverage_mean, sd=q.coverage_std,
                         n_seeds=int(q.n_seeds), red_flag_sd_gt_0_10=bool(q.coverage_std > 0.10)))
t = pd.DataFrame(rows)
p(t.round(4).to_string(index=False))
t.to_csv(os.path.join(O, "exp4_natops_sd.csv"), index=False)

open(os.path.join(O, "followup_results.txt"), "w").write("\n".join(lines) + "\n")
