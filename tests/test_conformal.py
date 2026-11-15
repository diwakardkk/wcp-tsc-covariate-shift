"""Unit tests for the conformal and feature code in run_experiment.py.

Run from the project root:  python -m pytest -q tests
"""
import math
import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_experiment import (conformal_thresholds, estimate_weights,  # noqa: E402
                            evaluate_sets, extract_features, weight_diagnostics)


# ----------------------------------------------------------------------
# 1. Unit weights reduce to standard split conformal
# ----------------------------------------------------------------------
def standard_split_threshold(scores, alpha):
    """S_(k) with k = ceil((n+1)(1-alpha)), computed in exact rational arithmetic."""
    n = len(scores)
    k = math.ceil((n + 1) * (1 - Fraction(str(alpha))))
    return np.inf if k > n else np.sort(scores)[k - 1]


@pytest.mark.parametrize("n", [1, 5, 9, 10, 19, 20, 30, 60, 90, 99, 100, 250])
@pytest.mark.parametrize("alpha", [0.05, 0.10, 0.20])
def test_unit_weights_equal_standard_quantile(n, alpha):
    rng = np.random.default_rng(1000 * n + int(alpha * 100))
    scores = rng.random(n)
    q = conformal_thresholds(scores, np.ones(n), np.ones(7), alpha)
    assert q.shape == (7,)
    expected = standard_split_threshold(scores, alpha)
    assert np.all(q == expected)


@pytest.mark.parametrize("n, alpha", [(10, 0.05), (18, 0.05), (8, 0.10), (3, 0.20), (1, 0.20)])
def test_unit_weights_infinite_threshold(n, alpha):
    # ceil((n+1)(1-alpha)) > n  <=>  n < 1/alpha - 1, so the threshold must be +inf.
    assert math.ceil((n + 1) * (1 - Fraction(str(alpha)))) > n
    q = conformal_thresholds(np.random.default_rng(0).random(n), np.ones(n), np.ones(3), alpha)
    assert np.all(np.isinf(q))


def test_unit_weights_boundary_is_finite():
    # n = 19, alpha = 0.05: k = ceil(20 * 0.95) = 19 = n -> the maximum score, not inf.
    scores = np.random.default_rng(1).random(19)
    q = conformal_thresholds(scores, np.ones(19), np.ones(1), 0.05)
    assert q[0] == scores.max()


def test_unit_weights_with_ties():
    scores = np.array([0.1, 0.2, 0.2, 0.2, 0.5, 0.7, 0.7, 0.9, 0.9, 0.9])
    for alpha in (0.05, 0.10, 0.20, 0.30, 0.50):
        q = conformal_thresholds(scores, np.ones(10), np.ones(2), alpha)
        assert np.all(q == standard_split_threshold(scores, alpha))


def test_weighted_quantile_by_hand():
    # Scores 1..4 with weights 1,1,1,1 and test weight 4 -> total mass 8.
    # Level 0.5 needs cumulative mass >= 4: reached at score 4 (cum 4).
    s = np.array([3.0, 1.0, 4.0, 2.0])
    q = conformal_thresholds(s, np.ones(4), np.array([4.0, 0.5]), 0.5)
    # test weight 0.5 -> total 4.5, need >= 2.25 -> score 3 (cum 3).
    assert q[0] == 4.0 and q[1] == 3.0
    # Heavy test weight pushes the quantile onto the point mass at +inf.
    assert np.isinf(conformal_thresholds(s, np.ones(4), np.array([10.0]), 0.5)[0])


def test_threshold_is_invariant_to_weight_scale():
    rng = np.random.default_rng(3)
    s, w, wt = rng.random(50), rng.exponential(size=50), rng.exponential(size=20)
    q1 = conformal_thresholds(s, w, wt, 0.1)
    q2 = conformal_thresholds(s, 7.3 * w, 7.3 * wt, 0.1)
    assert np.array_equal(q1, q2)


# ----------------------------------------------------------------------
# 2. Coverage under a known covariate shift (Monte Carlo)
# ----------------------------------------------------------------------
def _sigmoid(t):
    return 1.0 / (1.0 + np.exp(-t))


def _simulate(rng, n, lam):
    """Two-class problem whose labels get noisier as the shift covariate x grows.

    Covariates are (x, u). Source: x ~ N(0, 1); target: x ~ N(lam, 1), which is
    the exponential tilt exp(lam * x) of the source, so the exact density ratio
    is w(x) = exp(lam * x - lam^2 / 2). u ~ N(0, 1) under both.
    The fixed classifier predicts p(y=1) = sigmoid(2u) everywhere, but the true
    label follows it only for small x, so accuracy falls under the shift.
    """
    x = rng.normal(lam, 1.0, n)
    u = rng.normal(0.0, 1.0, n)
    p1 = _sigmoid(2.0 * u)
    P = np.column_stack([1.0 - p1, p1])
    strength = 4.0 * _sigmoid(-2.0 * x)          # ~4 for x << 0, ~0 for x >> 0
    y = (rng.random(n) < _sigmoid(2.0 * u * strength)).astype(int)
    return x, P, y


def _oracle_w(x, lam):
    return np.exp(lam * x - lam ** 2 / 2.0)


def test_weighted_cp_covers_and_standard_cp_undercovers():
    rng = np.random.default_rng(20261114)
    lam, alpha, ncal, ntest, reps = 1.0, 0.10, 400, 1000, 200

    cov_std, cov_wcp, acc_src, acc_tgt = [], [], [], []
    for _ in range(reps):
        xc, Pc, yc = _simulate(rng, ncal, 0.0)         # source calibration
        xt, Pt, yt = _simulate(rng, ntest, lam)        # tilted target test
        sc = 1.0 - Pc[np.arange(ncal), yc]

        q_std = conformal_thresholds(sc, np.ones(ncal), np.ones(ntest), alpha)
        q_wcp = conformal_thresholds(sc, _oracle_w(xc, lam), _oracle_w(xt, lam), alpha)
        cov_std.append(evaluate_sets(Pt, yt, q_std)["coverage"])
        cov_wcp.append(evaluate_sets(Pt, yt, q_wcp)["coverage"])
        acc_src.append((Pc.argmax(1) == yc).mean())
        acc_tgt.append((Pt.argmax(1) == yt).mean())

    cov_std, cov_wcp = np.array(cov_std), np.array(cov_wcp)
    se_std = cov_std.std(ddof=1) / np.sqrt(reps)
    se_wcp = cov_wcp.std(ddof=1) / np.sqrt(reps)

    # The shift really lowers accuracy.
    assert np.mean(acc_tgt) < np.mean(acc_src) - 0.05
    # Oracle-weighted CP: marginal coverage >= 1 - alpha up to Monte Carlo error.
    assert cov_wcp.mean() >= (1 - alpha) - 3 * se_wcp, (cov_wcp.mean(), se_wcp)
    # Standard CP: significantly below 1 - alpha.
    assert cov_std.mean() < (1 - alpha) - 3 * se_std, (cov_std.mean(), se_std)


def test_no_shift_standard_cp_covers():
    rng = np.random.default_rng(7)
    alpha, ncal, ntest, reps = 0.10, 200, 1000, 200
    cov = []
    for _ in range(reps):
        _, Pc, yc = _simulate(rng, ncal, 0.0)
        _, Pt, yt = _simulate(rng, ntest, 0.0)
        sc = 1.0 - Pc[np.arange(ncal), yc]
        q = conformal_thresholds(sc, np.ones(ncal), np.ones(ntest), alpha)
        cov.append(evaluate_sets(Pt, yt, q)["coverage"])
    cov = np.array(cov)
    se = cov.std(ddof=1) / np.sqrt(reps)
    assert abs(cov.mean() - (1 - alpha)) < 3 * se + 1.0 / (ncal + 1)


# ----------------------------------------------------------------------
# 3. Features
# ----------------------------------------------------------------------
@pytest.mark.parametrize("n, c, T", [(5, 1, 30), (12, 2, 45), (7, 6, 30), (4, 24, 51)])
def test_extract_features_shape_and_column0(n, c, T):
    X = np.random.default_rng(n * c + T).normal(size=(n, c, T)) + 3.0
    F = extract_features(X)
    assert F.shape == (n, 7 * c)
    np.testing.assert_allclose(F[:, 0], X[:, 0, :].mean(axis=1))


def test_extract_features_each_feature_matches_definition():
    rng = np.random.default_rng(5)
    n, c, T = 6, 3, 31
    X = rng.normal(size=(n, c, T)).cumsum(axis=2)
    F = extract_features(X)
    t = np.arange(T)
    for i in range(n):
        for ch in range(c):
            x = X[i, ch]
            power = np.abs(np.fft.rfft(x)) ** 2
            K = len(power) - 1                          # non-DC components
            k3 = K // 3
            expected = [x.mean(), x.std(ddof=0), x.min(), x.max(),
                        np.polyfit(t, x, 1)[0],
                        power[1:k3 + 1].sum(),           # k = 1 .. floor(K/3)
                        power[k3 + 1:K + 1].sum()]       # k = floor(K/3)+1 .. K
            np.testing.assert_allclose(F[i, 7 * ch:7 * ch + 7], expected, rtol=1e-9, atol=1e-9)


# ----------------------------------------------------------------------
# 4. Weight helpers
# ----------------------------------------------------------------------
def test_weight_diagnostics_uniform_and_skewed():
    d = weight_diagnostics(np.ones(40))
    assert d["ess"] == pytest.approx(40) and d["ess_ratio"] == pytest.approx(1)
    assert d["max_weight"] == pytest.approx(1 / 40)
    d = weight_diagnostics(np.array([1.0, 1.0, 1.0, 97.0]))
    assert d["ess"] == pytest.approx(100 ** 2 / (3 + 97 ** 2))
    assert d["max_weight"] == pytest.approx(0.97)


def test_estimate_weights_recovers_tilt_direction():
    rng = np.random.default_rng(11)
    src = rng.normal(0, 1, size=(2000, 1))
    tgt = rng.normal(1, 1, size=(1000, 1))
    grid = np.linspace(-2, 2, 9)[:, None]
    w = estimate_weights(src, tgt, grid, C=1.0, eps=1e-6)
    np.testing.assert_allclose(np.log(w), grid[:, 0] - 0.5, atol=0.25)


# ----------------------------------------------------------------------
# 5. v2 extensions: shift feature, weight accuracy, density-ratio options
# ----------------------------------------------------------------------
from run_experiment import estimate_weights_ex, tilt_feature, weight_accuracy  # noqa: E402


def test_tilt_feature_mean0_and_margin():
    rng = np.random.default_rng(0)
    F = rng.normal(size=(20, 14))
    P = rng.dirichlet(np.ones(4), size=20)
    np.testing.assert_array_equal(tilt_feature(F, P, "mean0"), F[:, 0])
    s = np.sort(P, axis=1)
    np.testing.assert_allclose(tilt_feature(F, P, "margin"), -(s[:, -1] - s[:, -2]))
    assert np.all(tilt_feature(F, P, "margin") <= 0)


def test_weight_accuracy():
    w = np.exp(np.linspace(-1, 1, 50))
    rho, nmae = weight_accuracy(3.0 * w, w)           # same shape, different scale
    assert rho == pytest.approx(1.0) and nmae == pytest.approx(0.0, abs=1e-12)
    rho, nmae = weight_accuracy(w[::-1], w)
    assert rho == pytest.approx(-1.0) and nmae > 0
    rho, nmae = weight_accuracy(w, np.ones(50))       # lambda = 0: oracle constant
    assert np.isnan(rho) and nmae > 0


def test_estimate_weights_ex_default_matches_original():
    rng = np.random.default_rng(1)
    src, tgt, grid = rng.normal(size=(200, 5)), rng.normal(0.5, 1, size=(200, 5)), rng.normal(size=(30, 5))
    w_new, C, _ = estimate_weights_ex(src, tgt, grid, 1.0, 1e-6)
    np.testing.assert_array_equal(w_new, estimate_weights(src, tgt, grid, 1.0, 1e-6))
    assert C == 1.0


def test_estimate_weights_ex_pca_and_cv():
    rng = np.random.default_rng(2)
    src, tgt = rng.normal(size=(200, 40)), rng.normal(size=(200, 40))
    tgt[:, 0] += 1.0
    w, C, _ = estimate_weights_ex(src, tgt, src, 1.0, 1e-6, pca_dim=5)
    assert w.shape == (200,) and np.all(w > 0) and C == 1.0
    w, C, _ = estimate_weights_ex(src, tgt, src, 1.0, 1e-6, cv_folds=5, seed=0)
    from run_experiment import C_GRID
    assert np.any(np.isclose(C, C_GRID))


def test_correctly_specified_ratio_recovers_oracle():
    # Tilt on z with lambda = 1: the true log-ratio is linear in z, so logistic
    # regression on z alone is correctly specified.
    rng = np.random.default_rng(3)
    pool = rng.normal(size=4000)
    p = np.exp(pool) / np.exp(pool).sum()
    src = pool[rng.choice(4000, 2000)][:, None]
    tgt = pool[rng.choice(4000, 2000, p=p)][:, None]
    w, _, _ = estimate_weights_ex(src, tgt, pool[:, None], 1.0, 1e-6)
    oracle = np.exp(pool) / np.exp(pool).mean()
    rho, nmae = weight_accuracy(w, oracle)
    assert rho > 0.99 and nmae < 0.15


# ----------------------------------------------------------------------
# v3: extended C grid, null fallback, uLSIF, vacuous-set metrics
# ----------------------------------------------------------------------
from run_experiment import C_GRID, c_grid, evaluate_sets  # noqa: E402


def test_c_grid_default_equals_v2_grid():
    np.testing.assert_allclose(c_grid(-3), C_GRID)
    g = c_grid(-5)
    assert len(g) == 15 and np.isclose(g[0], 1e-5) and np.isclose(g[-1], 100.0)


def test_extended_grid_can_select_below_1e3_without_shift():
    rng = np.random.default_rng(4)
    src, tgt = rng.normal(size=(200, 30)), rng.normal(size=(200, 30))
    _, C, _ = estimate_weights_ex(src, tgt, src, 1.0, 1e-6, cv_folds=5, seed=0, grid_low=-5)
    assert np.any(np.isclose(C, c_grid(-5)))


@pytest.mark.parametrize("method", ["logreg", "ulsif"])
def test_null_fallback_no_shift_gives_unit_weights(method):
    hits = 0
    for s in range(5):
        rng = np.random.default_rng(100 + s)
        src, tgt = rng.normal(size=(200, 10)), rng.normal(size=(200, 10))
        w, _, fb = estimate_weights_ex(src, tgt, src, 1.0, 1e-6, cv_folds=5, seed=s,
                                       grid_low=-5, null_fallback=True, method=method)
        if fb:
            assert np.all(w == 1.0)
            hits += 1
    assert hits >= 3          # no shift: the constant ratio should usually win


@pytest.mark.parametrize("method", ["logreg", "ulsif"])
def test_null_fallback_keeps_weights_under_strong_shift(method):
    rng = np.random.default_rng(7)
    src, tgt = rng.normal(size=(200, 5)), rng.normal(size=(200, 5))
    tgt[:, 0] += 1.5
    w, _, fb = estimate_weights_ex(src, tgt, src, 1.0, 1e-6, cv_folds=5, seed=0,
                                   grid_low=-5, null_fallback=True, method=method)
    assert not fb
    assert stats_spearman(w, src[:, 0]) > 0.8


def stats_spearman(a, b):
    from scipy import stats
    return stats.spearmanr(a, b)[0]


def test_ulsif_recovers_tilt():
    rng = np.random.default_rng(3)
    pool = rng.normal(size=4000)
    p = np.exp(pool) / np.exp(pool).sum()
    src = pool[rng.choice(4000, 400)][:, None]
    tgt = pool[rng.choice(4000, 400, p=p)][:, None]
    w, lam, fb = estimate_weights_ex(src, tgt, pool[:, None], 1.0, 1e-6, method="ulsif")
    oracle = np.exp(pool) / np.exp(pool).mean()
    rho, _ = weight_accuracy(w, oracle)
    assert not fb and rho > 0.9 and np.all(w > 0) and lam > 0


def test_evaluate_sets_vacuous_metrics():
    P = np.array([[0.7, 0.2, 0.1], [0.4, 0.35, 0.25], [0.1, 0.1, 0.8]])
    y = np.array([0, 2, 0])
    q = np.array([0.5, np.inf, 0.5])           # point 2 gets the full set
    m = evaluate_sets(P, y, q)
    assert m["full_set_rate"] == pytest.approx(1 / 3)
    assert m["coverage"] == pytest.approx(2 / 3)
    assert m["coverage_nonvacuous"] == pytest.approx(1 / 2)
    m = evaluate_sets(P, y, np.full(3, np.inf))
    assert np.isnan(m["coverage_nonvacuous"]) and m["full_set_rate"] == 1.0


# ----------------------------------------------------------------------
# v3b: APS score, cross-fitting, natural protocol, MiniRocket plumbing
# ----------------------------------------------------------------------
import types  # noqa: E402

import run_experiment as rx  # noqa: E402


def test_score_matrix_lac_and_aps_by_hand():
    P = np.array([[0.5, 0.3, 0.2]])
    np.testing.assert_allclose(rx.score_matrix(P, "lac"), 1 - P)
    np.testing.assert_allclose(rx.score_matrix(P, "aps", np.zeros(1)), [[0.0, 0.5, 0.8]])
    np.testing.assert_allclose(rx.score_matrix(P, "aps", np.ones(1)), [[0.5, 0.8, 1.0]])


def test_aps_split_conformal_coverage_monte_carlo():
    rng = np.random.default_rng(11)
    K, n, alpha, cov = 5, 200, 0.1, []
    for _ in range(300):
        logits = rng.normal(size=(n + 400, K)) * 1.5
        P = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        y = np.array([rng.choice(K, p=p) for p in P])
        S = rx.score_matrix(P, "aps", rng.random(len(P)))
        cal = S[np.arange(n), y[:n]]
        q = conformal_thresholds(cal, np.ones(n), np.ones(400), alpha)
        cov.append(rx.evaluate_sets(P[n:], y[n:], q, S[n:])["coverage"])
    assert abs(np.mean(cov) - (1 - alpha)) < 0.01


def _args(**kw):
    d = dict(logreg_cv_folds=5, logreg_C_cv=False, cv_grid_low=-3.0, null_fallback=False,
             ratio_method="logreg", ratio_crossfit=1, logreg_C=1.0, prob_eps=1e-6, cv_grid_n=None)
    d.update(kw)
    return types.SimpleNamespace(**d)


def test_ratio_weights_no_crossfit_matches_estimate_weights_ex():
    rng = np.random.default_rng(5)
    src, tgt, ev, cal = (rng.normal(size=(100, 4)) for _ in range(4))
    tgt += 0.5
    we, wc, _, fb = rx.ratio_weights(_args(), src, tgt, ev, cal, 0, 0, cal_in_src=False)
    w = estimate_weights(src, tgt, np.vstack([ev, cal]), 1.0, 1e-6)
    np.testing.assert_allclose(np.concatenate([we, wc]), w)
    assert fb == 0.0


def test_ratio_weights_crossfit_is_out_of_fold():
    rng = np.random.default_rng(6)
    src, tgt, ev, cal = (rng.normal(size=(120, 3)) for _ in range(4))
    tgt[:, 0] += 1.0
    we, wc, reg, fb = rx.ratio_weights(_args(ratio_crossfit=5), src, tgt, ev, cal, 0, 0, False)
    assert we.shape == (120,) and wc.shape == (120,) and np.all(wc > 0) and reg == 1.0
    # out-of-fold calibration weights differ from the in-sample fit
    _, wc_in, _, _ = rx.ratio_weights(_args(), np.vstack([src, cal]), tgt, ev, cal, 0, 0, False)
    assert not np.allclose(wc, wc_in)
    assert stats_spearman(wc, cal[:, 0]) > 0.8


def test_natural_protocol_end_to_end(tmp_path):
    rng = np.random.default_rng(0)
    X_tr, X_te = rng.normal(size=(60, 2, 20)), rng.normal(size=(80, 2, 20)) + 0.3
    y_tr, y_te = rng.choice(["a", "b", "c"], 60), rng.choice(["a", "b", "c"], 80)
    np.savez(tmp_path / "Toy.npz", X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
    import sys
    argv = sys.argv
    sys.argv = ["x", "--data-dir", str(tmp_path), "--datasets", "Toy", "--protocol", "natural",
                "--n-seeds", "2", "--ratio-crossfit", "3", "--score", "aps", "--n-jobs", "1"]
    try:
        df = rx.run(rx.parse_args())
    finally:
        sys.argv = argv
    assert set(df.method) == {"standard", "estimated", "estimated_clip10", "estimated_clip20"}
    assert (df.ncal == 30).all() and (df.n_test == 40).all()
    assert df.domain_auc.between(0, 1).all() and df.coverage.between(0, 1).all()


def test_minirocket_wrapper_plumbing(monkeypatch):
    class FakeMR:                      # stands in for aeon's MiniRocket (not installed locally)
        def __init__(self, random_state=None, n_jobs=1):
            pass

        def fit_transform(self, X):
            return self.transform(X)

        def transform(self, X):
            return np.concatenate([X.mean(2), X.std(2)], axis=1)
    monkeypatch.setattr(rx, "_minirocket", lambda: FakeMR)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(90, 3, 30))
    y = np.repeat([0, 1, 2], 30)
    X[y == 1, 0] += 2
    proba = rx.train_classifier(types.SimpleNamespace(classifier="minirocket", n_jobs=1),
                                X, None, np.arange(60), y, 0)
    P = proba(np.arange(90))
    assert P.shape == (90, 3) and np.allclose(P.sum(1), 1)


# ----------------------------------------------------------------------
# v3.1: 30-point grid
# ----------------------------------------------------------------------
def test_c_grid_v31_matches_spec():
    np.testing.assert_allclose(rx.c_grid(-5, 30), np.logspace(-5, 2, 30))


def test_cv_selects_from_v31_grid():
    rng = np.random.default_rng(9)
    src, tgt = rng.normal(size=(200, 6)), rng.normal(size=(200, 6))
    tgt[:, 0] += 1.0
    _, C, _ = estimate_weights_ex(src, tgt, src, 1.0, 1e-6, cv_folds=5, seed=0, grid_low=-5, grid_n=30)
    assert np.any(np.isclose(C, np.logspace(-5, 2, 30)))
