#!/usr/bin/env python3
"""Weighted conformal prediction under controlled covariate shift.

Compares standard split CP, oracle-weighted CP and estimated-weight CP
(logistic-regression density ratios, with optional weight clipping) on
UEA multivariate time-series datasets.

Usage:
    python run_experiment.py --data-dir data --out-dir results --device gpu
"""
import argparse
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

# "No shift" rather than "None": pandas.read_csv turns the string "None" into NaN.
SHIFT_NAMES = {0.0: "No shift", 0.5: "Mild", 1.0: "Moderate"}


# ----------------------------------------------------------------------
# Data and features
# ----------------------------------------------------------------------
def load_population(data_dir, name, population):
    """Return (X, y) for the finite empirical source population."""
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    if population == "train":
        X, y = d["X_train"], d["y_train"]
    else:  # "both": official train + test merged
        X = np.concatenate([d["X_train"], d["X_test"]])
        y = np.concatenate([d["y_train"], d["y_test"]])
    y = LabelEncoder().fit_transform(y)
    return X.astype(np.float64), y


def extract_features(X):
    """Seven summary features per channel. X: (n, channels, T) -> (n, 7*channels).

    Order per channel: mean, std, min, max, slope, E_low, E_high.
    Column 0 is therefore the mean of the first channel (the tilting feature).
    """
    X = np.nan_to_num(X)
    n, c, T = X.shape
    tc = np.arange(T) - (T - 1) / 2.0
    mean = X.mean(axis=2)
    std = X.std(axis=2)
    mn = X.min(axis=2)
    mx = X.max(axis=2)
    slope = ((X - mean[..., None]) * tc).sum(axis=2) / (tc ** 2).sum()
    power = np.abs(np.fft.rfft(X, axis=2)) ** 2
    power = power[..., 1:]                      # drop the DC component
    K = power.shape[2]
    k3 = K // 3
    e_low = power[..., :k3].sum(axis=2)         # k = 1 .. floor(K/3)
    e_high = power[..., k3:].sum(axis=2)        # k = floor(K/3)+1 .. K
    feats = np.stack([mean, std, mn, mx, slope, e_low, e_high], axis=2)
    return feats.reshape(n, c * 7)


# ----------------------------------------------------------------------
# Classifier
# ----------------------------------------------------------------------
def fit_classifier(F_train, y_train, seed, device, n_jobs):
    scaler = StandardScaler().fit(F_train)
    # 'binary' for two-class data (v3 adds SelfRegulationSCP1); LightGBM refuses
    # 'multiclass' with two classes. All v1/v2 datasets have > 2 classes.
    objective = "multiclass" if len(np.unique(y_train)) > 2 else "binary"
    params = dict(objective=objective, n_estimators=200, learning_rate=0.05,
                  num_leaves=15, min_child_samples=10, random_state=seed,
                  verbose=-1, n_jobs=n_jobs)
    if device != "cpu":
        params["device_type"] = device
    try:
        clf = lgb.LGBMClassifier(**params).fit(scaler.transform(F_train), y_train)
    except Exception as e:  # GPU build missing, etc.
        if device == "cpu":
            raise
        warnings.warn(f"LightGBM on '{device}' failed ({e}); falling back to CPU.")
        params.pop("device_type")
        clf = lgb.LGBMClassifier(**params).fit(scaler.transform(F_train), y_train)
    return scaler, clf


def _minirocket():
    """aeon's MiniRocket transform (multivariate); the class name changed in aeon 1.0."""
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket
    except ImportError:
        from aeon.transformations.collection.convolution_based import MiniRocketMultivariate as MiniRocket
    return MiniRocket


class MiniRocketLR:
    """MiniRocket features + standardized multinomial logistic regression.

    The usual MiniRocket head (RidgeClassifierCV) gives no probabilities, which the
    conformity score needs, so a logistic-regression head with fixed C is used.
    Input: raw series (n, channels, T).
    """

    def __init__(self, seed, n_jobs, C=1.0):
        self.seed, self.n_jobs, self.C = seed, n_jobs, C

    def fit(self, X, y):
        MR = _minirocket()
        try:
            self.tr = MR(random_state=self.seed, n_jobs=self.n_jobs)
        except TypeError:
            self.tr = MR(random_state=self.seed)
        Z = np.asarray(self.tr.fit_transform(X.astype(np.float32)), dtype=np.float64)
        self.sc = StandardScaler().fit(Z)
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(self.sc.transform(Z), y)
        self.classes_ = self.lr.classes_
        return self

    def predict_proba(self, X):
        Z = np.asarray(self.tr.transform(X.astype(np.float32)), dtype=np.float64)
        return self.lr.predict_proba(self.sc.transform(Z))


def predict_proba_full(scaler, clf, F, n_classes):
    """Probabilities over all classes, even if some were absent from training.

    scaler = None means clf takes its input as is (MiniRocket on raw series).
    """
    proba = clf.predict_proba(F if scaler is None else scaler.transform(F))
    full = np.zeros((F.shape[0], n_classes))
    full[:, clf.classes_.astype(int)] = proba
    return full


# ----------------------------------------------------------------------
# Conformal prediction
# ----------------------------------------------------------------------
def conformal_thresholds(cal_scores, cal_w, test_w, alpha):
    """Weighted conformal threshold for every test point (Eqs. 26-30).

    The test point carries mass test_w at +infinity. With all weights equal
    to one this reduces exactly to the standard split-conformal quantile
    S_(ceil((n+1)(1-alpha))) of Eq. 24, including the infinite case.
    """
    order = np.argsort(cal_scores)
    s = cal_scores[order]
    cw = np.cumsum(cal_w[order])
    target = (1.0 - alpha) * (cw[-1] + test_w)
    idx = np.searchsorted(cw, target * (1.0 - 1e-12), side="left")
    q = np.where(idx < len(s), s[np.minimum(idx, len(s) - 1)], np.inf)
    return q


def score_matrix(P, kind="lac", u=None):
    """Conformity scores for every label: (n, K).

    'lac' : 1 - p_y (the original score).
    'aps' : randomized adaptive prediction sets (Romano et al., 2020): mass of the
            labels ranked above y plus u * p_y, with u ~ U(0, 1) per point.
    """
    if kind == "lac":
        return 1.0 - P
    order = np.argsort(-P, axis=1, kind="stable")
    Ps = np.take_along_axis(P, order, axis=1)
    above = np.cumsum(Ps, axis=1) - Ps                     # mass strictly above, in sorted order
    S = np.empty_like(P)
    np.put_along_axis(S, order, above, axis=1)
    return S + u[:, None] * P


def evaluate_sets(P_test, y_test, q, S_test=None):
    scores = 1.0 - P_test if S_test is None else S_test
    in_set = scores <= q[:, None]
    size = in_set.sum(axis=1)
    covered = in_set[np.arange(len(y_test)), y_test]
    full = size == P_test.shape[1]               # vacuous set: every class included
    return dict(coverage=covered.mean(), set_size=size.mean(),
                singleton_rate=(size == 1).mean(), empty_rate=(size == 0).mean(),
                inf_threshold_rate=np.isinf(q).mean(),
                full_set_rate=full.mean(),
                coverage_nonvacuous=covered[~full].mean() if (~full).any() else np.nan)


def weight_diagnostics(w):
    wbar = w / w.sum()
    ess = 1.0 / np.sum(wbar ** 2)
    return dict(ess=ess, ess_ratio=ess / len(w), max_weight=wbar.max())


# ----------------------------------------------------------------------
# Density-ratio estimation
# ----------------------------------------------------------------------
C_GRID = np.logspace(-3, 2, 11)   # v2 grid for --logreg-C-cv (default --cv-grid-low -3)


def c_grid(low=-3.0, n=None):
    """Log-spaced C grid from 10**low to 10**2.

    n = None: two points per decade (low = -3 gives exactly C_GRID, the v2 grid;
    v3 uses low = -5). n given: exactly n points (v3.1: low = -5, n = 30, i.e.
    np.logspace(-5, 2, 30)).
    """
    if n is None:
        n = int(round(2 * (2.0 - low))) + 1
    return np.logspace(low, 2, n)


def _null_logloss(D, splits):
    """Per-fold CV log-loss of the intercept-only domain model (constant probability)."""
    out = []
    for tr, te in splits:
        p = np.clip(D[tr].mean(), 1e-12, 1 - 1e-12)
        out.append(-np.mean(D[te] * np.log(p) + (1 - D[te]) * np.log(1 - p)))
    return np.array(out)


def no_detectable_shift(improvement):
    """One-standard-error rule on per-fold CV improvements over the constant ratio.

    The model is kept only if its mean improvement exceeds one standard error;
    selecting the best of a grid makes the raw comparison optimistic.
    """
    imp = np.asarray(improvement, dtype=float)
    se = imp.std(ddof=1) / np.sqrt(len(imp)) if len(imp) > 1 else 0.0
    return imp.mean() <= se


def _ratio_preprocess(Xd, pca_dim, seed):
    steps = [StandardScaler()]
    if pca_dim and pca_dim < Xd.shape[1]:
        steps.append(PCA(n_components=pca_dim, random_state=seed))
    return steps


def estimate_weights_ex(F_src, F_tgt, F_all, C, eps, pca_dim=0, cv_folds=0, seed=0,
                        grid_low=-3.0, null_fallback=False, method="logreg", grid_n=None):
    """Density ratio w(x) = p_target(x) / p_source(x), evaluated on F_all.

    method 'logreg' : logistic-regression domain classifier with prior-odds
        correction (Eq. 33). pca_dim > 0 reduces the standardized inputs to that
        many principal components first. cv_folds > 1 chooses C from
        c_grid(grid_low) by stratified k-fold CV log-loss of the domain classifier
        (label-free: only the source/target indicator is used).
    method 'ulsif'  : unconstrained least-squares importance fitting (Kanamori et
        al., 2009) with Gaussian kernels; bandwidth and ridge chosen by k-fold CV of
        the uLSIF squared-loss objective (always cross-validated).
    null_fallback   : if the selected model does not beat the constant-ratio model
        in cross-validation by more than one standard error (logreg: intercept-only
        log-loss; uLSIF: w = 1), there is no detectable shift and unit weights are returned (standard CP).

    Returns (weights for F_all, regularization used, fallback flag). The
    regularization is C for logreg and the ridge parameter for uLSIF.
    """
    if method == "ulsif":
        return _ulsif_weights(F_src, F_tgt, F_all, pca_dim, max(cv_folds, 5), seed,
                              null_fallback, eps)
    Xd = np.vstack([F_src, F_tgt])
    D = np.concatenate([np.zeros(len(F_src)), np.ones(len(F_tgt))])
    steps = _ratio_preprocess(Xd, pca_dim, seed)
    use_cv = bool(cv_folds and cv_folds > 1)
    if use_cv:
        skf = StratifiedKFold(cv_folds, shuffle=True, random_state=seed)
        grid = C_GRID if (grid_low == -3.0 and grid_n is None) else c_grid(grid_low, grid_n)
        lr = LogisticRegressionCV(Cs=grid, cv=skf, scoring="neg_log_loss", max_iter=5000)
    else:
        lr = LogisticRegression(C=C, max_iter=5000)
    dom = make_pipeline(*steps, lr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)    # sklearn 1.10 API notices for LogisticRegressionCV
        dom.fit(Xd, D)
    C_used = float(np.atleast_1d(lr.C_)[0]) if hasattr(lr, "C_") else float(C)

    if null_fallback:
        # CV log-loss of the selected C versus the intercept-only model, same folds.
        skf = StratifiedKFold(max(cv_folds, 5) if not use_cv else cv_folds,
                              shuffle=True, random_state=seed)
        splits = list(skf.split(Xd, D))
        losses = []
        for tr, te in splits:
            m = make_pipeline(*_ratio_preprocess(Xd, pca_dim, seed),
                              LogisticRegression(C=C_used, max_iter=5000)).fit(Xd[tr], D[tr])
            p = np.clip(m.predict_proba(Xd[te])[:, 1], 1e-12, 1 - 1e-12)
            losses.append(-np.mean(D[te] * np.log(p) + (1 - D[te]) * np.log(1 - p)))
        if no_detectable_shift(_null_logloss(D, splits) - np.array(losses)):
            return np.ones(len(F_all)), C_used, True

    p = np.clip(dom.predict_proba(F_all)[:, 1], eps, 1.0 - eps)
    return p / (1.0 - p) * (len(F_src) / len(F_tgt)), C_used, False


ULSIF_SIGMA_MULT = (0.25, 0.5, 1.0, 2.0, 4.0)     # times the median pairwise distance
ULSIF_LAMBDAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
ULSIF_CENTERS = 100


def _gauss(A, B, sigma):
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-np.maximum(d2, 0.0) / (2 * sigma ** 2))


def _ulsif_fit(Ks, Kt, lam):
    H = Ks.T @ Ks / len(Ks)
    h = Kt.mean(axis=0)
    return np.linalg.solve(H + lam * np.eye(len(h)), h)


def _ulsif_objective(Ks, Kt, a):
    """uLSIF squared-loss objective J = E_s[w^2]/2 - E_t[w] (J = -0.5 for w = 1)."""
    ws, wt = np.maximum(Ks @ a, 0), np.maximum(Kt @ a, 0)
    return 0.5 * np.mean(ws ** 2) - np.mean(wt)


def _ulsif_weights(F_src, F_tgt, F_all, pca_dim, folds, seed, null_fallback, eps):
    Xd = np.vstack([F_src, F_tgt])
    pre = make_pipeline(*_ratio_preprocess(Xd, pca_dim, seed)).fit(Xd)
    S, T, A = pre.transform(F_src), pre.transform(F_tgt), pre.transform(F_all)
    rng = np.random.default_rng(seed)
    centers = T[rng.choice(len(T), size=min(ULSIF_CENTERS, len(T)), replace=False)]
    sub = Xd[rng.choice(len(Xd), size=min(300, len(Xd)), replace=False)]
    sub = pre.transform(sub)
    sq = (sub ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * sub @ sub.T, 0))
    med = np.median(d[np.triu_indices(len(sub), 1)])
    med = med if med > 0 else 1.0

    fs = np.array_split(rng.permutation(len(S)), folds)
    ft = np.array_split(rng.permutation(len(T)), folds)
    best = (np.inf, None, None, None)
    for mult in ULSIF_SIGMA_MULT:
        sigma = mult * med
        Ks, Kt = _gauss(S, centers, sigma), _gauss(T, centers, sigma)
        for lam in ULSIF_LAMBDAS:
            J = []
            for k in range(folds):
                trs = np.setdiff1d(np.arange(len(S)), fs[k])
                trt = np.setdiff1d(np.arange(len(T)), ft[k])
                a = np.maximum(_ulsif_fit(Ks[trs], Kt[trt], lam), 0)
                J.append(_ulsif_objective(Ks[fs[k]], Kt[ft[k]], a))
            if np.mean(J) < best[0]:
                best = (float(np.mean(J)), sigma, lam, np.array(J))
    _, sigma, lam, J_folds = best
    if null_fallback and no_detectable_shift(-0.5 - J_folds):   # J = -0.5 for w = 1
        return np.ones(len(F_all)), lam, True
    Ks, Kt = _gauss(S, centers, sigma), _gauss(T, centers, sigma)
    a = np.maximum(_ulsif_fit(Ks, Kt, lam), 0)
    w = np.maximum(_gauss(A, centers, sigma) @ a, eps)
    return w, lam, False


def estimate_weights(F_src, F_tgt, F_all, C, eps):
    """Original interface: all features, fixed C."""
    return estimate_weights_ex(F_src, F_tgt, F_all, C, eps)[0]


def ratio_weights(args, R_src, R_tgt, R_eval, R_cal, seed, pca_dim, cal_in_src):
    """Estimated weights for evaluation points and calibration points.

    cal_in_src=False (synthetic protocols): the ratio model is fitted on R_src vs
        R_tgt only; calibration weights are the fitted function at R_cal.
    cal_in_src=True  (natural protocol): the calibration sample is also the source
        sample of the ratio model (in-sample weights).
    args.ratio_crossfit = K > 1: the calibration points join the source sample, split
        into K folds; each calibration point is weighted by the model fitted without
        its fold (out-of-fold) and evaluation points by the mean of the K models.
    Returns (w_eval, w_cal, regularization, fallback rate).
    """
    kw = dict(pca_dim=pca_dim, cv_folds=args.logreg_cv_folds if args.logreg_C_cv else 0,
              seed=seed, grid_low=args.cv_grid_low, null_fallback=args.null_fallback,
              method=args.ratio_method, grid_n=args.cv_grid_n)
    K = args.ratio_crossfit
    if K <= 1:
        src = np.vstack([R_src, R_cal]) if cal_in_src else R_src
        w, reg, fb = estimate_weights_ex(src, R_tgt, np.vstack([R_eval, R_cal]),
                                         args.logreg_C, args.prob_eps, **kw)
        return w[:len(R_eval)], w[len(R_eval):], reg, float(fb)
    folds = np.array_split(np.random.default_rng([seed, 4242]).permutation(len(R_cal)), K)
    w_eval = np.zeros(len(R_eval))
    w_cal = np.zeros(len(R_cal))
    regs, fbs = [], []
    for f in folds:
        keep = np.setdiff1d(np.arange(len(R_cal)), f)
        src = np.vstack([R_src, R_cal[keep]])
        w, reg, fb = estimate_weights_ex(src, R_tgt, np.vstack([R_eval, R_cal[f]]),
                                         args.logreg_C, args.prob_eps, **kw)
        w_eval += w[:len(R_eval)] / K
        w_cal[f] = w[len(R_eval):]
        regs.append(reg)
        fbs.append(fb)
    return w_eval, w_cal, float(np.exp(np.mean(np.log(regs)))), float(np.mean(fbs))


def tilt_feature(F_pool, P_pool, kind):
    """Raw (unstandardized) tilting variable on the pool.

    'mean0'  : mean of the first channel (feature column 0) - the original design.
    'margin' : minus the classifier margin p(top1) - p(top2), so that a positive
               lambda moves mass towards observations the classifier finds hard.
               It depends only on x given the trained classifier, so P(Y|X) is
               unchanged and the shift is still a covariate shift.
    """
    if kind == "mean0":
        return F_pool[:, 0]
    top2 = np.sort(P_pool, axis=1)[:, -2:]
    return -(top2[:, 1] - top2[:, 0])


def weight_accuracy(w_hat, w_true):
    """Spearman rank correlation and normalized MAE of estimated vs oracle weights on the pool.

    Both weight vectors are rescaled to mean 1 over the pool before the MAE, so
    nMAE measures shape error, not scale. Spearman is undefined (NaN) when the
    oracle weights are constant (lambda = 0).
    """
    a, b = w_hat / w_hat.mean(), w_true / w_true.mean()
    nmae = float(np.mean(np.abs(a - b)))
    if np.ptp(w_true) < 1e-12 or np.ptp(w_hat) < 1e-12:
        return np.nan, nmae
    return float(stats.spearmanr(w_hat, w_true)[0]), nmae


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------
def train_classifier(args, X, F, idx, y, seed):
    """Fit the chosen classifier; returns a function giving full probabilities."""
    n_classes = len(np.unique(y))
    if args.classifier == "minirocket":
        clf = MiniRocketLR(seed, args.n_jobs).fit(X[idx], y[idx])
        return lambda rows: predict_proba_full(None, clf, X[rows], n_classes)
    scaler, clf = fit_classifier(F[idx], y[idx], seed, args.device, args.n_jobs)
    return lambda rows: predict_proba_full(scaler, clf, F[rows], n_classes)


def _row(name, seed, lam, ncal, alpha, mname, clip, m, diag, args, **extra):
    return dict(dataset=name, seed=seed, lam=lam, ncal=ncal, alpha=alpha, method=mname, clip=clip,
                **m, **diag, cov_dev=m["coverage"] - (1.0 - alpha), **extra,
                shift_feature=args.shift_feature if args.protocol != "natural" else "natural",
                classifier=args.classifier, score=args.score)


def run(args):
    if args.protocol == "natural":
        return run_natural(args)
    rows = []
    max_ncal = max(args.ncals)
    ratio_name = None

    for d_i, name in enumerate(args.datasets):
        X, y = load_population(args.data_dir, name, args.population)
        F = extract_features(X)
        N, n_classes = len(y), len(np.unique(y))
        print(f"\n=== {name}: population={N}, features={F.shape[1]}, classes={n_classes} ===")

        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            t0 = time.time()
            rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, d_i, seed]))
            # Separate stream for APS randomization, so the default draws are unchanged.
            rng_u = np.random.default_rng([args.base_seed, d_i, seed, 777])

            if args.protocol == "paper":
                # Everything sampled with replacement from the same population.
                pool = np.arange(N)
                tr_idx = rng.choice(N, size=args.n_train or N, replace=True)
            else:
                # Disjoint: classifier never sees the calibration/target pool.
                tr_idx, pool = train_test_split(
                    np.arange(N), test_size=0.5, stratify=y,
                    random_state=int(rng.integers(1_000_000_000)))

            proba = train_classifier(args, X, F, tr_idx, y, seed)

            F_pool, y_pool = F[pool], y[pool]
            M = len(pool)
            P_pool = proba(pool)

            # Tilting variable (default: mean of first channel), standardized on the pool.
            z_raw = tilt_feature(F_pool, P_pool, args.shift_feature)
            z = (z_raw - z_raw.mean()) / (z_raw.std() + 1e-12)

            # Inputs of the density-ratio model (default: all 7C features).
            R_pool = z_raw[:, None] if args.ratio_features == "tilt" else F_pool
            pca_dim = args.ratio_pca_dim if args.ratio_features == "pca" else 0
            ratio_name = f"pca{pca_dim}" if pca_dim else args.ratio_features

            # Source-side samples are shared across shift levels (paired design).
            cal_all = rng.choice(M, size=max_ncal, replace=True)
            ratio_src = rng.choice(M, size=args.n_ratio, replace=True)
            u_cal = rng_u.random(max_ncal)
            S_cal_all = score_matrix(P_pool[cal_all], args.score, u_cal)
            cal_scores_all = S_cal_all[np.arange(max_ncal), y_pool[cal_all]]

            for lam in args.lambdas:
                tilt = np.exp(lam * z)
                w_oracle = tilt / tilt.mean()            # Eq. 20, Z from the pool
                p_target = tilt / tilt.sum()
                adapt = rng.choice(M, size=args.n_adapt, replace=True, p=p_target)
                test = rng.choice(M, size=args.n_test, replace=True, p=p_target)
                S_test = score_matrix(P_pool[test], args.score, rng_u.random(len(test)))

                w_est, w_est_cal, ratio_C, fallback = ratio_weights(
                    args, R_pool[ratio_src], R_pool[adapt], R_pool, R_pool[cal_all],
                    seed, pca_dim, cal_in_src=False)
                if args.ratio_crossfit <= 1:
                    w_est_cal = w_est[cal_all]           # identical to the original code
                log_w_mae = np.mean(np.abs(np.log(w_est) - np.log(w_oracle)))
                target_acc = (P_pool[test].argmax(axis=1) == y_pool[test]).mean()

                # (name, weights on the pool, weights on cal_all, clip)
                methods = [("standard", np.ones(M), np.ones(max_ncal), np.nan),
                           ("oracle", w_oracle, w_oracle[cal_all], np.nan)]
                for c in args.clips:
                    label = "estimated" if np.isinf(c) else f"estimated_clip{c:g}"
                    methods.append((label, np.minimum(w_est, c), np.minimum(w_est_cal, c), c))
                # Accuracy of each estimated weight function against the oracle, on the pool.
                w_acc = {mn: weight_accuracy(w, w_oracle) for mn, w, _, _ in methods
                         if mn.startswith("estimated")}

                for ncal in args.ncals:
                    cal = cal_all[:ncal]
                    cal_scores = cal_scores_all[:ncal]
                    for mname, w, wc, clip in methods:
                        w_cal = wc[:ncal]
                        diag = weight_diagnostics(w_cal)
                        est = mname.startswith("estimated")
                        for alpha in args.alphas:
                            q = conformal_thresholds(cal_scores, w_cal, w[test], alpha)
                            m = evaluate_sets(P_pool[test], y_pool[test], q, S_test)
                            rows.append(_row(
                                name, seed, lam, ncal, alpha, mname, clip, m, diag, args,
                                log_w_mae=log_w_mae if est else np.nan,
                                w_spearman=w_acc[mname][0] if est else np.nan,
                                w_nmae=w_acc[mname][1] if est else np.nan,
                                ratio_C=ratio_C if est else np.nan,
                                ratio_fallback=fallback if est else np.nan,
                                ratio_method=args.ratio_method,
                                ratio_crossfit=args.ratio_crossfit,
                                n_empty=int(round(m["empty_rate"] * len(test))),
                                target_acc=target_acc,
                                ratio_features=ratio_name))
            print(f"  seed {seed}: done in {time.time() - t0:.1f}s")

    return pd.DataFrame(rows)


def load_splits(data_dir, name):
    """Official train and test splits with a common label encoding."""
    d = np.load(os.path.join(data_dir, f"{name}.npz"))
    enc = LabelEncoder().fit(np.concatenate([d["y_train"], d["y_test"]]))
    return (d["X_train"].astype(np.float64), enc.transform(d["y_train"]),
            d["X_test"].astype(np.float64), enc.transform(d["y_test"]))


def _split_half(idx, y, rng):
    try:
        return train_test_split(idx, test_size=0.5, stratify=y,
                                random_state=int(rng.integers(1_000_000_000)))
    except ValueError:                         # a class with a single member
        return train_test_split(idx, test_size=0.5,
                                random_state=int(rng.integers(1_000_000_000)))


def domain_auc(R_src, R_tgt, seed):
    """Label-free strength of a shift: 5-fold CV ROC AUC of a domain classifier
    (standardized logistic regression, C chosen by inner CV). 0.5 = no detectable shift."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    Xd = np.vstack([R_src, R_tgt])
    D = np.concatenate([np.zeros(len(R_src)), np.ones(len(R_tgt))])
    model = make_pipeline(StandardScaler(), LogisticRegressionCV(
        Cs=c_grid(-5), cv=StratifiedKFold(5, shuffle=True, random_state=seed),
        scoring="neg_log_loss", max_iter=5000))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        p = cross_val_predict(model, Xd, D, method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=seed + 1))[:, 1]
    return float(roc_auc_score(D, p))


def run_natural(args):
    """Natural shift: official train split = source, official test split = target.

    Per seed: the train split is halved (stratified) into a classifier set and a
    calibration set (which is also the source sample of the density-ratio model);
    the test split is halved into an unlabeled adaptation set and an evaluation set.
    No resampling and no oracle: the true density ratio is unknown. Rows use
    lam = 0 and ncal = size of the calibration set.
    """
    rows = []
    for d_i, name in enumerate(args.datasets):
        Xtr, ytr, Xte, yte = load_splits(args.data_dir, name)
        X = np.concatenate([Xtr, Xte])
        y = np.concatenate([ytr, yte])
        F = extract_features(X)
        itr, ite = np.arange(len(ytr)), len(ytr) + np.arange(len(yte))
        n_classes = int(y.max()) + 1
        print(f"\n=== {name} (natural shift): train={len(itr)}, test={len(ite)}, classes={n_classes} ===")
        for seed in range(args.seed_offset, args.seed_offset + args.n_seeds):
            t0 = time.time()
            rng = np.random.default_rng(np.random.SeedSequence([args.base_seed, d_i, seed, 99]))
            rng_u = np.random.default_rng([args.base_seed, d_i, seed, 777])
            clf_idx, cal = _split_half(itr, y[itr], rng)
            perm = rng.permutation(ite)
            adapt, test = perm[:len(perm) // 2], perm[len(perm) // 2:]

            proba = train_classifier(args, X, F, clf_idx, y, seed)
            P_cal, P_test = proba(cal), proba(test)
            S_cal = score_matrix(P_cal, args.score, rng_u.random(len(cal)))
            S_test = score_matrix(P_test, args.score, rng_u.random(len(test)))
            cal_scores = S_cal[np.arange(len(cal)), y[cal]]
            source_acc = (P_cal.argmax(1) == y[cal]).mean()
            target_acc = (P_test.argmax(1) == y[test]).mean()

            pca_dim = args.ratio_pca_dim if args.ratio_features == "pca" else 0
            # 'tilt' has no meaning without a synthetic tilt: use all features.
            w_test, w_cal, ratio_C, fallback = ratio_weights(
                args, F[cal][:0], F[adapt], F[test], F[cal], seed, pca_dim, cal_in_src=True)
            auc = domain_auc(F[cal], F[adapt], seed)

            methods = [("standard", np.ones(len(test)), np.ones(len(cal)), np.nan)]
            for c in args.clips:
                label = "estimated" if np.isinf(c) else f"estimated_clip{c:g}"
                methods.append((label, np.minimum(w_test, c), np.minimum(w_cal, c), c))
            for mname, wt, wc, clip in methods:
                diag = weight_diagnostics(wc)
                est = mname.startswith("estimated")
                for alpha in args.alphas:
                    q = conformal_thresholds(cal_scores, wc, wt, alpha)
                    m = evaluate_sets(P_test, y[test], q, S_test)
                    rows.append(_row(
                        name, seed, 0.0, len(cal), alpha, mname, clip, m, diag, args,
                        log_w_mae=np.nan, w_spearman=np.nan, w_nmae=np.nan,
                        ratio_C=ratio_C if est else np.nan,
                        ratio_fallback=fallback if est else np.nan,
                        ratio_method=args.ratio_method, ratio_crossfit=args.ratio_crossfit,
                        n_empty=int(round(m["empty_rate"] * len(test))),
                        target_acc=target_acc, source_acc=source_acc, domain_auc=auc,
                        n_test=len(test), ratio_features=args.ratio_features))
            print(f"  seed {seed}: done in {time.time() - t0:.1f}s (domain AUC {auc:.2f})")
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def summarize(df):
    keys = ["dataset", "lam", "ncal", "alpha", "method"]
    metrics = ["coverage", "cov_dev", "set_size", "singleton_rate", "empty_rate",
               "inf_threshold_rate", "ess", "ess_ratio", "max_weight",
               "log_w_mae", "target_acc"]
    # Columns added in v2; absent from results produced by the original code.
    metrics += [m for m in ("w_spearman", "w_nmae", "ratio_C", "n_empty",
                            "full_set_rate", "coverage_nonvacuous", "ratio_fallback",
                            "source_acc", "domain_auc")
                if m in df.columns]
    g = df.groupby(keys, sort=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out = out.reset_index()
    n = g.size().values
    out["n_seeds"] = n
    # 95% t-interval over seeds for every metric (spec item 10), not only coverage.
    tcrit = stats.t.ppf(0.975, np.maximum(n - 1, 1))
    for m in metrics:
        half = tcrit * out[f"{m}_std"] / np.sqrt(n)
        out[f"{m}_ci_low"] = out[f"{m}_mean"] - half
        out[f"{m}_ci_high"] = out[f"{m}_mean"] + half
    return out


def pm(mean, std, digits=3):
    return mean.map(lambda v: f"{v:.{digits}f}") + " ± " + std.map(lambda v: f"{v:.{digits}f}")


def make_tables(summary, args):
    s = summary[np.isclose(summary["alpha"], args.main_alpha)
                & (summary["ncal"] == args.main_ncal)].copy()
    if args.protocol == "natural":
        s = summary[np.isclose(summary["alpha"], args.main_alpha)].copy()   # ncal = calibration-set size
        s["shift"] = "Natural"
    else:
        s["shift"] = s["lam"].map(lambda v: SHIFT_NAMES.get(float(v), f"lambda={v:g}"))
    main = s[s["method"].isin(["standard", "oracle", "estimated"])].copy()

    # Table II: coverage
    main["cell"] = pm(main["coverage_mean"], main["coverage_std"])
    t2 = main.pivot(index=["dataset", "shift"], columns="method", values="cell")
    t2 = t2.reindex(index=main[["dataset", "shift"]].drop_duplicates().itertuples(index=False, name=None))
    # Only methods that were run (e.g. "estimated" is absent if --clips excludes inf).
    t2 = t2[[c for c in ["standard", "oracle", "estimated"] if c in t2.columns]].reset_index()

    # Table III: efficiency
    t3 = main[["dataset", "shift", "method"]].copy()
    t3["mean_size"] = pm(main["set_size_mean"], main["set_size_std"])
    t3["singleton_rate"] = pm(main["singleton_rate_mean"], main["singleton_rate_std"])

    # Table IV: weight diagnostics for estimated WCP under shift
    e = main[(main["method"] == "estimated") & ((main["lam"] > 0) | (args.protocol == "natural"))]
    t4 = e[["dataset", "shift"]].copy()
    t4["ess_ratio"] = pm(e["ess_ratio_mean"], e["ess_ratio_std"])
    t4["max_weight"] = pm(e["max_weight_mean"], e["max_weight_std"])

    return {"table2_coverage": t2, "table3_efficiency": t3, "table4_weights": t4}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--datasets", nargs="+", default=["Libras", "RacketSports", "NATOPS"])
    ap.add_argument("--population", choices=["train", "both"], default="train",
                    help="'train' = official train split (paper); 'both' = train+test merged")
    ap.add_argument("--protocol", choices=["paper", "disjoint", "natural"], default="paper",
                    help="'paper' = all samples drawn with replacement from one population; "
                         "'disjoint' = classifier trained on a separate half; "
                         "'natural' = official train split is the source, test split the target "
                         "(real shift, no oracle; --lambdas/--ncals/--population are ignored)")
    ap.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.05, 0.10, 0.20])
    ap.add_argument("--ncals", nargs="+", type=int, default=[30, 60, 90])
    ap.add_argument("--clips", nargs="+", type=float, default=[10.0, 20.0, float("inf")])
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--base-seed", type=int, default=2026)
    ap.add_argument("--n-train", type=int, default=0, help="0 = population size (paper protocol)")
    ap.add_argument("--n-ratio", type=int, default=200, help="source ratio-fitting sample size")
    ap.add_argument("--n-adapt", type=int, default=200, help="unlabeled target adaptation size")
    ap.add_argument("--n-test", type=int, default=500, help="target evaluation size")
    ap.add_argument("--logreg-C", type=float, default=1.0)
    ap.add_argument("--prob-eps", type=float, default=1e-6)
    # ---- optional extensions (defaults reproduce the original experiment) ----
    ap.add_argument("--shift-feature", choices=["mean0", "margin"], default="mean0",
                    help="tilting variable: 'mean0' = mean of first channel (original); "
                         "'margin' = minus classifier margin (shift towards hard series)")
    ap.add_argument("--ratio-features", choices=["all", "pca", "tilt"], default="all",
                    help="density-ratio inputs: 'all' 7C features (original); 'pca' first "
                         "--ratio-pca-dim principal components; 'tilt' only the tilting variable "
                         "(correctly specified model)")
    ap.add_argument("--ratio-pca-dim", type=int, default=14)
    ap.add_argument("--logreg-C-cv", action="store_true",
                    help="choose C by cross-validated log-loss of the domain classifier "
                         "(label-free); overrides --logreg-C")
    ap.add_argument("--logreg-cv-folds", type=int, default=5)
    # ---- v3 extensions (defaults reproduce v2) ----
    ap.add_argument("--cv-grid-low", type=float, default=-3.0,
                    help="lower end (log10) of the CV grid for C; grid runs to 10**2 with "
                         "two points per decade. -3 = v2 grid; v3 uses -5")
    ap.add_argument("--cv-grid-n", type=int, default=None,
                    help="number of grid points for CV of C (default: two per decade). "
                         "v3.1 uses --cv-grid-low -5 --cv-grid-n 30 = np.logspace(-5, 2, 30)")
    ap.add_argument("--null-fallback", action="store_true",
                    help="use unit weights (standard CP) when the density-ratio model does not "
                         "beat a constant ratio in cross-validation (label-free shift test)")
    ap.add_argument("--ratio-method", choices=["logreg", "ulsif"], default="logreg",
                    help="density-ratio estimator: logistic regression (default) or uLSIF "
                         "(Gaussian-kernel least squares, bandwidth/ridge by CV)")
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="first seed; seeds run from offset to offset+n_seeds-1. v3 uses 10 "
                         "so the confirmatory seeds are disjoint from the v2 seeds 0-9")
    ap.add_argument("--ratio-crossfit", type=int, default=1,
                    help="K > 1: cross-fit the density ratio over K folds of the calibration "
                         "sample, which then also serves as source data (out-of-fold weights)")
    ap.add_argument("--classifier", choices=["lgbm", "minirocket"], default="lgbm",
                    help="'lgbm' on 7C summary features (original); 'minirocket' = aeon MiniRocket "
                         "on raw series + logistic regression")
    ap.add_argument("--score", choices=["lac", "aps"], default="lac",
                    help="conformity score: 'lac' = 1 - p_y (original); 'aps' = randomized APS")
    ap.add_argument("--main-alpha", type=float, default=0.10)
    ap.add_argument("--main-ncal", type=int, default=90)
    ap.add_argument("--device", choices=["cpu", "gpu", "cuda"], default="cpu")
    ap.add_argument("--n-jobs", type=int, default=4)
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    df = run(args)
    df.to_csv(os.path.join(args.out_dir, "raw_results.csv"), index=False)
    summary = summarize(df)
    summary.to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    for tname, table in make_tables(summary, args).items():
        table.to_csv(os.path.join(args.out_dir, f"{tname}.csv"), index=False)
        print(f"\n--- {tname} (alpha={args.main_alpha}, ncal={args.main_ncal}) ---")
        print(table.to_string(index=False))
    print(f"\nAll outputs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
