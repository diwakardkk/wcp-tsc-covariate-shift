#!/usr/bin/env python3
"""Create fake .npz files with the same keys, shapes, dtypes and label format as
the real UEA datasets, for local end-to-end testing without any download.

Usage:  python make_synthetic_data.py --out-dir data_synthetic
        python run_experiment.py --data-dir data_synthetic --out-dir results_synthetic --n-seeds 2

Writes to data_synthetic/ by default so it never overwrites the real data/ folder.
"""
import argparse
import os

import numpy as np

# name: (n_train, n_test, channels, length, class labels) - same as the real files
SPECS = {
    "Libras": (180, 180, 2, 45, [f"{k}.0" for k in range(1, 16)]),
    "RacketSports": (151, 152, 6, 30, ["Badminton_Clear", "Badminton_Smash",
                                       "Squash_BackhandBoast", "Squash_ForehandBoast"]),
    "NATOPS": (180, 180, 24, 51, [f"{k}.0" for k in range(1, 7)]),
    # v3 extension datasets (real shapes; class labels are placeholders)
    "ArticularyWordRecognition": (275, 300, 9, 144, [f"{k}.0" for k in range(1, 26)]),
    "BasicMotions": (40, 40, 6, 100, ["Badminton", "Running", "Standing", "Walking"]),
    "Cricket": (108, 72, 6, 1197, [f"{k}.0" for k in range(1, 13)]),
    "Epilepsy": (137, 138, 3, 206, ["EPILEPSY", "RUNNING", "SAWING", "WALKING"]),
    "ERing": (30, 270, 4, 65, [str(k) for k in range(1, 7)]),
    "SelfRegulationSCP1": (268, 293, 6, 896, ["negativity", "positivity"]),
    "UWaveGestureLibrary": (120, 320, 3, 315, [f"{k}.0" for k in range(1, 9)]),
}


def make_split(rng, n, c, T, labels, protos):
    """Class-specific prototype + noise, so the classifier has something to learn.

    The first-channel offset also raises the noise level, so the exponential
    tilt on mean(channel 0) produces a target that is harder to classify.
    """
    y_idx = rng.integers(len(labels), size=n)
    offset = rng.normal(0.0, 1.0, size=n)                 # drives mean of channel 0
    noise_sd = 0.6 * np.exp(0.5 * offset)                  # harder when offset is high
    X = protos[y_idx] + noise_sd[:, None, None] * rng.normal(size=(n, c, T))
    X[:, 0, :] += offset[:, None]
    return X.astype(np.float64), np.asarray(labels)[y_idx].astype(str)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_synthetic")
    ap.add_argument("--datasets", nargs="+", default=list(SPECS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    for d_i, name in enumerate(args.datasets):
        n_tr, n_te, c, T, labels = SPECS[name]
        rng = np.random.default_rng([args.seed, d_i])
        t = np.linspace(0, 2 * np.pi, T)
        protos = np.stack([
            np.stack([np.sin((k % 4 + 1) * t + ch) * (1 + 0.2 * k) for ch in range(c)])
            for k in range(len(labels))])                  # (classes, c, T)
        X_tr, y_tr = make_split(rng, n_tr, c, T, labels, protos)
        X_te, y_te = make_split(rng, n_te, c, T, labels, protos)
        path = os.path.join(args.out_dir, f"{name}.npz")
        np.savez_compressed(path, X_train=X_tr, y_train=y_tr, X_test=X_te, y_test=y_te)
        print(f"{name}: X_train={X_tr.shape} X_test={X_te.shape} "
              f"classes={len(np.unique(np.concatenate([y_tr, y_te])))} -> {path}")


if __name__ == "__main__":
    main()
