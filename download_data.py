#!/usr/bin/env python3
"""Download the UEA datasets on the server and cache them as .npz files.

Usage:  python download_data.py --data-dir data
"""
import argparse
import os

import numpy as np

CORE = ["Libras", "RacketSports", "NATOPS"]
# v3: seven more equal-length UEA datasets for the scope extension
EXTRA = ["ArticularyWordRecognition", "BasicMotions", "Cricket", "Epilepsy",
         "ERing", "SelfRegulationSCP1", "UWaveGestureLibrary"]
DATASETS = CORE + EXTRA
# (train, test, channels, classes) from the UEA archive description (Bagnall et al., 2018)
EXPECTED = {
    "Libras": (180, 180, 2, 15),
    "RacketSports": (151, 152, 6, 4),
    "NATOPS": (180, 180, 24, 6),
    "ArticularyWordRecognition": (275, 300, 9, 25),
    "BasicMotions": (40, 40, 6, 4),
    "Cricket": (108, 72, 6, 12),
    "Epilepsy": (137, 138, 3, 4),
    "ERing": (30, 270, 4, 6),
    "SelfRegulationSCP1": (268, 293, 6, 2),
    "UWaveGestureLibrary": (120, 320, 3, 8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    args = ap.parse_args()

    from aeon.datasets import load_classification

    raw_dir = os.path.join(args.data_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    failed = []
    for name in args.datasets:
        path = os.path.join(args.data_dir, f"{name}.npz")
        if os.path.exists(path):
            print(f"{name}: already downloaded -> {path} [OK]")
            continue
        parts = {}
        try:
            for split in ("train", "test"):
                out = load_classification(name, split=split, extract_path=raw_dir)
                X = np.asarray(out[0], dtype=np.float64)
                y = np.asarray(out[1]).astype(str)
                if X.ndim != 3:
                    raise ValueError(f"{name}/{split}: expected 3D array, got {X.shape}")
                parts[f"X_{split}"] = X
                parts[f"y_{split}"] = y
        except Exception as e:          # keep going; the run script uses whatever is present
            print(f"{name}: [FAILED] {e}")
            failed.append(name)
            continue

        np.savez_compressed(path, **parts)

        n_classes = len(np.unique(np.concatenate([parts["y_train"], parts["y_test"]])))
        got = (len(parts["y_train"]), len(parts["y_test"]),
               parts["X_train"].shape[1], n_classes)
        status = "OK" if got == EXPECTED.get(name, got) else f"MISMATCH, expected {EXPECTED[name]}"
        print(f"{name}: train={got[0]} test={got[1]} channels={got[2]} "
              f"classes={got[3]} length={parts['X_train'].shape[2]} [{status}] -> {path}")
    if failed:
        print(f"\nFailed: {' '.join(failed)} (re-run the same command to retry)")


if __name__ == "__main__":
    main()
