#!/usr/bin/env bash
# Main study: Libras, RacketSports, NATOPS; seeds 0-9; 10 configurations -> results/main/
# Usage (from the repository root):  bash scripts/run_main.sh [data_dir]
# Resumable: a configuration whose summary.csv exists is skipped.
set -uo pipefail

DATA="${1:-data}"
OUT="${OUT_DIR:-results/main}"
EXTRA="${EXTRA_ARGS:-}"          # e.g. EXTRA_ARGS="--n-seeds 1" for a quick test
mkdir -p "$OUT/logs"
D="--protocol disjoint --population both"

run() {
    local name="$1"; shift
    if [[ -f "$OUT/$name/summary.csv" ]]; then echo "[skip] $name (already done)"; return; fi
    echo "[run ] $name: $*"
    local t0=$SECONDS
    if python run_experiment.py --data-dir "$DATA" --out-dir "$OUT/$name" "$@" $EXTRA > "$OUT/logs/$name.log" 2>&1; then
        echo "       done in $((SECONDS - t0)) s"
    else
        echo "[FAIL] $name - see $OUT/logs/$name.log"
    fi
}

run in_pool            --protocol paper                                  # in-pool protocol (comparison)
run disjoint           $D                                                # channel-mean tilt, C=1, clipping
run disjoint_C01       $D --logreg-C 0.1                                 # post hoc C=0.1 (reference)
run disjoint_Ccv       $D --logreg-C-cv                                  # CV-C, grid 1e-3..1e2
run disjoint_pca14     $D --ratio-features pca --ratio-pca-dim 14
run disjoint_pca42     $D --ratio-features pca --ratio-pca-dim 42
run disjoint_tilt      $D --ratio-features tilt                          # correctly specified (reference)
run margin             $D --shift-feature margin                         # low-margin tilt
run margin_Ccv         $D --shift-feature margin --logreg-C-cv
run margin_tilt        $D --shift-feature margin --ratio-features tilt
