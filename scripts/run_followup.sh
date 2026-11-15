#!/usr/bin/env bash
# Follow-up of the main study (seeds as in the main study):
#   CV grid for C extended to np.logspace(-5, 2, 30) for both tilts, and the low-margin run with 30 seeds
#   (seeds 0-29; seeds 0-9 reproduce results/main/margin exactly). -> results/followup/
# Usage (from the repository root):  bash scripts/run_followup.sh [data_dir]
set -uo pipefail

DATA="${1:-data}"
OUT="${OUT_DIR:-results/followup}"
EXTRA="${EXTRA_ARGS:-}"
mkdir -p "$OUT/logs"
export PYTHONWARNINGS=ignore
# Keep the dataset order: the random stream depends on the dataset's position.
D="--protocol disjoint --population both --datasets Libras RacketSports NATOPS"
GRID="--logreg-C-cv --cv-grid-low -5 --cv-grid-n 30"

run() {
    local name="$1"; shift
    if [[ -f "$OUT/$name/summary.csv" ]]; then echo "[skip] $name (already done)"; return; fi
    echo "[run ] $name: $*"
    local t0=$SECONDS
    if python run_experiment.py --data-dir "$DATA" --out-dir "$OUT/$name" $D "$@" $EXTRA > "$OUT/logs/$name.log" 2>&1; then
        echo "       done in $((SECONDS - t0)) s"
    else
        echo "[FAIL] $name - see $OUT/logs/$name.log"
    fi
}

run channel_Ccv_grid30   $GRID
run margin_Ccv_grid30    --shift-feature margin $GRID
run margin_30seeds       --shift-feature margin --n-seeds 30

python analysis/analyze_followup.py --out-dir "$OUT" && echo "Report: $OUT/followup_results.txt"
