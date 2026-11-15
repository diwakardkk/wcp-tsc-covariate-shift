#!/usr/bin/env bash
# Independent test of H2 on seeds 40-69, used by no other run (plan: docs/H2_fresh_plan.txt).
# Usage (from the repository root):  bash scripts/run_h2_fresh.sh [data_dir]
set -uo pipefail

DATA="${1:-data}"
OUT="${OUT_DIR:-results/h2_fresh}"
mkdir -p "$OUT/logs"
export PYTHONWARNINGS=ignore
NAME=margin_seeds40_69

if [[ -f "$OUT/$NAME/summary.csv" ]]; then
    echo "[skip] $NAME (already done)"
else
    echo "[run ] $NAME"
    t0=$SECONDS
    # Same flags as results/followup/margin_30seeds; only --seed-offset differs. Keep the dataset order.
    python run_experiment.py --data-dir "$DATA" --out-dir "$OUT/$NAME" \
        --protocol disjoint --population both --datasets Libras RacketSports NATOPS \
        --shift-feature margin --n-seeds 30 --seed-offset 40 \
        > "$OUT/logs/$NAME.log" 2>&1 || { echo "[FAIL] see $OUT/logs/$NAME.log"; exit 1; }
    echo "       done in $((SECONDS - t0)) s"
fi

python analysis/analyze_h2_fresh.py --out-dir "$OUT" && echo "Report: $OUT/h2_fresh_results.txt"
