#!/usr/bin/env bash
# Replication: ten UEA datasets, seeds 10-39, 17 configurations -> results/replication/
# Usage (from the repository root):  bash scripts/run_replication.sh [data_dir]
# JOBS configurations run in parallel. Resumable: a configuration with summary.csv is skipped.
set -uo pipefail

DATA="${1:-data}"
OUT="${OUT_DIR:-results/replication}"
JOBS="${JOBS:-4}"
N_JOBS="${N_JOBS:-4}"
EXTRA="${EXTRA_ARGS:-}"
mkdir -p "$OUT/logs"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore

# Keep this order: the random stream depends on the dataset's position.
ALL="Libras RacketSports NATOPS ArticularyWordRecognition BasicMotions Cricket Epilepsy ERing SelfRegulationSCP1 UWaveGestureLibrary"
DS=""
for d in $ALL; do
    if [[ -f "$DATA/$d.npz" ]]; then DS="$DS $d"; else echo "[warn] $DATA/$d.npz missing - skipped"; fi
done
echo "Datasets:$DS"

COMMON="--protocol disjoint --population both --n-seeds 30 --seed-offset 10 --n-jobs $N_JOBS --datasets$DS"
CV="--logreg-C-cv --cv-grid-low -5"          # CV-C, 15 log-spaced values in [1e-5, 1e2]
NAT="--protocol natural"
MR="--classifier minirocket"
APS="--score aps"

run() {
    local name="$1"; shift
    if [[ -f "$OUT/$name/summary.csv" ]]; then echo "[skip] $name (already done)"; return; fi
    (
        t0=$SECONDS
        echo "[start] $name: $*"
        if python run_experiment.py --data-dir "$DATA" --out-dir "$OUT/$name" $COMMON "$@" $EXTRA \
               > "$OUT/logs/$name.log" 2>&1; then
            echo "[done ] $name in $((SECONDS - t0)) s"
        else
            echo "[FAIL ] $name - see $OUT/logs/$name.log"
        fi
    ) &
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n 2>/dev/null || wait; done
}

# LightGBM, LAC score: both tilts
for tilt in mean0 margin; do
    run ${tilt}_base       --shift-feature $tilt                       # standard, oracle, C=1, clipping
    run ${tilt}_Ccv5       --shift-feature $tilt $CV
    run ${tilt}_ulsif      --shift-feature $tilt --ratio-method ulsif
done
# Natural shift: official train split = source, official test split = target (no oracle)
run nat_base               $NAT
run nat_Ccv5               $NAT $CV
# MiniRocket
run mr_mean0_base          $MR --shift-feature mean0
run mr_mean0_Ccv5          $MR --shift-feature mean0 $CV
run mr_margin_base         $MR --shift-feature margin
run mr_margin_Ccv5         $MR --shift-feature margin $CV
run mr_margin_ulsif        $MR --shift-feature margin --ratio-method ulsif
# APS score
run aps_mean0_base         $APS --shift-feature mean0
run aps_margin_base        $APS --shift-feature margin
run aps_margin_Ccv5        $APS --shift-feature margin $CV
run aps_nat_Ccv5           $APS $NAT $CV
wait

n_ok=$(ls "$OUT"/*/summary.csv 2>/dev/null | wc -l | tr -d " ")
echo "Finished: $n_ok of 17 configurations have summary.csv."
