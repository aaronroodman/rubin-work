#!/bin/bash
# Edge rules R0-R3 with the per-edge spider cut on night 1 (pass 6) and night 3, then the
# Batoid v4.02 / v4.03 rim-sensitivity test. Submit from the repo root.
# Usage: night3_edge_rules_batch.sh <workers>
set -euo pipefail
workers=$1
repo=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$repo"
study=$repo/aos/giant_donut_pupil
code=aos/giant_donut_pupil/code
test -f "$study/code/edge_rules.py"
export OMP_NUM_THREADS=1
dirs=()
for set in 20261023_pupil6:isr_phys_v402s 20261118_night2_20261107:v402s 20261118_night2_20261108:v402s; do
    run=${set%%:*}; fd=${set##*:}
    python $code/edge_rules.py --run "$run" --fit-dir "$fd" --workers "$workers"
    dirs+=(--data-rules "$(ls -td "$study"/output/*_edge_rules_${run}_${fd} | head -1)")
done
python $code/edge_rules_batoid.py --workers 8 "${dirs[@]}"
echo "edge rules done"
