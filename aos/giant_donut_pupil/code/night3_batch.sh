#!/bin/bash
# One night of night 3 in a Slurm job: night2_pipeline cache..report, then psi labels and the
# psi page for v402g and v402s, and the v402s gallery. Submit from the repo root.
# Usage: night3_batch.sh <day_obs> <workers>
set -euo pipefail
night=$1; workers=$2
# Slurm runs a spooled copy of this script, so under Slurm the repo root is the submit
# directory; submit from the repo root.
repo=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$repo"
study=$repo/aos/giant_donut_pupil
test -f "$study/code/night2_pipeline.py"
export OMP_NUM_THREADS=1 EFDAUTH=$HOME/.lsst/efdauth.json
code=aos/giant_donut_pupil/code
run=20261118_night2_$night
python $code/night2_pipeline.py --night "$night" --workers "$workers" \
    --steps cache,select,sample,stamps,fit,vanes,report
for fd in v402g v402s; do
    python $code/psi_labels.py --run "$run" --fit-dir $fd --workers "$workers"
    lab=$(basename "$(ls -td "$study"/output/*_psi_labels_${run}_${fd} | head -1)")
    python $code/psi_edges_report.py --run "$run" --fit-dir $fd --labels "$lab"
done
python $code/psi_gallery.py --run "$run" --fit-dir v402s --labels "$lab"
echo "night3 $night done"
