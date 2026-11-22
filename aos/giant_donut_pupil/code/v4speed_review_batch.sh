#!/bin/bash
# The v4_speed pairs review after v4speed_fits.py: psi labels at 1.6-1.725 and 0.25-0.5 deg
# for the three sources, then pairs_review models and PDFs. Submit from the repo root.
# Usage: v4speed_review_batch.sh <workers>
set -euo pipefail
repo=${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../../.." && pwd)}
cd "$repo"
code=aos/giant_donut_pupil/code
w=${1:-100}
export OMP_NUM_THREADS=1
for run in 20261023_pupil6 20261118_night2_20261107 20261118_night2_20261108; do
    python $code/psi_labels.py --run $run --fit-dir v4speed --workers "$w" --band 1.6 1.725
    python $code/psi_labels.py --run $run --fit-dir v4speed --workers "$w" --band 0.25 0.5
done
python $code/pairs_review.py --mask-set v4speed --steps models --workers "$w"
out=$(basename "$(ls -td aos/giant_donut_pupil/output/*_pairs_review_v4speed | head -1)")
python $code/pairs_review.py --mask-set v4speed --steps pdfs --out "$out"
echo "v4speed review done $out"
