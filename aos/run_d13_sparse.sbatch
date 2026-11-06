#!/bin/bash
#SBATCH --job-name=d13_sparse
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_sparse_%j.log
#
# fam_tables d13-sparse (Danish 1.3 blitz unpaired, sparse Zernike fit): FAM tables and
# the k=1..3 and k=1..6 DZ fits, set current; then the dzfit plots and the residual
# movies of both fits into dzfit/output/d13-sparse/.  ConsDB is reachable from torino.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_d13_sparse.sbatch
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/d13_sparse_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
# NOT named BUILD: the stack exports BUILD=x86_64-conda-linux-gnu.
B=${D13S_BUILD:-$(date -u +%Y%m%d)}
V=d13-sparse
P=/sdf/group/rubin/u/roodman/LSST/rubin-work/products/fam_tables/$V/$B
REPO=$(cd .. && pwd)

echo "== $(date)  fam_tables $V @ $B"
python -m rubinwork.products.fam_tables.builders.run_blitz_mktable \
    --param-set "$V" --build "$B" --fit --workers "$N" --overwrite
(cd "$REPO" && python -m rubinwork.products.catalog set-current fam_tables "$V" "$B")

# Each movie renders in its own directory: the JPEG frames and frame_list.txt have fixed
# names, so a shared directory would race.
O=dzfit/output/$V
mkdir -p "$O/movies/z1toz3" "$O/movies/z1toz6"
echo "== $(date)  plots and z1toz6 movie"
python dzfit/code/run_dz_plots.py "$P/donuts.parquet" "$P/fits.parquet" \
    --coord-sys OCS --movie-prefix z1toz6 --movie-name residuals_z1toz6.mp4 \
    --output-dir "$O/movies/z1toz6" > "logs/d13_sparse_plots_z1toz6.log" 2>&1 &
pid6=$!
echo "== $(date)  z1toz3 movie"
python dzfit/code/run_dz_plots.py "$P/donuts.parquet" "$P/fits.parquet" \
    --coord-sys OCS --movie-prefix z1toz3 --movie-name residuals_z1toz3.mp4 \
    --output-dir "$O/movies/z1toz3" > "logs/d13_sparse_plots_z1toz3.log" 2>&1 &
pid3=$!
wait $pid6; wait $pid3
tail -n 3 logs/d13_sparse_plots_z1toz6.log logs/d13_sparse_plots_z1toz3.log
ls -la "$O"/movies/*
echo "== $(date)  done"
