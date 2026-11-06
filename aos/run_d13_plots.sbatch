#!/bin/bash
#SBATCH --job-name=d13_plots
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_plots_%j.log
#
# dzfit plots and the k=1..3 and k=1..6 residual movies for the current fam_tables
# danish_1_3 build, into dzfit/output/danish_1_3/.  The fit-parameter and trio PDFs are
# made once, by the z1toz6 run.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_d13_plots.sbatch
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/d13_plots_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
V=danish_1_3
P=$(readlink -f /sdf/group/rubin/u/roodman/LSST/rubin-work/products/fam_tables/$V/current)
echo "== $(date)  $V build $P"

# Separate directories: the JPEG frames and frame_list.txt have fixed names.
O=dzfit/output/$V
mkdir -p "$O/movies/z1toz3" "$O/movies/z1toz6"
python dzfit/code/run_dz_plots.py "$P/donuts.parquet" "$P/fits.parquet" \
    --coord-sys OCS --movie-prefix z1toz6 --movie-name residuals_z1toz6.mp4 \
    --output-dir "$O/movies/z1toz6" > "logs/d13_plots_z1toz6.log" 2>&1 &
pid6=$!
python dzfit/code/run_dz_plots.py "$P/donuts.parquet" "$P/fits.parquet" \
    --coord-sys OCS --movie-prefix z1toz3 --movie-name residuals_z1toz3.mp4 \
    --no-fit-params --no-trio \
    --output-dir "$O/movies/z1toz3" > "logs/d13_plots_z1toz3.log" 2>&1 &
pid3=$!
rc=0
wait $pid6 || rc=1
wait $pid3 || rc=1
tail -n 3 logs/d13_plots_z1toz6.log logs/d13_plots_z1toz3.log
ls -la "$O"/movies/*
echo "== $(date)  done, rc=$rc"
exit $rc
