#!/bin/bash
#SBATCH --job-name=sim_donut_visit_fits
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=aos/logs/sim_donut_visit_fits_%j.log
#
# Build one sim_donut_visit_fits variant on 16 CPUs (about 0.5 CPU s a CWFS stamp at production tolerances, 6400 stamps).
#
# Submit from an s3df login node, from the repository root:
#   cd ~/notebooks/rubin-work && sbatch rubinwork/products/sim_donut_visit_fits/run_build.sbatch
# Variant and build: VARIANT (default v4_speed_cwfs8_vk_dense_prod), BUILD (default today);
# OVERWRITE=1 replaces a build nothing has read; SET_CURRENT=1 moves current.
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
V=${VARIANT:-v4_speed_cwfs8_vk_dense_prod}
B=${BUILD:-$(date +%Y%m%d)}
OW=""; [ "${OVERWRITE:-0}" = 1 ] && OW="--overwrite"
echo "== $(date)  sim_donut_visit_fits variant=$V build=$B jobs=$N $OW"
python -m rubinwork.products.sim_donut_visit_fits.builders.build --variant "$V" --build "$B" --jobs "$N" $OW
if [ "${SET_CURRENT:-0}" = 1 ]; then
    python -m rubinwork.products.catalog set-current sim_donut_visit_fits "$V" "$B"
fi
echo "== $(date)  done"
