#!/bin/bash
#SBATCH --job-name=sim_donut_fits
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=aos/logs/sim_donut_fits_%j.log
#
# Build one sim_donut_fits variant on 16 CPUs (about 2.5 CPU hours for v4_speed_sparse,
# about 4 times that unbinned).
#
# Submit from an s3df login node, from the repository root:
#   cd ~/notebooks/rubin-work && FIT_VARIANT=v4_speed_sparse sbatch rubinwork/products/sim_donut_fits/run_build.sbatch
# Variant and build: FIT_VARIANT (default v4_speed_sparse), FIT_BUILD (default today);
# FIT_OVERWRITE=1 replaces a build nothing has read.
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
V=${FIT_VARIANT:-v4_speed_sparse}
B=${FIT_BUILD:-$(date +%Y%m%d)}
OW=""; [ "${FIT_OVERWRITE:-0}" = 1 ] && OW="--overwrite"
echo "== $(date)  sim_donut_fits variant=$V build=$B jobs=$N $OW"
python -m rubinwork.products.sim_donut_fits.builders.build --variant "$V" --build "$B" --jobs "$N" $OW
echo "== $(date)  done"
