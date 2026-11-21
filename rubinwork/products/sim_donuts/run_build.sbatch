#!/bin/bash
#SBATCH --job-name=sim_donuts
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=aos/logs/sim_donuts_%j.log
#
# Build one sim_donuts variant on 16 CPUs (about 100 CPU minutes for v4_speed_mono).
#
# Submit from an s3df login node, from the repository root:
#   cd ~/notebooks/rubin-work && sbatch rubinwork/products/sim_donuts/run_build.sbatch
# Variant and build: SIM_VARIANT (default v4_speed_mono), SIM_BUILD (default today);
# SIM_OVERWRITE=1 replaces a build nothing has read.
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../../..}"
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
V=${SIM_VARIANT:-v4_speed_mono}
B=${SIM_BUILD:-$(date +%Y%m%d)}
OW=""; [ "${SIM_OVERWRITE:-0}" = 1 ] && OW="--overwrite"
echo "== $(date)  sim_donuts variant=$V build=$B jobs=$N $OW"
python -m rubinwork.products.sim_donuts.builders.build --variant "$V" --build "$B" --jobs "$N" $OW
echo "== $(date)  done"
