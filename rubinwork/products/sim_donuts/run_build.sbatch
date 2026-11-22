#!/bin/bash
#SBATCH --job-name=sim_donuts
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=aos/logs/sim_donuts_%A_%a.log
#
# Build one sim_donuts variant on 16 CPUs (about 100 CPU minutes for v4_speed_mono).
#
# Submit from an s3df login node, from the repository root:
#   cd ~/notebooks/rubin-work && sbatch rubinwork/products/sim_donuts/run_build.sbatch
# Variant and build: SIM_VARIANT (default v4_speed_mono), SIM_BUILD (default today);
# SIM_OVERWRITE=1 replaces a build nothing has read.
# DOF scan: submit with --array=0-21 and SIM_DOF_ARRAY=1, one partition per task, then
# one job with SIM_FINALIZE=1 (and SET_CURRENT=1 to move current) to merge them.
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
ARGS=(--variant "$V" --build "$B")
if [ "${SIM_FINALIZE:-0}" = 1 ]; then
    echo "== $(date)  sim_donuts variant=$V build=$B finalize"
    python -m rubinwork.products.sim_donuts.builders.build "${ARGS[@]}" --finalize
    if [ "${SET_CURRENT:-0}" = 1 ]; then
        python -m rubinwork.products.catalog set-current sim_donuts "$V" "$B"
    fi
else
    [ "${SIM_DOF_ARRAY:-0}" = 1 ] && ARGS+=(--dof-part "$SLURM_ARRAY_TASK_ID")
    echo "== $(date)  sim_donuts variant=$V build=$B jobs=$N $OW ${ARGS[*]}"
    python -m rubinwork.products.sim_donuts.builders.build "${ARGS[@]}" --jobs "$N" $OW
fi
echo "== $(date)  done"
