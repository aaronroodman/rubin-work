#!/bin/bash
#SBATCH --job-name=sim_donut_fits
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=aos/logs/sim_donut_fits_%A_%a.log
#
# Build one sim_donut_fits variant on 16 CPUs (about 2.5 CPU hours for v4_speed_sparse,
# about 4 times that unbinned).
#
# Submit from an s3df login node, from the repository root:
#   cd ~/notebooks/rubin-work && FIT_VARIANT=v4_speed_sparse sbatch rubinwork/products/sim_donut_fits/run_build.sbatch
# Variant and build: FIT_VARIANT (default v4_speed_sparse), FIT_BUILD (default today);
# FIT_OVERWRITE=1 replaces a build nothing has read.
# DOF scan: submit with --array=0-21 and FIT_DOF_ARRAY=1, one partition per task, then
# one job with FIT_FINALIZE=1 (and SET_CURRENT=1 to move current) to merge them.
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
ARGS=(--variant "$V" --build "$B")
if [ "${FIT_FINALIZE:-0}" = 1 ]; then
    echo "== $(date)  sim_donut_fits variant=$V build=$B finalize"
    python -m rubinwork.products.sim_donut_fits.builders.build "${ARGS[@]}" --finalize
    if [ "${SET_CURRENT:-0}" = 1 ]; then
        python -m rubinwork.products.catalog set-current sim_donut_fits "$V" "$B"
    fi
else
    [ "${FIT_DOF_ARRAY:-0}" = 1 ] && ARGS+=(--dof-part "$SLURM_ARRAY_TASK_ID")
    echo "== $(date)  sim_donut_fits variant=$V build=$B jobs=$N $OW ${ARGS[*]}"
    python -m rubinwork.products.sim_donut_fits.builders.build "${ARGS[@]}" --jobs "$N" $OW
fi
echo "== $(date)  done"
