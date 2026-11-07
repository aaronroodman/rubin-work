#!/bin/bash
#SBATCH --job-name=smatrix_dense
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=17
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/build_dense_%j.log
#
# Rebuild the 50-DOF DZ sensitivity matrix (51 batoid DZ evaluations, one per
# process) and compare it with the ts_ofc matrix.
#
# Submit from an s3df login node:
#   cd ~/notebooks/rubin-work/smatrix/ofc_reproduction && sbatch build_dense.sbatch
# Watch:
#   tail -f "$(ls -t ~/notebooks/rubin-work/smatrix/ofc_reproduction/logs/build_dense_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BATOID_RUBIN_DATA_DIR=/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data
cd code
python build_dense.py --jobs "${SLURM_CPUS_PER_TASK:-17}"
