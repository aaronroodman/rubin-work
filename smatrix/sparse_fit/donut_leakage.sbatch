#!/bin/bash
#SBATCH --job-name=sparse_leak
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=26
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/donut_leakage_%j.log
#
# Leakage matrix L and direct sparse sensitivity from Danish fits to model donuts:
# 26 configurations (8 corner half-chips, 9 science points x intra/extra), one per
# process, 50 DOF each.
#
# Submit from an s3df login node:
#   cd ~/notebooks/rubin-work/smatrix/sparse_fit && sbatch donut_leakage.sbatch
# Watch:
#   tail -f "$(ls -t ~/notebooks/rubin-work/smatrix/sparse_fit/logs/donut_leakage_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BATOID_RUBIN_DATA_DIR=/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data
cd code
python donut_leakage.py --jobs "${SLURM_CPUS_PER_TASK:-26}" --label "${LABEL:-donut_leakage}"
