#!/bin/bash
#SBATCH --job-name=sparse_refit
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=logs/refit_%j.log
#
# Refit the saved brute-force donuts of one DOF block with Danish: cold start, plus a
# warm start from the nominal fit when the cold fit looks lost (refit_images.py); keeps
# the lower-cost fit and logs cost and fwhm per image. RUNS (space-separated run
# directory names) and LABEL come from the environment.
#
# Submit (block 0) from an s3df login node:
#   cd ~/notebooks/rubin-work/smatrix/sparse_fit && \
#     RUNS=20261110_bf_dof00_04 LABEL=refit_dof00_04 sbatch --export=ALL refit.sbatch
# Watch:
#   tail -f "$(ls -t ~/notebooks/rubin-work/smatrix/sparse_fit/logs/refit_*.log | head -1)"
#
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BATOID_RUBIN_DATA_DIR=/sdf/group/rubin/u/roodman/LSST/packages/batoid_rubin_data
export PYTHONPATH=/sdf/group/rubin/u/roodman/LSST/packages/danish_1.4.0:$PYTHONPATH
python -c "import danish; assert danish.__version__ == '1.4.0', danish.__version__"
cd code
python refit_images.py --jobs "${SLURM_CPUS_PER_TASK:-48}" --label "$LABEL" \
    --runs $RUNS --warm retry ${EXTRA_ARGS:-}
