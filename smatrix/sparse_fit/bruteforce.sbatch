#!/bin/bash
#SBATCH --job-name=sparse_bf
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=48
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=logs/bruteforce_%j.log
#
# Brute-force ray-traced Danish sensitivity for one block of DOFs: all 940 field
# configurations (26 check points, 770 FAM grid, 144 CWFS grid), fit sets prod,
# prod_z22 and dense21, five amplitudes per DOF, donut images saved. DOFS (space
# separated DOF indices) and LABEL come from the environment; one job per block.
#
# Submit (block 0, DOF 0-4) from an s3df login node:
#   cd ~/notebooks/rubin-work/smatrix/sparse_fit && \
#     DOFS="0 1 2 3 4" LABEL=bf_dof00_04 sbatch --export=ALL bruteforce.sbatch
# Watch:
#   tail -f "$(ls -t ~/notebooks/rubin-work/smatrix/sparse_fit/logs/bruteforce_*.log | head -1)"
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
python donut_leakage.py --jobs "${SLURM_CPUS_PER_TASK:-48}" --label "$LABEL" \
    --config-set all --dofs $DOFS --truths rt --no-leakage --five-point --step-mult 2 \
    --save-images ${EXTRA_ARGS:-}
