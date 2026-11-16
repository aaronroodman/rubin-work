#!/bin/bash
#SBATCH --job-name=fsm_miw
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fsm_miw_%j.log
#
# Danish 1.3 MIW on the Danish-fitted sensitivity matrices (fitted_smatrix product), the
# nominal 50/34 scheme over the nine rotator bins. One variant per job, given as
# arguments VARIANT FAM_BUILD. No build is set current.
#
# Submit from an s3df login node, from the aos/ dir. --export=NONE keeps the
# submitting shell's stack out of the job (an inherited older stack hangs setup.sh):
#   cd ~/notebooks/rubin-work/aos && sbatch --export=NONE run_fsm_miw.sbatch \
#       d13s-A_50_34_i_fsm_sparse_no22 20261106
#   cd ~/notebooks/rubin-work/aos && sbatch --export=NONE run_fsm_miw.sbatch \
#       d13-A_50_34_i_fsm_dense21 20261028b
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/fsm_miw_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
B=${FSM_BUILD:-$(date +%Y%m%d)}
cd ..

smk() {
    echo "== $(date)  $*"
    snakemake -s "rubinwork/products/$1/Snakefile" --config "${@:2}" \
        -c"$N" --resources mem_mb=120000 --rerun-incomplete -q
}

smk miw variant="$1" build="$B" fam_build="$2"
echo "== $(date)  done"
