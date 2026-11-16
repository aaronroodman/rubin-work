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
# nominal 50/34 scheme over the nine rotator bins:
#   miw d13s-A_50_34_i_fsm_sparse_no22  from fam_tables d13-sparse@20261106
#   miw d13-A_50_34_i_fsm_dense21       from fam_tables danish_1_3@20261028b
# No build is set current.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_fsm_miw.sbatch
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
        -c"$N" --resources mem_mb=120000 -q
}

smk miw variant=d13s-A_50_34_i_fsm_sparse_no22 build="$B" fam_build=20261106
smk miw variant=d13-A_50_34_i_fsm_dense21 build="$B" fam_build=20261028b
echo "== $(date)  done"
