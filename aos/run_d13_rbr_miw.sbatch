#!/bin/bash
#SBATCH --job-name=d13_rbr_miw
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_rbr_miw_%j.log
#
# Danish 1.3 v3.14 MIW with Range-Bounded Recovery over all 50 DOF and 50 v-modes, from
# the current danish_1_3 fam_tables build:
#   miw d13-A_50_50_i_rbr, then d13-A_50_50_i_rbr_5rot (reuses the first one's grids).
# No build is set current.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_d13_rbr_miw.sbatch
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/d13_rbr_miw_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
B=${D13_BUILD:-$(date +%Y%m%d)}
cd ..

smk() {
    echo "== $(date)  $*"
    snakemake -s "rubinwork/products/$1/Snakefile" --config "${@:2}" \
        -c"$N" --resources mem_mb=120000 -q
}

smk miw variant=d13-A_50_50_i_rbr build="$B"
smk miw variant=d13-A_50_50_i_rbr_5rot build="$B" src_build="$B"
echo "== $(date)  done"
