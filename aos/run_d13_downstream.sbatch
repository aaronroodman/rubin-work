#!/bin/bash
#SBATCH --job-name=d13_downstream
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_downstream_%j.log
#
# Danish 1.3 v3.14 MIW from the current danish_1_3 fam_tables build (loose
# visit_quality_pass; bad_fit = more than 5 |DZ| > 2.0 um), then coadds against it:
#   1. miw d13-A_50_34_i, then d13-A_50_34_i_5rot (reuses the first one's grids)
#   2. coadds d13-A_50_34_i_5rot
# The other Danish 1.3 MIW variants keep their old-tree builds. No build is set current.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_d13_downstream.sbatch
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/d13_downstream_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
B=${D13_BUILD:-20261025}
cd ..

smk() {
    echo "== $(date)  $*"
    snakemake -s "rubinwork/products/$1/Snakefile" --config "${@:2}" \
        -c"$N" --resources mem_mb=120000 -q
}

smk miw variant=d13-A_50_34_i build="$B"
smk miw variant=d13-A_50_34_i_5rot build="$B" src_build="$B"
OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 \
    smk coadds variant=d13-A_50_34_i_5rot build="$B" miw_build="$B"
echo "== $(date)  done"
