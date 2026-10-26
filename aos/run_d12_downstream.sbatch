#!/bin/bash
#SBATCH --job-name=d12_downstream
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d12_downstream_%j.log
#
# Danish 1.2 products downstream of fam_tables, rebuilt from the current fam_tables build
# (loose visit_quality_pass; bad_fit = more than 5 |DZ| > 2.0 um), in dependency order:
#   1. miw d12-A_50_34_i, then d12-A_50_34_i_5rot (reuses the first one's grids)
#   2. dof_lut for both, coadds and bounce_tables for the MIW variants they had
# Every product's cuts are as before. No build is set current.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_d12_downstream.sbatch
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/d12_downstream_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
B=${D12_BUILD:-20261025}
cd ..

smk() {
    echo "== $(date)  $*"
    snakemake -s "rubinwork/products/$1/Snakefile" --config "${@:2}" \
        -c"$N" --resources mem_mb=120000 -q
}

smk miw variant=d12-A_50_34_i build="$B"
smk miw variant=d12-A_50_34_i_5rot build="$B" src_build="$B"
for v in d12-A_50_34_i d12-A_50_34_i_5rot; do
    smk dof_lut variant="$v" build="$B" miw_build="$B"
    smk bounce_tables variant="$v" build="$B" miw_build="$B"
done
OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 \
    smk coadds variant=d12-A_50_34_i_5rot build="$B" miw_build="$B"
echo "== $(date)  done"
