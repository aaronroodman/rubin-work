#!/bin/bash
#SBATCH --job-name=d13_downstream
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/d13_downstream_%j.log
#
# Danish 1.3 MIW builds from the current fam_tables builds (loose visit_quality_pass;
# bad_fit = more than 5 |DZ| > 2.0 um), then coadds against the d13 v3.14 MIW:
#   1. miw d13-*  (danish_1_3, v3.14 pupil), d13t-* (danish_1_3_test), d13v1000-*
#      (danish_1_3_v1000): each parent, then the _5rot variants that reuse its grids
#   2. coadds d13-A_50_34_i_5rot
# MIW cuts are as before. No build is set current.
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

for pair in d13-A_50_34_i:d13-A_50_34_i_5rot \
            d13t-A_50_34_i:d13t-A_50_34_i_5rot \
            d13v1000-A_50_34_i:d13v1000-A_50_34_i_5rot \
            d13v1000-A_50_34_i_rbr:d13v1000-A_50_34_i_rbr_5rot \
            d13v1000-A_50_50_i_rbr:d13v1000-A_50_50_i_rbr_5rot \
            d13v1000-A_22_12_i:d13v1000-A_22_12_i_5rot; do
    smk miw variant="${pair%%:*}" build="$B"
    smk miw variant="${pair##*:}" build="$B" src_build="$B"
done
OMP_NUM_THREADS=4 NUMEXPR_MAX_THREADS=4 \
    smk coadds variant=d13-A_50_34_i_5rot build="$B" miw_build="$B"
echo "== $(date)  done"
