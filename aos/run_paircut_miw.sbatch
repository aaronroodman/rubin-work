#!/bin/bash
#SBATCH --job-name=paircut_miw
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/paircut_miw_%j.log
#
# A fam_tables build without the donuts whose intra and extra fits disagree (15 nMAD,
# per visit; refit_build --pair-cut-nmad), its DZ fits, then one MIW on it. Arguments:
#   FAM_VARIANT FAM_SOURCE FAM_BUILD MIW_VARIANT MIW_BUILD
# No build is set current.
#
# Submit from an s3df login node, from the aos/ dir, with --export=NONE:
#   cd ~/notebooks/rubin-work/aos && sbatch --export=NONE run_paircut_miw.sbatch \
#       d13-sparse 20261106 20261116 d13s-A_50_34_i_fsm_sparse_no22 20261116b
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/paircut_miw_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
cd ..

echo "== $(date)  fam_tables $1@$3 from @$2, pair cut 15 nMAD"
python -m rubinwork.products.fam_tables.builders.refit_build \
    --variant "$1" --source "$2" --build "$3" --pair-cut-nmad 15

echo "== $(date)  miw $4@$5 on fam_tables $1@$3"
snakemake -s rubinwork/products/miw/Snakefile \
    --config variant="$4" build="$5" fam_build="$3" \
    -c"$N" --resources mem_mb=120000 --rerun-incomplete -q
echo "== $(date)  done"
