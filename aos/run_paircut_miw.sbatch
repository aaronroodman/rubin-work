#!/bin/bash
#SBATCH --job-name=paircut_miw
#SBATCH --partition=milano
#SBATCH --account=rubin:developers@milano
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/paircut_miw_%j.log
#
# One step of a pair-cut rebuild; the first argument picks it:
#   fam FAM_VARIANT FAM_SOURCE FAM_BUILD   a fam_tables build without the donuts whose
#                                          intra and extra fits disagree (15 nMAD, per
#                                          visit; refit_build --pair-cut-nmad), and its
#                                          DZ fits
#   miw FAM_BUILD MIW_VARIANT MIW_BUILD    one MIW on that fam_tables build
# No build is set current.
#
# Submit from an s3df login node, from the aos/ dir, with --export=NONE; the miw step
# waits on the fam step with --dependency:
#   cd ~/notebooks/rubin-work/aos && J=$(sbatch --parsable --export=NONE \
#       run_paircut_miw.sbatch fam d13-sparse 20261106 20261116)
#   sbatch --export=NONE --dependency=afterok:$J run_paircut_miw.sbatch \
#       miw 20261116 d13s-A_50_34_i_fsm_sparse_no22 20261116b
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

case "$1" in
fam)
    echo "== $(date)  fam_tables $2@$4 from @$3, pair cut 15 nMAD"
    python -m rubinwork.products.fam_tables.builders.refit_build \
        --variant "$2" --source "$3" --build "$4" --pair-cut-nmad 15 ;;
miw)
    echo "== $(date)  miw $3@$4 on fam_tables build $2"
    snakemake -s rubinwork/products/miw/Snakefile \
        --config variant="$3" build="$4" fam_build="$2" \
        -c"$N" --resources mem_mb=120000 --rerun-incomplete -q ;;
*)
    echo "first argument must be fam or miw, got '$1'" >&2; exit 2 ;;
esac
echo "== $(date)  done"
