#!/bin/bash
#SBATCH --job-name=d13_products
#SBATCH --partition=torino
#SBATCH --account=rubin:developers@torino
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/d13_products_%j.log
#
# Danish 1.3 v3.14 products, then the wfs_corner_compare builds, in one job:
#   1. fam_tables danish_1_3 (blitz FAM tables + DZ fits), 16 Butler-reading threads
#   2. cwfs_tables d13-blitz_unpaired (blitz corner tables paired to those FAM visits)
#   3. wfs_corner_compare for d12-refitWcs and d13-blitz_unpaired, 16 processes each
# Each step's build is set current when it completes.  ConsDB is reachable from torino;
# the EFD is not, which these builders do not need.
#
# Submit from an s3df login node, from the aos/ dir:
#   cd ~/notebooks/rubin-work/aos && sbatch run_danish13_products.sbatch
# Watch:  tail -f "$(ls -t ~/notebooks/rubin-work/aos/logs/d13_products_*.log | head -1)"
set -eo pipefail                 # NOT -u: the lsst/conda activate scripts use unbound vars
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
mkdir -p logs
set +e
source "$HOME/u/LSST/setup.sh"
set -e
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
N=${SLURM_CPUS_PER_TASK:-16}
B=${BUILD:-20261024}
P=/sdf/group/rubin/u/roodman/LSST/rubin-work/products
REPO=$(cd .. && pwd)

echo "== $(date)  fam_tables danish_1_3 @ $B"
python -m rubinwork.products.fam_tables.builders.run_blitz_mktable \
    --param-set danish_1_3 --build "$B" --fit --workers "$N" --overwrite
(cd "$REPO" && python -m rubinwork.products.catalog set-current fam_tables danish_1_3 "$B")

echo "== $(date)  cwfs_tables d13-blitz_unpaired @ $B"
(cd "$REPO" && snakemake -s rubinwork/products/cwfs_tables/Snakefile \
    --config variant=d13-blitz_unpaired build="$B" fam_build="$B" -c1 -q)
tail -4 "$P/cwfs_tables/d13-blitz_unpaired/$B/_work/logs/wfs_mktable.log"
(cd "$REPO" && python -m rubinwork.products.catalog set-current cwfs_tables d13-blitz_unpaired "$B")

for v in d12-refitWcs d13-blitz_unpaired; do
    echo "== $(date)  wfs_corner_compare $v @ $B"
    (cd "$REPO" && snakemake -s rubinwork/products/wfs_corner_compare/Snakefile \
        --config variant="$v" build="$B" workers="$N" -c"$N" -q)
    tail -2 "$P/wfs_corner_compare/$v/$B/_work/logs/build.log"
    (cd "$REPO" && python -m rubinwork.products.catalog set-current wfs_corner_compare "$v" "$B")
done
echo "== $(date)  done"
